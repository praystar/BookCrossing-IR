"""Ranking algorithms: TF-IDF, BM25, Semantic (Sentence Transformer) and
Reciprocal Rank Fusion."""
from collections import Counter

import numpy as np

from .config import BM25_B, BM25_K1, RRF_K, ST_MODEL
from .index import InvertedIndex
from .preprocess import preprocess


def top_k(scores: np.ndarray, k: int, positive_only: bool = False):
    """Indices and scores of the k best entries, best first."""
    k = min(k, len(scores))
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    if positive_only:
        idx = idx[scores[idx] > 0]
    return idx, scores[idx]


class _PostingsRanker:
    """Shared machinery: score = sum over query terms of precomputed
    per-posting weights, read straight from the inverted index."""

    def __init__(self, index: InvertedIndex):
        self.index = index
        self.counts = np.diff(index.matrix.indptr)

    def _accumulate(self, terms_with_weight, weights: np.ndarray) -> np.ndarray:
        m = self.index.matrix
        scores = np.zeros(self.index.n_docs, dtype=np.float32)
        for term, qw in terms_with_weight:
            tid = self.index.vocab.get(term)
            if tid is None:
                continue
            s, e = m.indptr[tid], m.indptr[tid + 1]
            scores[m.indices[s:e]] += qw * weights[s:e]
        return scores


class TfidfRanker(_PostingsRanker):
    """Cosine-normalised TF-IDF: w = (1 + log tf) * idf, idf = log((N+1)/(df+1)) + 1."""
    name = "tfidf"

    def __init__(self, index: InvertedIndex):
        super().__init__(index)
        m = index.matrix
        self.idf = (np.log((index.n_docs + 1) / (index.df + 1)) + 1).astype(np.float32)
        w = (1 + np.log(m.data)) * np.repeat(self.idf, self.counts)
        norms = np.sqrt(np.bincount(m.indices, weights=w ** 2, minlength=index.n_docs))
        norms[norms == 0] = 1
        self.weights = (w / norms[m.indices]).astype(np.float32)

    def rank(self, query: str, k: int):
        q = Counter(preprocess(query))
        terms = []
        for t, c in q.items():
            tid = self.index.vocab.get(t)
            if tid is not None:
                terms.append((t, (1 + np.log(c)) * self.idf[tid]))
        scores = self._accumulate(terms, self.weights)
        return top_k(scores, k, positive_only=True)


class Bm25Ranker(_PostingsRanker):
    """Okapi BM25 with Lucene-style non-negative IDF."""
    name = "bm25"

    def __init__(self, index: InvertedIndex, k1: float = BM25_K1, b: float = BM25_B):
        super().__init__(index)
        m = index.matrix
        n, df = index.n_docs, index.df
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = index.doc_len[m.indices]
        tf = m.data
        norm = k1 * (1 - b + b * dl / max(index.avgdl, 1e-9))
        self.weights = (np.repeat(self.idf, self.counts) * tf * (k1 + 1) / (tf + norm)).astype(np.float32)

    def rank(self, query: str, k: int):
        terms = [(t, 1.0) for t in set(preprocess(query))]
        scores = self._accumulate(terms, self.weights)
        return top_k(scores, k, positive_only=True)


# --------------------------------------------------------------------------
# Semantic ranker (dense embeddings)
# --------------------------------------------------------------------------
class SentenceTransformerEncoder:
    kind = "sentence-transformer"

    def __init__(self, model_name: str = ST_MODEL):
        self.model_name = model_name
        self._model = None
        self._load()  # fail early if the model cannot be loaded

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_docs(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        emb = self._load().encode(texts, batch_size=batch_size, show_progress_bar=True,
                                  normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        emb = self._load().encode(queries, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_model"] = None  # never pickle the network; reload by name
        return s


class LsaEncoder:
    """Offline fallback: char n-gram TF-IDF + truncated SVD (LSA).
    Not a neural model, but a dense, typo-tolerant representation, so the
    whole pipeline (incl. hybrid fusion) still runs without PyTorch."""
    kind = "lsa-fallback"
    name = "LSA fallback (char 3-4gram TF-IDF + SVD-128)"

    def __init__(self, dims: int = 128):
        self.dims = dims

    def encode_docs(self, texts: list[str], batch_size: int = 0) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                                   sublinear_tf=True, max_features=200_000, lowercase=True)
        x = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dims, x.shape[1] - 1), random_state=0)
        return self._norm(self.svd.fit_transform(x))

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._norm(self.svd.transform(self.vec.transform(queries)))

    @staticmethod
    def _norm(z: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(z, axis=1, keepdims=True)
        n[n == 0] = 1
        return (z / n).astype(np.float32)


def make_encoder(use_transformer: bool = True, model_name: str = ST_MODEL):
    if use_transformer:
        try:
            return SentenceTransformerEncoder(model_name)
        except Exception as exc:  # missing package, no internet, ...
            print(f"[warn] Sentence Transformer unavailable ({type(exc).__name__}: {exc}).")
            print("[warn] Falling back to LSA char-ngram embeddings. "
                  "Install `sentence-transformers` for the real neural model.")
    return LsaEncoder()


class SemanticRanker:
    name = "semantic"

    def __init__(self, encoder, embeddings: np.ndarray):
        self.encoder = encoder
        self.embeddings = embeddings

    def rank(self, query: str, k: int):
        q = self.encoder.encode_queries([query])[0]
        scores = self.embeddings @ q
        return top_k(scores, k)


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------
def rrf(rankings: dict[str, list[int]], k: int = RRF_K, top: int = 10):
    """Fuse ranked lists: score(d) = sum_r 1 / (k + rank_r(d)), rank from 1.

    Returns [(doc_id, fused_score, {ranker: rank})] best first.
    """
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, docs in rankings.items():
        for pos, d in enumerate(docs, start=1):
            d = int(d)
            fused[d] = fused.get(d, 0.0) + 1.0 / (k + pos)
            ranks.setdefault(d, {})[name] = pos
    order = sorted(fused, key=lambda d: (-fused[d], d))[:top]
    return [(d, fused[d], ranks[d]) for d in order]
