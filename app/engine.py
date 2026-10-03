"""SearchEngine: builds, saves, loads and queries all rankers."""
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, HYBRID_COMPONENTS, METHODS, POOL, RRF_K, ST_MODEL
from .data import load_books
from .index import InvertedIndex, doc_tokens
from .rankers import Bm25Ranker, SemanticRanker, TfidfRanker, make_encoder, rrf


def semantic_text(row) -> str:
    """Text fed to the neural encoder."""
    year = f", {row.year}" if row.year else ""
    return f"{row.title} by {row.author}. Published by {row.publisher}{year}."


class SearchEngine:
    def __init__(self, books: pd.DataFrame, index: InvertedIndex, encoder,
                 embeddings: np.ndarray, meta: dict):
        self.books = books
        self.index = index
        self.encoder = encoder
        self.embeddings = embeddings
        self.meta = meta
        self.rankers = {
            "tfidf": TfidfRanker(index),
            "bm25": Bm25Ranker(index),
            "semantic": SemanticRanker(encoder, embeddings),
        }
        self._records = books.to_dict("records")

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, limit: int | None = None, use_transformer: bool = True,
              model_name: str = ST_MODEL) -> "SearchEngine":
        t0 = time.time()
        books = load_books(limit=limit)

        print("Preprocessing + building metadata inverted index ...")
        toks = [doc_tokens(r.title, r.author, r.publisher, r.year)
                for r in books.itertuples(index=False)]
        index = InvertedIndex(toks)
        print("  ", index.stats())

        print("Encoding documents for the semantic ranker ...")
        encoder = make_encoder(use_transformer, model_name)
        texts = [semantic_text(r) for r in books.itertuples(index=False)]
        embeddings = encoder.encode_docs(texts)

        meta = {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "encoder": encoder.name,
            "encoder_kind": encoder.kind,
            "build_seconds": round(time.time() - t0, 1),
            **index.stats(),
        }
        print(f"Build finished in {meta['build_seconds']}s using: {encoder.name}")
        return cls(books, index, encoder, embeddings, meta)

    # ----------------------------------------------------------- persistence
    def save(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        self.books.to_pickle(ARTIFACT_DIR / "books.pkl")
        np.save(ARTIFACT_DIR / "embeddings.npy", self.embeddings)
        with open(ARTIFACT_DIR / "index.pkl", "wb") as f:
            pickle.dump({"index": self.index, "encoder": self.encoder, "meta": self.meta}, f, protocol=4)
        print(f"Saved artifacts to {ARTIFACT_DIR}")

    @classmethod
    def exists(cls) -> bool:
        return all((ARTIFACT_DIR / n).exists() for n in ("books.pkl", "embeddings.npy", "index.pkl"))

    @classmethod
    def load(cls) -> "SearchEngine":
        if not cls.exists():
            raise FileNotFoundError("No index found. Run `python run.py build` first.")
        books = pd.read_pickle(ARTIFACT_DIR / "books.pkl")
        embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")
        with open(ARTIFACT_DIR / "index.pkl", "rb") as f:
            blob = pickle.load(f)
        return cls(books, blob["index"], blob["encoder"], embeddings, blob["meta"])

    # ---------------------------------------------------------------- search
    def rank(self, query: str, method: str, k: int = 10, pool: int = POOL):
        """Return [(doc_id, score, details)] for the chosen method."""
        if method in ("tfidf", "bm25", "semantic"):
            ids, scores = self.rankers[method].rank(query, k)
            return [(int(i), float(s), {}) for i, s in zip(ids, scores)]
        if method == "hybrid":
            lists = {}
            for name in HYBRID_COMPONENTS:
                ids, _ = self.rankers[name].rank(query, max(pool, k))
                lists[name] = ids.tolist()
            return [(d, s, r) for d, s, r in rrf(lists, k=RRF_K, top=k)]
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}")

    def search(self, query: str, method: str = "hybrid", k: int = 10) -> dict:
        query = (query or "").strip()
        t0 = time.perf_counter()
        hits = self.rank(query, method, k) if query else []
        ms = (time.perf_counter() - t0) * 1000
        results = []
        for pos, (d, score, details) in enumerate(hits, start=1):
            rec = self._records[d]
            results.append({
                "rank": pos, "doc_id": d, "isbn": rec["isbn"], "title": rec["title"],
                "author": rec["author"], "year": rec["year"] or None,
                "publisher": rec["publisher"], "image": rec["image"],
                "avg_rating": rec["avg_rating"], "n_ratings": rec["n_ratings"],
                "score": round(score, 5), "source_ranks": details,
            })
        return {"query": query, "method": method, "label": METHODS[method],
                "took_ms": round(ms, 2), "results": results}

    def info(self) -> dict:
        return {**self.meta, "books": len(self.books), "methods": METHODS,
                "hybrid_components": list(HYBRID_COMPONENTS), "rrf_k": RRF_K}
