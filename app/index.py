"""Metadata inverted index.

Each book contributes one pseudo-document built from its metadata fields
(title, author, publisher, year). Fields are boosted by repetition, e.g.
a title term counts 3x and an author term 2x (see config.FIELD_BOOSTS).

The index is a term x document sparse matrix in CSR layout, which *is* an
inverted index: row t holds the postings list of term t
(`indices` = doc ids, `data` = term frequencies).
"""
from collections import Counter

import numpy as np
from scipy import sparse

from .config import FIELD_BOOSTS
from .preprocess import preprocess


def doc_tokens(title, author, publisher, year) -> list[str]:
    """Preprocess every metadata field and apply field boosts."""
    fields = {
        "title": preprocess(title),
        "author": preprocess(author),
        "publisher": preprocess(publisher),
        "year": [str(year)] if year else [],
    }
    out: list[str] = []
    for name, toks in fields.items():
        out.extend(toks * FIELD_BOOSTS[name])
    return out


class InvertedIndex:
    def __init__(self, token_lists: list[list[str]]):
        n = len(token_lists)
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        doc_len = np.zeros(n, dtype=np.float32)
        for d, toks in enumerate(token_lists):
            doc_len[d] = len(toks)
            for term, tf in Counter(toks).items():
                tid = vocab.setdefault(term, len(vocab))
                rows.append(tid)
                cols.append(d)
                vals.append(tf)
        self.vocab = vocab
        self.n_docs = n
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n else 0.0
        self.matrix = sparse.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(len(vocab), n),
        )
        self.df = np.diff(self.matrix.indptr).astype(np.float32)

    def postings(self, term: str):
        """Return (doc_ids, term_freqs) for a term, or empty arrays."""
        tid = self.vocab.get(term)
        if tid is None:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        s, e = self.matrix.indptr[tid], self.matrix.indptr[tid + 1]
        return self.matrix.indices[s:e], self.matrix.data[s:e]

    def stats(self) -> dict:
        return {
            "documents": self.n_docs,
            "vocabulary": len(self.vocab),
            "postings": int(self.matrix.nnz),
            "avg_doc_length": round(self.avgdl, 2),
        }
