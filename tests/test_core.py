"""Fast unit tests: python -m pytest -q"""
import numpy as np
from app.evaluate import query_metrics
from app.index import InvertedIndex
from app.preprocess import preprocess
from app.rankers import Bm25Ranker, TfidfRanker, rrf


def test_preprocess_stems_and_drops_stopwords():
    assert preprocess("The Lord of the Rings") == ["lord", "ring"]


def test_preprocess_keeps_all_stopword_titles():
    assert preprocess("It") == ["it"]


def _toy():
    return InvertedIndex([["dragon", "dragon", "king"], ["king", "queen"], ["queen", "garden"], ["dragon"]])


def test_inverted_index_postings():
    idx = _toy()
    ids, _ = idx.postings("dragon")
    assert sorted(ids.tolist()) == [0, 3]
    assert idx.postings("missing")[0].size == 0


def test_rankers_find_matching_docs():
    idx = _toy()
    for ranker in (TfidfRanker(idx), Bm25Ranker(idx)):
        ids, _ = ranker.rank("dragon", 3)
        assert set(ids.tolist()) == {0, 3}


def test_rrf_prefers_docs_in_both_lists():
    fused = rrf({"a": [1, 2, 3], "b": [3, 1, 9]}, k=60, top=3)
    assert [d for d, _, _ in fused][:2] == [1, 3]
    assert fused[0][2] == {"a": 1, "b": 2}


def test_metrics_perfect_and_empty():
    perfect = query_metrics([1, 2, 3] + [0] * 197, {1, 2, 3})
    assert perfect["MAP"] == 1.0 and perfect["nDCG@10"] == 1.0 and perfect["R@10"] == 1.0
    assert np.isclose(perfect["P@10"], 0.3)
    none = query_metrics([5, 6, 7], {1})
    assert none["MAP"] == 0 and none["MRR"] == 0
