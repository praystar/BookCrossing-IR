"""Central configuration. Change values here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
FRONTEND_DIR = ROOT / "frontend"
for _d in (RAW_DIR, ARTIFACT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Original Book-Crossing dump (Cai-Nicolas Ziegler, Univ. of Freiburg)
BX_URL = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"

# Corpus size. The full dataset has ~271k books; the 50k most-rated books keep
# neural encoding to a few minutes on a laptop CPU. Use --limit 0 for all.
DEFAULT_LIMIT = 50_000

# Metadata inverted index: field boosts (a term in the title counts 3x).
FIELD_BOOSTS = {"title": 3, "author": 2, "publisher": 1, "year": 1}

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Neural model
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reciprocal Rank Fusion
RRF_K = 60
HYBRID_COMPONENTS = ("bm25", "semantic")
POOL = 100  # candidates taken from each ranker before fusing

METHODS = {
    "tfidf": "TF-IDF",
    "bm25": "BM25",
    "semantic": "Semantic",
    "hybrid": "Hybrid (RRF)",
}
