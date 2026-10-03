# User Guide

## 1. Requirements

- Python 3.10 or newer
- About 2 GB free disk (PyTorch + model + dataset) and 4 GB RAM for 50k books
- Internet for the first run only (dataset + the `all-MiniLM-L6-v2` model, ~90 MB)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No GPU is needed. If installing `sentence-transformers` is a problem, delete that line from
`requirements.txt`; the project then uses an offline LSA fallback and shows a warning banner in the UI.

## 2. The pipeline in four commands

| Step | Command | What happens | Output |
|---|---|---|---|
| 1 | `python run.py setup` | Downloads and unzips Book-Crossing | `data/raw/*.csv` |
| 2 | `python run.py build` | Cleans metadata, builds the inverted index, encodes every book with the Sentence Transformer | `artifacts/` |
| 3 | `python run.py evaluate` | Generates test queries + relevance judgments, scores all four rankers | `results/` |
| 4 | `python run.py serve` | Starts the web app | http://localhost:8000 |

`python run.py all` does all four. `python run.py demo` does the same with synthetic data.
Steps 2 and 3 only need re-running if the data or settings change; `serve` just loads `artifacts/`.

Expected build time for 50,000 books on a laptop CPU: about 1 minute for the index plus 3 to 8 minutes for embeddings.

## 3. Using the web interface

### Search tab
1. Type a query (author, title words, or a topic) and press **Search**.
2. Choose the ranking method: **TF-IDF**, **BM25**, **Semantic** or **Hybrid (RRF)**. The result list reloads and the blue note under the buttons explains the method.
3. In Hybrid mode each result shows chips such as `BM25 #3` and `Semantic #1`, which are the positions it had in each list before fusion.
4. Matched query words are highlighted in the results.

Good queries to try: `stephen king`, `the lord of the rings`, `agatha christie murder`, `scary haunted house`, and a misspelling such as `tolkein`.

### Compare rankers tab
Runs the same query through all four methods in four columns. Hover a book to highlight where every ranker placed it and dim the rest. Use a typo query to show lexical rankers failing while Semantic and Hybrid recover.

### Evaluation tab
Shows the performance comparison table (best value per column is marked), the precision-recall graph, and a per-query-type breakdown. It reads `results/metrics.json`, so run `python run.py evaluate` first.

### How it works tab
One-page summary of the pipeline for the audience.

## 4. Evaluation outputs (`results/`)

| File | Content |
|---|---|
| `metrics.md` | Markdown table, paste into slides or a report |
| `metrics.csv` | Same table for Excel |
| `metrics.json` | Everything, including per-query-type scores and PR points (read by the UI) |
| `pr_curve.png` | High-resolution Precision-Recall graph for slides |
| `queries.json` | Every test query with its type and number of relevant books |

Metrics reported: Precision@10, Recall@10, Recall@100, MAP, nDCG@10, MRR, mean latency per query.

## 5. REST API

Base URL `http://localhost:8000`. Interactive docs at `/docs`.

| Endpoint | Description |
|---|---|
| `GET /api/search?q=...&method=hybrid&k=10` | One ranked list. `method` is `tfidf`, `bm25`, `semantic` or `hybrid` |
| `GET /api/compare?q=...&k=10` | All four methods for one query |
| `GET /api/info` | Corpus size, vocabulary size, encoder name |
| `GET /api/metrics` | Evaluation report |
| `GET /api/health` | `ok` or `no-index` |

```bash
curl "http://localhost:8000/api/search?q=agatha+christie&method=bm25&k=3"
```

## 6. Configuration (`app/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `DEFAULT_LIMIT` | 50000 | Books kept, chosen by rating count |
| `FIELD_BOOSTS` | title 3, author 2, publisher 1, year 1 | Field weighting in the inverted index |
| `BM25_K1`, `BM25_B` | 1.5, 0.75 | BM25 parameters |
| `ST_MODEL` | all-MiniLM-L6-v2 | Any sentence-transformers model name |
| `RRF_K` | 60 | RRF constant |
| `HYBRID_COMPONENTS` | bm25, semantic | Rankers fused by the hybrid model |

After changing boosts, BM25 values or the model, re-run `build` and `evaluate`.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not download the dataset` | Download manually (see README) and put the CSVs in `data/raw/` |
| UI says "Index not loaded" | Run `python run.py build`, then restart `serve` |
| Evaluation tab says no evaluation yet | Run `python run.py evaluate` |
| Warning: *Sentence Transformer unavailable* | `pip install sentence-transformers`; the first run needs internet to fetch the model. Rebuild afterwards |
| Build is slow or runs out of memory | `python run.py build --limit 20000` |
| Port already in use | `python run.py serve --port 9000` |
| Book covers are blank | Covers are loaded from Amazon image URLs in the dataset; many old ones no longer exist, a letter tile is shown instead |
| `pickle`/import errors after moving the folder | Run commands from the project root, then rebuild |
