# Book Search using Book-Crossing

An end-to-end information retrieval system for book catalog search.
Project 15 · Catalog Retrieval.

| Requirement | Where it lives |
|---|---|
| Text preprocessing (cleaning, tokenization, stop-words, stemming) | `app/preprocess.py` |
| Metadata inverted indexing | `app/index.py` |
| TF-IDF, BM25, Sentence Transformer rankers | `app/rankers.py` |
| Hybrid model: Reciprocal Rank Fusion | `app/rankers.py` (`rrf`), `app/engine.py` |
| Search interface | `frontend/index.html`, served by `app/server.py` |
| Performance comparison table + Precision-Recall graph | `app/evaluate.py` -> `results/` and the **Evaluation** tab |
| Precision@10, Recall, MAP, nDCG | `app/evaluate.py` |

## Quick start (about 2 minutes, no downloads needed)

```bash
cd book-search
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py demo                                       # synthetic data -> build -> evaluate -> serve
```

Open **http://localhost:8000**.

`demo` uses a small *synthetic* catalog so you can see the whole system work immediately.
Do not present its numbers as Book-Crossing results.

## Run on the real Book-Crossing data

```bash
python run.py all            # download -> build -> evaluate -> serve
```

Or step by step:

```bash
python run.py setup          # downloads the dataset into data/raw/
python run.py build          # preprocess, index, encode (50k most-rated books)
python run.py evaluate       # metrics table + PR graph into results/
python run.py serve          # web UI + API on http://localhost:8000
```

If the download fails (the university server is sometimes down), get the files by hand and put them in `data/raw/`:
the original `BX-Books.csv` + `BX-Book-Ratings.csv`, or the Kaggle `Books.csv` + `Ratings.csv`
(https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Both layouts are auto-detected.

### Useful options

| Command | Effect |
|---|---|
| `python run.py build --limit 0` | Index all ~271k books (encoding takes much longer on CPU) |
| `python run.py build --limit 20000` | Smaller, faster corpus |
| `python run.py build --no-transformer` | Skip PyTorch; use the offline LSA fallback |
| `python run.py evaluate --queries 150` | 150 queries per type instead of 75 |
| `python run.py serve --port 9000` | Different port |

## Project layout

```
book-search/
├── run.py                 CLI: setup | build | evaluate | serve | all | demo
├── requirements.txt
├── app/
│   ├── config.py          all tunable settings (boosts, k1/b, RRF k, model)
│   ├── data.py            download, load, clean; synthetic sample generator
│   ├── preprocess.py      clean -> tokenize -> stop-words -> Porter stemmer
│   ├── index.py           metadata inverted index (sparse postings)
│   ├── rankers.py         TF-IDF, BM25, semantic encoder, RRF
│   ├── engine.py          build / save / load / search
│   ├── evaluate.py        query + qrels generation, metrics, reports
│   └── server.py          FastAPI backend (+ serves the frontend)
├── frontend/index.html    search UI, ranker comparison, evaluation dashboard
├── tests/test_core.py     unit tests (python -m pytest -q)
├── docs/
│   ├── USER_GUIDE.md      how to use every part
│   └── PRESENTATION_GUIDE.md   slides outline, demo script, Q&A, limitations
├── data/raw/              dataset CSVs (git-ignored)
├── artifacts/             built index + embeddings (git-ignored)
└── results/               metrics.json/csv/md, pr_curve.png, queries.json
```

## Documentation

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md): installation, UI walkthrough, CLI, REST API, troubleshooting
- [docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md): architecture, how each part works, a 5-minute demo script, likely questions and honest limitations
