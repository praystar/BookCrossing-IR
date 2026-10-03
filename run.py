#!/usr/bin/env python
"""Command-line entry point.

  python run.py demo                 # offline synthetic data -> build -> evaluate -> serve
  python run.py setup                # download Book-Crossing
  python run.py build [--limit N]    # preprocess + index + embeddings
  python run.py evaluate             # metrics table + PR curve
  python run.py serve                # web UI + API on http://localhost:8000
  python run.py all                  # setup + build + evaluate + serve (real data)
"""
import argparse
import sys

MANUAL = """
Could not download the dataset automatically. Get it manually, then put the CSV
files into  data/raw/  and re-run:

  Option A (original):  http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip
                        -> unzip so BX-Books.csv and BX-Book-Ratings.csv are in data/raw/
  Option B (Kaggle):    https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
                        -> put Books.csv and Ratings.csv in data/raw/

Or try the offline demo:  python run.py demo
"""


def do_setup(a):
    from app import data
    if getattr(a, "sample", False):
        data.make_sample_data(getattr(a, "sample_size", 6000))
        return
    if data.raw_data_present():
        print("Dataset already present in data/raw/ - skipping download.")
        return
    try:
        data.download_bookcrossing()
    except Exception as exc:
        print(f"[error] {exc}")
        print(MANUAL)
        sys.exit(1)


def do_build(a):
    from app.config import DEFAULT_LIMIT, ST_MODEL
    from app.engine import SearchEngine
    limit = DEFAULT_LIMIT if a.limit is None else (a.limit or None)
    eng = SearchEngine.build(limit=limit, use_transformer=not a.no_transformer,
                             model_name=a.model or ST_MODEL)
    eng.save()


def do_evaluate(a):
    from app.engine import SearchEngine
    from app import evaluate as ev
    eng = SearchEngine.load()
    print("Generating queries and relevance judgments ...")
    qs = ev.build_queries(eng.books, n_per_type=a.queries, seed=a.seed)
    print(f"  {len(qs)} queries")
    report = ev.evaluate(eng, qs, depth=a.depth)
    ev.write_reports(report, qs)
    print("\n" + (ev.RESULTS_DIR / "metrics.md").read_text())


def do_serve(a):
    import uvicorn
    print(f"\n  Open http://localhost:{a.port}\n")
    uvicorn.run("app.server:app", host=a.host, port=a.port, log_level="info")


def main():
    p = argparse.ArgumentParser(description="Book Search IR project")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="download dataset (or --sample for synthetic data)")
    s.add_argument("--sample", action="store_true")
    s.add_argument("--sample-size", type=int, default=6000)

    b = sub.add_parser("build", help="build index and embeddings")
    b.add_argument("--limit", type=int, default=None, help="max books (default 50000, 0 = all)")
    b.add_argument("--no-transformer", action="store_true", help="use the LSA fallback")
    b.add_argument("--model", default=None, help="sentence-transformers model name")

    e = sub.add_parser("evaluate", help="run offline evaluation")
    e.add_argument("--queries", type=int, default=75, help="queries per type")
    e.add_argument("--depth", type=int, default=200)
    e.add_argument("--seed", type=int, default=42)

    v = sub.add_parser("serve", help="start web UI + API")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)

    for name, helptext in (("all", "setup + build + evaluate + serve (real data)"),
                           ("demo", "same, but with offline synthetic data")):
        x = sub.add_parser(name, help=helptext)
        x.add_argument("--limit", type=int, default=None)
        x.add_argument("--no-transformer", action="store_true")
        x.add_argument("--model", default=None)
        x.add_argument("--queries", type=int, default=75)
        x.add_argument("--depth", type=int, default=200)
        x.add_argument("--seed", type=int, default=42)
        x.add_argument("--host", default="127.0.0.1")
        x.add_argument("--port", type=int, default=8000)

    a = p.parse_args()
    if a.cmd == "setup":
        do_setup(a)
    elif a.cmd == "build":
        do_build(a)
    elif a.cmd == "evaluate":
        do_evaluate(a)
    elif a.cmd == "serve":
        do_serve(a)
    else:
        a.sample = a.cmd == "demo"
        a.sample_size = 6000
        do_setup(a)
        do_build(a)
        do_evaluate(a)
        do_serve(a)


if __name__ == "__main__":
    main()
