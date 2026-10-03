"""Offline evaluation.

Book-Crossing ships no queries or relevance judgments, so we build a
*metadata-derived* test collection (documented in docs/PRESENTATION_GUIDE.md):

  author        query = author name                        relevant = every book by that author
  title         query = 2-3 consecutive title words        relevant = every book whose title contains the phrase
  title_author  query = 1-2 title words + author surname   relevant = that author's books with those title words
  typo          author/title query with a typing mistake   relevant = same as the clean query

Metrics: Precision@10, Recall@10, Recall@100, MAP, nDCG@10, MRR, latency.
"""
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import METHODS, RESULTS_DIR
from .preprocess import STOP, clean

BAD_AUTHORS = {"unknown", "not applicable", "na", "n a", "anonymous", "various", "none", "unbekannt"}
RECALL_LEVELS = np.linspace(0, 1, 11)


@dataclass
class Query:
    qid: int
    qtype: str
    text: str
    relevant: set = field(default_factory=set)


# --------------------------------------------------------------------------
# Query generation
# --------------------------------------------------------------------------
def _corrupt(text: str, rng: random.Random) -> str:
    words = text.split()
    cand = [i for i, w in enumerate(words) if len(w) >= 5]
    if not cand:
        return text
    i = rng.choice(cand)
    w = words[i]
    p = rng.randrange(1, len(w) - 1)
    op = rng.choice(["delete", "swap", "substitute", "duplicate"])
    if op == "delete":
        w = w[:p] + w[p + 1:]
    elif op == "swap":
        w = w[:p] + w[p + 1] + w[p] + w[p + 2:]
    elif op == "substitute":
        w = w[:p] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[p + 1:]
    else:
        w = w[:p] + w[p] + w[p:]
    words[i] = w
    return " ".join(words)


def build_queries(books, n_per_type: int = 75, seed: int = 42) -> list[Query]:
    rng = random.Random(seed)
    norm_title = [clean(t) for t in books["title"]]
    norm_author = [clean(a) for a in books["author"]]

    by_author = defaultdict(list)
    for i, a in enumerate(norm_author):
        if len(a) >= 5 and a not in BAD_AUTHORS:
            by_author[a].append(i)
    authors = [a for a, ids in by_author.items() if 3 <= len(ids) <= 60]
    rng.shuffle(authors)

    padded_titles = [f" {t} " for t in norm_title]
    title_candidates = [i for i, t in enumerate(norm_title) if len(t.split()) >= 3]
    rng.shuffle(title_candidates)

    seen_text: set[str] = set()

    def author_q():
        for a in authors:
            if a in seen_text:
                continue
            seen_text.add(a)
            return a.title(), set(by_author[a])
        return None

    t_iter = iter(title_candidates)

    def title_q():
        for i in t_iter:
            toks = norm_title[i].split()
            n = rng.choice([2, 3])
            s = rng.randrange(0, len(toks) - n + 1)
            win = toks[s:s + n]
            if sum(w not in STOP for w in win) < 2:
                continue
            phrase = " ".join(win)
            if phrase in seen_text:
                continue
            rel = {j for j, t in enumerate(padded_titles) if f" {phrase} " in t}
            if 1 <= len(rel) <= 100:
                seen_text.add(phrase)
                return phrase, rel
        return None

    ta_iter = iter(rng.sample(range(len(books)), len(books)))

    def title_author_q():
        for i in ta_iter:
            a = norm_author[i]
            if a not in by_author or len(by_author[a]) > 60:
                continue
            words = [w for w in norm_title[i].split() if w not in STOP and len(w) >= 3]
            surname = a.split()[-1]
            if not words or len(surname) < 3:
                continue
            chosen = rng.sample(words, min(len(words), rng.choice([1, 2])))
            text = f"{' '.join(chosen)} {surname}"
            if text in seen_text:
                continue
            rel = {j for j in by_author[a] if all(w in norm_title[j].split() for w in chosen)}
            if rel:
                seen_text.add(text)
                return text, rel
        return None

    queries: list[Query] = []

    def collect(qtype, fn, n):
        got = 0
        while got < n:
            out = fn()
            if out is None:
                break
            text, rel = out
            queries.append(Query(len(queries), qtype, text, rel))
            got += 1

    collect("author", author_q, n_per_type)
    collect("title", title_q, n_per_type)
    collect("title_author", title_author_q, n_per_type)

    typo_n = 0
    attempts = 0
    while typo_n < n_per_type and attempts < n_per_type * 20:
        attempts += 1
        out = rng.choice([author_q, title_q])()
        if out is None:
            continue
        text, rel = out
        noisy = _corrupt(text, rng)
        if noisy != text:
            queries.append(Query(len(queries), "typo", noisy, rel))
            typo_n += 1
    return queries


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def query_metrics(ranked: list[int], rel: set) -> dict:
    hits = [1 if d in rel else 0 for d in ranked]
    n_rel = len(rel)

    ap, c = 0.0, 0
    prec_at_hit = []
    for i, h in enumerate(hits, start=1):
        if h:
            c += 1
            ap += c / i
            prec_at_hit.append((c / n_rel, c / i))
    ap /= n_rel

    dcg = sum(h / np.log2(i + 1) for i, h in enumerate(hits[:10], start=1))
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(n_rel, 10) + 1))
    first = next((i for i, h in enumerate(hits, start=1) if h), None)

    interp = []
    for lvl in RECALL_LEVELS:
        ps = [p for r, p in prec_at_hit if r >= lvl - 1e-12]
        interp.append(max(ps) if ps else 0.0)

    return {
        "P@10": sum(hits[:10]) / 10,
        "R@10": sum(hits[:10]) / n_rel,
        "R@100": sum(hits[:100]) / n_rel,
        "MAP": ap,
        "nDCG@10": dcg / idcg if idcg else 0.0,
        "MRR": 1 / first if first else 0.0,
        "_pr": interp,
    }


METRIC_ORDER = ["P@10", "R@10", "R@100", "MAP", "nDCG@10", "MRR"]


def evaluate(engine, queries: list[Query], depth: int = 200) -> dict:
    per = {m: [] for m in METHODS}
    latency = {m: [] for m in METHODS}
    for m in METHODS:
        print(f"  evaluating {METHODS[m]:<14} on {len(queries)} queries ...")
        for q in queries:
            t0 = time.perf_counter()
            hits = engine.rank(q.text, m, k=depth, pool=depth)
            latency[m].append((time.perf_counter() - t0) * 1000)
            res = query_metrics([d for d, _, _ in hits], q.relevant)
            res["qtype"] = q.qtype
            per[m].append(res)

    def agg(rows):
        out = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_ORDER}
        return out

    overall = {m: {**agg(rows), "latency_ms": float(np.mean(latency[m]))} for m, rows in per.items()}
    qtypes = sorted({q.qtype for q in queries})
    by_type = {
        t: {m: agg([r for r in per[m] if r["qtype"] == t]) for m in METHODS} for t in qtypes
    }
    pr = {m: np.mean([r["_pr"] for r in per[m]], axis=0).tolist() for m in METHODS}

    return {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_queries": len(queries),
            "queries_per_type": {t: sum(q.qtype == t for q in queries) for t in qtypes},
            "n_documents": len(engine.books),
            "encoder": engine.meta.get("encoder"),
            "encoder_kind": engine.meta.get("encoder_kind"),
            "depth": depth,
        },
        "methods": METHODS,
        "metric_order": METRIC_ORDER,
        "overall": overall,
        "by_type": by_type,
        "recall_levels": RECALL_LEVELS.tolist(),
        "pr_curve": pr,
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def write_reports(report: dict, queries: list[Query]) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    (RESULTS_DIR / "queries.json").write_text(json.dumps(
        [{"id": q.qid, "type": q.qtype, "query": q.text, "n_relevant": len(q.relevant)} for q in queries],
        indent=2))

    cols = METRIC_ORDER + ["latency_ms"]
    with open(RESULTS_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols)
        for m, row in report["overall"].items():
            w.writerow([METHODS[m]] + [round(row[c], 4) for c in cols])

    lines = ["| Method | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    best = {c: max(report["overall"][m][c] for m in METHODS) for c in METRIC_ORDER}
    for m, row in report["overall"].items():
        cells = []
        for c in cols:
            v = f"{row[c]:.4f}" if c != "latency_ms" else f"{row[c]:.1f}"
            if c in best and abs(row[c] - best[c]) < 1e-12:
                v = f"**{v}**"
            cells.append(v)
        lines.append(f"| {METHODS[m]} | " + " | ".join(cells) + " |")
    (RESULTS_DIR / "metrics.md").write_text("\n".join(lines) + "\n")

    colors = {"tfidf": "#94a3b8", "bm25": "#2563eb", "semantic": "#d97706", "hybrid": "#4338ca"}
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for m in METHODS:
        ax.plot(report["recall_levels"], report["pr_curve"][m], marker="o", ms=4,
                lw=2.4 if m == "hybrid" else 1.6, color=colors[m], label=METHODS[m])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Interpolated precision")
    ax.set_title("Precision-Recall curve (11-point, mean over queries)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curve.png")
    plt.close(fig)
    print(f"Reports written to {RESULTS_DIR}")
