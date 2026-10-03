"""Load and clean the Book-Crossing catalog.

Works with either the original dump (BX-Books.csv, ';' separated) or the
Kaggle copy (Books.csv, ',' separated). Both are auto-detected.
"""
import csv
import html
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BX_URL, RAW_DIR


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_bookcrossing(dest: Path = RAW_DIR) -> None:
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "BX-CSV-Dump.zip"
    print(f"Downloading Book-Crossing from {BX_URL} ...")
    with requests.get(BX_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="")
    print("\nExtracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    print("Done:", [p.name for p in dest.glob("*.csv")])


def _find_files(raw_dir: Path = RAW_DIR):
    books = ratings = None
    for p in raw_dir.rglob("*.csv"):
        name = p.name.lower()
        if "rating" in name:
            ratings = p
        elif "book" in name:
            books = p
    return books, ratings


def raw_data_present() -> bool:
    b, r = _find_files()
    return b is not None


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        head = f.readline().decode("latin-1")
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(
        path, sep=sep, encoding="latin-1", quotechar='"', escapechar="\\",
        on_bad_lines="skip", low_memory=False, dtype=str,
    )


def load_books(limit: int | None = None) -> pd.DataFrame:
    """Return a clean catalog DataFrame with a dense `doc_id` column.

    Columns: doc_id, isbn, title, author, year, publisher, image,
             n_ratings, avg_rating
    """
    books_path, ratings_path = _find_files()
    if books_path is None:
        raise FileNotFoundError(
            f"No books CSV found in {RAW_DIR}. Run `python run.py setup` first."
        )
    print(f"Reading {books_path.name} ...")
    df = _read_csv(books_path)
    df = df.rename(columns={
        "ISBN": "isbn", "Book-Title": "title", "Book-Author": "author",
        "Year-Of-Publication": "year", "Publisher": "publisher",
        "Image-URL-M": "image",
    })
    for col in ("isbn", "title", "author", "year", "publisher", "image"):
        if col not in df.columns:
            df[col] = ""
    df = df[["isbn", "title", "author", "year", "publisher", "image"]].fillna("")

    # Clean text: unescape &amp; etc., trim whitespace.
    for col in ("title", "author", "publisher"):
        df[col] = df[col].map(lambda s: html.unescape(str(s)).strip())
    yr = pd.to_numeric(df["year"], errors="coerce")
    df["year"] = yr.where((yr >= 1400) & (yr <= 2026)).fillna(0).astype(int)

    df = df[df["title"].str.len() > 0]
    df = df.drop_duplicates(subset="isbn").reset_index(drop=True)

    # Popularity signals from explicit ratings (rating > 0).
    df["n_ratings"] = 0
    df["avg_rating"] = 0.0
    if ratings_path is not None:
        print(f"Reading {ratings_path.name} ...")
        r = _read_csv(ratings_path)
        r["Book-Rating"] = pd.to_numeric(r["Book-Rating"], errors="coerce")
        r = r[r["Book-Rating"] > 0]
        stats = r.groupby("ISBN")["Book-Rating"].agg(["count", "mean"])
        df = df.drop(columns=["n_ratings", "avg_rating"]).merge(
            stats.rename(columns={"count": "n_ratings", "mean": "avg_rating"}),
            left_on="isbn", right_index=True, how="left",
        )
        df["n_ratings"] = df["n_ratings"].fillna(0).astype(int)
        df["avg_rating"] = df["avg_rating"].fillna(0.0).round(2)

    if limit:
        df = df.sort_values("n_ratings", ascending=False, kind="stable").head(limit)
    df = df.reset_index(drop=True)
    df.insert(0, "doc_id", np.arange(len(df)))
    print(f"Catalog ready: {len(df):,} books")
    return df


# --------------------------------------------------------------------------
# Synthetic sample (offline demo / tests). NOT real data.
# --------------------------------------------------------------------------
_GENRES = {
    "mystery": (["silent", "dead", "midnight", "hidden", "poisoned", "missing", "crooked", "vanishing"],
                ["murder", "detective", "witness", "alibi", "inspector", "manor", "clue", "letter"],
                ["Raven House", "Blackwood Press", "Scotland Lane Books"]),
    "romance": (["tender", "summer", "forbidden", "golden", "secret", "wild", "gentle", "distant"],
                ["heart", "kiss", "wedding", "promise", "garden", "letters", "embrace", "love"],
                ["Rosewater Books", "Heartland Press", "Silk Thread Publishing"]),
    "fantasy": (["dragon", "ancient", "shattered", "crystal", "lost", "enchanted", "frozen", "burning"],
                ["kingdom", "sorcerer", "throne", "prophecy", "sword", "realm", "wizard", "crown"],
                ["Dragonfire Books", "Moonstone Press", "Aetherwood"]),
    "scifi": (["galactic", "quantum", "last", "synthetic", "orbital", "distant", "silent", "infinite"],
              ["starship", "colony", "android", "planet", "signal", "empire", "voyage", "machine"],
              ["Nova Publishing", "Orbit Line", "Helix Books"]),
    "horror": (["haunted", "cursed", "dark", "whispering", "hollow", "restless", "bleeding", "forgotten"],
               ["house", "ghost", "graveyard", "nightmare", "mansion", "spirit", "asylum", "woods"],
               ["Nightfall Press", "Crypt & Co.", "Grimoire Books"]),
    "cooking": (["easy", "classic", "rustic", "everyday", "seasonal", "family", "quick", "healthy"],
                ["kitchen", "recipes", "baking", "dinner", "pasta", "soup", "bread", "dessert"],
                ["Hearth & Table", "Saucepan Press", "Pantry Books"]),
    "history": (["rise", "fall", "secret", "forgotten", "great", "imperial", "revolutionary", "lost"],
                ["empire", "war", "republic", "dynasty", "revolution", "frontier", "battle", "century"],
                ["Chronicle House", "Meridian Academic", "Old World Press"]),
    "children": (["little", "brave", "sleepy", "magic", "funny", "tiny", "happy", "curious"],
                 ["bunny", "bear", "adventure", "friends", "puppy", "rainbow", "forest", "balloon"],
                 ["Sunny Day Books", "Pebble Press", "Storytime Kids"]),
    "biography": (["life", "journey", "untold", "remarkable", "private", "public", "early", "final"],
                  ["story", "memoir", "legacy", "years", "portrait", "letters", "voice", "path"],
                  ["Lantern Biography", "Portrait Press", "Vantage Books"]),
}
_FIRST = ["Alice", "Brandon", "Carla", "Daniel", "Elena", "Frank", "Grace", "Henry", "Irene", "Jacob",
          "Karen", "Louis", "Maria", "Nathan", "Olivia", "Peter", "Quinn", "Rachel", "Samuel", "Tanya",
          "Umar", "Vera", "Walter", "Ximena", "Yusuf", "Zoe"]
_LAST = ["Abbott", "Barlow", "Castellan", "Dunmore", "Everhart", "Fairbanks", "Galloway", "Hargrove",
         "Ingram", "Jessop", "Kingsley", "Lockhart", "Montague", "Newcombe", "Oakley", "Pemberton",
         "Quayle", "Redfern", "Sandoval", "Thackeray", "Underhill", "Vance", "Whitaker", "Yarrow"]
_TEMPLATES = ["The {a} {n}", "{n} of the {a} {m}", "A {a} {n}", "The {n}'s {m}", "{a} {n}, {a2} {m}",
              "The Last {n}", "Return of the {a} {n}"]


def make_sample_data(n_books: int = 6000, seed: int = 7, dest: Path = RAW_DIR) -> None:
    """Write a synthetic catalog in the original BX-CSV format."""
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    dest.mkdir(parents=True, exist_ok=True)

    names = [f"{f} {l}" for f in _FIRST for l in _LAST]
    rng.shuffle(names)
    author_pool = []
    genres = list(_GENRES)
    for i, name in enumerate(names):
        author_pool.append((name, genres[i % len(genres)]))

    rows, used_isbn = [], set()
    ai = 0
    while len(rows) < n_books:
        name, genre = author_pool[ai % len(author_pool)]
        ai += 1
        adjs, nouns, pubs = _GENRES[genre]
        for _ in range(rng.randint(3, 14)):
            t = rng.choice(_TEMPLATES).format(
                a=rng.choice(adjs).title(), a2=rng.choice(adjs).title(),
                n=rng.choice(nouns).title(), m=rng.choice(nouns).title())
            editions = 2 if rng.random() < 0.08 else 1  # same title, several ISBNs
            for _e in range(editions):
                while True:
                    isbn = "".join(rng.choice("0123456789") for _ in range(10))
                    if isbn not in used_isbn:
                        used_isbn.add(isbn)
                        break
                rows.append([isbn, t, name, rng.randint(1960, 2003), rng.choice(pubs), ""])
    rows = rows[:n_books]

    books = pd.DataFrame(rows, columns=["ISBN", "Book-Title", "Book-Author",
                                        "Year-Of-Publication", "Publisher", "Image-URL-M"])
    books.to_csv(dest / "BX-Books.csv", sep=";", index=False, quoting=csv.QUOTE_ALL,
                 encoding="latin-1", escapechar="\\")

    # Ratings with a long-tail popularity distribution
    pop = np.minimum(nprng.pareto(1.1, len(books)) * 2, 400).astype(int)
    r_isbn, r_user, r_val = [], [], []
    for isbn, n in zip(books["ISBN"], pop):
        for _ in range(int(n)):
            r_isbn.append(isbn)
            r_user.append(int(nprng.integers(1, 5000)))
            r_val.append(int(nprng.integers(0, 11)))
    pd.DataFrame({"User-ID": r_user, "ISBN": r_isbn, "Book-Rating": r_val}).to_csv(
        dest / "BX-Book-Ratings.csv", sep=";", index=False, quoting=csv.QUOTE_ALL,
        encoding="latin-1")
    print(f"Synthetic sample written to {dest} ({len(books):,} books, {len(r_isbn):,} ratings).")
