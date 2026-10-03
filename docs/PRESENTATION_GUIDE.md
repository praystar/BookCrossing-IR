# Presentation Guide

## 1. One-sentence pitch
A search engine for the Book-Crossing catalog that compares classic lexical ranking (TF-IDF, BM25) with a neural Sentence Transformer and fuses them with Reciprocal Rank Fusion, evaluated with standard IR metrics.

## 2. Architecture

```mermaid
flowchart LR
  A[Book-Crossing CSVs<br/>270k books, 1.1M ratings] --> B[Clean + preprocess<br/>tokenize, stop-words, stem]
  B --> C[Metadata inverted index<br/>title x3, author x2, publisher, year]
  B --> D[Sentence Transformer<br/>MiniLM embeddings]
  C --> E[TF-IDF]
  C --> F[BM25]
  D --> G[Semantic cosine]
  F --> H[RRF fusion]
  G --> H
  E --> I[FastAPI /api/search]
  F --> I
  G --> I
  H --> I
  I --> J[Web UI]
  I --> K[Evaluation<br/>P@10, Recall, MAP, nDCG, PR curve]
```

## 3. What each part does (talking points)

**Data.** Book-Crossing has 271k books with title, author, year, publisher and cover URL, plus 1.1M explicit ratings. There are no text descriptions, so the "metadata description" of a book is its title + author + publisher + year. Ratings are used only for popularity (we keep the 50k most-rated books so the demo runs on a laptop).

**Preprocessing.** HTML entities and accents are normalised, text is lowercased and tokenized, English stop-words are removed, and the Porter stemmer reduces words to roots (*rings* -> *ring*). We use NLTK's classic stop list rather than scikit-learn's because the latter deletes plausible title words like *fire* and *system*. If a title is all stop-words ("It") it is kept as is.

**Metadata inverted index.** A sparse term-by-book matrix; each row is the postings list of one term. Fields are weighted by repetition (title x3, author x2) so a title match outranks a publisher match.

**TF-IDF.** `(1 + log tf) x idf`, cosine-normalised documents.
**BM25.** Term-frequency saturation (k1 = 1.5) and length normalisation (b = 0.75). Because records are short, length normalisation matters less than in web search.
**Semantic.** `all-MiniLM-L6-v2` embeds "Title by Author. Published by Publisher, Year." and the query; cosine similarity ranks books. Handles typos and loosely related wording.
**Hybrid (RRF).** `score(d) = sum over lists of 1 / (60 + rank)`. Needs no score normalisation, which is the reason to prefer it over weighted score averaging when mixing BM25 scores and cosine similarities.

## 4. Evaluation design (say this out loud, it is your credibility)

Book-Crossing is a recommendation dataset and has **no search queries or relevance judgments**. We therefore build a test collection from catalog metadata with a fixed random seed (42):

| Query type | Query | Relevant books |
|---|---|---|
| author | an author's name | all books by that author |
| title | 2-3 consecutive words from a title | all books whose title contains the phrase |
| title_author | 1-2 title words + author surname | that author's books with those words |
| typo | one of the above with a typing mistake | same as the clean query |

Defaults: 75 queries per type = 300 queries. Retrieval depth 200 per method.

Metrics: **Precision@10**, **Recall@10 / @100**, **MAP**, **nDCG@10** (binary gains), MRR, latency. The PR graph is the 11-point interpolated curve averaged over queries.

## 5. Suggested 5-minute demo

1. **Search tab, Hybrid.** Query `stephen king`. Point out the `BM25 #n / Semantic #n` chips.
2. **Switch to TF-IDF, then BM25.** Same author query, similar results: lexical methods are strong when words match exactly.
3. **Compare tab.** Query `tolkein` (misspelled). TF-IDF and BM25 return nothing or junk; Semantic and Hybrid still find the author. Hover a book to show it moving between columns.
4. **Compare tab, topic query.** `scary haunted house`: semantic similarity finds horror titles even without the exact words.
5. **Evaluation tab.** Walk the table (best per column is marked), then the PR graph, then switch the query-type dropdown to **typo** to show where hybrid wins.
6. **Close** with the limitations below.

## 6. What to say about the results

Read your own numbers rather than assuming. Typical, expected patterns (verify against your `results/metrics.md`):

- BM25 >= TF-IDF on most lexical query types.
- Lexical rankers are strongest on **title** and **author** queries because the judgments are word-based.
- Semantic is clearly better on **typo** queries.
- Hybrid is usually the most robust across all types even when not the top on any single one.

If hybrid does not win everywhere, that is a legitimate finding: say which query types lexical wins and why.

## 7. Honest limitations (examiners like hearing these first)

1. **Synthetic judgments.** Relevance comes from metadata rules, not human labels, so lexical methods are favoured on word-based queries. Real user studies or TREC-style pooling would be the next step.
2. **No descriptions.** Semantic models shine on descriptions or reviews; with only titles and authors they have little text to work with.
3. **Subset.** The default corpus is the 50k most-rated books. Use `--limit 0` for the whole catalog.
4. **Ratings are unused in ranking.** They are shown in the UI but do not influence scores. Popularity priors or collaborative filtering are natural extensions.
5. **P@10 ceiling.** Queries with fewer than 10 relevant books cannot reach P@10 = 1.0.
6. **Demo mode uses synthetic data.** Never quote `python run.py demo` numbers as Book-Crossing results.
7. If the UI shows the *LSA fallback* banner, the semantic results are not from a Sentence Transformer.

## 8. Likely questions

**Why RRF and not a weighted sum?** BM25 scores are unbounded and cosine scores lie in [-1, 1]; RRF avoids calibrating them by using ranks only. `k = 60` is the value from the original paper and damps the dominance of the top few ranks.

**Why does stemming matter?** It lets *dragons* match *dragon*, shrinking the vocabulary and improving recall, at the risk of occasional over-stemming.

**Why not fine-tune the transformer?** There are no labelled query-book pairs. With them, fine-tuning (or a cross-encoder reranker) would be the next improvement.

**How does it scale?** Lexical search touches only the postings of query terms (sub-millisecond here). Semantic search is a matrix-vector product over all embeddings; for millions of books use an ANN index such as FAISS.

**What are the Book-Crossing quirks?** Duplicate editions with different ISBNs, missing or zero publication years, HTML entities in text, and some malformed CSV rows. The loader handles all four.

**Future work.** Add Open Library descriptions, cross-encoder reranking, query spelling correction, popularity/rating priors, user-personalised recommendations from the ratings matrix.

## 9. Slide outline (7 slides)

1. Problem and dataset
2. Pipeline architecture (the diagram above)
3. Preprocessing and the metadata inverted index
4. Rankers: TF-IDF, BM25, Sentence Transformer
5. Hybrid RRF (formula + a worked example of two lists merging)
6. Evaluation setup and results (table + PR graph from `results/`)
7. Demo, limitations, future work
