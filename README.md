# Journal Finder for Computer Science Publications

CSE444 — Introduction to Data Mining, Final Project
Ceren Karadayı (20200808029)

A content-based recommender that takes the abstract of a paper and returns
the five most relevant journals from the *CompSciencePub* corpus
(23,034 articles across 428 Web-of-Science computer-science journals),
plus a K-Means topic-clustering view of the same corpus.

## Layout

```
project/
├── README.md
├── data/
│   └── CompSciencePub.sqlite        # input database (place yours here)
├── src/
│   ├── journal_finder.py            # the library: every function & class
│   └── try_recommender.py           # interactive terminal script
├── notebook/
│   └── journal_finder.ipynb         # end-to-end analysis with outputs
├── figures/                         # PNGs used by the notebook and report
├── results/                         # CSV exports written by the notebook
└── report/
    └── ceren_karadayi_report.pdf                   # compiled report
```

## What does `src/journal_finder.py` do?

This is the **library** that holds every reusable piece of the pipeline.
Nothing in this file runs by itself — it just defines functions and one
class that the notebook (and `try_recommender.py`) call. Splitting code
this way keeps the notebook readable and lets the same code be reused
from a script, the terminal, or a Python REPL without copy-pasting.

What's inside:

| Function / class | What it does |
|------------------|--------------|
| `load_data(db_path)`        | Joins the six SQLite tables and returns one DataFrame with one row per article (id, journal, abstract, kw, kwp, subject, text). |
| `clean_text(s)`             | Lower-cases the input and keeps only alphabetic tokens. |
| `build_tfidf(texts, ...)`   | Fits a TF-IDF vectoriser (unigrams + bigrams, sub-linear TF, top 15,000 features) on the texts. |
| `train_centroid(X, y)`      | Builds one mean TF-IDF vector per journal (the "Centroid + Cosine" model). |
| `predict_centroid(...)`     | Ranks journals for a query by cosine similarity to each centroid. |
| `predict_knn(...)`          | Similarity-weighted vote of the 25 nearest training articles. |
| `predict_classifier(...)`   | Ranks journals from the predicted probabilities of an sklearn classifier. |
| `topk_accuracy(...)`        | Computes Top-1 / Top-3 / Top-5 accuracy and Mean Reciprocal Rank. |
| `JournalFinder`             | The production wrapper class. Calling `finder.top_5(abstract)` returns a list of `(journal, score)` tuples. |
| `kmeans_topics(...)`        | Runs K-Means and returns labels + the top terms of each cluster. |
| `silhouette_scan(...)`      | Sweeps `k` and reports the silhouette score for each (used to pick the best `k`). |

## Three ways to use it

### 1) The notebook (recommended for the full report and figures)

```
jupyter notebook notebook/journal_finder.ipynb
```

Run every cell top to bottom. The notebook explains each step, trains the
four recommenders, compares them, picks the best, runs the K-Means
clustering, and at the end (Step 9) lets you paste your own abstract into
a `MY_ABSTRACT` variable and see the top 5 recommendations.

### 2) The terminal script (fastest way to just try the recommender)

```
python src/try_recommender.py
```

The script loads the corpus and trains the model once (this takes about a
minute), then enters an interactive prompt where you can paste an abstract
followed by an empty line and get the top 5 journals. Type `quit` to exit.
This is the simplest end-user experience and matches the I/O contract the
homework asks for.

Example session:

```
$ python src/try_recommender.py
Loading the corpus and training the recommender...
  loaded 23,034 articles across 428 journals
  ready.

============================================================
Journal Finder - paste an abstract, get the top 5 journals
============================================================
Type or paste your abstract, then press Enter twice (empty line) to submit.
Type 'quit' on a line by itself to exit.

Abstract:
We propose a transformer based language model for low resource
neural machine translation with cross lingual pre training.

Top 5 recommended journals:
------------------------------------------------------------
  1. COMPUTER SPEECH AND LANGUAGE  (score: 0.0261)
  2. NEURAL PROCESSING LETTERS  (score: 0.0231)
  3. NEURAL NETWORK WORLD  (score: 0.0134)
  4. JOURNAL OF ARTIFICIAL INTELLIGENCE RESEARCH  (score: 0.0110)
  5. ACM TRANSACTIONS ON ASIAN AND LOW-RESOURCE LANGUAGE INFORMATION PROCESSING  (score: 0.0104)
```

### 3) From Python code

```python
import sys
sys.path.insert(0, "src")

from journal_finder import load_data, JournalFinder

df = load_data("data/CompSciencePub.sqlite")
finder = JournalFinder().fit(df)

abstract = ("We propose a transformer based language model for low resource "
            "neural machine translation with cross lingual pre training.")

for journal, score in finder.top_5(abstract):
    print(f"{score:.3f}  {journal}")
```

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- The SQLite database `CompSciencePub.sqlite` placed in `data/`

Install:
```
pip install pandas numpy scikit-learn matplotlib
```

## Results

Stratified 80/20 hold-out, 428 journals:

| Model                | Top-1 | Top-3 | Top-5 | MRR   |
|----------------------|------:|------:|------:|------:|
| Centroid + Cosine    | 0.582 | 0.786 | 0.855 | 0.689 |
| KNN (k=25)           | 0.425 | 0.607 | 0.675 | 0.522 |
| Naive Bayes          | 0.437 | 0.612 | 0.683 | 0.530 |
| Logistic Regression  | **0.639** | **0.827** | **0.888** | **0.737** |

Random baseline for Top-5 is `5/428 ≈ 0.012`.

K-Means with `k = 14` (chosen by silhouette score on cosine distances)
recovers 14 sub-areas: telecommunications/electronics, semantic web,
information science, operations research, finite-element methods,
hardware/VLSI, software engineering and HCI, optimisation/evolutionary,
theoretical CS, cloud and security, computational biology, wireless
sensor networks, fuzzy decision making, and AI/neural learning.

## CSV outputs

When the notebook runs, it writes the following files to `results/`:

| file | what it contains |
|------|------------------|
| `eda.csv`                | basic dataset statistics |
| `results.csv`            | Top-1 / Top-3 / Top-5 / MRR for every model (Table I in the report) |
| `sample_predictions.csv` | top-5 predictions for the demo abstracts |
| `silhouette.csv`         | k vs silhouette score from the cluster-count scan |
| `clusters.csv`           | cluster id, size, top terms and top journals per cluster |
