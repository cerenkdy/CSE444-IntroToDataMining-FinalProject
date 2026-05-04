"""
CSE444 Introduction to Data Mining
Ceren Karadayi - 2020080829
Journal Finder - Computer Science Journal Recommender

Loads articles from CompSciencePub.sqlite, builds TF-IDF features,
trains four classifiers, and returns the top-5 journals for a given
abstract. Also runs K-Means clustering for topic discovery.
"""

import re
import sqlite3
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

# Data loading

def load_data(db_path):
    """Load articles, abstracts, keywords and subjects into a DataFrame."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            r.AcademicRecordID  AS id,
            p.Name              AS journal,
            a.AbstractText      AS abstract
        FROM AcademicRecord r
        JOIN Publication p              ON p.PublicationID    = r.PublicationId
        JOIN AcademicRecordAbstract a   ON a.AcademicRecordId = r.AcademicRecordID
    """
    df = pd.read_sql(query, conn)

    # Author keywords
    kw = pd.read_sql("""
        SELECT rk.AcademicRecordId AS id, k.Name AS kw
        FROM AcademicRecordKeyword rk
        JOIN AcademicKeyword k ON k.AcademicKeywordID = rk.AcademicKeywordId
    """, conn)
    kw = kw.groupby("id")["kw"].apply(lambda s: " ".join(s)).reset_index()

    # Keywords Plus
    kwp = pd.read_sql("""
        SELECT rk.AcademicRecordId AS id, k.Name AS kwp
        FROM AcademicRecordKeywordPlus rk
        JOIN AcademicKeywordPlus k ON k.AcademicKeywordPlusID = rk.AcademicKeywordPlusId
    """, conn)
    kwp = kwp.groupby("id")["kwp"].apply(lambda s: " ".join(s)).reset_index()

    # Subject categories
    sub = pd.read_sql("""
        SELECT rs.AcademicRecordId AS id, s.NameEn AS subject
        FROM AcademicRecordSubject rs
        JOIN AcademicSubject s ON s.AcademicSubjectID = rs.AcademicSubjectId
    """, conn)
    sub = sub.groupby("id")["subject"].apply(lambda s: " ".join(s)).reset_index()

    conn.close()

    df = df.merge(kw,  on="id", how="left")
    df = df.merge(kwp, on="id", how="left")
    df = df.merge(sub, on="id", how="left")
    df[["kw", "kwp", "subject"]] = df[["kw", "kwp", "subject"]].fillna("")

    # Combined text field used for vectorisation
    df["text"] = (df["abstract"].fillna("") + " " +
                  df["kw"]   + " " + df["kwp"] + " " + df["subject"])
    df["text"] = df["text"].apply(clean_text)

    # Drop empty abstracts and singleton journals (need >=2 for stratified split)
    df = df[df["abstract"].str.strip().astype(bool)].copy()
    counts = df["journal"].value_counts()
    df = df[df["journal"].isin(counts[counts >= 2].index)].reset_index(drop=True)
    return df

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Models

def build_tfidf(texts, max_features=20000):
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return vec, X


def train_centroid(X, y):
    """Build one centroid vector per journal (mean of its TF-IDF rows)."""
    journals = sorted(set(y))
    centroids = []
    for j in journals:
        rows = X[np.array(y) == j]
        c = np.asarray(rows.mean(axis=0)).ravel()
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        centroids.append(c)
    return np.array(journals), np.vstack(centroids)


def predict_centroid(query_vec, journals, centroids, k=5):
    sims = cosine_similarity(query_vec, centroids)[0]
    order = np.argsort(-sims)[:k]
    return [(journals[i], float(sims[i])) for i in order]


def train_knn(X, y):
    return X, np.array(y)


def predict_knn(query_vec, X_train, y_train, k_neighbors=25, k=5):
    sims = cosine_similarity(query_vec, X_train)[0]
    top_idx = np.argsort(-sims)[:k_neighbors]
    scores = {}
    for i in top_idx:
        j = y_train[i]
        scores[j] = scores.get(j, 0.0) + float(sims[i])
    ranked = sorted(scores.items(), key=lambda t: -t[1])[:k]
    return ranked


def predict_classifier(model, query_vec, k=5):
    probs = model.predict_proba(query_vec)[0]
    classes = model.classes_
    order = np.argsort(-probs)[:k]
    return [(classes[i], float(probs[i])) for i in order]

# Evaluation

def topk_accuracy(predictions, y_true, k_values=(1, 3, 5)):
    """Return a dict mapping k -> top-k accuracy."""
    out = {}
    for k in k_values:
        hits = 0
        for ranked, true in zip(predictions, y_true):
            top = [j for j, _ in ranked[:k]]
            if true in top:
                hits += 1
        out[f"top{k}"] = hits / len(y_true)
    # MRR
    rr = 0.0
    for ranked, true in zip(predictions, y_true):
        names = [j for j, _ in ranked]
        if true in names:
            rr += 1.0 / (names.index(true) + 1)
    out["mrr"] = rr / len(y_true)
    return out

# Production wrapper

class JournalFinder:
    """Trains the chosen model on the full corpus and exposes top_5()."""

    def __init__(self):
        self.vec = None
        self.model = None

    def fit(self, df):
        self.vec, X = build_tfidf(df["text"], max_features=15000)
        self.model = LogisticRegression(
            C=4.0, solver="saga", max_iter=200, tol=1e-3,
        )
        self.model.fit(X, df["journal"])
        return self

    def top_5(self, abstract):
        text = clean_text(abstract)
        q = self.vec.transform([text])
        return predict_classifier(self.model, q, k=5)
    
# Clustering

def kmeans_topics(X, k, vec, top_n=10, random_state=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    centers = km.cluster_centers_
    top_terms = []
    for i in range(k):
        idx = np.argsort(-centers[i])[:top_n]
        top_terms.append([terms[j] for j in idx])
    return labels, top_terms, km


def silhouette_scan(X, k_range, sample_size=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    Xs = X[sample_idx]
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
        labels = km.fit_predict(Xs)
        s = silhouette_score(Xs, labels, metric="cosine")
        scores.append(s)
    return list(k_range), scores
