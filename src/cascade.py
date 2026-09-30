"""
Hybrid theme classifier: a cheap local model for the easy tickets, the LLM only for the rest.

    ticket ──> TF-IDF + logistic regression (trained on cached LLM labels)
                 │ confident (p >= 0.6) AND familiar AND not "Other"  -> keep  (source 'tfidf', ₹0, stays local)
                 └ otherwise (low confidence / unfamiliar / "Other")   -> LLM if a key is set, else keyword rules

"Unfamiliar" = the ticket's TF-IDF vector is less similar to its nearest training ticket than
almost every training ticket is to its own nearest neighbour (2nd percentile). New kinds of
complaint look like this, and a classifier would otherwise force them into a known theme.
(Embedding-based novelty was tried in analysis/cluster_themes.py but needs torch, which is too
heavy for a clean-machine install; TF-IDF similarity does the same job with scikit-learn.)

The model trains in ~2s from cache/llm_labels.jsonl and is cached in cache/theme_model.joblib;
it retrains automatically when the labels change. Tickets in validation/check_100.csv are never
used for training.
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm import ROOT_DIR, PROMPT_VERSION, load_cache

MODEL_PATH = os.path.join(ROOT_DIR, "cache", "theme_model.joblib")
CHECK_SET_PATH = os.path.join(ROOT_DIR, "validation", "check_100.csv")

CONFIDENCE_THRESHOLD = 0.6
NOVELTY_PERCENTILE = 2.0
# Cap: on near-duplicate (templated) training data the percentile rule alone climbs towards 1.0
# and flags almost everything as novel. Real Vireo data gives ~0.32, so the cap only bites there.
NOVELTY_MAX_THRESHOLD = 0.35
MIN_EXAMPLES_PER_THEME = 3


def ticket_text(df):
    return (df["customer_message"].fillna("").astype(str) + " || " +
            df["agent_notes"].fillna("").astype(str)).str.lower()


def _build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_union
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),                 # words / phrases
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),  # typos, shorthand
    )
    return features, LogisticRegression(max_iter=2000, C=10)


def _max_similarity(X, X_train, chunk=2000, exclude_self=False):
    """Cosine similarity to the nearest training ticket (each TF-IDF block is L2-normalised, so /2)."""
    out = np.empty(X.shape[0])
    for start in range(0, X.shape[0], chunk):
        sims = (X[start:start + chunk] @ X_train.T).toarray() / 2.0
        if exclude_self:
            idx = np.arange(start, min(start + chunk, X.shape[0]))
            sims[idx - start, idx] = -1.0
        out[start:start + chunk] = sims.max(axis=1)
    return out


def load_check_ids(path=CHECK_SET_PATH):
    return set(pd.read_csv(path)["ticket_id"]) if os.path.exists(path) else set()


def train_model(df, labels, exclude_ids=None):
    """Fits TF-IDF + LR on tickets in df that have an LLM label. Returns a model dict."""
    exclude_ids = load_check_ids() if exclude_ids is None else set(exclude_ids)
    train = df[df["ticket_id"].isin(labels) & ~df["ticket_id"].isin(exclude_ids)].copy()
    train["y"] = train["ticket_id"].map(lambda t: labels[t]["theme"])
    counts = train["y"].value_counts()
    train = train[train["y"].isin(counts[counts >= MIN_EXAMPLES_PER_THEME].index)]
    if train["y"].nunique() < 2:
        raise ValueError(f"Need labelled tickets from at least 2 themes to train; have {len(train)} tickets.")

    features, clf = _build_pipeline()
    X = features.fit_transform(ticket_text(train))
    clf.fit(X, train["y"])
    nn_sim = _max_similarity(X, X, exclude_self=True)
    return {
        "features": features, "clf": clf, "X_train": X,
        "novelty_threshold": float(min(np.percentile(nn_sim, NOVELTY_PERCENTILE), NOVELTY_MAX_THRESHOLD)),
        "n_train": len(train), "themes": list(clf.classes_),
        "label_digest": _labels_digest(labels, exclude_ids),
    }


def _labels_digest(labels, exclude_ids):
    items = sorted((t, v["theme"]) for t, v in labels.items() if t not in exclude_ids)
    return hashlib.sha1(repr((PROMPT_VERSION, items)).encode()).hexdigest()[:16]


def get_model(df, labels=None, path=MODEL_PATH):
    """Cached model; retrains when the LLM labels (or the check set) change."""
    import joblib
    labels = load_cache() if labels is None else labels
    exclude_ids = load_check_ids()
    digest = _labels_digest(labels, exclude_ids)
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            if model.get("label_digest") == digest:
                return model
        except Exception:
            pass
    model = train_model(df, labels, exclude_ids)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return model


def predict(model, df):
    """Per-ticket TF-IDF theme, confidence, similarity and the routing decision."""
    X = model["features"].transform(ticket_text(df))
    proba = model["clf"].predict_proba(X)
    best = proba.argmax(axis=1)
    out = pd.DataFrame(index=df.index)
    out["tfidf_theme"] = np.array(model["themes"])[best]
    out["tfidf_confidence"] = proba[np.arange(len(df)), best]
    out["similarity"] = _max_similarity(X, model["X_train"])
    out["is_novel"] = out["similarity"] < model["novelty_threshold"]
    out["route_reason"] = np.select(
        [out["is_novel"], out["tfidf_confidence"] < CONFIDENCE_THRESHOLD, out["tfidf_theme"] == "Other"],
        ["novel", "low_confidence", "predicted_other"],
        default="confident",
    )
    return out


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from src.clean import clean_tickets_data
    df = clean_tickets_data()
    model = get_model(df)
    pred = predict(model, df)
    print(f"Trained on {model['n_train']:,} LLM-labelled tickets | {len(model['themes'])} themes | "
          f"novelty threshold {model['novelty_threshold']:.3f}")
    print("Routing of all tickets:")
    print(pred["route_reason"].value_counts(normalize=True).map("{:.1%}".format).to_string())
