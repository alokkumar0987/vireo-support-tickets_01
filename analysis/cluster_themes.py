"""
Unsupervised theme discovery: sentence embeddings + KMeans on customer messages.

Purpose: find complaint groups WITHOUT any labels, then compare them with the 19 themes
we designed (src/llm.THEMES) to see what the taxonomy misses or merges.
Runs locally (all-MiniLM-L6-v2, CPU); no API calls, no customer text leaves the machine.

    python analysis/cluster_themes.py            # k chosen by silhouette
    python analysis/cluster_themes.py --k 25

Writes analysis/cluster_report.md. Embeddings are cached in cache/embeddings_minilm.npy.
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.clean import clean_tickets_data
from src.llm import load_cache
from src.themes import attach_themes

EMB_PATH = os.path.join(ROOT, "cache", "embeddings_minilm.npy")
REPORT_PATH = os.path.join(ROOT, "analysis", "cluster_report.md")

# Boilerplate that would otherwise make clusters by writing style instead of by problem
BOILERPLATE = [
    r"\[ivr transcript\]",
    r"(dear|to the) (sir/madam|team|vireo customer care team|vireo)[,:]?",
    r"i am writ\w* (with|wiith) ref\w* to my order of .*?(placed on [^.]*)?\.",
    r"\b(hi|hello|hey|hii|helo|dear)\b( there| team| vireo| sir| ji)?[,!]*",
    # purchase preamble ("bought my X from vireo.in 2 weeks ago"): same in every ticket, says nothing about the problem
    r"\b(bought|ordered|got|purchased)\b[^.\n]{0,40}?(from (vireo\.in|flipkart|amazon)|\bago\b|\baround [a-z]+|\bon [a-z0-9 /-]+)",
    r"\bthis is regarding\b|\bregarding\b",
    # product names: the v1 run clustered by WHICH product a ticket mentions, not WHAT went wrong
    r"\b(pulse ?2?|pulss?e|nexa( fit)?( 2)?|nea|strata ?[23]?|orbit( mini| smart)?|airlite|airltie|fit band|"
    r"earbuds?|buds|headphones|smart ?watch|watch|wathc|speaker|speaekr|band|neckband|charging case|"
    r"spare case|cable|charger|65w|gan)\b",
    r"\b(product|order|purchased|issue|tried|expected|expectation):",
    r"\bvr\d+\b|\brma\d+\b|\b\d{1,2}[-/ ]\d{1,2}([-/ ]\d{2,4})?\b",
    r"\b(thanks|thank you|regards|waiting for your reply|please (help|advise|resolve)[^.]*|need this sorted[^.]*|"
    r"can (someone|you) fix this\??|what do i do now\??|how do i get this fixed\??|kindly look into it|"
    r"please call me on my registered number|anyone there\??|urgent|pls reply)\b|hello\?\?",
]
_BOILER_RX = [re.compile(p) for p in BOILERPLATE]


def clean_message(text):
    t = str(text or "").lower()
    for rx in _BOILER_RX:
        t = rx.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip() or "(empty)"


def embed(texts, path=EMB_PATH):
    import hashlib
    digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = path.replace(".npy", f"_{digest}.npy")  # new cleaning -> new cache file, never stale vectors
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, emb)
    return emb


def choose_k(emb, candidates=(12, 16, 20, 24, 28, 32), sample=3000, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    idx = np.random.RandomState(seed).choice(len(emb), min(sample, len(emb)), replace=False)
    scores = {}
    for k in candidates:
        labels = KMeans(k, n_init=5, random_state=seed).fit_predict(emb[idx])
        scores[k] = silhouette_score(emb[idx], labels)
        print(f"  k={k:<3} silhouette={scores[k]:.3f}")
    return max(scores, key=scores.get), scores


def top_terms(texts, labels, n=6):
    """c-TF-IDF: words that are frequent in a cluster but rare in the others."""
    from sklearn.feature_extraction.text import CountVectorizer
    docs = pd.Series(texts).groupby(labels).apply(" ".join)
    cv = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=3)
    counts = cv.fit_transform(docs).toarray().astype(float)
    tf = counts / counts.sum(axis=1, keepdims=True)
    idf = np.log(1 + counts.shape[0] / (1 + (counts > 0).sum(axis=0)))
    scores = tf * idf
    vocab = np.array(cv.get_feature_names_out())
    return {c: ", ".join(vocab[np.argsort(-scores[i])[:n]]) for i, c in enumerate(docs.index)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    from sklearn.cluster import KMeans

    df = attach_themes(clean_tickets_data()).reset_index(drop=True)
    df["msg_clean"] = df["customer_message"].map(clean_message)
    print(f"Embedding {len(df):,} cleaned customer messages (cached after first run)...")
    emb = embed(df["msg_clean"].tolist())

    if args.k:
        k = args.k
    else:
        print("Choosing k by silhouette on a 3,000-ticket sample:")
        k, _ = choose_k(emb)
    print(f"Clustering with k={k}")
    df["cluster"] = KMeans(k, n_init=10, random_state=0).fit_predict(emb)

    terms = top_terms(df["msg_clean"].tolist(), df["cluster"].values)
    llm = load_cache()
    df["llm_theme"] = df["ticket_id"].map(lambda t: llm.get(t, {}).get("theme"))

    rows, lines = [], [f"# Cluster report (k={k}, {len(df):,} tickets, all-MiniLM-L6-v2 + KMeans)\n",
                       "Purity = share of the cluster's tickets whose `theme` equals the cluster's most common theme.\n"]
    for c, g in df.groupby("cluster"):
        dom = g["theme"].value_counts()
        purity = dom.iloc[0] / len(g)
        bot = g["category"].value_counts()
        rows.append({"cluster": c, "size": len(g), "top_terms": terms[c], "main_theme": dom.index[0],
                     "purity": purity, "second_theme": dom.index[1] if len(dom) > 1 else "",
                     "main_bot_tag": bot.index[0], "bot_tag_share": bot.iloc[0] / len(g),
                     "repeat_rate": g["is_repeat_contact_30d"].mean()})
        lines.append(f"\n## Cluster {c}: {len(g):,} tickets | {terms[c]}")
        lines.append(f"- Our theme: **{dom.index[0]}** ({purity:.0%}); next: {dom.index[1] if len(dom) > 1 else '-'} "
                     f"| bot tag: {bot.index[0]} ({bot.iloc[0] / len(g):.0%}) | repeat rate {g['is_repeat_contact_30d'].mean():.0%}")
        # No example messages: this report is committed, and customer text stays out of the repo.
        # To read examples locally: g.sample(3)["customer_message"].

    summary = pd.DataFrame(rows).sort_values("size", ascending=False)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:2]) + "\n" + summary.to_markdown(index=False, floatfmt=".2f") + "\n" + "\n".join(lines[2:]))

    pd.set_option("display.width", 250, "display.max_colwidth", 55)
    print(summary[["cluster", "size", "top_terms", "main_theme", "purity", "main_bot_tag", "bot_tag_share", "repeat_rate"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nMean purity vs our themes: {np.average(summary['purity'], weights=summary['size']):.1%} | "
          f"vs bot tags: {np.average(summary['bot_tag_share'], weights=summary['size']):.1%}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
