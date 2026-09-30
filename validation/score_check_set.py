"""
Scores every classification method against the human labels in validation/check_100.csv.
Writes validation/results.md (the evidence for "how do you know it works, and how often is it not").

    python validation/score_check_set.py

Methods:
  bot_tag        intake bot category (counted correct if it is the natural bucket for the true theme)
  keyword        src/themes.py rules
  tfidf          src/cascade.py model alone (never trained on these tickets)
  llm            cached LLM label (prompt v3)
  hybrid_no_key  what a reviewer without an API key gets: confident TF-IDF, else keyword rules
  hybrid         production design: confident TF-IDF, else LLM
Accuracy comes with a 95% Wilson interval: with n=100, +/- several points is real uncertainty.
"""

import json
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.clean import clean_tickets_data
from src import cascade
from src.llm import THEMES, load_cache
from src.themes import attach_themes

CHECK_SET_PATH = os.path.join(ROOT, "validation", "check_100.csv")
RESULTS_PATH = os.path.join(ROOT, "validation", "results.md")

# The bot's 11 coarse categories -> the themes each one legitimately covers
BOT_TAG_COVERS = {
    "Delivery & Shipping": {"Delivery delayed / not delivered", "Address change / wrong address",
                            "Damaged in transit / dead on arrival", "Wrong item / variant received"},
    "Returns & Refunds": {"Return pickup not done", "Refund delayed / not received"},
    "Billing & Payments": {"Double charge / payment failed", "Coupon / discount / price", "Invoice / GST",
                           "Order cancellation"},
    "Connectivity": {"Pairing & connection drops"},
    "Charging & Battery": {"Battery drain / not charging"},
    "Audio Quality": {"Audio fault (one side, distortion, mic)"},
    "App & Firmware": {"App / firmware update failure"},
    "Warranty & Repair": {"Warranty claim / repair status", "Screen / touch / hardware fault"},
    "Account & Login": {"Account / OTP login"},
    "Product Enquiry": {"Product / compatibility question"},
    "Other": {"Other"},
}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


CHECK_SET_JSON_PATH = os.path.join(ROOT, "validation", "check_100.json")


def _read_check_file(path):
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            return pd.DataFrame(json.load(f)["tickets"])
    return pd.read_csv(path)


def _has_labels(frame):
    return frame["true_theme"].fillna("").astype(str).str.strip().ne("").any()


def load_human_labels(path=None):
    """Reads whichever of check_100.csv / check_100.json has been labelled (refuses if both have)."""
    if path is None:
        candidates = [p for p in (CHECK_SET_PATH, CHECK_SET_JSON_PATH)
                      if os.path.exists(p) and _has_labels(_read_check_file(p))]
        if len(candidates) > 1:
            sys.exit("Both check_100.csv and check_100.json contain labels. Keep one (clear the other).")
        path = candidates[0] if candidates else CHECK_SET_PATH
    print(f"Reading human labels from {os.path.basename(path)}")
    gold = _read_check_file(path)
    gold["true_theme"] = gold["true_theme"].fillna("").astype(str).str.strip()
    gold["true_theme"] = gold["true_theme"].replace("", "nan")
    labelled = gold[gold["true_theme"].isin(THEMES)].copy()
    bad = gold[gold["true_theme"].ne("nan") & ~gold["true_theme"].isin(THEMES)]
    return labelled, bad


def predictions(ids):
    df = clean_tickets_data()
    df = df[df["ticket_id"].isin(ids)].copy()
    llm_labels = load_cache()
    model = cascade.get_model(clean_tickets_data())
    pred = cascade.predict(model, df)
    no_key = attach_themes(df, use_llm=False, model=model)

    out = pd.DataFrame({"ticket_id": df["ticket_id"], "bot_tag": df["category"],
                        "keyword": no_key["keyword_theme"], "tfidf": pred["tfidf_theme"],
                        "route_reason": pred["route_reason"],
                        "llm": df["ticket_id"].map(lambda t: llm_labels.get(t, {}).get("theme")),
                        "llm_repeat": df["ticket_id"].map(lambda t: llm_labels.get(t, {}).get("is_repeat_claim")),
                        "keyword_repeat": no_key["is_repeat_claim"],
                        "hybrid_no_key": no_key["theme"]})
    confident = out["route_reason"] == "confident"
    out["hybrid"] = out["tfidf"].where(confident, out["llm"])
    return out


def main():
    gold, bad = load_human_labels()
    if len(bad):
        print(f"WARNING: {len(bad)} rows have a true_theme that is not in the theme list; ignored:")
        print(bad[["ticket_id", "true_theme"]].to_string(index=False))
    if len(gold) == 0:
        sys.exit("No human labels yet. Fill in true_theme in validation/check_100.csv (see validation/README.md).")

    res = gold.merge(predictions(set(gold["ticket_id"])), on="ticket_id")
    res["bot_tag_ok"] = [t in BOT_TAG_COVERS.get(b, set()) for t, b in zip(res["true_theme"], res["bot_tag"])]
    sure = res[res["unsure"].astype(str).str.upper().str.strip() != "Y"]

    labeller = (res["labelled_by"].dropna().iloc[0] if "labelled_by" in res and res["labelled_by"].notna().any()
                else "a person")
    lines = [f"# Check-set results ({len(res)} tickets)\n",
             f"Labelled by: **{labeller}**\n",
             f"Unsure even for the labeller: {len(res) - len(sure)} of {len(res)}. "
             f"LLM share in the hybrid: {(res['route_reason'] != 'confident').mean():.0%} of tickets.\n",
             "| Method | Correct | Accuracy | 95% range | Error rate | Accuracy on sure-only |",
             "|---|---|---|---|---|---|"]
    methods = {"bot_tag": "bot_tag_ok", "keyword": None, "tfidf": None, "llm": None,
               "hybrid_no_key": None, "hybrid": None}
    for name, ok_col in methods.items():
        ok = res[ok_col] if ok_col else res[name] == res["true_theme"]
        ok_sure = sure[ok_col] if ok_col else sure[name] == sure["true_theme"]
        k, n = int(ok.sum()), len(res)
        lo, hi = wilson(k, n)
        lines.append(f"| {name} | {k}/{n} | {k / n:.0%} | {lo:.0%}-{hi:.0%} | {1 - k / n:.0%} | "
                     f"{ok_sure.mean():.0%} (n={len(sure)}) |")

    if res["true_is_repeat_claim"].notna().any():
        truth = res["true_is_repeat_claim"].astype(str).str.upper().str.strip().eq("Y")
        has = res["true_is_repeat_claim"].notna()
        lines += ["\n## Repeat-claim flag", "| Method | Accuracy |", "|---|---|",
                  f"| llm | {(res['llm_repeat'].astype(bool) == truth)[has].mean():.0%} |",
                  f"| keyword | {(res['keyword_repeat'].astype(bool) == truth)[has].mean():.0%} |"]

    wrong = res[res["hybrid"] != res["true_theme"]]
    # ticket ids only: results.md is committed, and customer text stays out of the repo
    lines += [f"\n## Where the hybrid is wrong ({len(wrong)})", "| ticket | true | hybrid | via |", "|---|---|---|---|"]
    for r in wrong.itertuples():
        via = "tfidf" if r.route_reason == "confident" else "llm"
        lines.append(f"| {r.ticket_id} | {r.true_theme} | {r.hybrid} | {via} |")

    pairs = (res[res["hybrid"] != res["true_theme"]].groupby(["true_theme", "hybrid"]).size()
             .sort_values(ascending=False).head(5))
    lines += ["\n## Most common confusions", *[f"- {t} -> {p}: {c}" for (t, p), c in pairs.items()]]

    report = "\n".join(lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
