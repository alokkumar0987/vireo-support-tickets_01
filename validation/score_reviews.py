"""
Scores the case reviews of the rule-based outputs (repeat detection, double-dip detection).

    python validation/score_reviews.py

Verdicts: Y = tool right, N = tool wrong, ? = can't tell from the data. Precision is reported
two ways: counting "?" as wrong (pessimistic) and excluding it (decided cases only).
"""

import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def summarise(frame, group_col):
    rows = []
    for g, f in frame.groupby(group_col):
        v = f["verdict"].astype(str).str.strip()
        y, n_, q = (v == "Y").sum(), (v == "N").sum(), (v == "?").sum()
        lo, hi = wilson(y, len(f))
        rows.append({group_col: g, "cases": len(f), "right": y, "wrong": n_, "unclear": q,
                     "right_pessimistic": f"{y / len(f):.0%} ({lo:.0%}-{hi:.0%})",
                     "right_of_decided": f"{y / (y + n_):.0%}" if y + n_ else "-"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rep = pd.read_csv(os.path.join(ROOT, "validation", "review_repeats.csv"))
    dd = pd.read_csv(os.path.join(ROOT, "validation", "review_double_dips.csv"))
    print("Repeat detection (20 flagged + 20 not flagged, same customer+product within 30 days):")
    print(summarise(rep, "tool_says").to_string(index=False))
    print("\nDouble-dip detection (20 random flagged orders):")
    print(summarise(dd, "matched_on").to_string(index=False))
    print("\nWrong or unclear cases:")
    for f, label in [(rep, "repeat"), (dd, "double-dip")]:
        for r in f[f["verdict"].astype(str).str.strip() != "Y"].itertuples():
            print(f"  [{label}] {r.verdict}: {r.why}")
