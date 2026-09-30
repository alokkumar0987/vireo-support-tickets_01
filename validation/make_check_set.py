"""
Builds validation/check_100.csv: 100 tickets for a HUMAN to label, used to measure how often
the tool is wrong (brief deliverable 3). Run once; re-running refuses to overwrite labels.

Sampling: stratified by half-year and channel, so early and recent tickets are both covered
(the first LLM labels were mostly early 2025). Model outputs are deliberately NOT included:
labelling blind keeps the check independent of the models it is checking.
These ticket_ids are excluded from TF-IDF training (src/cascade.py reads CHECK_SET_PATH).

    python validation/make_check_set.py
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.clean import clean_tickets_data
from src.llm import THEMES

CHECK_SET_PATH = os.path.join(ROOT, "validation", "check_100.csv")
N = 100


def build(seed=42):
    df = clean_tickets_data()
    df["half"] = df["created_at_dt"].dt.year.astype(str) + "-H" + ((df["created_at_dt"].dt.month > 6) + 1).astype(str)
    strata = df.groupby(["half", "channel"])
    per_stratum = (strata.size() / len(df) * N).round().astype(int).clip(lower=1)
    parts = [g.sample(min(per_stratum[k], len(g)), random_state=seed) for k, g in strata]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).head(N)

    out = pd.DataFrame({
        "ticket_id": sample["ticket_id"],
        "created": sample["created_at_dt"].dt.strftime("%Y-%m-%d"),
        "channel": sample["channel"],
        "product": sample["product_name"],
        "customer_message": sample["customer_message"],
        "agent_notes": sample["agent_notes"],
        "true_theme": "",           # pick one of THEMES (see validation/README.md)
        "true_is_repeat_claim": "",  # Y / N: does the customer or note say it was raised before?
        "unsure": "",                # Y if even a human can't tell; these set the ceiling on accuracy
    })
    return out


if __name__ == "__main__":
    if os.path.exists(CHECK_SET_PATH):
        existing = pd.read_csv(CHECK_SET_PATH)
        if existing["true_theme"].notna().any():
            sys.exit(f"{CHECK_SET_PATH} already has labels; not overwriting.")
    out = build()
    # Text goes to a git-ignored sheet for labelling; the committed file carries ids and labels only.
    out.to_csv(CHECK_SET_PATH.replace(".csv", "_sheet.csv"), index=False, encoding="utf-8-sig")
    out.drop(columns=["customer_message", "agent_notes"]).to_csv(CHECK_SET_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} tickets to {CHECK_SET_PATH} (labels) and check_100_sheet.csv (text, not committed)")
    print(out.groupby([out['created'].str[:4], 'channel']).size().to_string())
    print("\nThemes to choose from:\n  " + "\n  ".join(THEMES))
