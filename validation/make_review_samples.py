"""
Exports random samples of the tool's RULE-BASED outputs for case-by-case review:

  review_repeats.csv       20 pairs flagged as a same-issue repeat (is it really the same issue?)
                         + 20 pairs of the same customer + product within 30 days that were NOT
                           flagged (did we miss a real repeat?)
  review_double_dips.csv   20 orders flagged as refund AND replacement (is it a real policy §5 breach?)

Fill in `verdict` (Y = tool is right, N = tool is wrong, ? = can't tell) and `why`, then run
validation/score_reviews.py. Seeds are fixed so the samples are reproducible.

    python validation/make_review_samples.py
"""

import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.clean import clean_tickets_data
from src.themes import attach_themes
from src.leakage import audit_double_dipping

OUT = os.path.join(ROOT, "validation")
N = 20
SEED = 7


def _one_line(text, n=220):
    return " ".join(str(text or "").split())[:n]


def repeat_pairs(df):
    d = df.sort_values(["customer_id", "product_sku", "created_at_dt"])
    g = d.groupby(["customer_id", "product_sku"])
    prev = {c: g[c].shift(1) for c in ["ticket_id", "created_at_dt", "resolved_at_dt", "customer_message",
                                       "agent_notes", "theme", "issue_family"]}
    in_window = prev["created_at_dt"].notna() & (
        d["created_at_dt"] <= prev["resolved_at_dt"].fillna(prev["created_at_dt"]) + pd.Timedelta(days=30))
    pairs = pd.DataFrame({
        "earlier_ticket": prev["ticket_id"], "later_ticket": d["ticket_id"],
        "days_after_earlier_closed": ((d["created_at_dt"] - prev["resolved_at_dt"].fillna(prev["created_at_dt"]))
                                      .dt.total_seconds() / 86400).round(1),
        "earlier_theme": prev["theme"], "later_theme": d["theme"],
        "earlier_message": prev["customer_message"].map(_one_line), "earlier_note": prev["agent_notes"].map(_one_line),
        "later_message": d["customer_message"].map(_one_line), "later_note": d["agent_notes"].map(_one_line),
        "same_family": prev["issue_family"] == d["issue_family"],
    })[in_window]
    flagged = pairs[pairs["same_family"]].sample(N, random_state=SEED).assign(tool_says="same issue (repeat)")
    unflagged = pairs[~pairs["same_family"]].sample(N, random_state=SEED).assign(tool_says="different issue (not a repeat)")
    out = pd.concat([flagged, unflagged]).drop(columns="same_family")
    out.insert(0, "tool_says", out.pop("tool_says"))
    return out.assign(verdict="", why="")


def double_dip_cases(df):
    dd = audit_double_dipping(df)["double_dip_orders_df"].sample(N, random_state=SEED)
    by_id = df.set_index("ticket_id")
    rows = []
    for r in dd.itertuples():
        tickets = [t.strip() for t in (r.refund_tickets + ", " + r.replacement_tickets).split(",")]
        detail = " || ".join(
            f"{t} [{by_id.loc[t, 'created_at_dt']:%Y-%m-%d}] refund={by_id.loc[t, 'refund_amount_inr']} "
            f"code={by_id.loc[t, 'refund_reason_code']} repl={by_id.loc[t, 'replacement_issued']} "
            f"note: {_one_line(by_id.loc[t, 'agent_notes'], 120)}"
            for t in dict.fromkeys(tickets))
        rows.append({"repeat_key": r.repeat_key, "matched_on": r.matched_on, "refund_inr": r.refund_inr,
                     "replacement_cost_inr": r.replacement_cost_inr, "leakage_inr": r.leakage_inr,
                     "refund_codes": r.refund_codes, "tickets": detail, "verdict": "", "why": ""})
    return pd.DataFrame(rows)


TEXT_COLUMNS = ["earlier_message", "earlier_note", "later_message", "later_note"]


def _strip_notes(detail):
    """'TK-1 [date] refund=.. code=.. repl=Y note: <agent text> || ...' -> drop the note text."""
    return re.sub(r" note: .*?(?= \|\| |$)", "", str(detail))


if __name__ == "__main__":
    themed = attach_themes(clean_tickets_data())
    for name, frame in [("review_repeats.csv", repeat_pairs(themed)),
                        ("review_double_dips.csv", double_dip_cases(themed))]:
        path = os.path.join(OUT, name)
        if os.path.exists(path) and pd.read_csv(path)["verdict"].notna().any():
            print(f"{name} already has verdicts; not overwriting")
            continue
        # Read the git-ignored *_sheet.csv; write verdicts into the committed file (same row order).
        frame.to_csv(path.replace(".csv", "_sheet.csv"), index=False, encoding="utf-8-sig")
        committed = frame.drop(columns=[c for c in TEXT_COLUMNS if c in frame])
        if "tickets" in committed:
            committed["tickets"] = committed["tickets"].map(_strip_notes)
        committed.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(frame)} rows to {path} (+ _sheet.csv with the text, not committed)")
