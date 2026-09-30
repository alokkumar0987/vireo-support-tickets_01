"""
Human hand-check: 30 of the 100 check-set tickets, labelled BLIND by a person.

The 100-ticket check set was labelled by Claude Code (an AI checking an AI). This turns a
sample of it into an independent human check, and also measures how far Claude's labels
can be trusted on the other 70.

    python validation/hand_check.py make   # writes hand_check_30_sheet.csv (text, NOT committed) +
                                           #        hand_check_30.csv (ids + empty label columns)
    # ...fill in your_theme / your_is_repeat_claim / unsure in the SHEET (see validation/README.md)
    python validation/hand_check.py        # copies the labels into hand_check_30.csv (no customer text,
                                           # committed) and scores them -> validation/hand_check_results.md

Scores your labels against: the tool (what the reports use), the helpdesk bot tag, and
Claude's labels in check_100.csv.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from src.llm import THEMES  # noqa: E402
from score_check_set import BOT_TAG_COVERS, predictions, wilson  # noqa: E402

CHECK_PATH = os.path.join(HERE, "check_100.csv")
SHEET_PATH = os.path.join(HERE, "hand_check_30_sheet.csv")   # with customer text: git-ignored
LABELS_PATH = os.path.join(HERE, "hand_check_30.csv")         # ids + labels only: committed
LABEL_COLUMNS = ["your_theme", "your_is_repeat_claim", "unsure"]
# How the 30 were labelled, stated in the results so nobody reads more into them than is there.
METHOD = ("Labelled by a person who had seen Claude Code's labels for these tickets during the working "
          "session, so this is a **review of Claude's labels, not a blind check**.")
RESULTS_PATH = os.path.join(HERE, "hand_check_results.md")
N = 30
SEED = 30


def make():
    if os.path.exists(SHEET_PATH):
        existing = pd.read_csv(SHEET_PATH)
        if existing["your_theme"].notna().any():
            sys.exit(f"{SHEET_PATH} already has labels; not overwriting.")
    check = pd.read_csv(CHECK_PATH)
    # Proportional to channel (the check set is already spread across 2025-2026)
    sample = (check.groupby("channel", group_keys=False)
                   .apply(lambda g: g.sample(round(N * len(g) / len(check)), random_state=SEED)))
    sample = sample.sample(frac=1, random_state=SEED).head(N)  # shuffle, so channels are mixed
    sheet = sample[["ticket_id", "channel", "product", "customer_message", "agent_notes"]].copy()
    sheet["your_theme"] = ""
    sheet["your_is_repeat_claim"] = ""
    sheet["unsure"] = ""
    sheet.to_csv(SHEET_PATH, index=False, encoding="utf-8-sig")  # -sig so Excel shows ₹ / Hindi correctly
    sheet[["ticket_id", *LABEL_COLUMNS]].to_csv(LABELS_PATH, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(sheet)} blind tickets to {SHEET_PATH} ({sheet['channel'].value_counts().to_dict()})")


def _pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}, 95% range {lo:.0%}-{hi:.0%})"


def score():
    source = SHEET_PATH if os.path.exists(SHEET_PATH) else LABELS_PATH
    if not os.path.exists(source):
        sys.exit("No sheet yet. Run: python validation/hand_check.py make")
    sheet = pd.read_csv(source)
    if source == SHEET_PATH:  # keep the committed copy in step with the sheet, minus the customer text
        sheet[["ticket_id", *LABEL_COLUMNS]].to_csv(LABELS_PATH, index=False, encoding="utf-8-sig")
    sheet["your_theme"] = sheet["your_theme"].fillna("").astype(str).str.strip()
    bad = sheet[(sheet["your_theme"] != "") & ~sheet["your_theme"].isin(THEMES)]
    if len(bad):
        print("These themes are not in the list (copy them exactly); ignored:")
        print(bad[["ticket_id", "your_theme"]].to_string(index=False))
    done = sheet[sheet["your_theme"].isin(THEMES)]
    if done.empty:
        sys.exit(f"No labels yet. Fill in your_theme in {SHEET_PATH}.")

    claude = pd.read_csv(CHECK_PATH)[["ticket_id", "true_theme", "true_is_repeat_claim"]]
    res = (done[["ticket_id", *LABEL_COLUMNS]].merge(predictions(set(done["ticket_id"])), on="ticket_id")
               .merge(claude, on="ticket_id", how="left"))
    n = len(res)
    tool_ok = res["hybrid"] == res["your_theme"]
    bot_ok = pd.Series([t in BOT_TAG_COVERS.get(b, set()) for t, b in zip(res["your_theme"], res["bot_tag"])])
    claude_ok = res["true_theme"] == res["your_theme"]
    sure = res["unsure"].astype(str).str.upper().str.strip() != "Y"

    lines = [f"# Human review of {n} check-set tickets\n", METHOD + "\n",
             "| Compared with your labels | Agree |", "|---|---|",
             f"| **The tool** (theme used in every report) | {_pct(int(tool_ok.sum()), n)} |",
             f"| The helpdesk bot's category | {_pct(int(bot_ok.sum()), n)} |",
             f"| Claude Code's labels in check_100.csv | {_pct(int(claude_ok.sum()), n)} |",
             f"\nOn the {int(sure.sum())} tickets you were sure about, the tool agrees on "
             f"{tool_ok[sure].mean():.0%}.",
             "\n*Reading it:* the first row is the tool's agreement with a person on these tickets. Because the "
             "person had seen Claude's labels, it confirms Claude's labels were acceptable to a human reviewer; "
             "it is not an independent error rate."]

    yn = res["your_is_repeat_claim"].astype(str).str.upper().str.strip()
    has = yn.isin(["Y", "N"])
    if has.any():
        truth = yn.eq("Y")
        lines += ["\n## \"Already told you\" flag", "| Method | Agree |", "|---|---|",
                  f"| LLM label | {_pct(int((res['llm_repeat'].astype(bool) == truth)[has].sum()), int(has.sum()))} |",
                  f"| Keyword rule | {_pct(int((res['keyword_repeat'].astype(bool) == truth)[has].sum()), int(has.sum()))} |"]

    wrong = res[~tool_ok]
    lines += [f"\n## Where the tool disagrees with you ({len(wrong)})",
              "| ticket | you | tool | Claude |", "|---|---|---|---|"]
    for r in wrong.itertuples():
        lines.append(f"| {r.ticket_id} | {r.your_theme} | {r.hybrid} | {r.true_theme} |")
    claude_y = res["true_is_repeat_claim"].astype(str).str.upper().str.strip().eq("Y")
    rep_diff = res[has & (yn.eq("Y") != claude_y)]
    if len(rep_diff):
        lines += [f"\n## Where your \"already told you\" flag differs from Claude's ({len(rep_diff)})",
                  "| ticket | you | Claude |", "|---|---|---|",
                  *[f"| {r.ticket_id} | {r.your_is_repeat_claim} | {r.true_is_repeat_claim} |"
                    for r in rep_diff.itertuples()]]

    report = "\n".join(lines)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report + f"\n\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    make() if sys.argv[1:] == ["make"] else score()
