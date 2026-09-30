# Human hand-check: 30 tickets (do this one first)

The 100 check-set labels below were written by Claude Code, so "99/100" is an AI checking an AI.
**Done:** a person labelled 30 of them and agreed on all 30 themes (`hand_check_results.md`). They had
seen Claude's labels first, so it is a review, not a blind check.

- `hand_check_30.csv` (committed): ticket IDs and the person's labels only.
- `hand_check_30_sheet.csv` (local, git-ignored): the same tickets with the customer text, for labelling.
  Recreate it with `python validation/hand_check.py make` (needs `data/`).
- Fill `your_theme` (copy a theme exactly from the table below), `your_is_repeat_claim` (`Y`/`N`) and
  `unsure` in the sheet, then run `python validation/hand_check.py`.

# Hand check: 100 tickets

The file `check_100.csv` holds 100 tickets. They are spread across 2025 and 2026 and across all four channels. **None of them were used to train the TF-IDF model.** A person labels them, and the results measure how often the tool is wrong.

## How to label (about 15 minutes)

No customer text is committed. `check_100.csv` holds ticket IDs and labels; the text to read is in the
local `check_100_sheet.csv` (git-ignored; `python validation/make_check_set.py` writes it from `data/`).
A local `check_100.json` copy, also git-ignored, may be used instead. The scorer reads whichever file has
labels, and stops if both do.

1. Read the tickets in `check_100_sheet.csv` and write labels in `check_100.csv` (same ticket IDs).
2. For each row, read `customer_message` (and `agent_notes` if the message is unclear), then fill in:
   - **`true_theme`**: one theme from the list below. Copy it exactly.
   - **`true_is_repeat_claim`**: `Y` if the customer or the note says the problem was raised before, otherwise `N`.
   - **`unsure`**: `Y` if you can't decide yourself. Still pick your best guess.
3. **Don't look at any model output while labelling.** Labelling blind is what keeps this check independent.
4. Save the file as CSV, keeping the same name.

## Themes

Pick the customer's **main** problem.

| Theme | Use it for |
|---|---|
| Delivery delayed / not delivered | order not arrived, tracking stuck, marked delivered but not received |
| Address change / wrong address | moved house, wrong pincode, redirect a shipment |
| Damaged in transit / dead on arrival | arrived cracked or broken, dead out of the box, missing parts |
| Wrong item / variant received | different product, colour or model delivered |
| Return pickup not done | courier never came to collect a return |
| Refund delayed / not received | money not back after a return, cancellation or approved refund |
| Double charge / payment failed | charged twice, money deducted but no order, gateway failure |
| Coupon / discount / price | promo code not working, discount not applied |
| Invoice / GST | invoice download, GST bill, GSTIN |
| Order cancellation | wants to cancel before dispatch |
| Pairing & connection drops | can't pair, not discoverable, disconnects, stutters |
| Battery drain / not charging | short battery life, device or case not charging |
| Audio fault (one side, distortion, mic) | one earbud silent, crackling, mic not working |
| App / firmware update failure | app crashes, firmware update stuck, device bricked after an update |
| Screen / touch / hardware fault | watch screen unresponsive, buttons, other physical faults |
| Warranty claim / repair status | chasing an RMA, repair or warranty replacement |
| Account / OTP login | can't log in, OTP not received |
| Product / compatibility question | pre-sales or how-to question, compatibility |
| Other | none of the above |

## Then

```
python validation/score_check_set.py
```

This scores the intake bot's tags, the keyword rules, TF-IDF, the LLM and the full hybrid against your labels. It reports accuracy, the error rate and the actual mistakes.
