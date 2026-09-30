# How we know it works, and how often it doesn't

This document answers the brief's question: *"How do you know your tool's output is correct, and how often is it not? Sample size, how you checked, error rate, and the kind of case it gets wrong."*

Every number here can be regenerated:
- `python validation/score_check_set.py` (classifier)
- `python validation/score_reviews.py` (rule-based outputs)

**Who checked:** the 100-ticket labels and the 60 case verdicts were produced by **Claude Code (Opus 5.5)**, not by a person.
- The labelling was blind: model outputs were not shown while labelling.
- One exception: check-set row 61 had been seen once earlier, during the model comparison.
- **Human review:** a person reviewed 30 of the 100 labels and agreed on all 30 themes, changing 2 "already told you" flags where the note says only "see prev". The reviewer had seen Claude's labels first, so this is a review, not a blind check. Results: `validation/hand_check_results.md`.
- A **blind** human check of fresh tickets is still the obvious next step; see the end of this document.

## 1. Complaint classifier (feeds the weekly digest)

**Sample:** 100 tickets, stratified by half-year and channel (59 from 2025, 41 from 2026). None of them were used for training.

| Method | Correct | 95% range | Error rate |
|---|---|---|---|
| Helpdesk intake-bot category (today's state) | 74/100 | 65–82% | **26%** |
| Keyword rules (offline fallback) | 78/100 | 69–85% | 22% |
| TF-IDF model alone | 100/100 | 96–100% | 0% |
| LLM alone (prompt v3) | 99/100 | 95–100% | 1% |
| **Hybrid, no API key** (what a reviewer runs) | 99/100 | 95–100% | 1% |
| **Hybrid, with key** | 99/100 | 95–100% | 1% |

**What it gets wrong:** tickets that fit two themes.
- **Example:** "right side works perfectly, left one just does not wake up, tried a different cable" could be a charging fault or an audio fault. The labeller marked it *unsure*, and the LLM picked the other one.
- **Weak themes (small samples):** "Damaged on arrival" and "Screen / hardware fault" had F1 scores of 0.55 and 0.75 in the TF-IDF study, and are easily confused with each other.
- **A model weakness:** a theme the model never trained on can be labelled *confidently wrong* if it shares words with a known theme. For example, "courier never came for the pickup" was labelled "refund delayed" when the only training examples containing "pickup" were refund tickets. The "unfamiliar ticket" check and a weekly hand check are the guards against this.

**Why 99% should be read with care:**
1. **The ticket text is heavily templated,** which makes it easy to classify. On real, messy tickets expect something closer to 85–90%.
2. **n = 100,** so "100%" means "at least about 96%".
3. **Claude labelled the check set,** and Claude is also a language model. If it shares a blind spot with the LLM, this check can't see it.

**How much goes to the LLM:** measured on held-out tickets, 2–6% of tickets reach the LLM (6% with the first model, 2% after retraining on 2,022 labels; `validation/results.md`). Costs are planned on 6%. After retraining on 2,022 labels, the in-sample routing shows 0.2%. That figure is flattering, because many of those tickets are now training data. **Plan cost on 6%.**

## 2. Repeat contacts (the business goal rests on this)

Policy §10 defines a repeat contact as *the same customer about the same issue within 30 days of resolution*.

**How the definition changed:**
- **The first version** counted any return by the same customer about the same product. That gave 29.8% of contacts.
- **The problem:** only 34% of those pairs were the same kind of problem. A "delivery late" ticket followed by a "battery drains" ticket counted as a repeat.
- **The current definition** adds the complaint theme, grouped into issue families: a refund chase after a cancellation counts as the same issue. That gives **12.9%**.

### Checked two ways

**(a) What the tickets say, on the 2,123 tickets the LLM read:**
- **63%** of the 273 tickets we flag as a repeat mention an earlier contact, by the customer or in the note ("third time", "was told it was fixed").
- Only **7%** of the other tickets say that.
- This signal is independent of how the flag is computed.
- **Why not all 11,875 tickets:** the rest get this signal from a keyword rule, which agrees with the LLM only 45% of the time when it fires (it also matches notes like "already dispatched"). On all tickets it gives 49% vs 17%, the same direction but noisier. An earlier draft quoted that figure.

**(b) Case review of 40 random pairs** (`validation/review_repeats.csv`):

| Tool says | Cases | Right | Wrong | Unclear |
|---|---|---|---|---|
| Same issue (repeat) | 20 | 19 | 1 | 0 |
| Different issue (not a repeat) | 20 | 19 | 0 | 1 |

**What it gets wrong:**
- **The one false repeat:** a "discount not applied" refund followed a few weeks later by an unrelated *return* refund. The "Money back" family groups them as one issue.
- **The unclear case:** the messages are about different things, but the agent's note says "issue recurred".

**Error rate:** about **5%** in each direction (95% range 1–24% on n = 20). That's too small a sample for a tight bound.

## 3. Refund *and* replacement on the same order (policy §5)

**Case review of 20 random flagged orders** (`validation/review_double_dips.csv`):

| How the order was matched | Cases | Right | Wrong | Unclear |
|---|---|---|---|---|
| Order ID on both tickets | 15 | **14** | 0 | 1 (goodwill credit, not a refund) |
| Customer + product only (no order ID quoted) | 5 | 0 | 1 | 4 (may be two different purchases) |

**What the review changed in the tool:**
1. **Only order-ID matches are counted now.** Customer + product matches are listed as "possible" (34 orders), not counted.
2. **Each distinct refund amount is counted once.** The same refund was being re-entered on chaser tickets ("refund confirmed credited (3499)") and summed up to 3 times.
3. **GW-OTHER goodwill credits are no longer treated as product refunds.** They are covered by the goodwill-cap check instead.

**Effect:** the headline went from 110 orders / ₹1.81L to **69 orders / ₹1.15L**, and every counted case now matches on order ID.

## 4. Weekly alerts and lot codes (digest)

**"Rising theme" alerts:**
- **The test:** a Poisson test against the previous 8 weeks, p < 0.01.
- **Backtest over 30 weeks × 19 themes:** 7 alerts, where **~6 would be expected from chance alone**. So a single alert means little, and the digest says so ("two weeks running = a real trend"). The only repeated signal was Audio fault in W49 and W50.
- **A real spike does trigger it:** a synthetic jump from 20 to 40 tickets fires the alert, and 20 to 24 does not (`tests/test_digest.py`).

**Manufacturing lots:**
- **The old check was noise:** it flagged about 13 lots that chance alone would produce.
- **The current check** uses hardware-fault tickets per unit sold with a Bonferroni correction across 1,270 lots. **No lot survives**, and the digest says "none stands out" rather than inventing an alert.

## 5. What is *not* validated

| Output | Why it isn't validated | What would validate it |
|---|---|---|
| 749 refunds that appear in notes with no amount recorded | They may be recorded in the payment gateway or a courier-claim system this export doesn't include | Reconcile against gateway settlements |
| 342 warranty replacements closed by Tier 1 | A Tier 2 approval may exist outside the ticket | A sample of RMA records |
| Scaling to 650 tickets a week | Assumes the export (≈150 a week) is a representative sample | Vireo confirms how the export was drawn |
| Business-goal lever (halving status chasers) | 50% is an assumption, not a measurement; the report shows 25%, 50% and 75% | Pilot proactive updates on one channel for 4 weeks |

## 6. To turn this into a blind human check (about 20 minutes)

1. Draw 20 fresh tickets that are not in `check_100.csv` and whose labels nobody has seen. Label them before looking at any tool output, then compare.
2. In the two `review_*.csv` files, check the rows marked `N` and `?`.
3. Run `python validation/score_check_set.py` and `python validation/score_reviews.py` again.
