# Prompt log: ticket classifier (`src/llm.py`)

The prompt used in production is `SYSTEM_PROMPT` in `src/llm.py`. The cache in `cache/llm_labels.jsonl` records the prompt version on every row. Changing `PROMPT_VERSION` makes every ticket get labelled again; old labels are never mixed with new ones.

**Endpoints and models used:** `temperature=0` and 20 tickets per call throughout.
- **First:** APInex `free/gpt-6-luna`, which labelled 839 tickets before its free 1M-token quota ran out.
- **Now:** xkiro `qwen/qwen3.8-max:free`. Same prompt v3; it agreed 40/40 with gpt-6-luna on a re-labelled sample, and labelled another 594 tickets.

## v1: discarded

- **Themes:** 14 broad themes, such as "Return or replacement request", "Payment or billing error" and "Other".
- **Flags:** `is_repeat_claim` and `resolved_per_note` (true/false).
- **Tested on:** 3 tickets, as a smoke test.

**Why it was thrown away:**
- A GST invoice request ("my accounts team needs the tax bill") was labelled **Other**, because no theme fit it.
- "Return or replacement request" mixed together three different problems: return pickups that never happened, refunds that never arrived, and genuine return requests. Each of these needs a different fix.
- `resolved_per_note` couldn't tell "refunded the customer" apart from "fixed the product". For failure demand, that is the whole question.

## v2: 18 root-cause themes, each with a one-line description, plus `outcome`

- **Themes:** themes split by **root cause**, based on reading about 55 real tickets across all 11 bot categories. Each theme came with a short description of what belongs in it.
- **Outcome:** `outcome` is one of `fixed`, `refund`, `replacement_or_reship`, `escalated`, `pending_or_deferred` or `unclear`. A lazy note ("done", "see prev", "cx ok") counts as `unclear`.
- **Glossary:** added an explanation of agent shorthand (cx, rslvd, rplc, pkp, crr, fw, pg).
- **Scope:** `is_repeat_claim` now also covers notes that say "repeat contact".

**Results on 40 randomly chosen tickets, checked by hand:** about 36 of 40 correct. It clearly beat the intake bot: the bot tagged 8 of the 40 as "Other" even though each had a clear issue (an invoice, a delivery, an address change, a cancellation, a compatibility question), and the LLM placed each of them.

**Errors found:**
1. "you picked up the item 15 days ago and my money hasn't come back" was labelled **Double charge**. It should be *Refund delayed*.
2. "sitting at home for 3 days waiting for your courier", with the note "pickup missed", was labelled **Delivery**. It should be *Return pickup not done*.
3. "ordered black, got white" was labelled **Other**, because no theme covered wrong items.

## v3: current

- Added the theme **"Wrong item / variant received"**.
- Sharpened three descriptions:
  - *Return pickup*: "courier never came to COLLECT… customer waiting at home for pickup".
  - *Refund delayed*: "money not back after a return pickup, cancellation or approved refund".
  - *Double charge*: "…(NOT waiting for a refund)".

**Result:** re-run on the same 40 tickets, all three errors were fixed. One ticket came back malformed; it was dropped and will be retried on the next run.

**Checked across models:** the same prompt run on two other free models, on 40 already-labelled tickets:
- `qwen/qwen3.8-max:free`: agreed 40/40.
- `cohere/command-a`: agreed 39/40.

Three models from different companies agreeing this closely suggests the labels are consistent. It does **not** prove they are correct; only the hand check does that.

## Discarded: labelling every ticket with the LLM

The first plan sent all 11,875 tickets to the LLM. It was stopped after 839 tickets, when the free quota ran out: **1M billed tokens were used by about 900 labelled tickets.** Billing counted roughly 3× more tokens than the API's own usage report showed.

That plan was replaced by the hybrid below.

## Current design: hybrid (`src/cascade.py`)

```
ticket -> TF-IDF + logistic regression, trained on the cached LLM labels (~2 s to train, 0.4 ms per ticket)
            confident (p >= 0.6), familiar, and not "Other"  -> keep the label (local, ₹0)
            otherwise                                          -> LLM if a key is set, else keyword rules
```

**How "familiar" is measured:**
- A ticket counts as familiar when its TF-IDF similarity to the nearest training ticket is at or above the 2nd percentile of training-set nearest-neighbour similarities.
- That threshold is capped at 0.35. The cap exists because on near-duplicate training data the percentile rule climbs towards 1.0 and flags everything as unfamiliar.

**Why not embeddings for the "familiar" check:** `analysis/cluster_themes.py` uses embeddings, but the embedding model needs torch, a download of several GB. That's too heavy for a clean-machine install.

**Where the training labels come from:**
- 839 early tickets (gpt-6-luna).
- 594 recent tickets (Qwen), added because the first 839 were mostly from early 2025.
- The 100 tickets in `validation/check_100.csv` are **never** trained on.

### Results against LLM labels, on the 100 held-out check tickets

| Method | Matches the LLM |
|---|---|
| Keyword rules | 79% |
| TF-IDF alone | 99% (99% on the 68 recent ones too) |
| Hybrid without a key (TF-IDF, else keywords) | 98% |
| Hybrid with a key (TF-IDF, else LLM) | 99%, with 6% of tickets sent to the LLM |

**Across all 11,875 tickets:** 94.1% are handled by TF-IDF, 4.2% are routed as low-confidence and 1.7% as unfamiliar.

Real accuracy comes from the human labels: `validation/score_check_set.py` writes `validation/results.md`.

### Known weakness

A theme the model never trained on can be **confidently wrong** if it shares words with a known theme. In a synthetic test, "courier never came for the pickup" was labelled Refund delayed at 67%, because the only training example with "pickup" was a refund ticket.

The two guards against this:
- the "unfamiliar" check;
- a weekly hand check of a small sample.

## Offline fallback (`src/themes.py`)

Keyword rules that produce the same 19 themes. They are used when there is no trained model and no key.
- They replaced `digest.COMPLAINT_TAXONOMY`, which matched on the first substring it found: "charged twice" went to Battery because it contains "charge", and "repair" went to Bluetooth because it contains "pair".
- One rule came from embedding clustering: 129 "two entries on my statement" and "bank says money went to you" tickets had been falling into "Other".

## Cost

**Measured on Qwen:** about 130 prompt + 40 completion tokens per ticket that reaches the LLM.

**At 650 tickets a week:**
- About 6% reach the LLM: roughly 40 tickets, 2 calls a week.
- Roughly 20k billed tokens a month, allowing for the ~3× billing factor seen on APInex.
- The model is free, so this costs **₹0 a month**.
- On a paid model: monthly cost ≈ 0.02 × (price per 1M tokens).

**Every run without a key:** ₹0. TF-IDF and the keyword rules run on the laptop.

**One-off labelling so far:** 1,433 tickets. It cost ₹0 in cash, but used up one free 1M-token quota.

