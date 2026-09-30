# Prompt log: the ticket classifier

The LLM does one job in this project: read a ticket (the customer's message and the agent's closing
note) and return a **complaint theme**, whether the ticket **mentions an earlier contact**, and what
the **note says happened**. Everything else (repeat contacts, the leaderboard, the money audit) is
plain code on top of those labels.

- **Where:** `SYSTEM_PROMPT` in `src/llm.py`, version `PROMPT_VERSION = "v3"`.
- **Versioning:** every cached label in `cache/llm_labels.jsonl` records the prompt version. Changing
  the prompt means bumping the version, which re-labels every ticket; labels from different versions
  are never mixed.
- **Settings throughout:** `temperature=0`, 20 tickets per call, message cut to 1,200 characters and
  note to 600.

## At a glance

| Version | Themes | What changed | Checked on | Result | Status |
|---|---|---|---|---|---|
| v1 | 14 broad | first attempt: theme + `is_repeat_claim` + `resolved_per_note` | 3-ticket smoke test, then reading | GST invoices fell into "Other"; one theme mixed 3 different problems | **thrown away** |
| v2 | 18 by root cause | one-line description per theme, agent-shorthand glossary, `outcome` instead of resolved yes/no | 40 random tickets, by hand | ~36/40 right, 3 error types | **replaced** |
| v3 | 19 | added *Wrong item / variant*, sharpened 3 descriptions | the same 40 tickets | all 3 error types fixed | **in use** |

```
  v1  14 broad themes ──────────► thrown away: "Other" catches invoices; returns/refunds/pickups mixed
   │
   ▼
  v2  18 root-cause themes ──────► 40-ticket review: 3 kinds of error
   │     + descriptions, glossary, outcome
   ▼
  v3  19 themes ─────────────────► same 40 tickets: all 3 fixed; 3 models agree 39-40 of 40
         + "Wrong item", sharper refund / pickup / double-charge lines

  architecture:  send every ticket to the LLM  ──►  quota gone after 839 tickets  ──►  thrown away
                 LLM labels a sample once, a local model learns from it, only unsure tickets go
                 back to the LLM  ──►  in use (see "Current design")
```

---

## The prompt in use (v3)

System prompt, exactly as sent (generated from `src/llm.py`):

```text
You label customer-support tickets for Vireo Audio, an Indian consumer-audio brand
(earbuds, headphones, speakers, smartwatches, chargers).
Each ticket has the customer's opening message and the agent's closing note. Messages can have
typos, Hinglish or be IVR transcripts; notes use shorthand (cx=customer, rslvd=resolved,
rplc=replacement, pkp=pickup, crr=courier, fw=firmware, pg=payment gateway).

Themes (pick exactly one, for the customer's MAIN problem; use the note if the message is vague):
{
 "Delivery delayed / not delivered": "order not arrived, tracking stuck, courier says delivered but not received",
 "Address change / wrong address": "moved house, wrong pincode, redirect a shipment",
 "Damaged in transit / dead on arrival": "arrived cracked/broken, dead out of the box, missing parts",
 "Wrong item / variant received": "different product, wrong colour or model delivered",
 "Return pickup not done": "courier never came to COLLECT a return/replacement (customer waiting at home for pickup)",
 "Refund delayed / not received": "money not back after a return pickup, cancellation or approved refund",
 "Double charge / payment failed": "charged twice for one order, money deducted but no order created, gateway failure (NOT waiting for a refund)",
 "Coupon / discount / price": "promo code not working, discount not applied, price drop",
 "Invoice / GST": "invoice download, GST bill, GSTIN correction",
 "Order cancellation": "wants to cancel before dispatch, cancel button not working",
 "Pairing & connection drops": "cannot pair, not discoverable, disconnects, audio stutters from signal",
 "Battery drain / not charging": "short battery life, case or device not charging, charger/cable faults",
 "Audio fault (one side, distortion, mic)": "one earbud silent, crackling, buzzing, low mic",
 "App / firmware update failure": "app crashes or won't open, firmware update stuck, device bricked after update",
 "Screen / touch / hardware fault": "watch screen unresponsive, buttons, physical fault not covered above",
 "Warranty claim / repair status": "chasing an RMA, repair or warranty replacement already raised",
 "Account / OTP login": "cannot log in, OTP not received, account locked",
 "Product / compatibility question": "pre-sales or how-to question, compatibility with a phone/TV",
 "Other": "none of the above"
}

Return ONLY a JSON array, one object per ticket, same order, no prose:
{"ticket_id": str,
  "theme": one of the theme names above, spelled exactly,
  "is_repeat_claim": true if the customer OR the note says this was raised before
                     ("already told", "third time", "was told it was resolved", "repeat contact"),
  "outcome": one of ["fixed", "refund", "replacement_or_reship", "escalated", "pending_or_deferred", "unclear"] - what the note says happened;
             "unclear" if the note is empty or says nothing specific ("done", "see prev", "cx ok")}
```

**User message:** a JSON array of up to 20 tickets (invented example):

```json
[{"ticket_id": "TK-000001",
  "customer_message": "earbuds die in 2 hours even after full charge, pls help",
  "agent_notes": "cx rptd poor battery backup. fw update pushed, adv to monitor"}]
```

**Expected reply:** one object per ticket, same order:

```json
[{"ticket_id": "TK-000001", "theme": "Battery drain / not charging",
  "is_repeat_claim": false, "outcome": "fixed"}]
```

**How replies are checked** (`_label_batch` in `src/llm.py`): a row is kept only if its `ticket_id`
was in the batch and its `theme` is spelled exactly as one of the 19. Invented themes, unknown IDs
and duplicate rows are dropped, and those tickets are retried on the next run. An unknown `outcome`
becomes `unclear`.

**Why it is written this way:**
- **A description per theme, not just a name.** v1 had bare names; the model then guessed at the
  edges ("is a refund after a pickup a return problem or a refund problem?").
- **The glossary.** Agent notes are shorthand (`rplc`, `pkp`, `crr`, `rslvd`); without it the note,
  which often settles the theme, was wasted.
- **"Use the note if the message is vague".** Many messages are one line ("hello pls help"); the note
  says what the ticket was actually about.
- **`is_repeat_claim` covers the note as well as the customer.** Agents write "recontact" or "second
  contact on same fault"; that is Neha's "I already told your colleague" in the agent's words.
- **Strict JSON, no prose.** The output is parsed by code, and anything off-list is rejected.

---

## v1: thrown away

- **Themes:** 14 broad ones, such as "Return or replacement request", "Payment or billing error" and "Other".
- **Flags:** `is_repeat_claim` and `resolved_per_note` (true/false).
- **Checked on:** a 3-ticket smoke test, then reading its labels against the messages.

**Why it was thrown away:**
- A GST invoice request ("my accounts team needs the tax bill") was labelled **Other**: no theme fit.
- "Return or replacement request" mixed three problems that need different fixes: return pickups
  that never happened, refunds that never arrived, and genuine return requests.
- `resolved_per_note` could not tell "refunded the customer" from "fixed the product". For failure
  demand, that is the whole question.

## v2: 18 root-cause themes

- **Themes** split by **root cause**, from reading about 55 real tickets across all 11 bot categories,
  each with a one-line description of what belongs in it.
- **`outcome`** replaced resolved yes/no: `fixed`, `refund`, `replacement_or_reship`, `escalated`,
  `pending_or_deferred` or `unclear` (a lazy note such as "done", "see prev", "cx ok").
- **Glossary** of agent shorthand (cx, rslvd, rplc, pkp, crr, fw, pg).
- **`is_repeat_claim`** extended to notes that say "repeat contact".

**Checked on 40 random tickets, by hand:** about 36 of 40 right. It clearly beat the intake bot,
which tagged 8 of the 40 as "Other" although each had a clear issue (an invoice, a delivery, an
address change, a cancellation, a compatibility question); the LLM placed all 8.

**Errors found:**
1. "you picked up the item 15 days ago and my money hasn't come back" → **Double charge**; should be *Refund delayed*.
2. "sitting at home for 3 days waiting for your courier" + note "pickup missed" → **Delivery**; should be *Return pickup not done*.
3. "ordered black, got white" → **Other**; there was no theme for wrong items.

## v3: in use

| Change | Aimed at |
|---|---|
| New theme **Wrong item / variant received** | error 3 |
| *Return pickup not done*: "courier never came to **COLLECT** … customer waiting at home for pickup" | error 2 |
| *Refund delayed*: "money not back **after a return pickup, cancellation or approved refund**" | error 1 |
| *Double charge*: "… **(NOT waiting for a refund)**" | error 1 |

**Result:** re-run on the same 40 tickets, all three errors were fixed. One reply row came back
malformed; it was dropped and retried on the next run, as designed.

**Checked across models:** the same prompt on 40 already-labelled tickets:

| Model | Agrees with the first model |
|---|---|
| `qwen/qwen3.8-max:free` | 40/40 |
| `cohere/command-a` | 39/40 |

Models from different companies agreeing this closely means the prompt is **unambiguous**. It does
not prove the labels are **correct**; the check set does that (below).

---

## Thrown away

| What | Why |
|---|---|
| Prompt v1 | coarse themes; see above |
| Prompt v2 descriptions | 3 error types in the 40-ticket review |
| **Sending every ticket to the LLM** | stopped after 839 tickets, when the free 1M-token quota ran out: about 900 labelled tickets used 1M *billed* tokens, roughly 3× what the API's own usage report said. The quota running out showed the design was wrong: a per-ticket bill is exactly what Arjun said he does not want. |
| Embedding-based "is this ticket unfamiliar?" check | needs torch (several GB) for a clean-machine install; TF-IDF similarity does the job with scikit-learn |
| The first version's keyword taxonomy | matched on the first substring: "charged twice" → Battery (contains "charge"), "repair" → Bluetooth (contains "pair") |
| `resolved_per_note` | replaced by `outcome` in v2 |

---

## Current design: the LLM labels a sample, a local model does the rest

```
  2,123 tickets labelled once by the LLM (prompt v3), cached
        │
        ▼
  TF-IDF + logistic regression, trained on 2,022 of them        (check-set tickets never used)
        │   ~2 s to train, ~0.4 ms per ticket, ₹0
        ▼
  every ticket ──► confident (p ≥ 0.6), familiar, not "Other"? ──yes──► local label
                        │ no  (2–6% of new tickets)
                        ▼
                  LLM, if --llm and a key are set; else keyword rules
```

"Familiar" means the ticket's TF-IDF similarity to its nearest training ticket is at least the 2nd
percentile of training nearest-neighbour similarities (0.309 on this data), capped at 0.35: on
near-duplicate training data the percentile alone climbs towards 1.0 and flags everything.

**Where the 2,123 labels came from:**

| Model | Tickets | Why |
|---|---|---|
| APInex `free/gpt-6-luna` | 839 | the first full-labelling run, until the free quota ran out (mostly early 2025) |
| xkiro `qwen/qwen3.8-max:free` | 594 | recent weeks, so the model also knows 2026 tickets |
| xkiro `qwen/qwen3.8-max:free` | 690 | the tickets the first local model was unsure about (`run.py --llm`) |

**How well it works, on the 100 check-set tickets** (never trained on):

| Method | Matches the LLM's labels | Matches the check-set labels |
|---|---|---|
| Keyword rules only | 79% | 78/100 |
| TF-IDF model alone | 99% | 100/100 |
| Hybrid, no key (TF-IDF, else keyword rules) | 100% | 99/100 |
| Hybrid, with key (TF-IDF, else LLM) | 100% | 99/100 |
| *Helpdesk bot `category`, for comparison* | | *74/100* |

The check-set labels were written by Claude Code; a person then reviewed 30 of them and agreed on
all 30 (they had seen Claude's labels first, so this is a review, not a blind check). Details and
confidence ranges: `validation/results.md`, `validation/hand_check_results.md`, `docs/validation.md`.

**Share of tickets sent to the LLM:** 6% on held-out tickets with the first local model (trained on
1,433 labels), 2% after retraining on 2,022. Costs below are planned on 6%. On the historical tickets
the figure is 0.2%, but that is flattering: most of them are now training data.

---

## Known weaknesses

- **Confidently wrong on a theme it never saw.** In a synthetic test, "courier never came for the
  pickup" was labelled *Refund delayed* at 67% confidence, because the only training example with
  "pickup" was a refund ticket. Guards: the "unfamiliar" check, and a weekly hand check of a few tickets.
- **Two-sided tickets.** The one check-set miss: "right side works perfectly, left one just does not
  wake up… tried a different cable". The LLM said *Audio fault*; the note ("l bud no charge in case")
  says *Battery / not charging*. The prompt asks for one main problem, and this one has two readings.
- **The keyword fallback's "already told you" flag is noisy.** Right only ~45% of the time when it
  fires, against the LLM (it matches "already dispatched" in notes). The LLM's flag is used for every
  headline figure; the keyword flag only where no LLM label exists, and the digest calls it a rough signal.
- **Personal data.** Messages were sent to free third-party proxies without masking names, order
  numbers or phone numbers. Acceptable for this synthetic pack; masking would be needed for real data.

---

## Cost

**Measured on Qwen:** about 129 prompt + 40 completion tokens per ticket that reaches the LLM (the
system prompt is shared across the 20 tickets in a call).

| | Tickets to the LLM | Calls | Tokens (API-reported) | Cost on the free model |
|---|---|---|---|---|
| A normal run, no `--llm` | 0 | 0 | 0 | ₹0 |
| One week at 650 tickets, with `--llm` (planned at 6%) | ≈39 | 2 | ≈6,600 | ₹0 |
| One month | ≈170 | ≈9 | ≈29,000 | ₹0 |

- **Billing overhead:** APInex billed about 3× the API-reported tokens, so budget up to **0.1M tokens a month**.
- **On a paid model:** about 0.1 × (price per 1M tokens) a month, e.g. ₹10 at ₹100 per 1M.
- **No surprise bill:** labels are cached per ticket, so re-running a report never pays twice.
- **Spent so far on labelling:** ₹0 cash for 2,123 tickets, but one free 1M-token quota used up.

---

## Changing the prompt

1. Edit `THEME_GUIDE` or `SYSTEM_PROMPT` in `src/llm.py`.
2. Bump `PROMPT_VERSION` (e.g. `"v4"`); old labels are then ignored, not mixed in.
3. Re-label with `python -m src.llm --all --workers 6` (resumable; about 2 hours on the free tier), or
   only recent weeks with `--weeks 8`.
4. Check it: `python validation/score_check_set.py`, and read 20 of the new labels yourself.
5. Add a section to this log: what changed, why, what you checked it on, and what it got wrong.
