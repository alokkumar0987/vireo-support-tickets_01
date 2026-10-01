# Vireo Audio: support ticket digest, leaderboard and business case

Reads Vireo's 18-month helpdesk export (11,875 tickets) and produces, every week:

| Report | For | What it answers |
|---|---|---|
| **Weekly complaint digest** | Priya (Head of CX) | What are customers complaining about, what rose, who had to come back? |
| **Agent leaderboard** | Priya, team leads | Tickets closed per week, ranked **within each team**; Escalations & Warranty measured in days |
| **Business case + money audit** | Arjun (Finance) | What does fixing it save, and where is money leaking against policy? |

Runs **offline, with no API key, for ₹0**. An optional LLM step handles the 2–6% of tickets the local model is unsure about.

### At a glance

| | |
|---|---|
| **Goal** | Cut same-issue repeat contacts from **12.9% → 9.8%**, about **₹67,000 a quarter** (≈₹60,000 net of message costs) at 650 tickets a week, with proactive status updates |
| **Money for Finance** | **₹4.2 lakh a year** in confirmed refund-policy breaches, plus 3 findings for review |
| **Accuracy** | Complaint themes **99/100** on held-out tickets; the helpdesk's own tag gets 74/100 |
| **Cost to run** | ₹0 (local model); the optional LLM step uses a free model |

---

## Quick start (clean machine, Python 3.10+; tested on 3.11 and 3.13)

```bash
pip install -r requirements.txt

# Copy the brief's pack files into data/ (any filename prefix is fine):
#   tickets.csv  agents.csv  orders.csv  customers.csv  products.csv
python run.py --report all             # digest + leaderboard + business case, latest complete week
python run.py --dashboard              # the same numbers in a Streamlit dashboard
python -m pytest tests/ -q             # 88 tests, all offline, ~1 min
```

On the first run the local classifier trains itself in about 2 seconds from the committed LLM labels
(`cache/llm_labels.jsonl`). Nothing is downloaded, and no network call is made unless you pass `--llm`.

| Command | What it does |
|---|---|
| `python run.py` | Weekly digest only (latest complete ISO week) |
| `python run.py --week 2026-W24 --report all --output report.md` | A specific week, saved to a file |
| `python run.py --report scorecard` / `--report leakage` | Leaderboard only / business case only |
| `python run.py --llm` | Also send unsure tickets to the LLM (needs `.env`, copy `.env.example`). Off by default |
| `python validation/score_check_set.py` | Accuracy of every classifier on the 100-ticket check set |
| `python validation/score_reviews.py` | Precision of the repeat-contact and refund-breach rules |

---

## How it works

```
  data/  (the pack's CSV exports)
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. CLEAN                                        src/clean.py │
│    12,528 rows ──► 11,875 tickets                            │
│    • 653 Freshdesk-migration duplicates dropped              │
│      (helpdesk copy kept: its times are already IST)         │
│    • legacy resolution times  UTC ──► IST (+5:30)            │
│    • legacy CSAT "0" = no response ──► blank (3.32, not 2.49)│
│    • joined to agents, products, customers, orders           │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. CLASSIFY                   src/themes.py + src/cascade.py │
│    • one of 19 complaint themes per ticket (diagram below)   │
│    • repeat contacts per policy §10 (diagram below)          │
└──────────────────────────────┬───────────────────────────────┘
          ┌────────────────────┼─────────────────────┐
          ▼                    ▼                     ▼
┌──────────────────┐ ┌───────────────────┐ ┌────────────────────┐
│ 3a. DIGEST       │ │ 3b. LEADERBOARD   │ │ 3c. BUSINESS CASE  │
│ src/digest.py    │ │ src/scorecard.py  │ │ src/leakage.py     │
│                  │ │                   │ │                    │
│ themes + quotes  │ │ closed per week,  │ │ goal 12.9% ► 9.8%  │
│ rising alerts    │ │ within team,      │ │ ₹ saved / quarter  │
│ repeat contacts  │ │ 4-week average    │ │ policy breaches    │
│ product watch    │ │ Tier 2 in days    │ │ items for review   │
└────────┬─────────┘ └─────────┬─────────┘ └─────────┬──────────┘
         └─────────────────────┼─────────────────────┘
                               ▼
        run.py ──► markdown report        src/app.py ──► dashboard
```

### How a ticket gets its theme

The helpdesk's own `category` tag is wrong on about 1 ticket in 4, so every ticket is re-read.
Cheap sources go first; the LLM only sees what the local model can't decide.

```
  ticket = customer message + agent's closing note
     │
     ▼
┌─────────────────────────────┐  yes
│ LLM label already cached?   │ ─────► use it          2,123 tickets   paid once, reused free
└──────────────┬──────────────┘
               │ no
               ▼
┌─────────────────────────────┐  yes
│ local TF-IDF model is       │ ─────► use it          9,735 tickets   ₹0, offline
│ • confident (p ≥ 0.6)       │
│ • has seen similar tickets  │
│ • not predicting "Other"    │
└──────────────┬──────────────┘
               │ no   (2–6% of new tickets)
               ▼
┌─────────────────────────────┐  yes
│ run with --llm and a key?   │ ─────► ask the LLM, cache the answer
└──────────────┬──────────────┘
               │ no
               ▼
        keyword rules                                17 tickets
```

The TF-IDF model is trained on the cached LLM labels, never on the 100 check-set tickets, and
retrains itself when the labels change. Counts are for the current export; on new weeks, 2–6% of
tickets go to the LLM (≈2 calls a week at 650 tickets).

### What counts as a repeat contact (policy §10)

```
  same customer + same product + same issue family

  ticket A ──────────── A closed by agent X ───────────── + 30 days ──┤
                              │                                       │
                              └──── ticket B arrives ─────────────────┘
                                         │
                                         ├─► B is a REPEAT contact (costed at B's channel)
                                         └─► A counts against agent X's first-contact resolution
```

"Issue family" is the theme, except that cancellation, double charge, coupon and refund themes
count as one issue ("Money back"), and pickup, wrong item and damage as another ("Return &
replacement"): a refund chased after a cancellation is the same problem coming back.
Chasing a still-open ticket also counts. This gives **12.9%** of contacts, against 29.8% if any
return about the same product counted.

---

## Decisions worth knowing

| Question | What we decided | Why |
|---|---|---|
| What counts as a repeat contact? | Same customer, product and **issue**, within 30 days of the earlier ticket closing: **12.9%** | Policy §10 says "the same issue". Counting any return about the same product gives 29.8% (the first version's 34.3% counted from creation, not resolution), and only a third of those pairs were the same problem. On the 2,123 tickets the LLM read, 63% of flagged repeats mention an earlier contact, against 7% of other tickets. |
| Complaint categories | Our own theme classifier, not the helpdesk's `category` | The intake-bot tag was wrong on 26 of 100 checked tickets. |
| Leaderboard | Delivered as asked: tickets closed per week, ranked **within team** on a 4-week average; Tier 2 separate, in days | Counted by the week a ticket was **closed**. A weekly rank barely predicts next week's (rank correlation 0.13 in Chat). One Billing agent is alone on the Day shift, which explains that agent's volume. Policy §6 bars ranking Tier 2 on volume. |
| Cost per contact | Policy §4 per-channel figures (chat ₹210 … voice ₹520) | The email thread settles ₹290 blended over Arjun's ₹180. We use the channel each repeat actually came in on. |
| Scale | Totals × ~4.3 | The export has ~150 tickets a week; the brief says ~650. We assume it samples whole customers, so rates carry over (see limitations). |
| `GW-OTHER` refunds | Not counted as goodwill breaches | The code means "Goodwill / Other". None of the 37 `GW-OTHER` refunds over the ₹500 cap mention goodwill in the note. 40 `GW-OTHER` refunds are for product faults; they are reported separately, for review. |
| Refund + replacement breaches | Only counted when both tickets quote the same order ID | In the case review, 14 of 15 order-ID matches were real, and 0 of 5 customer + product matches could be confirmed. |
| Legacy money values | Treated as rupees | Policy §9 warns about a "native unit", but refund ÷ order value falls between 0.1 and 1.0 in both systems, and migrated duplicates carry identical amounts. |
| Client data in the repo | Not published | `data/` is git-ignored, and committed validation files hold ticket IDs and labels only. Scripts that need the text read it from `data/` into local `*_sheet.csv` files (git-ignored). |

---

## Results (latest complete week, 2026-W26)

- **Goal:** cut same-issue repeat contacts from **12.9% to 9.8%**, about **₹67,000 a quarter** at 650 tickets a week (≈₹60,000 net of an estimated ₹28,000 a year in messages). 47% of repeats are customers chasing a delivery, refund, pickup, repair or double-charge refund; proactive updates answer those before they ask.
- **Money for Finance:** about **₹4.2 lakh a year** in confirmed policy breaches (refund *and* replacement on the same order; goodwill over ₹500 hidden under the `RETURN-QC-OK` code). For review, not counted:
  - **40 product-fault refunds** under the catch-all `GW-OTHER` code, mostly past the 7-day DOA window and by Tier 1: they look like warranty buy-backs without Tier 2 approval;
  - **749 refunds** that appear in agent notes with **no amount recorded** anywhere: reconcile against the payment gateway;
  - **342 warranty replacements** closed by Tier 1 agents: check against RMA approvals.
- **Accuracy:** complaint themes 99/100 on held-out tickets (labels by Claude Code; a person reviewed 30 and agreed on all 30); repeat flag 19/20 each way; refund breaches 14/15. Details, including what is *not* validated, in [`docs/validation.md`](docs/validation.md).

---

## Project layout

```
run.py                   entry point: reports, --llm, --dashboard
src/
  clean.py               load, de-duplicate, fix time zones and CSAT, join, repeat contacts
  llm.py                 prompt (v3), OpenAI-compatible client, label cache
  cascade.py             TF-IDF + logistic regression, confidence and novelty routing
  themes.py              theme cascade, keyword rules, issue families
  digest.py              weekly digest (Poisson alerts, lot check with Bonferroni correction)
  scorecard.py           leaderboard: Tier 1 within team, Tier 2 in days, FCR
  leakage.py             business case, money audit, sample scaling
  analytics.py           SLA / transfer KPIs for the dashboard
  app.py                 Streamlit dashboard
tests/                   88 offline tests
cache/llm_labels.jsonl   2,123 LLM labels (committed: the tool runs without an API key)
validation/              check set, human review, case reviews and their scorers
docs/prompt_log.md       prompt versions v1 → v3, what was thrown away, costs
docs/validation.md       all accuracy evidence
analysis/                optional embedding clustering used to check the theme list
                         (pip install sentence-transformers tabulate)
data/                    the pack goes here (git-ignored)
```

---
## 🎥 Demo

[![Watch the Demo](https://img.youtube.com/vi/CrIZeg2EuaE/maxresdefault.jpg)](https://youtu.be/CrIZeg2EuaE)
## Known limitations

- **Accuracy will be lower on real tickets.** The ticket text is heavily templated, so expect worse than 99% on real, messier text.
- **Who checked:** Claude Code labelled the check set and reviewed the cases. A person then reviewed 30 of the 100 labels and agreed on all 30 themes, changing 2 "already told you" flags where the note says only "see prev". The reviewer had seen Claude's labels first, so this is a review, not a blind check. A blind human check of fresh tickets is described in `docs/validation.md`.
- **Customer messages were sent to free third-party LLM proxies** without masking personal details. Fine for a synthetic exercise; not for production.
- **One theme per ticket.** Multi-issue tickets get their main issue only.
- **The "Money back" issue family occasionally groups two unrelated refunds** as one repeat (1 of 20 reviewed).
- **The ×4.3 scaling assumes the export samples whole customers.** If it sampled individual tickets, most repeat pairs would be cut apart and the true repeat rate would be well above 12.9%.
- **The keyword "already told you" rule is noisy** (right 45% of the time when it fires, against the LLM). Headline figures use the LLM-read tickets; the digest's weekly figure is labelled a rough signal.
- **7% of non-repeat tickets (LLM-read) still mention an earlier contact.** Customer IDs may split across channels. Not investigated.
