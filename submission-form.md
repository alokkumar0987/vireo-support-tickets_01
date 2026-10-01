# Submission form: Vireo Audio, Support Tickets (Set A)

---

### What did you build, and what business outcome does it move?* State the number and the money.*

**What we built:** a command-line tool (with an optional dashboard) that reads the 18-month export and produces three weekly reports.
1. **A one-page complaint digest for Priya.** Every ticket gets one of 19 complaint themes, taken from the customer's message and the agent's note, not the chatbot tag, which is wrong on 26% of tickets. The digest shows the three lines worth reading, a theme table with real quotes, repeat contacts, and alerts that fire only on statistically unusual changes.
2. **The tickets-closed leaderboard Priya asked for.** It is ranked within each team on a 4-week average, alongside first-contact resolution and CSAT. Escalations & Warranty is shown separately, measured in days.
3. **A business case and money audit for Arjun.**

**The outcome:**
> **Cut repeat contacts about the same issue from 12.9% to 9.8% of all contacts, worth about ₹67,000 a quarter (₹2.7 lakh a year; about ₹60,000 a quarter net of message costs), by sending proactive status updates for refunds, deliveries, pickups and repairs.**

**The arithmetic:**
- **How big the problem is:** 1,526 of 11,875 tickets (12.9%) are the same customer returning about the same product and issue within 30 days of the earlier ticket closing (policy §10). 720 of those (47%) are status chasers: delivery, refund, pickup, warranty status, or a double charge waiting for its auto-refund.
- **The lever:** halving the status chasers takes 6.1% × 50% = 3.0 points off the rate.
- **At Vireo's volume:** 650 × 52 = 33,800 contacts a year, × 3.03% = 1,025 contacts avoided. At ₹263 each (policy §4 channel costs, weighted by the channels the chasers actually use) that is **₹2,69,817 a year, or ₹67,454 a quarter**.
- **If the lever works less or better:** removing 25% of chasers saves ₹33,727 a quarter; removing 75% saves ₹1,01,181.
- **The messages cost money too:** about 37,000 orders a year × 3 updates × ₹0.25 (assumed SMS / WhatsApp rate) ≈ ₹27,764 a year, so the net saving is about **₹60,513 a quarter**.

**Money found for Finance** (export totals scaled ×4.26 to 650 tickets a week and annualised):
- About **₹4.2 lakh a year** in confirmed policy breaches:
  - refund **and** replacement on the same order (69 orders, each valued at the smaller of the two): ₹3.3 lakh a year;
  - goodwill over the ₹500 cap hidden under the `RETURN-QC-OK` code (9 tickets, every note says "as goodwill"): ₹1.0 lakh a year.
- **Flagged for review, not counted:**
  - **40 product-fault refunds filed under `GW-OTHER` ("Goodwill / Other").** 27 of the 29 with a known order date came after the 7-day DOA window, and 31 were made by Tier 1. Policy §5/§6 allows repair, replacement or a Tier 2 buy-back (`WTY-BUYBACK`), so these look like buy-backs without Tier 2 approval. An earlier draft counted them as "goodwill over the cap" (₹7.3 lakh headline); none of their notes mention goodwill, so we removed them from the total;
  - **749 refunds** that appear in agent notes but have no amount recorded anywhere in the data;
  - **342 warranty replacements** closed by Tier 1 agents (policy §6 reserves these for Tier 2).

---

### What does one run cost, and what would a month cost at Vireo's volume (roughly 650 tickets a week)?* Show the arithmetic. If you used no paid calls, say so.*

**No paid calls were used.** Cash cost to date: **₹0**. The only spend was Claude Code, used as the coding assistant: **[fill in your plan and its cost]**.

**A normal run (`python run.py`): ₹0.** Everything runs on the laptop: cleaning, the TF-IDF classifier, the keyword fallback and every report. The classifier trains in about 2 seconds from the LLM labels saved in the repo.

**With the optional LLM step (`python run.py --llm`):**
- **Share of tickets:** only tickets the local model is unsure about or finds unfamiliar go to the LLM. That was **2–6%** on held-out tickets (6% with the first model, 2% after retraining on more labels). The cost below uses 6%, to be safe.
- **Per week:** 650 × 6% ≈ 39 tickets. At 20 per call that is 2 calls.
- **Tokens:** measured on `qwen/qwen3.8-max:free`, about 129 prompt + 40 completion tokens per ticket. So 39 × 169 ≈ 6,600 tokens a week, or **≈ 29,000 tokens a month**.
- **Billing overhead:** on the first provider, billing counted about 3× what the API reported, so allow **≤ 0.1M tokens a month**.
- **Monthly cost:** the model is free, so ₹0 a month. On a paid model it would be about 0.1 × (price per 1M tokens) a month. For example, at ₹100 per 1M tokens, that is ₹10 a month.
- **Why the bill can't surprise Arjun:** labels are cached per ticket, so re-running a report never pays twice.

**One-off labelling so far:** 2,123 tickets labelled by the LLM (prompt v3). All used free tiers:
- 839 on APInex `free/gpt-6-luna`, which used up its **1M-token free quota**;
- 1,284 on xkiro `qwen/qwen3.8-max:free`.

---

### How do you know it works?* Sample size, how you checked, error rate, and the kind of case it gets wrong.*

Every output the business case relies on was checked against individually reviewed cases. Full detail is in `docs/validation.md`; the numbers can be regenerated with `validation/score_check_set.py` and `validation/score_reviews.py`.

| Output | Sample | Right | Error rate (95% range) | What it gets wrong |
|---|---|---|---|---|
| Complaint theme (hybrid classifier) | 100 held-out tickets, stratified by half-year and channel | 99/100 | 1% (0–5%) | Tickets that fit two themes, e.g. "left bud won't wake up, tried another cable" (charging or audio?) |
| *Chatbot `category` tag, for comparison* | same 100 | 74/100 | **26%** | Many "Other" tags on invoices, cancellations, deliveries |
| *Human review of the theme labels* | 30 of the 100 | 30/30 themes; 2 "already told you" flags changed | n/a | The reviewer had seen Claude's labels: a review, not a blind check (`validation/hand_check_results.md`) |
| "Same-issue repeat" flag | 20 random flagged pairs | 19/20 | 5% | A discount refund, then an unrelated return refund, grouped as one "money back" issue |
| "Not a repeat" | 20 random unflagged same-customer pairs | 19/20, 0 clear misses | ≤5% | One pair where the note says "issue recurred" but the messages differ |
| Refund + replacement breach, matched on order ID | 15 random flagged orders | 14/15 | 7% | A goodwill credit mistaken for a refund |
| Same breach, matched on customer + product | 5 | 0/5 confirmed | n/a | Can't tell whether it was the same purchase |

**Independent check on repeats:** on the 2,123 tickets the LLM read, 63% of the 273 flagged repeats mention an earlier contact ("I already told you", "third time"), against 7% of other tickets. The keyword rule used for the remaining tickets is weaker (right 45% of the time when it fires) and gives 49% vs 17%.

**Checks on the digest's alerts:**
- **Rising-theme alert:** a backtest over the latest 30 weeks gave 8 alerts where at most about 5 would come from chance (541 theme-weeks at p < 0.01). Across all 69 testable weeks there were 20, so the alert fires somewhat more often than chance, not much more. The digest says a single alert means "worth a look"; two weeks running is a trend.
- **Lot-defect check:** it is corrected for testing 1,270 lots, and flags none.

**Caveats, stated plainly:**
- **Who checked:** the 100 labels and 60 verdicts were produced by Claude Code, blind to model output; a person then reviewed 30 of the 100 labels and agreed on all 30 themes, changing 2 "already told you" flags where the note says only "see prev". The reviewer had seen Claude's labels first, so this is a review, not a blind check (see the next question).
- **The ticket text is templated,** so real-world accuracy will be lower than 99%.
- **Small sample:** n = 20 per repeat group is too few for tight error bounds.

---

### Did you change, narrow, or push back on the client's ask?* What, when, and why. [can only raise your score]*

1. Leaderboard: delivered, with guardrails, not refused. Priya said "the leaderboard stays", so it does, as tickets closed per week. We changed how it is ranked because the data made a raw ranking misleading:
   - Weekly ranks are mostly luck. Agents close 2–9 tickets a week, and this week's order barely predicts next week's (rank correlation 0.13 in Chat). So it is ranked on a 4-week average.
   - "Closed" means closed. Tickets are counted in the week they were resolved, not opened; 65% of Tier 2 cases close in a later week than they opened.
   - Queues differ between teams. Team averages range from 2.9 to 6.5 tickets a week, so agents are ranked only within their team.
   - Some volume is structural. The top Billing closer is the only Billing agent on the Day shift, so that agent is flagged 🧍 rather than crowned.
   - Tier 2 is separate. On one combined list, 5 of the 6 bottom places would be Escalations & Warranty (Neha's warning, and policy §6). They are measured in days.
   - We dropped the "balanced score" from our own first version (35/35/30 weights we had invented). The columns are shown instead.
2. "Repeat contact" was narrowed to the policy's own definition.
   - Our first version reported 34.3% and promised ₹14 lakh a year by cutting it to 20%. It counted any second ticket within 30 days.
   - Policy §10 says "the same issue… within 30 days of resolution". Applying that, the rate is 12.9%, and the honest saving is ₹2.7 lakh a year.
   - We chose the smaller number because Arjun asked "what does it save?", and the old answer would not survive his first question.
3. Cost per contact. Arjun said ₹180; Priya corrected that to ₹290 blended. We used the policy's per-channel costs, weighted by the channel each repeat actually came in on, as policy §10 says.
4. SLA breach credits are shown as a cost, not "leakage".** They are the price of late first responses, not a policy breach. Our first version had added them to the leakage total.

---

### What is wrong with what you are handing us?* Be specific: bugs, shortcuts, things you know are off. [can only raise your score]*

1. **The validation was done by an AI, not a person.** Claude Code labelled the 100-ticket check set and gave the 60 case verdicts. It did so blind to model outputs, except check-set row 61, which it had seen once. It is itself a language model, so it may share blind spots with the LLM being checked. A person reviewed 30 of the labels afterwards and agreed with all 30 themes, but had seen Claude's labels first. So there is still **no blind human check**: labelling 20 fresh tickets before seeing any output is the first thing to do next.
2. **Customer messages were sent to free third-party LLM proxies** (APInex, xkiro) **without masking** names, order numbers or phone numbers. Acceptable for synthetic assessment data; not acceptable for Vireo in production. Masking before sending is about 20 lines of code and was not done.
3. **The rupee figures assume the export is a random sample.** It has about 150 tickets a week, while the brief says about 650, so totals are scaled ×4.26. That assumes the export samples whole customers. If it sampled individual tickets, most repeat pairs would be split apart: the true repeat rate would be well above 12.9% and the goal worth more. If it was drawn some other way, the rupee figures are off.
4. **The 50% reduction in status chasers is an assumption,** not a measurement. The sensitivity table shows 25% and 75%.
5. **The "Money back" issue family sometimes groups two unrelated refunds** as one repeat (1 of 20 reviewed).
6. **7% of non-repeat tickets (LLM-read) still mention an earlier contact.** Customer IDs may be split across channels (chat vs email identities), so the repeat rate may be **under**-counted. We did not investigate. The keyword rule behind the digest's weekly "mention an earlier contact" line is noisy (45% precision), and the digest says so.
7. **The 749 "refund in note, no amount recorded" cases are a heuristic** based on note wording ("refund initiated…"). Some may be refunds that were promised but never executed.
8. **The 342 Tier 1 warranty replacements only cover tickets with a known order date** (805 of 1,202 replacements). The count is a lower bound, and some may have been approved by Tier 2 outside the ticket.
9. **The LLM labels come from two models** (gpt-6-luna, then qwen3.8-max), using the same prompt v3. They agreed 40/40 on an overlap sample, but they are still two sources.
10. **The in-sample routing figure is flattering.** After retraining on 2,022 labels, the model routes only 0.2% of historical tickets to the LLM, because many are now its own training data. Plan on the held-out ~6%.
11. **The dashboard was wrong until the final review, and is still less tested than the CLI.**
    - A late code review found that the dashboard disagreed with the report:
      - it ran on data without themes, so every agent's FCR was off by up to 29 points;
      - its repeat tile showed the discarded 29.8%;
      - two trend figures ("▼ -8.4%") were hardcoded and never calculated;
      - its category chart used the chatbot tag we tell Priya not to trust.
    - All four are fixed. Every report now goes through one function that adds themes (`themes.ensure_themes`), and a regression test covers it.
    - The dashboard is still only smoke-tested: each view loads, and spot values were checked against `report.md`. Its sidebar filters apply only to the Overview tiles, and it says so.
12. **One theme per ticket.** Multi-issue tickets get their main issue only.
13. **The "weekly ranks are noise" argument partly depends on the sample.**
    - In the export, agents close 2–9 tickets a week. If the export is about a quarter of real volume (point 3), the real weekly counts are about 4× higher, and weekly ranks would be steadier than the 0.13 correlation suggests.
    - The within-team ranking and the Tier 2 separation still hold. The 4-week average may be more caution than full data needs.
    - Re-check it on the complete export before telling Priya that weekly ranks are meaningless.
14. **Late fixes from our own final review** (all in the numbers above): the ₹7.3 lakh breach headline became ₹4.2 lakh (`GW-OTHER` fault refunds were wrongly called goodwill); the leaderboard counted tickets by the week they were opened, not closed; the "already told you" evidence came from a 45%-precision keyword rule; FCR for a past week could see later tickets. Each has a test now.

---

### What did you deliberately leave out, and why that rather than something else?*

- **Live helpdesk integration.** The tool runs on weekly exports. The value is in the definitions and the numbers, and an integration adds nothing to them in five hours.
- **Merging customer identities across channels.** It would probably raise the repeat rate (see point 6 above), but it needs matching rules we could not validate from this data.
- **Staffing and shift analysis of SLA breaches** (email breaches at 11.6%). Real money, about ₹9.9 lakh a year in credits, but it is a rostering project, not a complaint digest.
- **Multi-label themes, and dashboard polish.** Neither changes the business number.
- **An LLM-written narrative digest.** The three-line summary is built from the numbers, so it cannot misquote a count. An LLM would make it read better, but it would add a risk of wrong numbers and a per-week call.
- **Why these rather than something else:** we spent the time on getting the *definitions* right (what a repeat is, what counts as a breach, what an alert means). A wrong definition makes every downstream number wrong, and that is what the first version got wrong.

---

### Anything you built or found that nobody asked for?*

1. **749 refunds exist only in agent notes.** Their notes say "refund processed / initiated", but no refund amount is recorded on that ticket or any other ticket for the same order. That is up to ₹18.5 lakh of order value invisible to support's own reports.
2. **Goodwill hidden under another refund code.** Refunds "processed without pickup as goodwill" were coded `RETURN-QC-OK` ("return received and passed QC"). That is a contradiction, and it hides them from any report that filters on the goodwill code. 9 of them exceed the ₹500 cap.
3. **342 warranty replacements closed by Tier 1 agents,** against policy §6.
4. **Recorded refunds are counted several times in the data.** The same refund is re-entered on the customer's follow-up tickets ("refund confirmed credited (3499)"). A naive sum overstates refunds; our first version did exactly that.
5. **Unsupervised clustering of all 11,875 messages** (`analysis/cluster_themes.py`, local embeddings, no API):
   - It confirmed that our 19 themes cover what customers actually write (74% agreement, against 70% for the chatbot tags).
   - It found 129 "two entries on my statement" double-charge tickets our rules had filed as "Other".
   - It showed that refund and pickup chasers have the highest repeat rates.
6. **Dataset traps handled:**
   - 653 migration duplicates removed;
   - legacy resolution times moved from UTC to IST;
   - CSAT "0 = no response" (true CSAT 3.32, not 2.49);
   - checked that legacy money values are rupees, despite policy §9's "native unit" warning.
7. **Two "findings" from our own first version, shown to be noise and removed:**
   - a "65W GaN charger defect lot" (a charger's tickets being about charging is expected);
   - 13 "anomalous" manufacturing lots (about what chance produces across 1,270 lots).

---

### What did you use AI for?* Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.*

**Tools and models:**
- **[v1 of the code: fill in the assistant you used for the first version, e.g. Antigravity / Gemini]**
- **Claude Code (Opus 5.5):** coding assistant for the audit, rebuild, tests and documents.
- **APInex `free/gpt-6-luna`:** ticket labelling (839 tickets).
- **xkiro `qwen/qwen3.8-max:free`:** ticket labelling (1,284 tickets).
- **`cohere/command-a`:** a cross-check only (40 tickets).
- **Local `all-MiniLM-L6-v2` embeddings:** clustering.
- **scikit-learn TF-IDF + logistic regression:** the local classifier.

**Where AI helped:**
- **Labelling:** the LLM labelled 2,123 tickets, which trained a free local model that matches it 99% of the time.
- **Auditing our own first version:** Claude Code found the circular accuracy test, the FCR metric charged to the wrong agent, the refund double-counting, and the noise alerts.
- **Reading at scale:** reading every ticket surfaced findings a person would not dig out by hand, such as refunds that exist only in notes and goodwill hidden under the QC code.

**Where it wasted time:**
- **Models vanished.** Free models were renamed or put behind a paywall mid-task (402/404 errors on 2 of the first 3 models tried).
- **The free quota ran out mid-run.** It ran out after 839 of 11,875 tickets, which killed the plan to label everything.
- **Shell escaping corrupted code** in a few edits: `\n` and `\b` inside quoted scripts.
- **The novelty threshold needed a fix** after tests exposed that it breaks on repetitive data.

**What we threw away** (details in `docs/prompt_log.md`):
- **Prompt v1 (14 themes):** GST invoices landed in "Other".
- **Prompt v2 (18 themes):** 3 error types in a 40-ticket review.
- **The plan to send all tickets to the LLM:** replaced by the hybrid.
- **Embedding-based novelty detection:** needed torch, several GB for a clean-machine install.
- **From the v1 codebase:**
  - the circular "95% precision" benchmark;
  - the first-substring keyword taxonomy ("charged" → Battery);
  - the "balanced score";
  - the 34.3% repeat definition and the ₹14L claim;
  - the raw-count lot alerts;
  - the GaN spotlight;
  - SLA credits counted as leakage.

**Screen recording:** ["https://drive.google.com/file/d/1WWr3yuSgmrUyaOkf0J0gRILY3tET22zc/view?usp=drivesdk"]

---

### Your Public Google Drive Link

["https://drive.google.com/file/d/1WWr3yuSgmrUyaOkf0J0gRILY3tET22zc/view?usp=drivesdk"]

---

### Someone picks this up on Monday and you are unreachable.* The three things they need to know.*

1. **How to run it.**
   - `pip install -r requirements.txt`, copy the pack files into `data/`, then run `python run.py --report all`.
   - No key is needed. `python -m pytest tests/ -q` should show 88 passing.
   - For new weekly data, run with `--llm` (and a `.env` from `.env.example`) so the 2–6% of unclear tickets get LLM labels. They are cached, and the local model retrains itself.
2. **The definitions are the product. Don't change them casually.**
   - A **repeat contact** means same customer, product and *issue* within 30 days of resolution (12.9%), not "any second ticket" (29.8% or 34.3%).
   - **Never use the chatbot `category`** for complaint counts; it is wrong 1 time in 4.
   - **Never rank agents across teams, or on a single week,** and never rank Tier 2 on counts.
   - `docs/validation.md` explains how each definition was checked.
3. **Maintenance.**
   - **Free LLM models disappear.** If `--llm` fails with 402/404, change `LLM_MODEL` in `.env`: list the models and pick one ending in `:free`.
   - **If you edit the prompt,** bump `PROMPT_VERSION` in `src/llm.py`, which re-labels everything, and log it in `docs/prompt_log.md`.
   - **Hand-check 20 tickets a week** to catch drift.

---

### Honest hours spent.* One number.*

**[6]**

---

### Github Repo Link*

https://github.com/alokkumar0987/vireo-support-tickets_01
