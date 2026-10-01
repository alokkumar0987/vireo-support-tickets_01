# Memo

**To:** Priya Raman, Head of Customer Experience, Vireo Audio
**Cc:** Arjun Mehta, Neha Kulkarni
**Re:** What 18 months of tickets say, and what we suggest doing about it
**Reading time:** about 6 minutes

---

You asked for a weekly digest of what customers complain about and a leaderboard of tickets closed. Both are built and run every week at no cost. Reading all 11,875 tickets, including the ones nobody reads, also turned up three things you should know.

## 1. What you get every Monday

- **A one-page digest.**
  - It opens with three lines: the most common complaint, anything that rose more than normal, and how many customers had to come back.
  - Below that is a table of 19 complaint types, each with a real customer quote.
  - The complaint types come from reading each message. **The category your chatbot assigns is wrong on about 1 ticket in 4**, so we don't use it.
- **The leaderboard you asked for.**
  - Tickets closed per agent per week, shown team by team.
  - Each agent's first-contact resolution and customer rating sit beside the count.
  - The warranty team has its own table, measured in days to resolve.

## 2. What the tickets say

**Customers keep coming back about the same thing, and mostly just to ask where things stand.**
- 13% of all contacts are a customer returning about an issue we had already closed, within 30 days.
- Nearly half of those are chasing a status: *where is my refund, my delivery, my pickup, my repair?*
- On the tickets our AI read in full, nearly two in three of these repeat tickets say so ("third time now", "I was told it was resolved"), against 1 in 14 of other tickets.
- Neha's frontline were right that this isn't just noise.

**The biggest complaints are about getting the product and the money, not the product itself.**
- Late or missing deliveries are the single largest complaint, at 13%.
- Refunds, double charges, pickups and cancellations together are larger still.
- Among product faults, Bluetooth pairing, battery and one-sided audio lead. No product or manufacturing batch stands out as unusually faulty once you allow for how many were sold.

**A tickets-closed ranking needs reading with care.** We built it the way you asked, with three guardrails:
- **Ranked within each team, over four weeks.** Agents close only a handful of tickets a week, so one week's order is mostly luck.
- **Queue size is shown.** For example, one Billing agent is the only person on the Day shift, so that agent closes about twice as many as colleagues on other shifts. That's the queue, not extra effort.
- **The warranty team is kept separate,** as Neha asked and your own policy requires. On a single list, five of the six bottom places would be theirs.

## 3. The goal we propose

> **Cut repeat contacts about the same issue from 12.9% to 9.8% of all contacts, worth about ₹67,000 a quarter (₹2.7 lakh a year) at your volume, by telling customers where their refund, delivery, pickup or repair stands before they have to ask.**

That needs no new staff. It means an automatic message when a refund is processed, when a courier updates, when a pickup is booked, and when a repair moves stage. The messages themselves cost roughly ₹28,000 a year, so the net saving is about ₹60,000 a quarter. The saving assumes half of status-chasing contacts disappear; a quarter would still save about ₹34,000 a quarter.

**Our suggestion:** try it on refunds and deliveries for four weeks and watch the repeat number in the Monday digest.

## 4. For Arjun: money worth a look

- **About ₹4.2 lakh a year in clear breaches of your refund policy.**
  - Customers who got both a refund and a replacement on the same order (69 cases in the export).
  - Goodwill refunds over the ₹500 cap recorded under a "return passed QC" code, even though the note says the item was never collected and the refund was goodwill.
- **40 refunds for faulty products filed under "Goodwill / Other".** Almost all came after the 7-day dead-on-arrival window, and most were made by frontline agents. Your policy says repair or replacement, or a warranty buy-back approved by the warranty team. These look like buy-backs that skipped that approval. We have not counted them as a loss; worth a review.
- **749 refunds that appear only in agent notes.** The note says the refund was processed, but no amount is recorded anywhere in the helpdesk. The money may be recorded in the payment system, but your support reports can't see it. We recommend matching these against the payment gateway.
- **342 warranty replacements closed by frontline agents,** where policy reserves that approval for the warranty team. They may have been approved elsewhere; worth a sample check.

## 5. How sure we are

- **Checking:** we checked the tool against 100 tickets it had never seen, plus 60 of its money and repeat findings, one by one. It was right 99 times in 100 on complaint type, 19 in 20 on repeats, and 14 in 15 on refund breaches.
- **Caveats:**
  - The export looks like about a quarter of your real volume, so we scaled rupee figures up to 650 tickets a week.
  - Our AI tooling did the checking. One of our team then reviewed 30 of its answers and agreed with all of them, but had seen the AI's answers first. A blind 20-minute check by one of your team leads would make this stronger.
- **Cost to run:** nothing. The optional AI step for unclear tickets uses a free model today.

## What we need from you

1. **A yes** to the four-week trial of proactive refund and delivery updates.
2. **Someone in Finance** to match the 749 note-only refunds against the payment gateway.
3. **A decision** on whether the leaderboard goes to agents or stays with team leads. We'd suggest team leads first.
