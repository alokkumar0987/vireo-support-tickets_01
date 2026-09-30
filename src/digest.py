"""
Weekly complaint digest for Priya Raman (Head of CX): what customers complained about this week,
what changed, and what it cost, in a form a busy person will actually read.

Design choices (see docs/prompt_log.md for the classifier):
- Themes come from src/themes.attach_themes (hybrid TF-IDF / LLM / keyword), NOT the intake bot's
  `category`, which is wrong on ~26% of tickets (validation/results.md).
- "Rising" and "product" alerts need a statistically unusual count (Poisson test vs the previous
  8 weeks, p < 0.01, and >= 3 tickets above expected). A theme with ~20 tickets a week moves by
  +/-5 on its own; flagging every wobble would teach the reader to ignore the digest.
- Lot-code alerts use hardware-fault tickets only, per unit sold, with a Bonferroni correction
  for the ~1,270 lots tested. The previous version flagged raw ticket counts, which was noise.
- Customer quotes are shown with order / RMA numbers masked.

    python -m src.digest                 # latest complete week
    python -m src.digest --week 2026-W24
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clean import clean_tickets_data, latest_complete_week, DEFAULT_DATA_DIR  # noqa: F401 (re-exported for app.py)
from src.themes import ensure_themes

BASELINE_WEEKS = 8
ALERT_P_VALUE = 0.01
ALERT_MIN_EXCESS = 3
HARDWARE_THEMES = {
    "Battery drain / not charging", "Audio fault (one side, distortion, mic)", "Pairing & connection drops",
    "Screen / touch / hardware fault", "Damaged in transit / dead on arrival", "App / firmware update failure",
}


# ----------------------------------------------------------------------------- helpers

def _poisson_sf(k, mu):
    """P(X >= k) for X ~ Poisson(mu); scipy-free so the clean-machine install stays small."""
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    from math import exp, lgamma, log
    cdf = sum(exp(i * log(mu) - mu - lgamma(i + 1)) for i in range(int(k)))
    return max(0.0, 1.0 - cdf)


def mask_quote(text, max_len=110):
    """Short customer quote with order/RMA/phone numbers masked and whitespace collapsed."""
    t = re.sub(r"(?i)\bvr\d+\b", "VR•••", str(text or ""))
    t = re.sub(r"(?i)\brma\d+\b", "RMA•••", t)
    t = re.sub(r"\b\d{10}\b", "••••••••••", t)
    t = re.sub(r"\[ivr transcript\]\s*", "", t, flags=re.I)
    t = " ".join(t.split())
    return t if len(t) <= max_len else t[:max_len - 1].rsplit(" ", 1)[0] + "…"


def _pick_quote(msgs):
    """Prefer a short, self-contained message (long ones are mostly greetings and signatures)."""
    msgs = [m for m in msgs if isinstance(m, str) and m.strip()]
    if not msgs:
        return ""
    short = [m for m in msgs if 25 <= len(m) <= 140]
    return mask_quote(sorted(short or msgs, key=len)[len(short or msgs) // 2])


def _weeks_before(df, week, n):
    weeks = sorted(df["year_week"].unique())
    i = weeks.index(week)
    return weeks[max(0, i - n):i]


def _unusual_rise(count, expected):
    p = _poisson_sf(count, expected)
    return bool(p < ALERT_P_VALUE and count - expected >= ALERT_MIN_EXCESS), p


# ----------------------------------------------------------------------------- sections

def theme_table(df, week):
    """Per theme: tickets this week, share, expected from the 8-week baseline share, repeat rate, a quote."""
    cur = df[df["year_week"] == week]
    base = df[df["year_week"].isin(_weeks_before(df, week, BASELINE_WEEKS))]
    base_share = base["theme"].value_counts(normalize=True)
    rows = []
    for theme, g in cur.groupby("theme"):
        expected = base_share.get(theme, 0.0) * len(cur)
        alert, p = _unusual_rise(len(g), expected)
        rows.append({
            "theme": theme, "tickets": len(g), "share": len(g) / len(cur), "expected": expected,
            "change_vs_normal": len(g) - expected, "rising_alert": alert, "p_value": p,
            "repeat_rate": g["is_repeat_contact_30d"].mean(),
            "repeat_cost_inr": g["repeat_cost_inr"].sum(),
            "quote": _pick_quote(g["customer_message"].tolist()),
        })
    return pd.DataFrame(rows).sort_values("tickets", ascending=False).reset_index(drop=True)


def product_watch(df, week):
    """Per product: hardware-fault tickets this week vs expected from its own 8-week average."""
    cur = df[(df["year_week"] == week) & df["theme"].isin(HARDWARE_THEMES)]
    prior_weeks = _weeks_before(df, week, BASELINE_WEEKS)
    base = df[df["year_week"].isin(prior_weeks) & df["theme"].isin(HARDWARE_THEMES)]
    rows = []
    for product, g in cur.groupby("product_name"):
        expected = (base["product_name"] == product).sum() / max(len(prior_weeks), 1)
        alert, p = _unusual_rise(len(g), expected)
        rows.append({"product": product, "hardware_tickets": len(g), "expected": expected,
                     "top_fault": g["theme"].value_counts().index[0], "alert": alert, "p_value": p})
    return pd.DataFrame(rows).sort_values("hardware_tickets", ascending=False).reset_index(drop=True)


def detect_manufacturing_lot_anomalies(df_clean, data_dir=DEFAULT_DATA_DIR, min_units=20, family_alpha=0.05):
    """
    Lots with more HARDWARE-fault tickets per unit sold than their SKU's average, tested with a
    Poisson test and a Bonferroni correction across all lots (so ~0 false alarms, not ~13).
    Only tickets that quote an order_id carry a lot code; the rest cannot be attributed.
    """
    df = ensure_themes(df_clean)
    orders = pd.read_csv(glob.glob(os.path.join(data_dir, "*orders*"))[0])
    units = orders.groupby(["sku", "lot_code"])["qty"].sum().rename("units")
    hw = df[df["theme"].isin(HARDWARE_THEMES)].dropna(subset=["lot_code"])
    tickets = hw.groupby(["product_sku", "lot_code"]).size().rename_axis(["sku", "lot_code"]).rename("hw_tickets")
    lots = pd.concat([units, tickets], axis=1).fillna(0).reset_index()
    sku_rate = lots.groupby("sku")["hw_tickets"].sum() / lots.groupby("sku")["units"].sum()
    lots["expected"] = lots["units"] * lots["sku"].map(sku_rate)
    lots["ratio"] = lots["hw_tickets"] / lots["expected"].replace(0, np.nan)
    lots["p_value"] = [_poisson_sf(k, mu) for k, mu in zip(lots["hw_tickets"], lots["expected"])]
    threshold = family_alpha / max(len(lots), 1)
    lots["is_anomaly"] = (lots["p_value"] < threshold) & (lots["units"] >= min_units)
    names = df.drop_duplicates("product_sku").set_index("product_sku")["product_name"]
    lots["product_name"] = lots["sku"].map(names)
    lots.attrs.update({"lots_tested": len(lots), "p_threshold": threshold})
    return lots.sort_values("p_value").reset_index(drop=True)


# ----------------------------------------------------------------------------- digest

def build_weekly_digest(df, week):
    df = ensure_themes(df)
    cur = df[df["year_week"] == week]
    if len(cur) == 0:
        raise ValueError(f"No tickets in week {week}")
    prior = _weeks_before(df, week, BASELINE_WEEKS)
    prev_week = prior[-1] if prior else None
    prev_n = (df["year_week"] == prev_week).sum() if prev_week else 0
    avg_n = df["year_week"].isin(prior).sum() / max(len(prior), 1)

    themes = theme_table(df, week)
    repeats = cur[cur["is_repeat_contact_30d"]]
    repeat_by_theme = repeats.groupby("theme")["repeat_cost_inr"].agg(["size", "sum"]).sort_values("size", ascending=False)
    novel = cur[cur.get("route_reason", pd.Series("", index=cur.index)) == "novel"]
    lots = detect_manufacturing_lot_anomalies(df)

    return {
        "week": week,
        "date_range": f"{cur['created_at_dt'].min():%d %b} – {cur['created_at_dt'].max():%d %b %Y}",
        "days_covered": cur["created_at_dt"].dt.date.nunique(),
        "tickets": len(cur), "prev_week_tickets": int(prev_n), "avg_weekly_tickets": avg_n,
        "themes": themes,
        "rising": themes[themes["rising_alert"]],
        "repeat_contacts": len(repeats), "repeat_share": len(repeats) / len(cur),
        "repeat_cost_inr": float(repeats["repeat_cost_inr"].sum()),
        "repeat_by_theme": repeat_by_theme,
        "says_already_contacted": float(cur["is_repeat_claim"].mean()),
        "vague_agent_notes": float((cur["note_outcome"] == "unclear").mean()),
        "products": product_watch(df, week),
        "lot_alerts": lots[lots["is_anomaly"]],
        "lots_tested": lots.attrs["lots_tested"],
        "novel": novel,
        "theme_sources": cur["theme_source"].value_counts().to_dict(),
        "avg_csat": cur["csat_score"].mean(), "csat_responses": int(cur["csat_score"].notna().sum()),
        "bot_tag_disagrees": float(np.mean([t not in _BOT_TO_THEMES.get(c, set())
                                            for t, c in zip(cur["theme"], cur["category"])])),
    }


# bot category -> themes it can legitimately mean (same mapping as validation/score_check_set.py)
_BOT_TO_THEMES = {
    "Delivery & Shipping": {"Delivery delayed / not delivered", "Address change / wrong address",
                            "Damaged in transit / dead on arrival", "Wrong item / variant received"},
    "Returns & Refunds": {"Return pickup not done", "Refund delayed / not received"},
    "Billing & Payments": {"Double charge / payment failed", "Coupon / discount / price", "Invoice / GST",
                           "Order cancellation"},
    "Connectivity": {"Pairing & connection drops"},
    "Charging & Battery": {"Battery drain / not charging"},
    "Audio Quality": {"Audio fault (one side, distortion, mic)"},
    "App & Firmware": {"App / firmware update failure"},
    "Warranty & Repair": {"Warranty claim / repair status", "Screen / touch / hardware fault"},
    "Account & Login": {"Account / OTP login"},
    "Product Enquiry": {"Product / compatibility question"},
    "Other": {"Other"},
}


def _pct(x):
    return f"{x:.0%}"


def _inr(x):
    return f"₹{x:,.0f}"


def format_digest_markdown(d):
    t = d["themes"]
    top = t.iloc[0]
    md = [f"# Vireo support: weekly complaint digest, {d['week']}",
          f"*{d['date_range']} · {d['tickets']} tickets "
          f"(last week {d['prev_week_tickets']}, 8-week average {d['avg_weekly_tickets']:.0f})*"]
    if d["days_covered"] < 7:
        md.append(f"\n> ⚠️ Only {d['days_covered']} days of data in this week: numbers are partial.")

    # --- the three lines to read if you read nothing else
    md.append("\n## This week in three lines")
    md.append(f"1. **Most common complaint: {top['theme']}**, {top['tickets']} tickets ({_pct(top['share'])}). "
              f"A typical one: _“{top['quote']}”_")
    if len(d["rising"]):
        r = d["rising"].sort_values("change_vs_normal", ascending=False).iloc[0]
        md.append(f"2. **Rising: {r['theme']}**, {r['tickets']} tickets against about {r['expected']:.0f} "
                  f"in a normal week. _“{r['quote']}”_")
    else:
        md.append("2. **Nothing rose beyond normal week-to-week variation.**")
    if len(d["repeat_by_theme"]):
        rt = d["repeat_by_theme"].index[0]
        md.append(f"3. **{d['repeat_contacts']} customers came back about a problem we had already handled** "
                  f"({_pct(d['repeat_share'])} of tickets, {_inr(d['repeat_cost_inr'])} in handling cost). "
                  f"The biggest source was **{rt}** ({d['repeat_by_theme'].iloc[0]['size']} repeats).")

    # --- theme table
    md.append("\n## What customers complained about")
    md.append("| Complaint | Tickets | Share | vs a normal week | Repeat contacts | Example |")
    md.append("|---|---|---|---|---|---|")
    for r in t.itertuples():
        delta = r.change_vs_normal
        trend = ("⬆️ " if r.rising_alert else "") + (f"+{delta:.0f}" if delta >= 0.5 else f"{delta:.0f}" if delta <= -0.5 else "≈")
        md.append(f"| {r.theme} | {r.tickets} | {_pct(r.share)} | {trend} | {_pct(r.repeat_rate)} | _{r.quote}_ |")
    md.append("\n⬆️ = more than normal variation explains (Poisson test against the previous 8 weeks, p < 0.01). "
              "Across 19 themes that still means roughly one false alarm every five weeks, so treat a single ⬆️ "
              "as \"worth a look\" and two weeks running as a real trend. *Repeat contacts* = share of this "
              "week's tickets from customers who had already contacted us about the same issue in the last "
              "30 days.")

    # --- failure demand
    md.append("\n## Customers who had to come back")
    md.append(f"- **{d['repeat_contacts']} repeat contacts** this week: same customer, same product, same issue, within 30 days "
              f"of the earlier ticket closing (policy §10). Handling cost: **{_inr(d['repeat_cost_inr'])}**.")
    md.append(f"- **{_pct(d['says_already_contacted'])}** of this week's tickets mention an earlier contact "
              f"(\"third time\", \"was told it was fixed\"), in the customer's words or the agent's note. "
              f"Rough signal: mostly keyword-based, which also catches notes like \"already dispatched\".")
    md.append(f"- **{_pct(d['vague_agent_notes'])}** of closing notes say nothing usable (\"done\", \"see prev\", "
              f"\"cx ok\"), so the next agent starts from zero.")
    if len(d["repeat_by_theme"]):
        md.append("- Where repeats came from: " + ", ".join(
            f"{th} ({int(r['size'])}, {_inr(r['sum'])})" for th, r in d["repeat_by_theme"].head(3).iterrows()))

    # --- products
    md.append("\n## Product watch (hardware faults only)")
    p = d["products"]
    flagged = p[p["alert"]]
    if len(flagged):
        for r in flagged.itertuples():
            md.append(f"- ⬆️ **{r.product}**: {r.hardware_tickets} hardware tickets against about {r.expected:.0f} "
                      f"normally; mostly {r.top_fault}.")
    else:
        md.append("- No product had an unusual number of hardware-fault tickets this week.")
    md.append("- Most hardware tickets: " + ", ".join(
        f"{r.product} ({r.hardware_tickets}, mostly {r.top_fault.split(' (')[0].lower()})" for r in p.head(3).itertuples()))
    if len(d["lot_alerts"]):
        for r in d["lot_alerts"].itertuples():
            md.append(f"- 🏭 Lot **{r.lot_code}** ({r.product_name}): {int(r.hw_tickets)} hardware tickets from "
                      f"{int(r.units)} units, {r.ratio:.1f}× the normal rate for this product.")
    else:
        md.append(f"- Manufacturing lots: {d['lots_tested']:,} checked; none has more hardware faults than chance "
                  f"explains once you allow for how many were checked.")

    # --- unfamiliar
    if len(d["novel"]):
        md.append(f"\n## Tickets that don't look like anything before ({len(d['novel'])})")
        md.append("These might be the start of a new kind of problem. Worth a human look:")
        for m in d["novel"]["customer_message"].head(3):
            md.append(f"- _“{mask_quote(m)}”_")

    # --- provenance
    src = d["theme_sources"]
    md.append("\n---")
    md.append(f"*How this was made: every ticket was read by the classifier ({src.get('tfidf', 0)} by the local "
              f"model, {src.get('llm', 0)} by the LLM, {src.get('keyword', 0)} by keyword rules). On 100 held-out "
              f"tickets labelled by Claude Code (30 reviewed by a person) it was right on 99; the helpdesk's own "
              f"category tag on 74 (validation/results.md). This week the tag disagreed with the ticket text on "
              f"{_pct(d['bot_tag_disagrees'])} of tickets. CSAT {d['avg_csat']:.2f}/5 from "
              f"{d['csat_responses']} responses.*")
    return "\n".join(md)


# ----------------------------------------------------------------------------- entry point

def generate_weekly_digest(df_clean, target_week=None):
    """Kept for run.py / app.py: themes are attached here if the caller has not done it."""
    df = ensure_themes(df_clean)
    week = target_week if target_week in set(df["year_week"]) else latest_complete_week(df)
    return build_weekly_digest(df, week)


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default=None)
    args = ap.parse_args()
    print(format_digest_markdown(generate_weekly_digest(clean_tickets_data(), args.week)))
