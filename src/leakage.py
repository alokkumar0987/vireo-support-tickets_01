"""
Business case and money audit for Vireo Audio: Arjun Mehta's "Before we spend on this - what does
it save? ... If it takes contacts out of the queue I'm interested."

1. BUSINESS GOAL (contacts out of the queue): repeat contacts about the same issue (policy §10),
   and the share of them that are STATUS CHASERS ("where is my refund / delivery / pickup / repair").
   A proactive status message answers those before the customer has to ask.
2. MONEY FOUND (policy breaches and gaps; for Finance to review, not all of it is recoverable):
   - refund AND replacement on the same order (policy §5), valued at the smaller of the two
   - goodwill over the Rs 500 cap (policy §5), including goodwill hidden under other reason codes
   - product-fault refunds filed under the catch-all GW-OTHER ("Goodwill / Other") code (for review)
   - refunds the agent note says were processed but with no amount recorded anywhere
   - warranty replacements closed by Tier 1 agents (policy §6: only Tier 2 may approve them)
3. COST CONTEXT (not leakage): SLA breach credits (policy §3).

All totals are measured on this export and then scaled to Vireo's real volume: the export has ~150
tickets a week, the brief says ~650, so it looks like a sample. Rates and per-ticket amounts carry
over; totals are multiplied by VOLUME_SCALE and annualised. That assumption is stated in every output.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clean import clean_tickets_data, DEFAULT_DATA_DIR
from src.themes import ensure_themes

POLICY_GOODWILL_CAP_INR = 500.0            # policy §5
SLA_CREDIT_PER_BREACH_INR = 350.0          # policy §3
INTERNAL_TRANSFER_COST_INR = 305.0         # policy §4
REPLACEMENT_LOGISTICS_INR = 340.0          # policy §5: unit cost + Rs 340 per replacement
BLENDED_CONTACT_COST_INR = 290.0           # policy §4 (Priya corrected Arjun's Rs 180 in the email thread)
VIREO_WEEKLY_VOLUME = 650                  # brief: "roughly 650 tickets a week"

# Refund codes that are not "the customer got their money back for the product":
# a duplicate payment, a price adjustment or a pre-dispatch cancellation can legitimately
# coexist with a replacement on the same order. GW-OTHER goodwill credits were added after the
# case review (validation/review_double_dips.csv #11, #16): they are covered by the goodwill-cap
# check instead.
NOT_PRODUCT_REFUND_CODES = {"DUP-PAYMENT", "PRICE-ADJ", "CANCEL", "GW-OTHER"}

STATUS_CHASER_THEMES = {
    "Delivery delayed / not delivered", "Refund delayed / not received", "Return pickup not done",
    "Warranty claim / repair status", "Double charge / payment failed",
}
HARDWARE_FAULT_THEMES = {
    "Battery drain / not charging", "Audio fault (one side, distortion, mic)", "Pairing & connection drops",
    "Screen / touch / hardware fault", "App / firmware update failure",
}
DEFAULT_CHASER_REDUCTION = 0.5
# A GW-OTHER refund is goodwill only if the note says so: the code is "Goodwill / Other" (policy §5).
GOODWILL_NOTE_PATTERN = r"goodwill|gesture|courtesy|compensat"
# Cost of the proactive updates the goal relies on (ASSUMPTIONS, stated in the output):
MESSAGE_COST_INR = 0.25        # one SMS / WhatsApp utility message
UPDATES_PER_ORDER = 3          # e.g. dispatched, out for delivery / pickup booked, refund processed


# ----------------------------------------------------------------------------- scaling

def sample_scale(df, weekly_volume=VIREO_WEEKLY_VOLUME):
    """Multiply an 18-month total in this export by this to get a per-year figure at Vireo's volume."""
    weeks = (df["created_at_dt"].max() - df["created_at_dt"].min()).days / 7.0
    sample_weekly = len(df) / weeks
    return {"weeks": weeks, "sample_weekly": sample_weekly,
            "volume_factor": weekly_volume / sample_weekly,
            "annual_factor": (52.0 / weeks) * (weekly_volume / sample_weekly)}


def _products(data_dir=DEFAULT_DATA_DIR):
    return pd.read_csv(glob.glob(os.path.join(data_dir, "*products*"))[0]).set_index("sku")


# ----------------------------------------------------------------------------- 1. business goal

def audit_repeat_contacts_and_savings(df_clean, weekly_ticket_volume=VIREO_WEEKLY_VOLUME,
                                      chaser_reduction=DEFAULT_CHASER_REDUCTION, target_repeat_rate=None):
    """
    Repeat contacts per policy §10 (same customer, same product, same issue family, within 30 days
    of the earlier ticket's resolution), each costed at the channel it came back on.
    Goal: cut the STATUS-CHASER repeats by `chaser_reduction` with proactive status updates.
    """
    df = ensure_themes(df_clean)
    n = len(df)
    rep = df[df["is_repeat_contact_30d"]]
    rate = len(rep) / n if n else 0.0
    avg_cost = rep["repeat_cost_inr"].mean() if len(rep) else BLENDED_CONTACT_COST_INR

    chasers = rep[rep["theme"].isin(STATUS_CHASER_THEMES)]
    chaser_rate = len(chasers) / n
    chaser_cost = chasers["repeat_cost_inr"].mean() if len(chasers) else avg_cost
    target = rate - chaser_rate * chaser_reduction if target_repeat_rate is None else target_repeat_rate

    annual_volume = weekly_ticket_volume * 52
    contacts_saved = annual_volume * max(rate - target, 0.0)
    annual_savings = contacts_saved * chaser_cost
    matured = df[df["repeat_window_complete"]]

    sensitivity = []
    for red in (0.25, 0.5, 0.75):
        t = rate - chaser_rate * red
        saved = annual_volume * (rate - t)
        sensitivity.append({"reduction": red, "target_rate": t, "contacts_saved": saved,
                            "annual_savings_inr": saved * chaser_cost})

    # "I already told you" (Neha's question). Measured on the tickets the LLM read: the keyword rule
    # used for the rest is right only ~45% of the time when it fires (it also matches "already
    # dispatched" in agent notes). The rule-based figure over all tickets is kept for comparison.
    llm_read = df[df["theme_source"] == "llm"]
    llm_rep = llm_read[llm_read["is_repeat_contact_30d"]]

    return {
        "definition": "same customer + product + issue family (policy §10)",
        "total_tickets": n,
        "repeat_tickets_count": len(rep),
        "current_repeat_rate": rate,
        "status_chaser_repeats": len(chasers),
        "status_chaser_rate": chaser_rate,
        "status_chaser_share_of_repeats": len(chasers) / len(rep) if len(rep) else float("nan"),
        "chaser_repeats_by_theme": chasers["theme"].value_counts(),
        "already_told_repeats_llm": float(llm_rep["is_repeat_claim"].mean()) if len(llm_rep) else float("nan"),
        "already_told_others_llm": float(llm_read.loc[~llm_read["is_repeat_contact_30d"], "is_repeat_claim"].mean())
                                   if len(llm_read) else float("nan"),
        "llm_read_tickets": len(llm_read), "llm_read_repeats": len(llm_rep),
        "already_told_repeats_rule": float(rep["is_repeat_claim"].mean()),
        "already_told_others_rule": float(df.loc[~df["is_repeat_contact_30d"], "is_repeat_claim"].mean()),
        "lower_bound_repeat_rate": df["is_repeat_same_tag_30d"].mean(),
        "same_product_repeat_rate": df["is_repeat_same_product_30d"].mean(),
        "old_any_recontact_rate": df["is_any_recontact_30d"].mean(),
        "fcr_rate": 1.0 - matured["caused_repeat_30d"].mean() if len(matured) else float("nan"),
        "chaser_reduction": chaser_reduction,
        "target_repeat_rate": target,
        "avg_repeat_cost_inr": avg_cost,
        "avg_chaser_cost_inr": chaser_cost,
        "total_historical_repeat_cost_inr": rep["repeat_cost_inr"].sum(),
        "annual_volume": annual_volume,
        "annual_contacts_saved": contacts_saved,
        "annual_savings_inr": annual_savings,
        "quarterly_savings_inr": annual_savings / 4.0,
        "sensitivity": sensitivity,
    }


# ----------------------------------------------------------------------------- 2. money found

def audit_double_dipping(df_clean, data_dir=DEFAULT_DATA_DIR):
    """
    Policy §5: "In no case is a customer to receive both a refund and a replacement for the same order".
    Matches on repeat_key (order_id, else customer + product: 34% of tickets quote no order_id),
    ignores refunds that are not product refunds (duplicate payment, price adjustment, cancellation),
    and values each case at the smaller of the refund and the replacement cost (unit cost + Rs 340):
    whichever the customer should not have had.
    """
    df = df_clean
    products = _products(data_dir)
    refunds = df[(df["refund_amount_inr"] > 0) & ~df["refund_reason_code"].isin(NOT_PRODUCT_REFUND_CODES)]
    replacements = df[df["replacement_issued"] == "Y"]
    keys = sorted(set(refunds["repeat_key"]) & set(replacements["repeat_key"]))

    rows = []
    for key in keys:
        rf, rp = refunds[refunds["repeat_key"] == key], replacements[replacements["repeat_key"] == key]
        sku = rp["product_sku"].iloc[0]
        repl_cost = (products.loc[sku, "unit_cost_inr"] + REPLACEMENT_LOGISTICS_INR) if sku in products.index else np.nan
        # The same refund is often re-entered on the customer's chaser tickets ("refund confirmed
        # credited (3499)"): count each distinct amount once (case review #3, #4, #13).
        refund = rf["refund_amount_inr"].drop_duplicates().sum()
        rows.append({
            "repeat_key": key, "matched_on": "order_id" if "|" not in str(key) else "customer+product",
            "product_sku": sku, "refund_inr": refund, "replacement_cost_inr": repl_cost,
            "leakage_inr": np.nanmin([refund, repl_cost]),
            "refund_codes": ", ".join(sorted(rf["refund_reason_code"].dropna().unique())),
            "refund_tickets": ", ".join(rf["ticket_id"]), "replacement_tickets": ", ".join(rp["ticket_id"]),
            "agents": ", ".join(sorted(set(rf["agent_id"]) | set(rp["agent_id"]))),
            "same_ticket": bool(set(rf["ticket_id"]) & set(rp["ticket_id"])),
        })
    orders = pd.DataFrame(rows)
    # Case review: 14/15 order-ID matches were real breaches; 0/5 customer+product matches could be
    # confirmed (a customer may simply have bought the product twice). Only order-ID matches count
    # towards the headline; the rest are listed as "possible".
    confirmed = orders[orders["matched_on"] == "order_id"] if len(orders) else orders
    possible = orders[orders["matched_on"] != "order_id"] if len(orders) else orders
    return {
        "double_dip_orders_df": orders,
        "double_dip_orders_count": len(confirmed),
        "matched_on_order_id": len(confirmed),
        "possible_count": len(possible),
        "possible_leakage_inr": float(possible["leakage_inr"].sum()) if len(possible) else 0.0,
        "same_ticket_double_count": int(orders["same_ticket"].sum()) if len(orders) else 0,
        "total_leakage_inr": float(confirmed["leakage_inr"].sum()) if len(confirmed) else 0.0,
    }


def audit_goodwill_cap_violations(df_clean):
    """
    Policy §5: goodwill capped at Rs 500 per ticket, Team Lead approval required.
    (a) GW-OTHER refunds above the cap whose note calls them goodwill. GW-OTHER is "Goodwill / Other":
        in this export every GW-OTHER refund over the cap is a product-fault refund whose note never
        mentions goodwill, so those are reported by audit_fault_refunds_coded_other() instead.
    (b) refunds the NOTE calls goodwill / "without pickup" but coded as something else. Those are
        invisible to a report that filters on the GW-OTHER code. (All 9 notes say "as goodwill".)
    """
    df = df_clean
    notes = df["agent_notes"].fillna("").str.lower()
    says_goodwill = notes.str.contains(GOODWILL_NOTE_PATTERN)
    coded = df[(df["refund_reason_code"] == "GW-OTHER")]
    coded_over = coded[(coded["refund_amount_inr"] > POLICY_GOODWILL_CAP_INR) & says_goodwill[coded.index]]
    hidden = df[notes.str.contains(r"goodwill|without (?:pickup|pkp)") & (df["refund_reason_code"] != "GW-OTHER")
                & (df["refund_amount_inr"] > POLICY_GOODWILL_CAP_INR)]
    excess = lambda frame: float((frame["refund_amount_inr"] - POLICY_GOODWILL_CAP_INR).clip(lower=0).sum())
    return {
        "gw_violations_df": coded_over,
        "total_gw_tickets": len(coded),
        "gw_violations_count": len(coded_over),
        "total_gw_refunded_inr": float(coded["refund_amount_inr"].sum()),
        "total_excess_goodwill_inr": excess(coded_over),
        "hidden_goodwill_df": hidden,
        "hidden_goodwill_count": len(hidden),
        "hidden_goodwill_codes": hidden["refund_reason_code"].value_counts().to_dict(),
        "hidden_goodwill_excess_inr": excess(hidden),
        "max_goodwill_ticket_inr": float(coded["refund_amount_inr"].max()) if len(coded) else 0.0,
    }


def audit_fault_refunds_coded_other(df_clean, doa_days=7):
    """
    Refunds coded GW-OTHER ("Goodwill / Other") whose note is not goodwill: in this export they are
    refunds for product faults (pairing, battery, audio...). Policy §5: an in-warranty fault gets a
    repair or replacement, a refund only when dead on arrival (7 days); buy-backs have their own code
    (WTY-BUYBACK) and warranty decisions belong to Tier 2 (§6). So these look like warranty buy-backs
    made without Tier 2 under a catch-all code. For review, NOT counted as recoverable: what Tier 2
    would have approved (repair, replacement or buy-back) is not in the data.
    """
    df = ensure_themes(df_clean)
    notes = df["agent_notes"].fillna("").str.lower()
    fault_themes = HARDWARE_FAULT_THEMES | {"Warranty claim / repair status"}
    f = df[(df["refund_reason_code"] == "GW-OTHER") & (df["refund_amount_inr"] > 0)
           & ~notes.str.contains(GOODWILL_NOTE_PATTERN) & df["theme"].isin(fault_themes)].copy()
    f["days_since_order"] = (f["created_at_dt"] - pd.to_datetime(f["order_date"])).dt.days
    return {
        "fault_refunds_df": f,
        "count": len(f),
        "amount_inr": float(f["refund_amount_inr"].sum()),
        "order_date_known": int(f["days_since_order"].notna().sum()),
        "past_doa_window": int((f["days_since_order"] > doa_days).sum()),
        "by_tier_1": int((f["tier"] == 1).sum()),
        "themes": f["theme"].value_counts().to_dict(),
    }


def proactive_message_cost(df_clean, data_dir=DEFAULT_DATA_DIR, scale=None):
    """Yearly cost of the proactive updates at Vireo's volume: orders a year x updates x message cost.
    Orders in the export's ticket window are scaled by the same factor as tickets (an assumption)."""
    scale = scale or sample_scale(df_clean)
    orders = pd.read_csv(glob.glob(os.path.join(data_dir, "*orders*"))[0], parse_dates=["order_date"])
    start, end = df_clean["created_at_dt"].min(), df_clean["created_at_dt"].max()
    in_window = orders[(orders["order_date"] >= start.normalize()) & (orders["order_date"] <= end)]
    orders_per_year = len(in_window) * scale["annual_factor"]
    return {"orders_per_year": orders_per_year, "updates_per_order": UPDATES_PER_ORDER,
            "message_cost_inr": MESSAGE_COST_INR,
            "annual_cost_inr": orders_per_year * UPDATES_PER_ORDER * MESSAGE_COST_INR}


def audit_unrecorded_refunds(df_clean):
    """
    Notes that say a refund was PROCESSED (not merely confirmed/chased) where no refund amount is
    recorded on that ticket or on any other ticket for the same order / customer+product.
    Not a loss estimate: the money may be recorded in the payment gateway or a courier-claim system
    outside this export. It is a control gap: the helpdesk's refund field does not show it.
    """
    df = df_clean
    notes = df["agent_notes"].fillna("").str.lower()
    processed = notes.str.contains(r"(?:refund|rfnd)\w* (?:processed|issued|reprocessed|initiated)|refunded|"
                                   r"(?:full|partial) (?:refund|rfnd)|without (?:pickup|pkp)")
    chaser = notes.str.contains(r"confirmed credited|arn shared")
    blank = df["refund_amount_inr"].isna() | (df["refund_amount_inr"] == 0)
    keys_with_refund = set(df.loc[~blank, "repeat_key"])
    gap = df[processed & ~chaser & blank & ~df["repeat_key"].isin(keys_with_refund)].copy()
    gap["reason"] = notes[gap.index].str.extract(
        r"(lost in transit|without (?:pickup|pkp)|goodwill|cancel|duplicate|discount|partial refund|full refund)")[0]
    gap["reason"] = gap["reason"].replace({"without pkp": "without pickup"}).fillna("other")
    return {
        "unrecorded_df": gap,
        "count": len(gap),
        "by_reason": gap["reason"].value_counts().to_dict(),
        "order_value_upper_bound_inr": float(gap["order_value_inr"].sum()),
        "order_value_known_for": int(gap["order_value_inr"].notna().sum()),
    }


def audit_tier1_warranty_replacements(df_clean, data_dir=DEFAULT_DATA_DIR, doa_days=7):
    """
    Policy §6: only Tier 2 may approve warranty replacements. Flags replacements on hardware-fault
    tickets resolved by a Tier 1 agent more than 7 days after the order (past the dead-on-arrival
    window, so warranty territory). For review: a Tier 2 approval may exist outside this export.
    """
    df = ensure_themes(df_clean)
    products = _products(data_dir)
    rep = df[(df["replacement_issued"] == "Y")].copy()
    rep["days_since_order"] = (rep["created_at_dt"] - pd.to_datetime(rep["order_date"])).dt.days
    flagged = rep[(rep["tier"] == 1) & rep["theme"].isin(HARDWARE_FAULT_THEMES) & (rep["days_since_order"] > doa_days)].copy()
    flagged["replacement_cost_inr"] = flagged["product_sku"].map(products["unit_cost_inr"]) + REPLACEMENT_LOGISTICS_INR
    return {
        "flagged_df": flagged,
        "count": len(flagged),
        "cost_inr": float(flagged["replacement_cost_inr"].sum()),
        "by_team": flagged["team"].value_counts().to_dict(),
        "order_date_known_for": int(rep["days_since_order"].notna().sum()),
        "replacements_total": len(rep),
        "tier2_replacements": int((rep["tier"] == 2).sum()),
    }


# ----------------------------------------------------------------------------- 3. cost context

def audit_sla_penalties(df_clean):
    """Policy §3: Rs 350 store credit on resolution of every ticket that missed its first-response target."""
    df = df_clean
    breached_resolved = df[df["is_breached"] & df["status"].isin(["resolved", "closed"])]
    by_channel = df.groupby("channel").agg(
        total_tickets=("ticket_id", "count"), breaches=("is_breached", "sum"),
        breach_rate=("is_breached", "mean"), sla_credits_inr=("sla_credit_inr", "sum")).reset_index()
    return {"breached_resolved_count": len(breached_resolved),
            "total_sla_credits_inr": len(breached_resolved) * SLA_CREDIT_PER_BREACH_INR,
            "breach_by_channel": by_channel}


# ----------------------------------------------------------------------------- synthesis

def generate_comprehensive_financial_audit(df_clean=None):
    df = ensure_themes(clean_tickets_data() if df_clean is None else df_clean)
    scale = sample_scale(df)
    audit = {
        "scale": scale,
        "repeat": audit_repeat_contacts_and_savings(df),
        "double_dip": audit_double_dipping(df),
        "goodwill": audit_goodwill_cap_violations(df),
        "fault_refunds_other": audit_fault_refunds_coded_other(df),
        "unrecorded_refunds": audit_unrecorded_refunds(df),
        "tier1_warranty": audit_tier1_warranty_replacements(df),
        "sla": audit_sla_penalties(df),
        "base_contact_spend_inr": float(df["contact_cost_inr"].sum()),
        "transfer_cost_inr": float(df["transfers"].sum() * INTERNAL_TRANSFER_COST_INR),
    }
    audit["messages"] = proactive_message_cost(df, scale=scale)
    audit["net_annual_savings_inr"] = audit["repeat"]["annual_savings_inr"] - audit["messages"]["annual_cost_inr"]
    audit["direct_ops_spend_inr"] = (audit["base_contact_spend_inr"] + audit["transfer_cost_inr"]
                                     + audit["sla"]["total_sla_credits_inr"])
    # Policy-breach money that Finance can act on (sample, 18 months). SLA credits are a cost of
    # running late, not a breach, so they are NOT in here (the old version added them).
    audit["total_direct_leakage_inr"] = (audit["double_dip"]["total_leakage_inr"]
                                         + audit["goodwill"]["total_excess_goodwill_inr"]
                                         + audit["goodwill"]["hidden_goodwill_excess_inr"])
    audit["total_direct_leakage_annual_inr"] = audit["total_direct_leakage_inr"] * scale["annual_factor"]
    return audit


def _inr(x):
    return f"₹{x:,.0f}"


def _lakh(x):
    return f"₹{x / 1e5:.1f} lakh"


def format_business_case_markdown(a):
    r, s = a["repeat"], a["scale"]
    f = s["annual_factor"]
    dd, gw, ur, t1 = a["double_dip"], a["goodwill"], a["unrecorded_refunds"], a["tier1_warranty"]
    fo, msg = a["fault_refunds_other"], a["messages"]
    md = ["# Business case: what this saves (for Arjun Mehta)",
          f"*Measured on the export ({s['sample_weekly']:.0f} tickets a week over {s['weeks']:.0f} weeks), scaled to "
          f"Vireo's ~{VIREO_WEEKLY_VOLUME} tickets a week: rates carry over, totals ×{s['volume_factor']:.1f}. "
          f"Contact costs are policy §4's per-channel figures.*"]

    md.append("\n## The goal")
    top = r["chaser_repeats_by_theme"]
    md.append(f"> **Cut repeat contacts about the same issue from {r['current_repeat_rate']:.1%} to "
              f"{r['target_repeat_rate']:.1%} of all contacts, worth about {_inr(r['quarterly_savings_inr'])} a quarter "
              f"({_lakh(r['annual_savings_inr'])} a year), by sending customers a status update before they have to "
              f"chase a refund, delivery, pickup or repair.**")
    md.append(f"\n- Today **{r['current_repeat_rate']:.1%}** of contacts ({r['repeat_tickets_count']:,} in the export) are the "
              f"same customer coming back about the same issue within 30 days of it being closed (policy §10).")
    md.append(f"- **{r['status_chaser_share_of_repeats']:.0%}** of those are status chasers: "
              + ", ".join(f"{t.split(' /')[0].lower()} ({n})" for t, n in top.head(5).items())
              + ". Nothing is broken; the customer just does not know where things stand.")
    md.append(f"- The tickets confirm it: of the {r['llm_read_tickets']:,} tickets the LLM read, "
              f"**{r['already_told_repeats_llm']:.0%}** of the {r['llm_read_repeats']} repeat contacts mention an earlier "
              f"contact (the customer or the agent's note), against {r['already_told_others_llm']:.0%} of other tickets. "
              f"(The keyword rule used for the other tickets is noisier: {r['already_told_repeats_rule']:.0%} vs "
              f"{r['already_told_others_rule']:.0%}.)")
    md.append(f"- Halving status chasers takes **{r['annual_contacts_saved']:,.0f} contacts a year** out of the queue at "
              f"~{_inr(r['avg_chaser_cost_inr'])} each (their channel mix).")
    md.append(f"- The updates are not free: about {msg['orders_per_year']:,.0f} orders a year × {msg['updates_per_order']} "
              f"messages × ₹{msg['message_cost_inr']:.2f} (assumed SMS / WhatsApp rate) ≈ **{_inr(msg['annual_cost_inr'])} a "
              f"year**, so the net saving is about **{_inr(a['net_annual_savings_inr'] / 4)} a quarter**.")
    md.append("\n| If proactive updates remove… | Repeat rate | Contacts saved / year | Saving / year | Saving / quarter |")
    md.append("|---|---|---|---|---|")
    for x in r["sensitivity"]:
        md.append(f"| {x['reduction']:.0%} of status chasers | {x['target_rate']:.1%} | {x['contacts_saved']:,.0f} | "
                  f"{_inr(x['annual_savings_inr'])} | {_inr(x['annual_savings_inr'] / 4)} |")
    md.append(f"\n*Why not the bigger numbers: counting any return by the same customer about the same product gives "
              f"{r['same_product_repeat_rate']:.1%} (a delivery ticket then a battery ticket would count), and the "
              f"old version of this tool claimed 34.3%. Policy §10 says \"the same issue\", so {r['current_repeat_rate']:.1%} "
              f"is the honest base. First-contact resolution today: {r['fcr_rate']:.1%}.*")

    md.append("\n## Money Finance should look at")
    md.append("| Finding | Cases (export) | Export, 18 months | Per year at Vireo volume | Confidence |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Refund **and** replacement on the same order (policy §5), valued at the smaller of the two | "
              f"{dd['double_dip_orders_count']} | {_inr(dd['total_leakage_inr'])} | "
              f"{_inr(dd['total_leakage_inr'] * f)} | High: 14 of 15 reviewed cases confirmed |")
    md.append(f"| Possible refund + replacement where the ticket quotes no order ID (matched on customer + product) | "
              f"{dd['possible_count']} | {_inr(dd['possible_leakage_inr'])} | — | Low: 0 of 5 reviewed could be "
              f"confirmed; not counted |")
    md.append(f"| Goodwill over the ₹500 cap, coded GW-OTHER and called goodwill in the note | {gw['gw_violations_count']} | "
              f"{_inr(gw['total_excess_goodwill_inr'])} | {_inr(gw['total_excess_goodwill_inr'] * f)} | "
              f"{'High' if gw['gw_violations_count'] else 'None: GW-OTHER refunds over the cap are fault refunds (see the fault-refund row)'} |")
    md.append(f"| Goodwill over the cap hidden under other codes (note says goodwill / \"without pickup\"; coded "
              f"{', '.join(gw['hidden_goodwill_codes'])}) | {gw['hidden_goodwill_count']} | "
              f"{_inr(gw['hidden_goodwill_excess_inr'])} | {_inr(gw['hidden_goodwill_excess_inr'] * f)} | High |")
    md.append(f"| Refunds for product faults filed under the catch-all GW-OTHER (\"Goodwill / Other\") code: "
              f"{fo['past_doa_window']} of {fo['order_date_known']} with a known order date were past the 7-day DOA window, "
              f"{fo['by_tier_1']} by Tier 1. Policy §5/§6: repair, replacement or a Tier 2 buy-back (WTY-BUYBACK) | "
              f"{fo['count']} | {_inr(fo['amount_inr'])} refunded | {_inr(fo['amount_inr'] * f)} refunded | "
              f"Review: look like buy-backs without Tier 2; not counted |")
    md.append(f"| Warranty replacements closed by **Tier 1** (policy §6: Tier 2 only), hardware faults >7 days after order | "
              f"{t1['count']} | {_inr(t1['cost_inr'])} | {_inr(t1['cost_inr'] * f)} | Review: approval may exist elsewhere |")
    md.append(f"| Refunds the note says were processed, **no amount recorded** anywhere | {ur['count']} | "
              f"up to {_inr(ur['order_value_upper_bound_inr'])} (order value, known for {ur['order_value_known_for']}) | "
              f"— | Control gap, not a loss estimate |")
    md.append(f"\n**Recoverable policy breaches:** {_inr(a['total_direct_leakage_inr'])} in the export, about "
              f"**{_lakh(a['total_direct_leakage_annual_inr'])} a year** at Vireo's volume. The unrecorded refunds "
              f"({', '.join(f'{k} {v}' for k, v in list(ur['by_reason'].items())[:3])}…) need a reconciliation "
              f"against the payment gateway before anyone puts a number on them.")

    md.append("\n## Cost context (not leakage)")
    md.append(f"- SLA breach credits (₹350 each, policy §3): {_inr(a['sla']['total_sla_credits_inr'])} in the export, "
              f"about {_lakh(a['sla']['total_sla_credits_inr'] * f)} a year. This is the price of late first responses, "
              f"not a policy breach, so it is not counted above.")
    md.append(f"- Team transfers (₹305 each): {_inr(a['transfer_cost_inr'])} in the export.")
    return "\n".join(md)


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(format_business_case_markdown(generate_comprehensive_financial_audit()))
