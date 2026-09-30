"""
Tests for the business case and money audit (src/leakage.py).
Rule tests use tiny synthetic frames; the headline numbers are pinned on the real export.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import clean_tickets_data
from src.themes import attach_themes
from src.leakage import (
    audit_double_dipping, audit_goodwill_cap_violations, audit_unrecorded_refunds, audit_sla_penalties,
    audit_repeat_contacts_and_savings, generate_comprehensive_financial_audit, format_business_case_markdown,
    sample_scale,
)


@pytest.fixture(scope="module")
def themed():
    return attach_themes(clean_tickets_data())


@pytest.fixture(scope="module")
def audit(themed):
    return generate_comprehensive_financial_audit(themed)


def _t(**kw):
    base = {"ticket_id": "T", "repeat_key": "O1", "product_sku": "VA-EB-PL2", "refund_amount_inr": np.nan,
            "refund_reason_code": np.nan, "replacement_issued": "N", "agent_id": "A1", "agent_notes": "",
            "order_value_inr": 3000.0}
    base.update(kw)
    return base


# ---------------------------------------------------------------- double dip rules

def test_double_dip_valued_at_the_smaller_of_refund_and_replacement():
    df = pd.DataFrame([_t(ticket_id="T1", refund_amount_inr=100000.0, refund_reason_code="RETURN-QC-OK"),
                       _t(ticket_id="T2", replacement_issued="Y")])
    out = audit_double_dipping(df)
    row = out["double_dip_orders_df"].iloc[0]
    assert out["double_dip_orders_count"] == 1
    assert row["leakage_inr"] == row["replacement_cost_inr"] < 100000, "cheap replacement caps the loss"


def test_duplicate_payment_or_cancellation_refunds_are_not_double_dips():
    df = pd.DataFrame([_t(ticket_id="T1", refund_amount_inr=999.0, refund_reason_code="DUP-PAYMENT"),
                       _t(ticket_id="T2", refund_amount_inr=999.0, refund_reason_code="CANCEL", repeat_key="O2"),
                       _t(ticket_id="T3", replacement_issued="Y"),
                       _t(ticket_id="T4", replacement_issued="Y", repeat_key="O2")])
    assert audit_double_dipping(df)["double_dip_orders_count"] == 0


def test_customer_product_matches_are_only_possible_not_counted():
    # case review: 0 of 5 such matches could be confirmed (same customer may have bought twice)
    df = pd.DataFrame([_t(ticket_id="T1", repeat_key="C9|VA-EB-PL2", refund_amount_inr=500.0, refund_reason_code="DOA-REPL"),
                       _t(ticket_id="T2", repeat_key="C9|VA-EB-PL2", replacement_issued="Y")])
    out = audit_double_dipping(df)
    assert out["double_dip_orders_count"] == 0 and out["total_leakage_inr"] == 0.0
    assert out["possible_count"] == 1 and out["possible_leakage_inr"] == 500.0


def test_refund_repeated_on_chaser_tickets_is_counted_once():
    df = pd.DataFrame([_t(ticket_id="T1", refund_amount_inr=399.0, refund_reason_code="RETURN-QC-OK"),
                       _t(ticket_id="T2", refund_amount_inr=399.0, refund_reason_code="RETURN-QC-OK",
                          agent_notes="refund confirmed credited"),
                       _t(ticket_id="T3", replacement_issued="Y")])
    assert audit_double_dipping(df)["double_dip_orders_df"]["refund_inr"].iloc[0] == 399.0


def test_goodwill_credit_next_to_a_replacement_is_not_a_double_dip():
    df = pd.DataFrame([_t(ticket_id="T1", refund_amount_inr=195.0, refund_reason_code="GW-OTHER"),
                       _t(ticket_id="T2", replacement_issued="Y")])
    assert audit_double_dipping(df)["double_dip_orders_count"] == 0


# ---------------------------------------------------------------- goodwill + unrecorded refunds

def test_goodwill_hidden_under_another_code_is_caught():
    df = pd.DataFrame([_t(ticket_id="T1", refund_amount_inr=2500.0, refund_reason_code="RETURN-QC-OK",
                          agent_notes="Refund processed without pickup as goodwill"),
                       _t(ticket_id="T2", refund_amount_inr=800.0, refund_reason_code="GW-OTHER",
                          agent_notes="goodwill credit for the delay"),
                       _t(ticket_id="T3", refund_amount_inr=300.0, refund_reason_code="GW-OTHER",
                          agent_notes="goodwill"),
                       # "Goodwill / Other": a fault refund under the catch-all code is not goodwill
                       _t(ticket_id="T4", refund_amount_inr=3999.0, refund_reason_code="GW-OTHER",
                          agent_notes="no audio one side -> refund of rs 3999 initiated to source")])
    out = audit_goodwill_cap_violations(df)
    assert out["gw_violations_count"] == 1 and out["total_excess_goodwill_inr"] == 300.0
    assert out["hidden_goodwill_count"] == 1 and out["hidden_goodwill_excess_inr"] == 2000.0


def test_unrecorded_refunds_skip_chasers_and_refunds_recorded_elsewhere():
    df = pd.DataFrame([
        _t(ticket_id="T1", repeat_key="O1", agent_notes="rfnd initiated as lost in transit"),          # gap
        _t(ticket_id="T2", repeat_key="O2", agent_notes="refund confirmed credited, arn shared"),      # chaser
        _t(ticket_id="T3", repeat_key="O3", agent_notes="full refund processed"),                     # recorded on T4
        _t(ticket_id="T4", repeat_key="O3", refund_amount_inr=999.0, refund_reason_code="RETURN-QC-OK"),
    ])
    out = audit_unrecorded_refunds(df)
    assert out["count"] == 1 and out["unrecorded_df"]["ticket_id"].tolist() == ["T1"]
    assert out["by_reason"] == {"lost in transit": 1}


# ---------------------------------------------------------------- real export: headline numbers

def test_business_goal_uses_policy_same_issue_definition(audit):
    r = audit["repeat"]
    assert r["definition"].startswith("same customer + product + issue family")
    assert r["repeat_tickets_count"] == 1526 and 0.128 <= r["current_repeat_rate"] <= 0.129
    assert r["current_repeat_rate"] < r["same_product_repeat_rate"] < r["old_any_recontact_rate"]
    assert r["status_chaser_repeats"] == 720
    assert 0.098 <= r["target_repeat_rate"] <= 0.0985
    assert round(r["annual_savings_inr"]) == 269817
    assert r["already_told_repeats_llm"] > 5 * r["already_told_others_llm"], \
        "tickets the LLM read should confirm the repeats (Neha's 'I already told your colleague')"


def test_sensitivity_is_monotonic(audit):
    s = audit["repeat"]["sensitivity"]
    assert [x["reduction"] for x in s] == [0.25, 0.5, 0.75]
    assert s[0]["annual_savings_inr"] < s[1]["annual_savings_inr"] < s[2]["annual_savings_inr"]
    assert s[1]["annual_savings_inr"] == pytest.approx(audit["repeat"]["annual_savings_inr"])


def test_money_found_on_real_export(audit):
    assert audit["double_dip"]["double_dip_orders_count"] == 69
    assert round(audit["double_dip"]["total_leakage_inr"]) == 114821
    assert audit["double_dip"]["possible_count"] == 34
    assert audit["goodwill"]["gw_violations_count"] == 0          # no GW-OTHER note over the cap says goodwill
    assert audit["fault_refunds_other"]["count"] >= 37              # ...they are fault refunds, reported for review
    assert audit["goodwill"]["hidden_goodwill_count"] == 9
    assert audit["unrecorded_refunds"]["count"] == 749
    assert audit["tier1_warranty"]["count"] == 342


def test_sla_credits_are_cost_not_leakage(audit):
    assert audit["sla"]["total_sla_credits_inr"] == 347200.0
    expected = (audit["double_dip"]["total_leakage_inr"] + audit["goodwill"]["total_excess_goodwill_inr"]
                + audit["goodwill"]["hidden_goodwill_excess_inr"])
    assert audit["total_direct_leakage_inr"] == pytest.approx(expected)


def test_scaling_to_vireo_volume(themed):
    s = sample_scale(themed)
    assert 145 < s["sample_weekly"] < 160
    assert s["volume_factor"] == pytest.approx(650 / s["sample_weekly"])


def test_unthemed_data_gets_themes_instead_of_a_looser_definition():
    # No silent fallback to "same product" (29.8%): themes are attached, so the goal is always policy §10.
    r = audit_repeat_contacts_and_savings(clean_tickets_data())
    assert r["definition"].startswith("same customer + product + issue family")
    assert r["repeat_tickets_count"] == 1526


def test_recoverable_headline_excludes_fault_refunds_under_the_catch_all_code(audit):
    # GW-OTHER = "Goodwill / Other": refunds over the cap there are fault refunds, not goodwill breaches.
    assert audit["goodwill"]["total_excess_goodwill_inr"] == 0.0
    fo = audit["fault_refunds_other"]
    assert fo["count"] == 40 and fo["past_doa_window"] >= 0.9 * fo["order_date_known"]
    assert round(audit["total_direct_leakage_annual_inr"] / 1e5, 1) == 4.2
    assert 0 < audit["messages"]["annual_cost_inr"] < 0.15 * audit["repeat"]["annual_savings_inr"]


def test_report_states_the_goal_as_a_number(audit):
    md = format_business_case_markdown(audit)
    assert "Cut repeat contacts about the same issue from 12.9% to 9.8%" in md
    assert "a quarter" in md and "Money Finance should look at" in md
