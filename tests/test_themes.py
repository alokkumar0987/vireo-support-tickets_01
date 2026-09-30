"""
Tests for the offline theme rules and the LLM/keyword merge (src/themes.py).
Each case is a real message pattern from tickets.csv that an earlier version got wrong.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.themes import keyword_theme, keyword_outcome, keyword_repeat_claim, attach_themes


@pytest.mark.parametrize("message, expected", [
    # old taxonomy: "charge" substring sent billing to Battery
    ("I was charged twice for one order", "Double charge / payment failed"),
    ("card charged two times", "Double charge / payment failed"),
    # found by embedding clustering: these fell into "Other"
    ("two entries of Rs 1799 on my statement for one pair of earbuds", "Double charge / payment failed"),
    ("my bank says Rs 2499 went to you but your site says I have no orders", "Double charge / payment failed"),
    # old taxonomy: "pair" substring sent repairs to Bluetooth
    ("no update on my repair, claim number RMA66066", "Warranty claim / repair status"),
    # LLM v2 errors, fixed in prompt v3 and mirrored here
    ("you picked up the item 15 days ago and my money hasn't come back", "Refund delayed / not received"),
    ("nobody came for the pickup", "Return pickup not done"),
    ("ordered black, got white, not what i asked for", "Wrong item / variant received"),
    # the intake bot put these in "Other"
    ("my company accounts team is asking for the tax bill", "Invoice / GST"),
    ("i moved houses yesterday and the order is going to the old flat", "Address change / wrong address"),
    # "can I connect" is a question, not a pairing fault
    ("Can I connect two the Pulse 2 earbuds together", "Product / compatibility question"),
    ("left side not charging in the case", "Battery drain / not charging"),
    ("update failed and now it won't turn on", "App / firmware update failure"),
    ("box was crushed and the AirLite is cracked", "Damaged in transit / dead on arrival"),
    ("not receiving otp to login", "Account / OTP login"),
    ("hey my order has not been delivered yet", "Delivery delayed / not delivered"),
])
def test_keyword_theme_regressions(message, expected):
    assert keyword_theme(message) == expected


def test_vague_message_falls_back_to_note():
    assert keyword_theme("hello?? please help", "1. Payment debited, no order") == "Double charge / payment failed"
    assert keyword_theme("hello?? please help", "") == "Other"


@pytest.mark.parametrize("note, expected", [
    ("done ~Diya", "unclear"),
    ("see prev [closed]", "unclear"),
    ("cx ok (SOP 3.1)", "unclear"),
    ("-", "unclear"),
    ("Full refund processed (4998).", "refund"),
    ("re-shipped from warehouse", "replacement_or_reship"),
    ("1. Battery draining fast\n2. Checked FW\n3. Escalated to warranty", "escalated"),
    ("Pickup rescheduled for 26 Jun.", "pending_or_deferred"),
    ("walked through forget + re-pair, paired ok, rslvd", "fixed"),
])
def test_keyword_outcome(note, expected):
    assert keyword_outcome(note) == expected


def test_repeat_claim_detects_customer_and_note_signals():
    assert keyword_repeat_claim("Raised this last month and was told it was resolved")
    assert keyword_repeat_claim("screen dead", "repeat contact - previous fix did not hold")
    assert not keyword_repeat_claim("my order has not been delivered yet", "re-shipped")


def test_attach_themes_prefers_llm_labels_and_falls_back():
    df = pd.DataFrame({
        "ticket_id": ["T1", "T2"],
        "customer_message": ["card charged two times", "nobody came for the pickup"],
        "agent_notes": ["duplicate refunded", "pickup rescheduled"],
    })
    labels = {"T1": {"theme": "Coupon / discount / price", "is_repeat_claim": True, "outcome": "fixed"}}
    out = attach_themes(df, labels=labels, use_model=False).set_index("ticket_id")

    assert out.loc["T1", "theme_source"] == "llm" and out.loc["T1", "theme"] == "Coupon / discount / price"
    assert out.loc["T1", "keyword_theme"] == "Double charge / payment failed", "baseline kept for comparison"
    assert out.loc["T1", "is_repeat_claim"] and out.loc["T1", "note_outcome"] == "fixed"

    assert out.loc["T2", "theme_source"] == "keyword" and out.loc["T2", "theme"] == "Return pickup not done"
    assert out.loc["T2", "note_outcome"] == "pending_or_deferred"


def test_attach_themes_offline_mode_ignores_cache():
    df = pd.DataFrame({"ticket_id": ["T1"], "customer_message": ["otp not coming"], "agent_notes": [""]})
    out = attach_themes(df, use_llm=False, use_model=False)
    assert out["theme_source"].tolist() == ["keyword"]
