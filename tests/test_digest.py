"""
Tests for the weekly complaint digest (src/digest.py).
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import clean_tickets_data
from src.themes import attach_themes
from src.digest import (
    _poisson_sf, mask_quote, theme_table, build_weekly_digest, format_digest_markdown,
    detect_manufacturing_lot_anomalies, latest_complete_week,
)


@pytest.fixture(scope="module")
def themed():
    return attach_themes(clean_tickets_data())


def test_poisson_tail_matches_closed_form():
    assert math.isclose(_poisson_sf(1, 1.0), 1 - math.exp(-1), rel_tol=1e-9)
    assert math.isclose(_poisson_sf(2, 2.0), 1 - 3 * math.exp(-2), rel_tol=1e-9)
    assert _poisson_sf(0, 5.0) == 1.0 and _poisson_sf(3, 0.0) == 0.0


def test_quotes_mask_order_and_rma_numbers():
    q = mask_quote("[IVR transcript] order VR894781 and claim RMA42319, call 9876543210")
    assert "894781" not in q and "42319" not in q and "9876543210" not in q
    assert "IVR" not in q and "VR•••" in q
    assert len(mask_quote("word " * 100)) <= 110


def _synthetic_weeks(spike):
    """9 weeks of 100 tickets: 20 Battery / 80 Other, then a final week with `spike` Battery tickets."""
    rows = []
    for w in range(1, 10):
        n_batt = spike if w == 9 else 20
        for i in range(100):
            rows.append({"year_week": f"2026-W{w:02d}", "theme": "Battery drain / not charging" if i < n_batt else "Other",
                         "customer_message": "battery dies fast", "is_repeat_contact_30d": False, "repeat_cost_inr": 0})
    return pd.DataFrame(rows)


def test_rising_alert_fires_on_a_real_spike_only():
    spiked = theme_table(_synthetic_weeks(spike=40), "2026-W09").set_index("theme")
    normal = theme_table(_synthetic_weeks(spike=24), "2026-W09").set_index("theme")
    assert spiked.loc["Battery drain / not charging", "rising_alert"], "20 -> 40 is not noise"
    assert not normal.loc["Battery drain / not charging", "rising_alert"], "20 -> 24 is ordinary variation"


def test_digest_counts_are_consistent(themed):
    d = build_weekly_digest(themed, "2026-W26")
    assert d["tickets"] == 199 and d["days_covered"] == 7
    assert d["themes"]["tickets"].sum() == d["tickets"]
    assert d["repeat_contacts"] == int(themed.loc[themed["year_week"] == "2026-W26", "is_repeat_contact_30d"].sum())
    assert 0 <= d["bot_tag_disagrees"] <= 1


def test_partial_week_is_flagged(themed):
    assert latest_complete_week(themed) == "2026-W26"
    md = format_digest_markdown(build_weekly_digest(themed, "2026-W27"))
    assert "Only 2 days of data" in md


def test_markdown_has_the_sections_priya_reads(themed):
    md = format_digest_markdown(build_weekly_digest(themed, "2026-W26"))
    for section in ["This week in three lines", "What customers complained about",
                    "Customers who had to come back", "Product watch", "How this was made"]:
        assert section in md
    assert "VR8" not in md and "VR9" not in md, "order numbers must be masked in quotes"


def test_lot_check_corrects_for_multiple_testing(themed):
    lots = detect_manufacturing_lot_anomalies(themed)
    assert lots.attrs["lots_tested"] > 1000
    # with ~1,270 lots, an uncorrected p<0.01 rule would flag ~13 by chance alone
    assert lots["is_anomaly"].sum() <= 2
    assert (lots.loc[lots["is_anomaly"], "units"] >= 20).all()
