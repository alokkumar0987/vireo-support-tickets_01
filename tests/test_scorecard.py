"""
Tests for the agent leaderboard (src/scorecard.py).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import clean_tickets_data
from src.themes import attach_themes
from src.scorecard import (
    demonstrate_raw_leaderboard_bias, build_tier_1_scorecard, build_tier_2_scorecard,
    weekly_leaderboard, format_leaderboard_markdown, generate_weekly_agent_leaderboards,
)

WEEK = "2026-W26"


@pytest.fixture(scope="module")
def raw_data():
    return clean_tickets_data()


@pytest.fixture(scope="module")
def clean_data(raw_data):
    # themed, as run.py and app.py pass it (policy §10 issue-level repeats)
    return attach_themes(raw_data)


@pytest.fixture(scope="module")
def lb(clean_data):
    return weekly_leaderboard(clean_data, WEEK)


def test_flat_list_would_bury_tier_2(clean_data):
    bias = demonstrate_raw_leaderboard_bias(clean_data)
    assert bias["all_bottom_tier_2"], "a single tickets-closed list puts Tier 2 at the bottom"
    assert bias["tier_2_in_bottom_n"] >= 5


def test_tier_1_ranks_only_within_team(lb):
    t1 = lb["tier_1"]
    assert len(t1) == 38 and (t1["tier"] == 1).all()
    for team, g in t1.groupby("team"):
        assert g["rank_in_team"].min() == 1, f"{team} ranks must restart at 1"
        assert g.sort_values("rank_in_team")["closed_per_week_4wk"].is_monotonic_decreasing


def test_tier_1_counts_match_the_tickets(clean_data, lb):
    t1 = lb["tier_1"]
    attended = clean_data[clean_data["status"].isin(["resolved", "closed"]) & (clean_data["tier"] == 1)]
    # counted by the week the ticket was CLOSED (Priya: "tickets closed per week")
    assert t1["closed_this_week"].sum() == (attended["resolved_week"] == WEEK).sum()
    in_window = attended[attended["resolved_week"].isin(lb["window"])]
    assert len(lb["window"]) == 4
    assert abs(t1["closed_per_week_4wk"].sum() - len(in_window) / 4) < 1e-9


def test_sole_agent_on_shift_is_flagged(lb):
    t1 = lb["tier_1"].set_index("agent_id")
    assert t1.loc["A3033", "sole_agent_on_shift"], "Diya Singh is the only Billing agent on the Day shift"
    assert not t1.loc["A3032", "sole_agent_on_shift"]


def test_quality_columns_are_in_range(lb):
    t1 = lb["tier_1"]
    assert t1["fcr_pct"].between(0, 100).all()
    assert t1["vague_notes_pct"].dropna().between(0, 100).all()
    assert t1["csat"].dropna().between(1, 5).all()


def test_tier_2_measured_in_days_not_volume(clean_data):
    t2 = build_tier_2_scorecard(clean_data, WEEK)
    assert len(t2) == 6 and (t2["tier"] == 2).all()
    assert "rank" not in " ".join(t2.columns), "Tier 2 is not ranked on counts (policy §6)"
    assert (t2["median_resolution_days"] > 2.0).all(), "warranty cases take days by design"
    assert t2["median_resolution_days"].is_monotonic_increasing


def test_weekly_order_is_unstable_which_is_why_we_use_4_weeks(lb):
    assert lb["stability"]["Chat Frontline"] < 0.5


def test_markdown_keeps_tier_2_out_of_team_tables(lb):
    md = format_leaderboard_markdown(lb)
    tier1_part, tier2_part = md.split("## Escalations & Warranty (Tier 2)")
    for name in lb["tier_2"]["name"]:
        assert name not in tier1_part and name in tier2_part
    assert "How to read this" in md and "🧍" in md


def test_wrapper_used_by_run_and_app(clean_data):
    out = generate_weekly_agent_leaderboards(clean_data, WEEK)
    assert {"tier_1_scorecard", "tier_2_scorecard", "leaderboard"} <= set(out)
    assert len(build_tier_1_scorecard(clean_data)) == 38


def test_fcr_is_the_same_whether_or_not_the_caller_attached_themes(raw_data, clean_data):
    # Regression: the dashboard once passed unthemed data and every agent's FCR moved by up to 29 points.
    a = build_tier_1_scorecard(raw_data, WEEK).set_index("agent_id")["fcr_pct"]
    b = build_tier_1_scorecard(clean_data, WEEK).set_index("agent_id")["fcr_pct"]
    assert a.round(6).equals(b.round(6))
