"""
Agent leaderboard: tickets closed per week, as Priya asked, with the context that keeps it fair.

What the data showed (see weekly_leaderboard docstring for the numbers it recomputes):
- Weekly counts are tiny (2-9 tickets per agent in this export) and a weekly rank barely predicts
  next week's (rank correlation ~0.13 within Chat Frontline). So the table shows THIS week's count
  but ranks on the 4-week average, within team.
- Volume follows the queue, not effort: Billing's Day shift has one agent, who closes ~2x their
  Morning-shift teammates. Agents alone on their team's shift are flagged.
- Tier 2 (Escalations & Warranty) closes 1-2 cases a week by design. Policy §6: measured on
  resolution in days, never compared with Tier 1 on volume. They get their own table, no count rank.
- The earlier "balanced score" (35% CSAT / 35% FCR / 30% SLA) was dropped: the weights were ours,
  not Vireo's, and a single opaque number is harder to act on than the columns it hides.
- SLA breaches are left out on purpose: policy §3 charges them to the RESOLVING agent, but the
  first response is often made by someone else or missed while nobody was on shift.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clean import clean_tickets_data, latest_complete_week, week_end
from src.themes import ensure_themes

WINDOW_WEEKS = 4
FCR_LOOKBACK_DAYS = 90
ATTENDED = ["resolved", "closed"]  # policy §10: attendance = resolved or closed (incl. 72h auto-close)


def compute_trailing_fcr(df, target_week=None, lookback_days=FCR_LOOKBACK_DAYS):
    """
    Per-agent first-contact resolution (policy §10) over tickets whose 30-day repeat window has
    closed: 1 - share of the agent's resolved tickets that were followed by a repeat. A week's FCR
    is only known 30 days later, so with a target week this uses the preceding `lookback_days`.
    """
    matured = df[df["repeat_window_complete"]]
    if target_week:
        end = week_end(target_week)  # bounded both sides: a past week must not see later tickets
        matured = matured[(matured["created_at_dt"] > end - pd.Timedelta(days=lookback_days)) &
                          (matured["created_at_dt"] < end)]
    return matured.groupby("agent_id").agg(
        fcr_pct=("caused_repeat_30d", lambda c: (1.0 - c.mean()) * 100.0),
        fcr_basis_tickets=("ticket_id", "count"),
    ).reset_index()


def _window(df, week, n=WINDOW_WEEKS):
    weeks = sorted(df["year_week"].unique())
    i = weeks.index(week)
    return weeks[max(0, i - n + 1):i + 1]


def _vague_note_share(notes):
    from src.themes import keyword_outcome
    return float(np.mean([keyword_outcome(n) == "unclear" for n in notes])) if len(notes) else np.nan


def _agents(df):
    return (df.dropna(subset=["agent_id"])
              .drop_duplicates("agent_id")[["agent_id", "name", "team", "tier", "site", "shift"]])


def weekly_rank_stability(df, teams_df, n_weeks=26):
    """Mean Spearman correlation of agents' closed counts between consecutive weeks, per Tier 1 team."""
    att = df[df["status"].isin(ATTENDED)]
    weeks = sorted(df["year_week"].unique())[-(n_weeks + 2):-1]
    counts = (att[att["resolved_week"].isin(weeks)].groupby(["agent_id", "resolved_week"]).size()
                .unstack(fill_value=0).reindex(columns=weeks, fill_value=0))
    out = {}
    for team, g in teams_df[teams_df["tier"] == 1].groupby("team"):
        c = counts.reindex(g["agent_id"]).fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):  # weeks where everyone closed the same count
            cors = [c[a].rank().corr(c[b].rank()) for a, b in zip(weeks[:-1], weeks[1:])]
        out[team] = float(np.nanmean(cors))
    return out


def build_tier_1_scorecard(df, target_week=None):
    """Tier 1: closed this week + 4-week average, ranked within team, with quality context."""
    df = ensure_themes(df)
    week = target_week or latest_complete_week(df)
    window = _window(df, week)
    agents = _agents(df)
    t1 = agents[agents["tier"] == 1].copy()
    att = df[df["status"].isin(ATTENDED)]
    # "Closed" = the week the ticket was resolved, not the week it was opened.
    this_week = att[att["resolved_week"] == week].groupby("agent_id").size()
    in_window = att[att["resolved_week"].isin(window)]
    per_week = in_window.groupby("agent_id").size() / len(window)

    t1["closed_this_week"] = t1["agent_id"].map(this_week).fillna(0).astype(int)
    t1["closed_per_week_4wk"] = t1["agent_id"].map(per_week).fillna(0.0)
    t1["team_avg_per_week"] = t1.groupby("team")["closed_per_week_4wk"].transform("mean")
    t1["vs_team_avg"] = t1["closed_per_week_4wk"] / t1["team_avg_per_week"].replace(0, np.nan)
    t1["rank_in_team"] = (t1.groupby("team")["closed_per_week_4wk"]
                            .rank(ascending=False, method="min").astype(int))
    t1["agents_on_same_shift"] = t1.groupby(["team", "shift"])["agent_id"].transform("count")
    t1["sole_agent_on_shift"] = t1["agents_on_same_shift"] == 1

    quality = in_window.groupby("agent_id").agg(
        csat=("csat_score", "mean"), csat_responses=("csat_score", lambda s: int(s.notna().sum())),
        vague_notes_pct=("agent_notes", lambda n: _vague_note_share(n.tolist()) * 100))
    t1 = t1.merge(quality, left_on="agent_id", right_index=True, how="left")
    t1 = t1.merge(compute_trailing_fcr(df[df["tier"] == 1], week), on="agent_id", how="left")
    return t1.sort_values(["team", "rank_in_team"]).reset_index(drop=True)


def build_tier_2_scorecard(df, target_week=None):
    """Tier 2 (policy §6): cases resolved and resolution in DAYS; no volume ranking against anyone."""
    df = ensure_themes(df)
    week = target_week or latest_complete_week(df)
    window = _window(df, week)
    agents = _agents(df)
    t2 = agents[agents["tier"] == 2].copy()
    att = df[df["status"].isin(ATTENDED)]
    end = week_end(week)
    recent = att[(att["resolved_at_dt"] > end - pd.Timedelta(days=FCR_LOOKBACK_DAYS)) &
                 (att["resolved_at_dt"] < end)]

    t2["cases_resolved_this_week"] = t2["agent_id"].map(att[att["resolved_week"] == week].groupby("agent_id").size()).fillna(0).astype(int)
    t2["cases_resolved_4wk"] = t2["agent_id"].map(att[att["resolved_week"].isin(window)].groupby("agent_id").size()).fillna(0).astype(int)
    days = recent.groupby("agent_id")["handle_time_hours"]
    t2["median_resolution_days"] = t2["agent_id"].map(days.median() / 24.0)
    t2["cases_90d"] = t2["agent_id"].map(days.count()).fillna(0).astype(int)
    t2["replacements_approved_90d"] = t2["agent_id"].map(recent[recent["replacement_issued"] == "Y"].groupby("agent_id").size()).fillna(0).astype(int)
    t2["csat"] = t2["agent_id"].map(recent.groupby("agent_id")["csat_score"].mean())
    t2 = t2.merge(compute_trailing_fcr(df[df["tier"] == 2], week), on="agent_id", how="left")
    return t2.sort_values("median_resolution_days").reset_index(drop=True)


def demonstrate_raw_leaderboard_bias(df):
    """What a single all-agent 'tickets closed' list would do: who ends up at the bottom."""
    att = df[df["status"].isin(ATTENDED)]
    ranked = (att.groupby(["agent_id", "name", "team", "tier"]).size().rename("tickets_closed")
                 .reset_index().sort_values("tickets_closed", ascending=False).reset_index(drop=True))
    ranked["raw_rank"] = ranked.index + 1
    n_t2 = int((ranked["tier"] == 2).sum())
    bottom = ranked.tail(n_t2)
    return {"raw_ranked_df": ranked, "bottom_5": ranked.tail(5),
            "all_bottom_tier_2": bool((ranked.tail(5)["tier"] == 2).all()),
            "tier_2_in_bottom_n": int((bottom["tier"] == 2).sum()), "n_tier_2": n_t2}


def weekly_leaderboard(df, target_week=None):
    df = ensure_themes(df)  # FCR must use the issue-level repeat definition
    week = target_week if target_week in set(df["year_week"]) else latest_complete_week(df)
    t1 = build_tier_1_scorecard(df, week)
    return {
        "week": week, "window": _window(df, week),
        "tier_1": t1, "tier_2": build_tier_2_scorecard(df, week),
        "stability": weekly_rank_stability(df, _agents(df)),
        "flat_list": demonstrate_raw_leaderboard_bias(df),
    }


def format_leaderboard_markdown(lb):
    w, t1, t2 = lb["week"], lb["tier_1"], lb["tier_2"]
    stab = lb["stability"]
    team_avg = t1.groupby("team")["team_avg_per_week"].first()
    md = [f"# Agent leaderboard: tickets closed, {w}",
          f"*Tickets counted in the week they were closed. Ranked within each team on the average of the last "
          f"{len(lb['window'])} weeks ({lb['window'][0]} to {w}); this week's count is shown alongside.*",
          "\n## How to read this",
          f"- **A single week is too few tickets to rank people.** Agents close 2–9 tickets a week in this data, "
          f"and this week's order barely predicts next week's (rank correlation {stab.get('Chat Frontline', np.nan):.2f} "
          f"in Chat Frontline, where 1.0 would be a stable order). Hence the 4-week column.",
          f"- **Compare people only within a team.** Teams get very different queues: team averages range from "
          f"{team_avg.min():.1f} ({team_avg.idxmin()}) to {team_avg.max():.1f} ({team_avg.idxmax()}) closed per "
          f"agent per week.",
          "- **🧍 = only agent on that team's shift.** Their volume is the whole shift's queue, not extra effort.",
          f"- **Escalations & Warranty is not on this list** (policy §6, and Neha's request). On one combined list "
          f"{lb['flat_list']['tier_2_in_bottom_n']} of the {lb['flat_list']['n_tier_2']} bottom places would be theirs, "
          f"because their cases take days by design. They are measured on days to resolve, below.",
          "- **FCR** = share of the agent's resolved tickets where the customer did *not* come back about the same "
          "issue within 30 days (policy §10; last 90 days, only tickets old enough to know). **Vague notes** = closing notes "
          "like \"done\" or \"see prev\" that leave the next agent nothing to work from."]

    for team, g in t1.groupby("team", sort=True):
        md.append(f"\n## {team}  (team average {g['team_avg_per_week'].iloc[0]:.1f} closed per agent per week)")
        md.append("| # | Agent | Shift | Closed this week | Per week (4 wk) | vs team | FCR | CSAT (n) | Vague notes |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in g.itertuples():
            sole = " 🧍" if r.sole_agent_on_shift else ""
            fcr = f"{r.fcr_pct:.0f}%" if pd.notna(r.fcr_pct) else "–"
            csat = f"{r.csat:.2f} ({r.csat_responses})" if pd.notna(r.csat) else f"– ({r.csat_responses or 0})"
            vague = f"{r.vague_notes_pct:.0f}%" if pd.notna(r.vague_notes_pct) else "–"
            md.append(f"| {r.rank_in_team} | {r.name} | {r.site} {r.shift}{sole} | {r.closed_this_week} | "
                      f"{r.closed_per_week_4wk:.1f} | {r.vs_team_avg:.0%} | {fcr} | {csat} | {vague} |")

    md.append("\n## Escalations & Warranty (Tier 2): measured in days, not tickets")
    md.append("| Agent | Shift | Cases this week | Cases (4 wk) | Median days to resolve (90 d) | Replacements approved (90 d) | FCR | CSAT |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in t2.itertuples():
        md.append(f"| {r.name} | {r.site} {r.shift} | {r.cases_resolved_this_week} | {r.cases_resolved_4wk} | "
                  f"{r.median_resolution_days:.1f} | {r.replacements_approved_90d} | "
                  f"{(f'{r.fcr_pct:.0f}%' if pd.notna(r.fcr_pct) else '–')} | "
                  f"{(f'{r.csat:.2f}' if pd.notna(r.csat) else '–')} |")
    return "\n".join(md)


def generate_weekly_agent_leaderboards(df_clean, target_week=None):
    """Kept for run.py / app.py."""
    lb = weekly_leaderboard(df_clean, target_week)
    return {"tier_1_scorecard": lb["tier_1"], "tier_2_scorecard": lb["tier_2"],
            "bias_demo": lb["flat_list"], "leaderboard": lb}


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
    print(format_leaderboard_markdown(weekly_leaderboard(clean_tickets_data(), args.week)))
