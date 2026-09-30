"""
Vireo Audio Support Intelligence System — Command-Line Interface (CLI Runner)
Clean-machine execution entrypoint.

Usage:
    python run.py                           # Runs default analysis for latest week
    python run.py --week 2026-W24           # Runs weekly complaint digest for target week
    python run.py --report leakage          # Generates financial leakage & business case audit
    python run.py --report scorecard        # Generates balanced agent scorecard
    python run.py --report all              # Generates full operational & financial report
    python run.py --dashboard               # Launches the interactive Streamlit UI
"""

import argparse
import os
import sys
import subprocess

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.clean import clean_tickets_data, latest_complete_week
from src.digest import generate_weekly_digest, format_digest_markdown
from src.scorecard import generate_weekly_agent_leaderboards, format_leaderboard_markdown
from src.leakage import generate_comprehensive_financial_audit, format_business_case_markdown
from src.themes import attach_themes

def main():
    parser = argparse.ArgumentParser(description="Vireo Audio Support Intelligence System")
    parser.add_argument("--week", type=str, default=None, help="Target ISO week (e.g. 2026-W24)")
    parser.add_argument("--report", type=str, choices=["digest", "scorecard", "leakage", "all"], default="digest", help="Type of report to generate")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive Streamlit web dashboard")
    parser.add_argument("--output", type=str, default=None, help="Path to save markdown output")
    parser.add_argument("--llm", action="store_true",
                        help="Send tickets the local model is unsure about to the LLM (needs .env). "
                             "Labels are cached, so only new unsure tickets are sent: ~6%% of a new week. "
                             "Off by default: no network calls unless asked.")

    args = parser.parse_args()

    if args.dashboard:
        print("Launching Vireo Audio Support Analytics Dashboard...")
        # sys.executable -m: works in a fresh venv where the `streamlit` script is not on PATH,
        # and an absolute path works from any working directory.
        app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app])
        return

    print("Loading and reconciling Vireo Audio support datasets...")
    try:
        df_clean = clean_tickets_data()
    except FileNotFoundError as e:
        sys.exit(f"{e}\nCopy the brief's pack files (tickets.csv, agents.csv, orders.csv, customers.csv, "
                 "products.csv) into data/ first - see README 'Quick start'.")
    print(f"Loaded {len(df_clean):,} unique tickets across {df_clean['created_at'].min()[:10]} to {df_clean['created_at'].max()[:10]}.\n")

    output_blocks = []
    target_week = args.week if args.week else latest_complete_week(df_clean)

    # Every report works on the themed data, so "repeat contact" means the same thing everywhere:
    # same customer + product + issue (policy §10), which needs the complaint themes.
    print("Classifying complaints (cached LLM labels + local model; --llm to send unsure ones to the LLM)...")
    df_themed = attach_themes(df_clean, call_llm=args.llm)

    # 1. Weekly Digest
    if args.report in ["digest", "all"]:
        digest = generate_weekly_digest(df_themed, target_week=target_week)
        output_blocks.append(format_digest_markdown(digest))

    # 2. Agent leaderboard (tickets closed per week, within team; Tier 2 separately in days)
    if args.report in ["scorecard", "all"]:
        scorecards = generate_weekly_agent_leaderboards(df_themed, target_week=target_week)
        output_blocks.append(format_leaderboard_markdown(scorecards["leaderboard"]))

    # 3. Business case and money audit (for Arjun)
    if args.report in ["leakage", "all"]:
        audit = generate_comprehensive_financial_audit(df_themed)
        output_blocks.append(format_business_case_markdown(audit))

    full_output = "\n\n---\n\n".join(output_blocks) + "\n"
    print(full_output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(full_output)
        print(f"\nReport saved to: {args.output}")

if __name__ == "__main__":
    main()
