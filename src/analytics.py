"""
Deterministic Operational Analytics & SLA Engine for Vireo Audio.
100% deterministic Pandas/NumPy aggregations ensuring zero floating-point hallucination.

Calculates:
1. Channel-level SLA breaches, First Response Times (FRT), and auto-credits.
2. Internal team transfers and re-handling cost (policy §4: Rs 305 each).
3. Weekly and monthly time-series aggregations with WoW movement.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clean import clean_tickets_data

TRANSFER_UNIT_COST_INR = 305.0

def compute_channel_sla_metrics(df_clean):
    """
    Computes SLA targets, first response times, breach counts, breach rates,
    and automatic store credit liabilities broken down by channel.
    """
    channel_summary = df_clean.groupby('channel').agg(
        total_tickets=('ticket_id', 'count'),
        median_frt_min=('first_response_time_min', 'median'),
        mean_frt_min=('first_response_time_min', 'mean'),
        p90_frt_min=('first_response_time_min', lambda x: x.quantile(0.90)),
        breaches=('is_breached', 'sum'),
        breach_rate=('is_breached', 'mean'),
        resolved_breaches=('ticket_id', lambda x: (df_clean.loc[x.index, 'is_breached'] & df_clean.loc[x.index, 'status'].isin(['resolved', 'closed'])).sum()),
        sla_credits_inr=('sla_credit_inr', 'sum'),
        avg_csat=('csat_score', 'mean')
    ).reset_index()

    channel_summary['pct_of_volume'] = channel_summary['total_tickets'] / len(df_clean) * 100.0
    return channel_summary

def compute_weekly_time_series(df_clean):
    """
    Groups data into ISO weeks (year_week) with WoW volume changes,
    breaches, SLA credits, repeat contacts, and CSAT.
    """
    weekly = df_clean.groupby('year_week').agg(
        start_date=('created_at_dt', 'min'),
        end_date=('created_at_dt', 'max'),
        tickets=('ticket_id', 'count'),
        breaches=('is_breached', 'sum'),
        sla_credits_inr=('sla_credit_inr', 'sum'),
        repeat_contacts=('is_repeat_contact_30d', 'sum'),
        transfers=('transfers', 'sum'),
        avg_csat=('csat_score', 'mean')
    ).reset_index().sort_values('year_week')

    # Compute WoW metrics
    weekly['tickets_wow_pct'] = weekly['tickets'].pct_change() * 100.0
    weekly['breach_rate'] = weekly['breaches'] / weekly['tickets']
    weekly['repeat_contact_rate'] = weekly['repeat_contacts'] / weekly['tickets']
    weekly['12w_moving_avg'] = weekly['tickets'].rolling(window=12, min_periods=1).mean()

    return weekly

def get_overall_operational_kpis(df_clean=None):
    """
    Produces the top-level KPI dictionary matching executive dashboard cards.
    """
    if df_clean is None:
        df_clean = clean_tickets_data()

    total_tickets = len(df_clean)
    weekly_ts = compute_weekly_time_series(df_clean)
    avg_tickets_per_week = weekly_ts['tickets'].mean()
    
    total_breaches = df_clean['is_breached'].sum()
    breach_rate = total_breaches / total_tickets if total_tickets > 0 else 0
    total_sla_credits = df_clean['sla_credit_inr'].sum()
    
    total_transfers = df_clean['transfers'].sum()
    total_transfer_cost = total_transfers * TRANSFER_UNIT_COST_INR
    
    total_repeat_contacts = df_clean['is_repeat_contact_30d'].sum()
    repeat_rate = total_repeat_contacts / total_tickets if total_tickets > 0 else 0
    
    avg_csat = df_clean['csat_score'].mean()
    csat_responses = df_clean['csat_score'].notna().sum()

    return {
        "total_tickets": total_tickets,
        "avg_tickets_per_week": avg_tickets_per_week,
        "total_breaches": total_breaches,
        "breach_rate": breach_rate,
        "total_sla_credits_inr": total_sla_credits,
        "total_transfers": total_transfers,
        "total_transfer_cost_inr": total_transfer_cost,
        "total_repeat_contacts": total_repeat_contacts,
        "repeat_rate": repeat_rate,
        "avg_csat": avg_csat,
        "csat_responses": csat_responses
    }

if __name__ == "__main__":
    print("Running Part 3: Operational Analytics & SLA Engine...")
    df = clean_tickets_data()
    kpis = get_overall_operational_kpis(df)
    
    print("\n" + "="*80)
    print("                    EXECUTIVE OPERATIONAL KPIS")
    print("="*80)
    print(f"Total Unique Tickets:        {kpis['total_tickets']:,}")
    print(f"Avg Tickets / Week:          {kpis['avg_tickets_per_week']:.1f}")
    print(f"SLA Breaches:                {kpis['total_breaches']:,} ({kpis['breach_rate']*100:.2f}%)")
    print(f"Automatic SLA Credits:       ₹{kpis['total_sla_credits_inr']:,.2f}")
    print(f"Internal Transfers:          {kpis['total_transfers']:,} (Cost: ₹{kpis['total_transfer_cost_inr']:,.2f})")
    print(f"Repeat Contacts (30d):       {kpis['total_repeat_contacts']:,} ({kpis['repeat_rate']*100:.2f}%)")
    print(f"Average CSAT:                {kpis['avg_csat']:.2f} / 5.00 ({kpis['csat_responses']:,} responses)")
    print("="*80)
    
    print("\nCHANNEL PERFORMANCE:")
    print(compute_channel_sla_metrics(df)[['channel', 'total_tickets', 'median_frt_min', 'breach_rate', 'sla_credits_inr']])
