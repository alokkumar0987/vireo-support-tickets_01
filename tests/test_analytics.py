"""
Unit tests for Part 3: Operational Analytics & SLA Engine
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import clean_tickets_data
from src.analytics import (
    compute_channel_sla_metrics,
    compute_weekly_time_series,
    get_overall_operational_kpis
)

@pytest.fixture(scope="module")
def clean_data():
    return clean_tickets_data()

def test_overall_kpis(clean_data):
    kpis = get_overall_operational_kpis(clean_data)
    assert kpis['total_tickets'] == 11875
    assert kpis['total_breaches'] == 1051
    assert 0.088 <= kpis['breach_rate'] <= 0.089
    assert kpis['total_sla_credits_inr'] == 347200.0
    assert kpis['total_transfers'] == 1169
    assert kpis['total_transfer_cost_inr'] == 356545.0
    assert kpis['total_repeat_contacts'] == 3536  # clean.py's product-level definition; themes.ensure_themes narrows it to the §10 issue level (12.9%)
    assert 3.30 <= kpis['avg_csat'] <= 3.35

def test_channel_sla_metrics(clean_data):
    channels = compute_channel_sla_metrics(clean_data)
    assert len(channels) == 4
    
    # Verify Email has highest breach rate
    email_row = channels[channels['channel'] == 'email'].iloc[0]
    assert email_row['total_tickets'] == 3807
    assert email_row['breaches'] == 440
    assert 0.115 <= email_row['breach_rate'] <= 0.116
    assert email_row['median_frt_min'] == 165.0
    
    # Verify Chat has lowest median FRT
    chat_row = channels[channels['channel'] == 'chat'].iloc[0]
    assert chat_row['median_frt_min'] == 4.0
    assert chat_row['total_tickets'] == 5161

def test_weekly_time_series(clean_data):
    weekly = compute_weekly_time_series(clean_data)
    assert len(weekly) >= 75, f"Expected ~78 weeks, got {len(weekly)}"
    assert weekly['tickets'].sum() == 11875
    assert '12w_moving_avg' in weekly.columns
    assert 'tickets_wow_pct' in weekly.columns

if __name__ == "__main__":
    pytest.main(["-v", __file__])
