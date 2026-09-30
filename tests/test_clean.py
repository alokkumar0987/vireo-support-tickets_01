"""
Unit tests for Part 1: Data Ingestion, Reconciliation & Integrity Pipeline
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import clean_tickets_data, load_raw_data

def test_deduplication():
    raw_data = load_raw_data()
    raw_tickets_count = len(raw_data['tickets'])
    assert raw_tickets_count == 12528, f"Expected 12,528 raw ticket rows, got {raw_tickets_count}"
    
    clean_df = clean_tickets_data(raw_data)
    unique_tickets_count = len(clean_df)
    assert unique_tickets_count == 11875, f"Expected 11,875 unique tickets after deduplication, got {unique_tickets_count}"
    assert clean_df['ticket_id'].nunique() == 11875, "Ticket IDs are not strictly unique!"
    assert (raw_tickets_count - unique_tickets_count) == 653, "Expected exactly 653 duplicate ticket pairs purged!"

def test_csat_sanitization():
    clean_df = clean_tickets_data()
    # Ensure no 0.0 values remain in csat_score
    assert (clean_df['csat_score'] == 0).sum() == 0, "CSAT still contains 0.0 values! Must be NaN per Policy §8."
    # Ensure valid CSAT scores are between 1 and 5
    valid_scores = clean_df['csat_score'].dropna()
    assert valid_scores.min() >= 1.0, f"Minimum CSAT should be >= 1.0, got {valid_scores.min()}"
    assert valid_scores.max() <= 5.0, f"Maximum CSAT should be <= 5.0, got {valid_scores.max()}"
    # Authentic average should be ~3.32
    assert 3.30 <= clean_df['csat_score'].mean() <= 3.35, f"Expected CSAT mean ~3.32, got {clean_df['csat_score'].mean():.2f}"

def test_timezone_reconciliation():
    clean_df = clean_tickets_data()
    resolved_tickets = clean_df.dropna(subset=['resolved_at_dt'])
    # In resolved tickets, resolved_at must be >= created_at
    negative_resolutions = (resolved_tickets['resolved_at_dt'] < resolved_tickets['created_at_dt']).sum()
    assert negative_resolutions == 0, f"Found {negative_resolutions} tickets where resolved_at is before created_at!"

def test_operational_features_presence():
    clean_df = clean_tickets_data()
    expected_cols = [
        'first_response_time_min',
        'sla_target_min',
        'is_breached',
        'sla_credit_inr',
        'handle_time_hours',
        'contact_cost_inr',
        'is_repeat_contact_30d',
        'product_name',
        'unit_cost_inr',
        'team',
        'tier'
    ]
    for col in expected_cols:
        assert col in clean_df.columns, f"Missing engineered column: {col}"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
