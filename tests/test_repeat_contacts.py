"""
Rule tests for the Policy §10 repeat-contact definition (src/clean.py::add_repeat_contact_features),
on tiny synthetic ticket histories so each rule is checked in isolation.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import add_repeat_contact_features


def _tickets(rows):
    """rows: (ticket_id, customer, sku, created, resolved_or_None, status, category)"""
    df = pd.DataFrame(rows, columns=['ticket_id', 'customer_id', 'product_sku', 'created_at_dt',
                                     'resolved_at_dt', 'status', 'category'])
    df['created_at_dt'] = pd.to_datetime(df['created_at_dt'])
    df['resolved_at_dt'] = pd.to_datetime(df['resolved_at_dt'])
    df['contact_cost_inr'] = 210
    return add_repeat_contact_features(df).set_index('ticket_id')


def test_repeat_is_charged_to_the_earlier_ticket_not_the_later_one():
    df = _tickets([
        ('T1', 'C1', 'SKU-A', '2026-01-01', '2026-01-02', 'resolved', 'Audio'),
        ('T2', 'C1', 'SKU-A', '2026-01-20', '2026-01-21', 'resolved', 'Audio'),
        ('T9', 'C9', 'SKU-A', '2026-12-31', None, 'open', 'Audio'),  # pushes data end out
    ])
    assert df.loc['T1', 'caused_repeat_30d'] and not df.loc['T1', 'is_repeat_contact_30d']
    assert df.loc['T2', 'is_repeat_contact_30d'] and not df.loc['T2', 'caused_repeat_30d']
    assert df.loc['T2', 'repeat_cost_inr'] == 210 and df.loc['T1', 'repeat_cost_inr'] == 0


def test_window_runs_from_resolution_not_creation():
    # 40 days after creation but 25 days after a slow resolution -> still a repeat
    df = _tickets([
        ('T1', 'C1', 'SKU-A', '2026-01-01', '2026-01-16', 'resolved', 'Audio'),
        ('T2', 'C1', 'SKU-A', '2026-02-10', '2026-02-11', 'resolved', 'Audio'),
    ])
    assert df.loc['T2', 'is_repeat_contact_30d']
    assert not df.loc['T2', 'is_any_recontact_30d'], "old creation-to-creation rule misses it"


def test_after_30_days_or_other_product_is_not_a_repeat():
    df = _tickets([
        ('T1', 'C1', 'SKU-A', '2026-01-01', '2026-01-02', 'resolved', 'Audio'),
        ('T2', 'C1', 'SKU-A', '2026-02-05', '2026-02-06', 'resolved', 'Audio'),  # 34 days later
        ('T3', 'C1', 'SKU-B', '2026-02-07', '2026-02-08', 'resolved', 'Audio'),  # other product
    ])
    assert not df.loc['T2', 'is_repeat_contact_30d']
    assert not df.loc['T3', 'is_repeat_contact_30d']
    assert df.loc['T3', 'is_any_recontact_30d'], "old rule counted any product"


def test_chasing_an_open_ticket_counts_as_a_repeat():
    df = _tickets([
        ('T1', 'C1', 'SKU-A', '2026-01-01', '2026-01-10', 'resolved', 'Delivery'),
        ('T2', 'C1', 'SKU-A', '2026-01-05', '2026-01-10', 'resolved', 'Delivery'),  # before T1 resolved
    ])
    assert df.loc['T2', 'is_repeat_contact_30d']
    assert df.loc['T1', 'caused_repeat_30d']


def test_same_tag_variant_is_stricter():
    df = _tickets([
        ('T1', 'C1', 'SKU-A', '2026-01-01', '2026-01-02', 'resolved', 'Delivery'),
        ('T2', 'C1', 'SKU-A', '2026-01-10', '2026-01-11', 'resolved', 'Battery'),
    ])
    assert df.loc['T2', 'is_repeat_contact_30d']
    assert not df.loc['T2', 'is_repeat_same_tag_30d']


def test_recent_tickets_are_excluded_from_fcr_until_window_closes():
    df = _tickets([
        ('T1', 'C1', 'SKU-A', '2026-01-01', '2026-01-02', 'resolved', 'Audio'),
        ('T2', 'C2', 'SKU-A', '2026-03-01', '2026-03-02', 'resolved', 'Audio'),  # data ends 2026-03-15
        ('T3', 'C3', 'SKU-A', '2026-03-15', None, 'open', 'Audio'),
    ])
    assert df.loc['T1', 'repeat_window_complete']
    assert not df.loc['T2', 'repeat_window_complete'], "only 13 days observed: cannot claim FCR yet"
    assert not df.loc['T3', 'repeat_window_complete'], "open tickets are not attended"
