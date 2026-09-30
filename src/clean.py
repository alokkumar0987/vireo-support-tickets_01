"""
Data Ingestion, Reconciliation & Integrity Pipeline for Vireo Audio.
Handles:
1. Deduplication of Freshdesk-to-Helpdesk migration reconciliation duplicates (653 pairs / 1306 rows).
2. Timezone reconciliation (+5:30h conversion from UTC to IST for legacy_fd reconstructed timestamps).
3. CSAT score sanitization (re-mapping 0.0 non-responses to NaN per Policy §8).
4. Relational denormalization across tickets, agents, products, customers, and orders.
5. Operational feature engineering (FRT, SLA breaches, SLA credits, handle time, repeat contacts).
"""

import os
import glob
import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SLA_TARGETS_MIN = {
    'chat': 15,
    'voice': 120,
    'social': 240,
    'email': 480
}

CHANNEL_COSTS_INR = {
    'chat': 210,
    'email': 260,
    'voice': 520,
    'social': 240
}

REPEAT_WINDOW_DAYS = 30  # Policy §10

def find_file(pattern, data_dir=DEFAULT_DATA_DIR):
    matches = glob.glob(os.path.join(data_dir, f"*{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"Could not find file matching '{pattern}' in {data_dir}")
    return matches[0]

def latest_complete_week(df):
    """Latest ISO week with tickets created on all 7 days (the export ends mid-week)."""
    days = df.groupby("year_week")["created_at_dt"].apply(lambda s: s.dt.date.nunique())
    complete = days[days == 7]
    return complete.index.max() if len(complete) else df["year_week"].max()


def week_end(week):
    """Exclusive end of an ISO week string like '2026-W26' (the following Monday 00:00)."""
    year, wk = week.split("-W")
    return pd.Timestamp.fromisocalendar(int(year), int(wk), 7).normalize() + pd.Timedelta(days=1)


def load_raw_data(data_dir=DEFAULT_DATA_DIR):
    """Loads all raw datasets from the data directory."""
    tickets_path = find_file("tickets", data_dir)
    agents_path = find_file("agents", data_dir)
    customers_path = find_file("customers", data_dir)
    products_path = find_file("products", data_dir)
    orders_path = find_file("orders", data_dir)

    df_tickets = pd.read_csv(tickets_path)
    df_agents = pd.read_csv(agents_path)
    df_customers = pd.read_csv(customers_path)
    df_products = pd.read_csv(products_path)
    df_orders = pd.read_csv(orders_path)

    return {
        "tickets": df_tickets,
        "agents": df_agents,
        "customers": df_customers,
        "products": df_products,
        "orders": df_orders
    }

def add_repeat_contact_features(df, window_days=REPEAT_WINDOW_DAYS, issue_col=None):
    """
    Policy §10: a ticket is resolved at first contact if the same customer does not
    contact again about the same issue within 30 days of resolution; a return contact
    in that window is a repeat contact, costed at the contact cost of the channel used.

    "Same issue":
      issue_col=None   same customer + same product_sku. Used by clean_tickets_data(), before
                       themes exist. Over-counts: a delivery ticket then a battery ticket on the
                       same product counts as a repeat (29.8% of contacts; only 34% of these
                       pairs share a theme).
      issue_col='issue_family' (set by themes.attach_themes) adds the complaint family, which
                       is policy §10 proper (12.9% of contacts).
    A contact made while the earlier ticket is still open is also a repeat (the customer is
    chasing). If the earlier ticket was never resolved, its window runs from creation.

    Adds:
      is_repeat_contact_30d   this contact is a repeat of an earlier one (the failure demand)
      repeat_cost_inr         channel cost of that repeat contact (0 otherwise)
      caused_repeat_30d       this attended ticket was followed by a repeat (FCR failure),
                              i.e. the metric to charge to the agent who resolved it
      repeat_window_complete  attended ticket whose full 30-day window lies inside the export;
                              filter on it before computing FCR, or recent tickets look perfect
      is_repeat_same_tag_30d  stricter variant: same product AND same bot category (lower bound)
      is_any_recontact_30d    old definition: any contact within 30 days of any earlier
                              creation, for comparison only
    """
    df = df.copy()
    window = pd.Timedelta(days=window_days)

    def _flag_repeats(frame, keys):
        frame = frame.sort_values(keys + ['created_at_dt'])
        grp = frame.groupby(keys)
        prev_created = grp['created_at_dt'].shift(1)
        prev_window_start = grp['resolved_at_dt'].shift(1).fillna(prev_created)
        is_repeat = prev_created.notna() & (frame['created_at_dt'] <= prev_window_start + window)

        next_created = grp['created_at_dt'].shift(-1)
        attended = frame['status'].isin(['resolved', 'closed'])
        caused = attended & next_created.notna() & (next_created <= frame['resolved_at_dt'] + window)
        return is_repeat.reindex(df.index), caused.reindex(df.index)

    product_keys = ['customer_id', 'product_sku']
    issue_keys = product_keys + ([issue_col] if issue_col else [])
    df['is_repeat_contact_30d'], df['caused_repeat_30d'] = _flag_repeats(df, issue_keys)
    df['is_repeat_same_tag_30d'], _ = _flag_repeats(df, product_keys + ['category'])
    if issue_col:
        df['is_repeat_same_product_30d'], df['caused_repeat_same_product_30d'] = _flag_repeats(df, product_keys)

    df['repeat_cost_inr'] = np.where(df['is_repeat_contact_30d'], df['contact_cost_inr'], 0)

    data_end = df['created_at_dt'].max()
    df['repeat_window_complete'] = (
        df['status'].isin(['resolved', 'closed']) & (df['resolved_at_dt'] + window <= data_end)
    )

    by_customer = df.sort_values(['customer_id', 'created_at_dt'])
    prev_any = by_customer.groupby('customer_id')['created_at_dt'].shift(1)
    df['is_any_recontact_30d'] = (
        prev_any.notna() & (by_customer['created_at_dt'] - prev_any <= window)
    ).reindex(df.index)

    return df

def clean_tickets_data(raw_data=None, data_dir=DEFAULT_DATA_DIR):
    """
    Cleans, deduplicates, reconciles timezones, sanitizes CSAT, and enriches
    the tickets dataset with relational metadata and operational features.
    """
    if raw_data is None:
        raw_data = load_raw_data(data_dir)

    df_tickets = raw_data["tickets"].copy()
    df_agents = raw_data["agents"].copy()
    df_customers = raw_data["customers"].copy()
    df_products = raw_data["products"].copy()
    df_orders = raw_data["orders"].copy()

    # --- STEP 1: DEDUPLICATION ---
    # Migration reconciliation caused 1,306 duplicate rows (653 ticket pairs).
    # Prioritize 'helpdesk' source system over 'legacy_fd'
    df_tickets['source_system_priority'] = df_tickets['source_system'].map({'helpdesk': 0, 'legacy_fd': 1}).fillna(2)
    df_clean = df_tickets.sort_values(
        by=['ticket_id', 'source_system_priority'],
        ascending=[True, True]
    ).drop_duplicates(subset=['ticket_id'], keep='first').drop(columns=['source_system_priority']).copy()

    # --- STEP 2: CSAT SANITIZATION (Policy §8 Compliance) ---
    # Freshdesk recorded non-responses as 0.0. Policy §8 dictates blanks are non-responses and must be excluded.
    df_clean['raw_csat_score'] = df_clean['csat_score']
    df_clean.loc[df_clean['csat_score'] == 0, 'csat_score'] = np.nan

    # --- STEP 3: DATETIME PARSING & TIMEZONE RECONCILIATION ---
    df_clean['created_at_dt'] = pd.to_datetime(df_clean['created_at'])
    df_clean['first_response_at_dt'] = pd.to_datetime(df_clean['first_response_at'])
    df_clean['resolved_at_dt'] = pd.to_datetime(df_clean['resolved_at'])

    # Tickets originating from legacy_fd had resolved_at reconstructed in UTC,
    # whereas created_at and first_response_at are in IST (UTC+5:30).
    legacy_mask = (df_clean['source_system'] == 'legacy_fd') & df_clean['resolved_at_dt'].notna()
    df_clean.loc[legacy_mask, 'resolved_at_dt'] = df_clean.loc[legacy_mask, 'resolved_at_dt'] + pd.Timedelta(hours=5, minutes=30)

    # --- STEP 4: OPERATIONAL METRICS & SLA CALCULATIONS ---
    # First response time in minutes
    df_clean['first_response_time_min'] = (
        df_clean['first_response_at_dt'] - df_clean['created_at_dt']
    ).dt.total_seconds() / 60.0

    # SLA targets
    df_clean['sla_target_min'] = df_clean['channel'].map(SLA_TARGETS_MIN)
    df_clean['is_breached'] = df_clean['first_response_time_min'] > df_clean['sla_target_min']

    # SLA breach penalty credit: ₹350 on resolution (Policy §3)
    df_clean['sla_credit_inr'] = np.where(
        df_clean['is_breached'] & df_clean['status'].isin(['resolved', 'closed']),
        350,
        0
    )

    # Handle time in hours
    df_clean['handle_time_hours'] = (
        df_clean['resolved_at_dt'] - df_clean['first_response_at_dt']
    ).dt.total_seconds() / 3600.0

    # Contact costs per policy §4
    df_clean['contact_cost_inr'] = df_clean['channel'].map(CHANNEL_COSTS_INR)

    # --- STEP 5: REPEAT CONTACT / FAILURE DEMAND (Policy §10) ---
    df_clean = add_repeat_contact_features(df_clean)

    # Time groupings
    # ISO-8601 week (Monday start, ISO year) so '2026-W24' means what the CLI help says
    df_clean['year_week'] = df_clean['created_at_dt'].dt.strftime('%G-W%V')
    df_clean['year_month'] = df_clean['created_at_dt'].dt.strftime('%Y-%m')
    # Week the ticket was CLOSED (blank if open/pending): "tickets closed per week" counts on this,
    # not on year_week (13% of attended tickets, 65% in Tier 2, close in a later week than they opened).
    df_clean['resolved_week'] = df_clean['resolved_at_dt'].dt.strftime('%G-W%V')

    # --- STEP 6: RELATIONAL JOINS ---
    # Merge Products
    df_clean = df_clean.merge(
        df_products[['sku', 'product_name', 'family', 'unit_cost_inr', 'retail_price_inr', 'warranty_months']],
        left_on='product_sku',
        right_on='sku',
        how='left'
    )

    # Merge Agents
    df_clean = df_clean.merge(
        df_agents[['agent_id', 'name', 'site', 'team', 'shift', 'tier']],
        on='agent_id',
        how='left',
        suffixes=('', '_agent_meta')
    )

    # Merge Customers
    df_clean = df_clean.merge(
        df_customers[['customer_id', 'name', 'city', 'state', 'care_plus']],
        on='customer_id',
        how='left',
        suffixes=('', '_customer_meta')
    )

    # Merge Orders (primary join on order_id; fallback: customer_id + product_sku)
    df_clean = df_clean.merge(
        df_orders[['order_id', 'order_date', 'channel', 'qty', 'order_value_inr', 'lot_code']],
        on='order_id',
        how='left',
        suffixes=('', '_order_meta')
    )

    # Issue key for order-level audits: order_id when quoted, else the documented
    # customer_id + product_sku fallback (README: ~34% of tickets carry no order_id)
    df_clean['repeat_key'] = df_clean['order_id'].where(
        df_clean['order_id'].notna(),
        df_clean['customer_id'] + '|' + df_clean['product_sku']
    )

    return df_clean

if __name__ == "__main__":
    print("Running Vireo Audio Data Pipeline (Part 1)...")
    cleaned_df = clean_tickets_data()
    print("Data ingestion and sanitization successful!")
    print(f"Unique Tickets: {len(cleaned_df):,}")
    print(f"Date Range: {cleaned_df['created_at'].min()} to {cleaned_df['created_at'].max()}")
    print(f"Authentic CSAT Average: {cleaned_df['csat_score'].mean():.2f} / 5.00 (from {cleaned_df['csat_score'].notna().sum():,} responses)")
    print(f"SLA Breaches: {cleaned_df['is_breached'].sum():,} ({cleaned_df['is_breached'].mean()*100:.2f}%)")
    print(f"Repeat contacts (Policy §10, same product): {cleaned_df['is_repeat_contact_30d'].sum():,} ({cleaned_df['is_repeat_contact_30d'].mean()*100:.2f}% of contacts)")
    full = cleaned_df[cleaned_df['repeat_window_complete']]
    print(f"First-contact resolution: {(1 - full['caused_repeat_30d'].mean())*100:.2f}% (of {len(full):,} attended tickets with a full 30-day window)")
