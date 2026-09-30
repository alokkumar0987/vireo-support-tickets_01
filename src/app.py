"""
Vireo Audio — Support Analytics Dashboard (Streamlit Application)
The same numbers as `python run.py --report all`, clickable: every view reads the THEMED data
(attach_themes), so repeat contacts and FCR use the policy §10 issue-level definition.

To run:
    streamlit run src/app.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clean import clean_tickets_data
from src.analytics import (
    compute_channel_sla_metrics,
    compute_weekly_time_series,
    get_overall_operational_kpis
)
from src.digest import (
    generate_weekly_digest,
    format_digest_markdown,
    detect_manufacturing_lot_anomalies,
    latest_complete_week,
)
from src.themes import attach_themes
from src.scorecard import build_tier_1_scorecard, generate_weekly_agent_leaderboards, format_leaderboard_markdown
from src.leakage import generate_comprehensive_financial_audit, format_business_case_markdown

# Page configuration
st.set_page_config(
    page_title="Vireo Audio — Support Analytics",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Navy / Slate Theme CSS matching the mockup
st.markdown("""
<style>
    /* Dark Slate & Navy Theme */
    .stApp {
        background-color: #0b1120;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0px 20px 0px;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 20px;
    }
    .header-logo {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #38bdf8;
    }
    .header-subtitle {
        font-size: 13px;
        color: #94a3b8;
    }

    /* KPI Cards */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .kpi-title {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-val {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
    }
    .kpi-sub {
        font-size: 11px;
        color: #ef4444;
        font-weight: 500;
    }
    .kpi-sub-neutral {
        font-size: 11px;
        color: #64748b;
    }

    /* Section Cards */
    .section-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .section-title {
        font-size: 14px;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 12px;
    }

    /* Sidebar summary */
    .sidebar-summary {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 12px;
        font-size: 12px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prep_data():
    return clean_tickets_data()

@st.cache_data
def load_themed_data():
    # cached LLM labels + local TF-IDF model + keyword rules; never makes network calls from the UI
    return attach_themes(load_and_prep_data())

@st.cache_data
def load_audit():
    return generate_comprehensive_financial_audit(load_themed_data())

# Themed data everywhere: the unthemed frame carries the product-level repeat columns (29.8%),
# which contradict the report's policy §10 figure (12.9%) and every agent's FCR.
df_clean = load_themed_data()
LATEST_WEEK = latest_complete_week(df_clean)
DATE_FROM, DATE_TO = df_clean['created_at_dt'].min(), df_clean['created_at_dt'].max()
DATE_RANGE = f"{DATE_FROM:%b %d, %Y} – {DATE_TO:%b %d, %Y}"

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🎧 **VIREO AUDIO**")
    st.caption("Crystal Sound. Better Support.")
    
    st.markdown("---")
    nav = st.radio(
        "Navigation",
        [
            "Overview",
            "Weekly Digest",
            "Agent Scorecards",
            "Financial Leakages",
            "Product Spotlights"
        ]
    )

    st.markdown("---")
    st.markdown("#### 🔍 Filter Controls")
    
    # Filter: Channel
    channels = ["All"] + sorted(df_clean['channel'].unique().tolist())
    sel_channel = st.selectbox("Channel", channels)
    
    # Filter: Product Family
    families = ["All"] + sorted(df_clean['family'].dropna().unique().tolist())
    sel_family = st.selectbox("Product Family", families)

    # Filter: Team
    teams = ["All"] + sorted(df_clean['assigned_team'].dropna().unique().tolist())
    sel_team = st.selectbox("Team", teams)

    # Apply filters
    filtered_df = df_clean.copy()
    if sel_channel != "All":
        filtered_df = filtered_df[filtered_df['channel'] == sel_channel]
    if sel_family != "All":
        filtered_df = filtered_df[filtered_df['family'] == sel_family]
    if sel_team != "All":
        filtered_df = filtered_df[filtered_df['assigned_team'] == sel_team]

    st.caption("Filters apply to the Overview tiles and charts. The leaderboard, digest and audits "
               "always use all tickets, so they match the report.")

    st.markdown(f"""
    <div class="sidebar-summary">
        <b>📊 Dataset Summary</b><br>
        • Unique tickets (migration duplicates removed): {len(df_clean):,}<br>
        • Dates: {DATE_RANGE}<br>
        • Agents: {df_clean['agent_id'].nunique()}<br>
        • Products: {df_clean['product_sku'].nunique()} SKUs<br>
        • Latest complete week: {LATEST_WEEK}
    </div>
    """, unsafe_allow_html=True)

# ----------------- TOP HEADER BAR -----------------
st.markdown(f"""
<div class="header-container">
    <div>
        <div style="font-size: 22px; font-weight: 700; color: #ffffff;">Support Analytics Dashboard</div>
        <div class="header-subtitle">Complaint themes, leaderboard and business case from the helpdesk export</div>
    </div>
    <div style="background: #1e293b; border: 1px solid #334155; padding: 6px 14px; border-radius: 6px; font-size: 13px; color: #94a3b8;">
        📅 {DATE_RANGE}
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- VIEW 1: OVERVIEW -----------------
if nav == "Overview":
    # 6 Top KPI Cards matching mockup
    kpis = get_overall_operational_kpis(filtered_df)
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🎧 Total Support Tickets</div>
            <div class="kpi-val">{kpis['total_tickets']:,}</div>
            <div class="kpi-sub-neutral">in the export (a sample of Vireo's volume)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💬 Avg. Tickets / Week</div>
            <div class="kpi-val">{kpis['avg_tickets_per_week']:.1f}</div>
            <div class="kpi-sub-neutral">export; Vireo says ~650 in reality</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">⏱️ SLA Breaches</div>
            <div class="kpi-val">{kpis['total_breaches']:,}</div>
            <div class="kpi-sub">{kpis['breach_rate']*100:.2f}% <span class="kpi-sub-neutral">breach rate</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🪙 SLA Credits (Auto)</div>
            <div class="kpi-val">₹{kpis['total_sla_credits_inr']:,.0f}</div>
            <div class="kpi-sub-neutral">₹350 per breach</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🔄 Total Transfers</div>
            <div class="kpi-val">{kpis['total_transfers']:,}</div>
            <div class="kpi-sub-neutral">₹{kpis['total_transfer_cost_inr']:,.0f} @ ₹305/ea (§4); new helpdesk only (§9)</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">👥 Same-issue repeats (30d, §10)</div>
            <div class="kpi-val">{kpis['total_repeat_contacts']:,}</div>
            <div class="kpi-sub">{kpis['repeat_rate']*100:.1f}% <span class="kpi-sub-neutral">of tickets</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Main Grid (Row 1: Weekly Volume & Category Donut & Category Trends)
    g1, g2 = st.columns([1.6, 1.4])

    with g1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Weekly Ticket Volume (18 Months)</div>', unsafe_allow_html=True)
        weekly_df = compute_weekly_time_series(filtered_df)
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=weekly_df['year_week'],
            y=weekly_df['tickets'],
            mode='lines',
            name='Weekly Tickets',
            line=dict(color='#38bdf8', width=2)
        ))
        fig_vol.add_trace(go.Scatter(
            x=weekly_df['year_week'],
            y=weekly_df['12w_moving_avg'],
            mode='lines',
            name='12-Week Moving Avg',
            line=dict(color='#ef4444', width=1.5, dash='dash')
        ))
        fig_vol.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            xaxis=dict(showgrid=True, gridcolor='#1e293b', tickangle=-45, nticks=12),
            yaxis=dict(showgrid=True, gridcolor='#1e293b'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top complaint themes (read from the message, not the bot tag)</div>', unsafe_allow_html=True)
        theme_counts = filtered_df['theme'].value_counts().head(10).sort_values().reset_index()
        theme_counts.columns = ['Theme', 'Tickets']
        theme_counts['Share'] = theme_counts['Tickets'] / max(len(filtered_df), 1) * 100

        fig_cat = px.bar(theme_counts, x='Tickets', y='Theme', orientation='h',
                         hover_data={'Share': ':.1f'}, color_discrete_sequence=['#38bdf8'])
        fig_cat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            yaxis_title=None
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.caption("The intake bot's category disagreed with the message on 26 of 100 checked tickets, so it is not used.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Top Products, SLA by Channel, and Top Tier 1 Agents
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Products by Tickets</div>', unsafe_allow_html=True)
        prod_tbl = filtered_df['product_name'].value_counts().head(5).reset_index()
        prod_tbl.columns = ['Product', 'Tickets']
        prod_tbl['Share'] = (prod_tbl['Tickets'] / len(filtered_df) * 100.0).map('{:.1f}%'.format)
        st.dataframe(prod_tbl, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">SLA Breach Rate by Channel</div>', unsafe_allow_html=True)
        ch_tbl = compute_channel_sla_metrics(filtered_df)[['channel', 'total_tickets', 'breach_rate', 'sla_credits_inr']]
        ch_tbl.columns = ['Channel', 'Tickets', 'Breach Rate', 'SLA Credits']
        ch_tbl['Breach Rate'] = (ch_tbl['Breach Rate'] * 100).map('{:.2f}%'.format)
        ch_tbl['SLA Credits'] = ch_tbl['SLA Credits'].map('₹{:,.0f}'.format)
        st.dataframe(ch_tbl, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Most tickets closed, per team (4-week avg)</div>', unsafe_allow_html=True)
        t1 = build_tier_1_scorecard(df_clean)
        leaders = t1[t1['rank_in_team'] == 1][['team', 'name', 'closed_per_week_4wk', 'fcr_pct']].copy()
        leaders.columns = ['Team', 'Agent', 'Closed / wk', 'FCR %']
        st.dataframe(leaders.round(1), use_container_width=True, hide_index=True)
        st.caption("Ranked within team only: queues differ by team and shift. Tier 2 is measured in days (Agent Scorecards).")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: product watch (latest complete week) and the business goal
    b1, b2 = st.columns([1.2, 1.8])

    with b1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔎 Product Watch (latest complete week)</div>', unsafe_allow_html=True)
        week_digest = generate_weekly_digest(df_clean, LATEST_WEEK)
        pw = week_digest["products"]
        flagged = pw[pw["alert"]]
        headline = (f"{len(flagged)} product(s) with unusual hardware faults" if len(flagged)
                    else "No product has unusual hardware faults this week")
        st.markdown(f"**{headline}** · {week_digest['lots_tested']:,} manufacturing lots checked, "
                    f"{len(week_digest['lot_alerts'])} flagged after correcting for multiple testing.")
        st.dataframe(pw[["product", "hardware_tickets", "expected", "top_fault", "alert"]].round(1),
                     use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💰 The goal and what it is worth</div>', unsafe_allow_html=True)
        audit_home = load_audit()
        rep = audit_home['repeat']
        st.metric("Same-issue repeat contacts (policy §10)", f"{rep['current_repeat_rate']:.1%}",
                  f"target {rep['target_repeat_rate']:.1%}", delta_color="off")
        st.markdown(f"Halving status chasers (refund / delivery / pickup / repair) with proactive updates saves "
                    f"**₹{rep['quarterly_savings_inr']:,.0f} a quarter** at Vireo's ~650 tickets a week. "
                    f"Recoverable policy breaches found: **₹{audit_home['total_direct_leakage_annual_inr'] / 1e5:.1f} lakh a year** "
                    f"(see Financial Leakages).")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- VIEW 2: WEEKLY DIGEST -----------------
elif nav == "Weekly Digest":
    st.markdown("### 📋 Automated Weekly Complaint Digest")
    st.caption("Designed for Priya Raman (Head of CX) to replace manual reading of 650 weekly tickets.")
    
    # Default to the latest COMPLETE week: the export ends mid-week, and a partial week reads as a volume drop.
    weeks = sorted(df_clean['year_week'].unique(), reverse=True)
    selected_week = st.selectbox("Select Target Week", weeks, index=weeks.index(LATEST_WEEK),
                                 format_func=lambda w: w if w <= LATEST_WEEK else f"{w} (incomplete)")

    digest = generate_weekly_digest(df_clean, target_week=selected_week)
    md_report = format_digest_markdown(digest)
    
    st.markdown(md_report)
    
    st.download_button(
        label="📥 Download Weekly Digest (Markdown)",
        data=md_report,
        file_name=f"Vireo_Digest_{selected_week}.md",
        mime="text/markdown"
    )

# ----------------- VIEW 3: AGENT SCORECARDS -----------------
elif nav == "Agent Scorecards":
    st.markdown("### 🏆 Agent leaderboard: tickets closed per week")
    lb_weeks = sorted(df_clean['year_week'].unique(), reverse=True)
    lb_week = st.selectbox("Week", lb_weeks, index=lb_weeks.index(LATEST_WEEK))
    lb = generate_weekly_agent_leaderboards(df_clean, lb_week)
    t1_scorecard, t2_scorecard = lb['tier_1_scorecard'], lb['tier_2_scorecard']

    tab1, tab2, tab3 = st.tabs([f"Tier 1 by team ({len(t1_scorecard)} agents)",
                                f"Escalations & Warranty ({len(t2_scorecard)} agents)", "How to read this"])
    with tab1:
        for team, g in t1_scorecard.groupby('team'):
            st.markdown(f"#### {team}")
            st.dataframe(
                g[['rank_in_team', 'name', 'site', 'shift', 'sole_agent_on_shift', 'closed_this_week',
                   'closed_per_week_4wk', 'vs_team_avg', 'fcr_pct', 'csat', 'csat_responses', 'vague_notes_pct']].round(2),
                use_container_width=True, hide_index=True)
    with tab2:
        st.info("Policy §6: Tier 2 cases are multi-touch and measured on resolution in days, never compared with Tier 1 on volume.")
        st.dataframe(
            t2_scorecard[['name', 'site', 'shift', 'cases_resolved_this_week', 'cases_resolved_4wk',
                          'median_resolution_days', 'replacements_approved_90d', 'fcr_pct', 'csat']].round(2),
            use_container_width=True, hide_index=True)
    with tab3:
        md = format_leaderboard_markdown(lb['leaderboard'])
        st.markdown(md.split("\n## How to read this")[1].split("\n## ")[0])

# ----------------- VIEW 4: FINANCIAL LEAKAGES -----------------
elif nav == "Financial Leakages":
    st.markdown("### 💸 Financial Leakage & Policy Breach Audit")
    st.caption("Detailed audit for Finance Controller Arjun Mehta: Recoverable cash leakages and policy violations.")

    audit = load_audit()
    st.markdown(format_business_case_markdown(audit))

    st.markdown("---")
    dd = audit['double_dip']
    st.markdown(f"#### Orders with BOTH a refund and a replacement (policy §5): {dd['double_dip_orders_count']}")
    st.dataframe(dd['double_dip_orders_df'], use_container_width=True, hide_index=True)
    st.markdown(f"#### Refunds in notes with no recorded amount: {audit['unrecorded_refunds']['count']}")
    st.dataframe(audit['unrecorded_refunds']['unrecorded_df'][['ticket_id', 'created_at', 'team', 'agent_id', 'reason',
                                                                'order_value_inr', 'agent_notes']],
                 use_container_width=True, hide_index=True)

# ----------------- VIEW 5: PRODUCT SPOTLIGHTS -----------------
elif nav == "Product Spotlights":
    st.markdown("### 🔍 Manufacturing Lot & Product Defect Spotlights")
    lots = detect_manufacturing_lot_anomalies(df_clean)

    st.markdown("#### Manufacturing lots: hardware-fault tickets per unit sold")
    st.caption(f"{lots.attrs['lots_tested']:,} lots tested. A lot is flagged only if its excess survives a "
               f"Bonferroni correction (p < {lots.attrs['p_threshold']:.1e}) and it sold ≥ 20 units; "
               f"an uncorrected test would flag ~13 lots by chance alone.")
    st.metric("Flagged lots", int(lots["is_anomaly"].sum()))
    st.dataframe(
        lots.head(15)[['lot_code', 'product_name', 'units', 'hw_tickets', 'expected', 'ratio', 'p_value', 'is_anomaly']].round(4),
        use_container_width=True,
        hide_index=True
    )
