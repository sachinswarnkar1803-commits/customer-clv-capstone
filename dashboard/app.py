"""Interactive Streamlit Command Dashboard for BDS-34 Capstone Project.

Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments.
Author: BDS-34 Capstone Team (T.Y. B.Sc. Data Science - Semester V)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Customer Intelligence Platform | BDS-34",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (Dark/Glassmorphic Modern UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 6px;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #38bdf8;
        margin-top: 4px;
        font-weight: 500;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-champions { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-risk { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-growing { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .badge-loyal { background: rgba(139, 92, 246, 0.2); color: #a78bfa; border: 1px solid #8b5cf6; }
    .badge-new { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
    .badge-dormant { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid #64748b; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Data loading helpers with Streamlit caching
@st.cache_data
def load_dashboard_data():
    """Load analytical tables and reports."""
    base_dir = Path(__file__).resolve().parent.parent

    # Processed datasets
    clv_path = base_dir / "data/processed/customer_clv_segments.parquet"
    if not clv_path.exists():
        clv_path = base_dir / "data/processed/customer_clv_segments.csv"

    clean_tx_path = base_dir / "data/processed/clean_transactions.parquet"
    if not clean_tx_path.exists():
        clean_tx_path = base_dir / "data/processed/clean_transactions.csv"

    ret_matrix_path = base_dir / "data/processed/cohort_retention_matrix.csv"
    cohort_summary_path = base_dir / "data/processed/cohort_summary.csv"

    # Reports
    eval_report_path = base_dir / "reports/model_evaluation_report.json"
    drift_report_path = base_dir / "reports/monitoring_report.json"
    quality_report_path = base_dir / "reports/data_quality_report.md"

    # Read data
    df_customers = pd.read_parquet(clv_path) if str(clv_path).endswith(".parquet") else pd.read_csv(clv_path)
    df_clean_tx = pd.read_parquet(clean_tx_path) if str(clean_tx_path).endswith(".parquet") else pd.read_csv(clean_tx_path)

    df_retention = pd.read_csv(ret_matrix_path, index_col=0) if ret_matrix_path.exists() else None
    df_cohort_summary = pd.read_csv(cohort_summary_path) if cohort_summary_path.exists() else None

    eval_data = {}
    if eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

    drift_data = {}
    if drift_report_path.exists():
        with open(drift_report_path, "r", encoding="utf-8") as f:
            drift_data = json.load(f)

    return df_customers, df_clean_tx, df_retention, df_cohort_summary, eval_data, drift_data


try:
    df_customers, df_clean_tx, df_retention, df_cohort_summary, eval_data, drift_data = load_dashboard_data()
except Exception as e:
    st.error(f"Error loading analytical data: {e}. Please ensure `python -m src.pipeline` has been executed.")
    st.stop()


# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=600&q=80", use_container_width=True)
st.sidebar.title("💎 Customer Intelligence")
st.sidebar.caption("Capstone Project BDS-34 • Sem V")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Overview",
        "👥 Cohort & Retention Dynamics",
        "🔍 Customer 360 Explorer",
        "🎯 Action Segments",
        "⚡ Next-Best-Action Engine",
        "🎲 Campaign Simulator",
        "🛡️ Model Monitoring & Health",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Academic Provenance**\n"
    "- Dataset: UCI Online Retail II\n"
    "- Models: BG/NBD, Gamma-Gamma, Weibull\n"
    "- Horizon: 90 Days (Configurable)\n"
    "- Uncertainty: 80% Bootstrap CI"
)

# -----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
if nav_choice == "📊 Executive Overview":
    st.markdown("""
    <div class="main-header">
        <h1>Executive Customer Intelligence Overview</h1>
        <p>Probabilistic Customer Lifetime Value, Inactivity Hazard, and Prescriptive Next-Best-Action Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Bar
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Customers</div>
            <div class="metric-value">{len(df_customers):,}</div>
            <div class="metric-subtitle">Training Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        tot_rev = df_clean_tx["revenue"].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Clean Spend</div>
            <div class="metric-value">£{tot_rev/1e6:.2f}M</div>
            <div class="metric-subtitle">779,421 verified txns</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        avg_clv = df_customers["clv_expected_90d"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg 90-Day CLV</div>
            <div class="metric-value">£{avg_clv:.2f}</div>
            <div class="metric-subtitle">Discounted Expected</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        champions_cnt = (df_customers["action_segment"] == "Champions / High Value Active").sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Champions (VIP)</div>
            <div class="metric-value">{champions_cnt:,}</div>
            <div class="metric-subtitle">{(champions_cnt/len(df_customers))*100:.1f}% of base</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        risk_cnt = (df_customers["nba_risk_level"] == "HIGH").sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High Churn Hazard</div>
            <div class="metric-value">{risk_cnt:,}</div>
            <div class="metric-subtitle">Immediate intervention</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    col_left, col_right = st.columns([6, 4])

    with col_left:
        st.subheader("🎯 Customer Distribution by Action Segment")
        seg_counts = df_customers["action_segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig_seg = px.bar(
            seg_counts,
            x="Count",
            y="Segment",
            orientation="h",
            color="Segment",
            color_discrete_sequence=px.colors.qualitative.Prism,
            text="Count",
        )
        fig_seg.update_layout(
            template="plotly_dark",
            showlegend=False,
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis={'categoryorder': 'total ascending'},
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_right:
        st.subheader("⚡ Recommended Next-Best-Actions")
        action_counts = df_customers["nba_recommended_action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Customers"]

        fig_pie = px.pie(
            action_counts,
            names="Action",
            values="Customers",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_pie.update_layout(
            template="plotly_dark",
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Value vs Risk Quadrant
    st.subheader("🌐 Value-Risk Strategic Map")
    st.caption("Visualizing customer positioning across 90-Day Expected CLV and Inactivity Hazard")

    sample_viz = df_customers.sample(n=min(1200, len(df_customers)), random_state=42)
    fig_scatter = px.scatter(
        sample_viz,
        x="inactivity_prob_90d",
        y="clv_expected_90d",
        color="action_segment",
        size="frequency",
        hover_data=["customer_id", "total_revenue", "p_alive", "nba_recommended_action"],
        labels={"inactivity_prob_90d": "Inactivity Probability (90 Days)", "clv_expected_90d": "Expected 90-Day CLV (£)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_scatter.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, min(sample_viz['clv_expected_90d'].quantile(0.99) * 1.2, 5000)]),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 2: COHORT & RETENTION DYNAMICS
# -----------------------------------------------------------------------------
elif nav_choice == "👥 Cohort & Retention Dynamics":
    st.markdown("""
    <div class="main-header">
        <h1>Acquisition Cohort & Retention Dynamics</h1>
        <p>Tracking customer longevity, repeat cadence, and cumulative revenue decay across monthly acquisition cohorts</p>
    </div>
    """, unsafe_allow_html=True)

    if df_retention is not None:
        tab1, tab2, tab3 = st.tabs(["🔥 Retention Rate Matrix", "📈 Retention Decay Curves", "💰 Cohort Cumulative Revenue"])

        with tab1:
            st.subheader("Monthly Retention Matrix (%)")
            st.caption("Percentage of acquired cohort customers active in subsequent months after acquisition (Month 0 = 100%)")

            # Format to percentage display
            ret_pct = (df_retention * 100.0).round(1)
            # Limit to first 12 periods for neat display
            cols_to_show = [c for c in ret_pct.columns[:13]]

            fig_hm = px.imshow(
                ret_pct[cols_to_show],
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Purples",
                labels=dict(x="Months Since First Acquisition", y="Cohort Acquisition Month", color="Retention (%)"),
            )
            fig_hm.update_layout(
                template="plotly_dark",
                height=550,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        with tab2:
            st.subheader("Cohort Retention Decay Curves")
            st.caption("Comparing customer drop-off curves across acquisition cohorts")

            # Plot top 6 largest cohorts
            cohort_order = df_retention.index[:8]
            fig_line = go.Figure()
            for ch in cohort_order:
                curve = df_retention.loc[ch].dropna() * 100.0
                fig_line.add_trace(go.Scatter(
                    x=curve.index.astype(int),
                    y=curve.values,
                    mode="lines+markers",
                    name=ch,
                ))

            fig_line.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Months Elapsed Since Acquisition",
                yaxis_title="Retention Rate (%)",
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with tab3:
            st.subheader("Cumulative Revenue per Cohort")
            if df_cohort_summary is not None:
                fig_cum = px.line(
                    df_cohort_summary[df_cohort_summary["cohort_period"] <= 12],
                    x="cohort_period",
                    y="cumulative_clv",
                    color="cohort_month_str",
                    labels={"cohort_period": "Cohort Period (Months)", "cumulative_clv": "Cumulative Spend / Customer (£)", "cohort_month_str": "Cohort"},
                )
                fig_cum.update_layout(
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig_cum, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 3: CUSTOMER 360 EXPLORER
# -----------------------------------------------------------------------------
elif nav_choice == "🔍 Customer 360 Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>Individual Customer 360 Explorer</h1>
        <p>Audit individual customer transaction trajectories, probabilistic life expectancy, and Next-Best-Action prescriptive briefs</p>
    </div>
    """, unsafe_allow_html=True)

    # Search and Filter Controls
    fcol1, fcol2, fcol3 = st.columns([3, 3, 3])
    with fcol1:
        seg_filter = st.selectbox(
            "Filter by Action Segment",
            ["All"] + sorted(df_customers["action_segment"].unique().tolist()),
        )
    with fcol2:
        country_filter = st.selectbox(
            "Filter by Country",
            ["All"] + sorted(df_customers["country"].dropna().unique().tolist()[:20]),
        )
    with fcol3:
        search_cust = st.text_input("Search Customer ID", placeholder="e.g. 12346")

    # Filtered dataframe
    df_filtered = df_customers.copy()
    if seg_filter != "All":
        df_filtered = df_filtered[df_filtered["action_segment"] == seg_filter]
    if country_filter != "All":
        df_filtered = df_filtered[df_filtered["country"] == country_filter]
    if search_cust.strip():
        df_filtered = df_filtered[df_filtered["customer_id"].astype(str).str.contains(search_cust.strip())]

    st.write(f"Showing **{len(df_filtered):,}** matching customers.")

    if len(df_filtered) == 0:
        st.warning("No customers match the current criteria.")
    else:
        # Select customer
        selected_cust_id = st.selectbox(
            "Select Customer to Inspect:",
            df_filtered["customer_id"].tolist()[:50],
        )

        cust_row = df_customers[df_customers["customer_id"] == selected_cust_id].iloc[0]

        # Customer Header Profile
        st.markdown("---")
        pcol1, pcol2, pcol3, pcol4 = st.columns([3, 2, 2, 3])
        with pcol1:
            st.markdown(f"### Customer ID: `{cust_row['customer_id']}`")
            st.markdown(f"**Country**: {cust_row['country']} | **Cohort**: {cust_row['cohort_month']}")
            st.markdown(f"**Action Segment**: `{cust_row['action_segment']}`")
        with pcol2:
            st.metric("P(Alive)", f"{cust_row['p_alive']*100:.1f}%")
            st.metric("Inactivity Hazard (90d)", f"{cust_row['inactivity_prob_90d']*100:.1f}%")
        with pcol3:
            st.metric("Expected 90d CLV", f"£{cust_row['clv_expected_90d']:.2f}")
            st.metric("80% Value Interval", f"£{cust_row['clv_lower_80pct']:.1f} - £{cust_row['clv_upper_80pct']:.1f}")
        with pcol4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #6366f1;">
                <div class="metric-title">Next-Best-Action Brief</div>
                <div style="font-size: 18px; font-weight: 800; color: #a5b4fc; margin-top: 4px;">{cust_row['nba_recommended_action']}</div>
                <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">Channel: <b>{cust_row['nba_channel'].upper()}</b></div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">{cust_row['nba_offer']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Historical transactions of this customer
        st.subheader("📜 Historical Transaction Basket Log")
        cust_tx = df_clean_tx[df_clean_tx["customer_id"] == str(selected_cust_id)].sort_values(
            "transaction_date", ascending=False
        )

        st.dataframe(
            cust_tx[["invoice_id", "transaction_date", "stock_code", "description", "quantity", "unit_price", "revenue"]].head(25),
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# PAGE 4: ACTION SEGMENTS
# -----------------------------------------------------------------------------
elif nav_choice == "🎯 Action Segments":
    st.markdown("""
    <div class="main-header">
        <h1>Action-Oriented Customer Segments</h1>
        <p>Operational customer clustering informed by statistical longevity, repeat velocity, and expected customer lifetime value</p>
    </div>
    """, unsafe_allow_html=True)

    # Segment Aggregation Table
    seg_summary = df_customers.groupby("action_segment").agg(
        customers=("customer_id", "count"),
        avg_revenue=("total_revenue", "mean"),
        total_segment_rev=("total_revenue", "sum"),
        avg_clv_90d=("clv_expected_90d", "mean"),
        avg_p_alive=("p_alive", "mean"),
        avg_inactivity_hazard=("inactivity_prob_90d", "mean"),
    ).reset_index()

    seg_summary["pct_customers"] = (seg_summary["customers"] / len(df_customers) * 100).round(1)
    seg_summary["avg_revenue"] = seg_summary["avg_revenue"].round(2)
    seg_summary["avg_clv_90d"] = seg_summary["avg_clv_90d"].round(2)
    seg_summary["avg_p_alive"] = (seg_summary["avg_p_alive"] * 100).round(1)
    seg_summary["avg_inactivity_hazard"] = (seg_summary["avg_inactivity_hazard"] * 100).round(1)

    st.dataframe(
        seg_summary.rename(columns={
            "action_segment": "Segment",
            "customers": "Customer Count",
            "pct_customers": "% of Base",
            "avg_revenue": "Avg Past Spend (£)",
            "total_segment_rev": "Total Revenue (£)",
            "avg_clv_90d": "Expected 90d CLV (£)",
            "avg_p_alive": "Avg P(Alive) %",
            "avg_inactivity_hazard": "Avg Inactivity Risk %",
        }),
        use_container_width=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Distribution Plots
    sc1, sc2 = st.columns(2)
    with sc1:
        st.subheader("Revenue Contribution by Segment")
        fig_rev = px.pie(
            seg_summary,
            names="action_segment",
            values="total_segment_rev",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.45,
        )
        fig_rev.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_rev, use_container_width=True)

    with sc2:
        st.subheader("Expected CLV Boxplot by Segment")
        fig_box = px.box(
            df_customers,
            x="action_segment",
            y="clv_expected_90d",
            color="action_segment",
            points=False,
        )
        fig_box.update_layout(
            template="plotly_dark",
            height=380,
            showlegend=False,
            yaxis=dict(range=[0, df_customers['clv_expected_90d'].quantile(0.98)]),
            xaxis_title="",
            yaxis_title="Expected 90d CLV (£)",
        )
        st.plotly_chart(fig_box, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 5: NEXT-BEST-ACTION ENGINE
# -----------------------------------------------------------------------------
elif nav_choice == "⚡ Next-Best-Action Engine":
    st.markdown("""
    <div class="main-header">
        <h1>Next-Best-Action (NBA) Prescriptive Engine</h1>
        <p>Operational CRM orchestration: Translating lifetime valuations and churn risk into prioritized contact strategies</p>
    </div>
    """, unsafe_allow_html=True)

    # Action Summary Tiles
    acts = df_customers["nba_recommended_action"].value_counts()
    ncol1, ncol2, ncol3, ncol4 = st.columns(4)
    with ncol1:
        st.metric("Total Retention Touches", f"{acts.get('RETENTION', 0) + acts.get('WIN_BACK', 0):,}")
    with ncol2:
        st.metric("Upsell & Cross-Sell Opportunities", f"{acts.get('UPSELL', 0) + acts.get('CROSS_SELL', 0):,}")
    with ncol3:
        st.metric("Loyalty & VIP Touches", f"{acts.get('LOYALTY_REWARD', 0):,}")
    with ncol4:
        st.metric("Suppressed / No Action (Budget Saved)", f"{acts.get('NO_ACTION', 0):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # High Priority Action Queue
    st.subheader("🚨 Priority Action Queue (Top Ranked Contacts)")
    st.caption("Sorted by highest expected incremental yield and urgency")

    action_filter = st.selectbox(
        "Filter Action Queue",
        ["All High Priority"] + sorted(df_customers["nba_recommended_action"].unique().tolist()),
    )

    queue_df = df_customers.copy()
    if action_filter == "All High Priority":
        queue_df = queue_df[queue_df["nba_priority"] == "HIGH"]
    else:
        queue_df = queue_df[queue_df["nba_recommended_action"] == action_filter]

    queue_df = queue_df.sort_values("nba_expected_incremental_value", ascending=False)

    st.dataframe(
        queue_df[[
            "customer_id", "action_segment", "nba_recommended_action", "nba_priority",
            "nba_channel", "clv_expected_90d", "nba_expected_incremental_value",
            "nba_estimated_cost", "nba_offer", "nba_reason"
        ]].head(50).rename(columns={
            "customer_id": "Customer ID",
            "action_segment": "Segment",
            "nba_recommended_action": "Action",
            "nba_priority": "Priority",
            "nba_channel": "Channel",
            "clv_expected_90d": "CLV 90d (£)",
            "nba_expected_incremental_value": "Expected Yield (£)",
            "nba_estimated_cost": "Contact Cost (£)",
            "nba_offer": "Incentive / Offer",
            "nba_reason": "Decision Rationale",
        }),
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# PAGE 6: CAMPAIGN SCENARIO SIMULATOR
# -----------------------------------------------------------------------------
elif nav_choice == "🎲 Campaign Simulator":
    st.markdown("""
    <div class="main-header">
        <h1>Interactive Marketing Campaign Simulator</h1>
        <p>What-if scenario planning: Simulate marketing budget allocations, response rates, discount mechanics, and expected ROI</p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ **ACADEMIC INTEGRITY NOTICE**: All campaign response probabilities and uplifts simulated on this page are generated from a **synthetic simulation layer**. The UCI Online Retail II dataset is purely transactional; simulation data is segregated to demonstrate decision science workflows.")

    sim_col1, sim_col2 = st.columns([4, 6])

    with sim_col1:
        st.subheader("⚙️ Scenario Parameters")
        target_seg = st.selectbox(
            "Target Segment",
            df_customers["action_segment"].unique().tolist(),
            index=0,
        )
        avail = int((df_customers["action_segment"] == target_seg).sum())
        st.info(f"Available population in `{target_seg}`: **{avail:,} customers**")

        camp_action = st.selectbox("Campaign Objective / Action", ["RETENTION", "UPSELL", "LOYALTY_REWARD", "REACTIVATION", "WIN_BACK"])
        channel = st.selectbox("Contact Channel", ["email", "sms", "direct_mail", "push"])
        camp_size = st.slider("Targeted Customers Count", min_value=10, max_value=max(10, avail), value=min(500, avail), step=10)
        resp_rate = st.slider("Assumed Response Rate (%)", min_value=1.0, max_value=40.0, value=15.0, step=0.5) / 100.0
        inc_aov = st.slider("Expected Incremental Spend / Order (£)", min_value=10.0, max_value=200.0, value=65.0, step=5.0)
        disc_pct = st.slider("Voucher / Discount Level (%)", min_value=0.0, max_value=30.0, value=12.0, step=1.0) / 100.0
        creative_cost = st.number_input("Fixed Creative / Setup Cost (£)", min_value=0.0, value=150.0, step=25.0)

    with sim_col2:
        st.subheader("📈 Projected Financial Return")
        from src.simulation.campaign_simulator import ScenarioSimulator
        simulator = ScenarioSimulator()
        sim_res = simulator.simulate_campaign(
            segment_df=df_customers,
            target_segment=target_seg,
            action=camp_action,
            campaign_size=camp_size,
            channel=channel,
            base_response_rate=resp_rate,
            avg_incremental_aov=inc_aov,
            discount_pct=disc_pct,
            fixed_creative_cost=creative_cost,
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Expected Responders", f"{sim_res['expected_responders']:,}", f"80% Range: {sim_res['responders_80pct_range'][0]}-{sim_res['responders_80pct_range'][1]}")
        with r2:
            st.metric("Total Campaign Cost", f"£{sim_res['total_campaign_cost_gbp']:,.2f}")
        with r3:
            roi_color = "normal" if sim_res['roi_percent'] > 0 else "inverse"
            st.metric("Projected ROI", f"{sim_res['roi_percent']:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            "Expected Net Profit (£)",
            f"£{sim_res['expected_net_value_gbp']:,.2f}",
            f"80% CI: [£{sim_res['net_value_80pct_range'][0]:,.2f}, £{sim_res['net_value_80pct_range'][1]:,.2f}]",
        )

        # Cost Breakdown Waterfall
        breakdown_df = pd.DataFrame({
            "Cost Component": ["Fixed Creative", "Channel Outreach", "Discount Voucher", "Expected Gross Revenue"],
            "Amount (£)": [
                sim_res["fixed_cost_gbp"],
                sim_res["channel_cost_gbp"],
                sim_res["discount_cost_gbp"],
                sim_res["gross_incremental_revenue_gbp"],
            ],
            "Type": ["Cost", "Cost", "Cost", "Revenue"],
        })
        fig_bar = px.bar(
            breakdown_df,
            x="Cost Component",
            y="Amount (£)",
            color="Type",
            text="Amount (£)",
            color_discrete_map={"Cost": "#ef4444", "Revenue": "#10b981"},
        )
        fig_bar.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 7: MODEL MONITORING & HEALTH
# -----------------------------------------------------------------------------
elif nav_choice == "🛡️ Model Monitoring & Health":
    st.markdown("""
    <div class="main-header">
        <h1>Model Evaluation, Calibration & Production Drift Monitoring</h1>
        <p>Quantitative holdout benchmarking, interval coverage verification, calibration reliability, and Population Stability Index (PSI)</p>
    </div>
    """, unsafe_allow_html=True)

    mtab1, mtab2, mtab3 = st.tabs(["📊 Holdout Benchmarks", "🎯 Calibration & Interval Coverage", "📡 Drift Monitoring (PSI)"])

    with mtab1:
        st.subheader("Holdout Revenue Error: Model Benchmarking")
        st.caption("Comparing Probabilistic CLV vs Baseline A (Historical Extrapolation) vs Baseline B (RFM) on ground truth holdout spend")

        if eval_data and "benchmark_results" in eval_data:
            bench_df = pd.DataFrame(eval_data["benchmark_results"])
            st.dataframe(bench_df, use_container_width=True)

            # Bar plot of Spearman Rank Correlation
            fig_corr = px.bar(
                bench_df,
                x="Model",
                y="Spearman Rank Corr",
                color="Model",
                text="Spearman Rank Corr",
                title="Spearman Rank Correlation with Future Spend (Higher is Better)",
            )
            fig_corr.update_layout(template="plotly_dark", height=320, showlegend=False)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Evaluation benchmark data loading...")

    with mtab2:
        c_left, c_right = st.columns(2)
        with c_left:
            st.subheader("80% CLV Prediction Interval Coverage")
            if eval_data and "interval_coverage" in eval_data:
                cov = eval_data["interval_coverage"]
                st.metric("Empirical Coverage Rate", f"{cov['empirical_coverage_pct']}%", f"Nominal Target: {cov['nominal_target']*100:.0f}%")
                st.info(f"**Audit Status**: {cov['status']}")
                st.write(f"- Outcomes Below Lower Bound: **{cov['below_lower_bound_pct']}%**")
                st.write(f"- Outcomes Above Upper Bound: **{cov['above_upper_bound_pct']}%**")

        with c_right:
            st.subheader("Inactivity Calibration Reliability")
            if eval_data and "calibration" in eval_data:
                calib = eval_data["calibration"]
                st.metric("Brier Score", f"{calib['brier_score']:.4f}", calib["brier_interpretation"])
                calib_bins_df = pd.DataFrame(calib["calibration_bins"])
                st.dataframe(calib_bins_df, use_container_width=True)

        st.subheader("Sparse-History Sensitivity Analysis")
        if eval_data and "sparse_history_sensitivity" in eval_data:
            st.dataframe(pd.DataFrame(eval_data["sparse_history_sensitivity"]), use_container_width=True)

    with mtab3:
        st.subheader("Production Drift Monitoring (Population Stability Index)")
        if drift_data:
            overall = drift_data.get("overall_health", {})
            st.metric("Maximum PSI Across Features", f"{overall.get('max_psi', 0.0):.4f}", f"Status: {overall.get('overall_status', 'HEALTHY')}")

            # PSI Table
            psi_rows = []
            for k, v in drift_data.items():
                if isinstance(v, dict) and "psi_value" in v:
                    psi_rows.append({
                        "Metric / Feature": k.replace("psi_", ""),
                        "PSI Value": v["psi_value"],
                        "Status": v["status"],
                    })
            if psi_rows:
                st.dataframe(pd.DataFrame(psi_rows), use_container_width=True)

            if "segment_population_drift" in drift_data:
                st.subheader("Segment Population Shifts Across Time")
                st.dataframe(pd.DataFrame(drift_data["segment_population_drift"]), use_container_width=True)
