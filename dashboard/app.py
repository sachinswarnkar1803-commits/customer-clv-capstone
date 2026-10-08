"""Professional interactive Streamlit dashboard for BDS-34."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence | BDS-34",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent.parent
# Make the repository root importable when Streamlit launches this file directly.
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stSidebar"] {border-right: 1px solid #e5e7eb;}
      .hero {padding: 1.5rem 1.75rem; border: 1px solid #e5e7eb; border-radius: 16px; background: linear-gradient(135deg,#f8fafc,#eef2ff); margin-bottom: 1.25rem;}
      .hero h1 {margin:0; font-size:2rem; letter-spacing:-0.03em; color:#111827;}
      .hero p {margin:.45rem 0 0; color:#4b5563; font-size:.95rem;}
      .section-note {color:#6b7280; font-size:.88rem; margin-top:-.4rem; margin-bottom:1rem;}
      .status {padding:.7rem .9rem; border-radius:10px; background:#f8fafc; border:1px solid #e5e7eb; color:#374151;}
      .decision {padding:1rem; border-left:4px solid #4f46e5; background:#f8fafc; border-radius:8px;}
      .small {font-size:.82rem;color:#6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_table(stem: str) -> pd.DataFrame:
    pq = BASE / "data" / "processed" / f"{stem}.parquet"
    csv = BASE / "data" / "processed" / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(stem)


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    data = {}
    data["customers"] = _read_table("customer_clv_segments")
    data["transactions"] = _read_table("clean_transactions")
    for name in ["cohort_retention_matrix", "cohort_summary"]:
        path = BASE / "data" / "processed" / f"{name}.csv"
        data[name] = pd.read_csv(path, index_col=0) if name == "cohort_retention_matrix" and path.exists() else (pd.read_csv(path) if path.exists() else None)

    for name in ["model_evaluation_report.json", "monitoring_report.json", "data_quality_report.md"]:
        path = BASE / "reports" / name
        if path.exists():
            if path.suffix == ".json":
                data[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                data[name] = path.read_text(encoding="utf-8")
        else:
            data[name] = None

    validation_path = BASE / "data" / "validation" / "audit_trail.json"
    data["validation_audit"] = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None
    return data


try:
    d = load_dashboard_data()
except Exception as exc:
    st.markdown('<div class="hero"><h1>Customer Intelligence Platform</h1><p>BDS-34 capstone dashboard</p></div>', unsafe_allow_html=True)
    st.error("Analytical artifacts are not available yet.")
    st.write("Run the pipeline once, then reload this page:")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.caption(f"Details: {exc}")
    st.stop()

customers = d["customers"].copy()
tx = d["transactions"].copy()
retention = d["cohort_retention_matrix"]
cohort_summary = d["cohort_summary"]
evaluation = d["model_evaluation_report.json"] or {}
monitoring = d["monitoring_report.json"] or {}

for col in ["transaction_date", "first_purchase_date", "last_purchase_date"]:
    if col in tx.columns:
        tx[col] = pd.to_datetime(tx[col], errors="coerce")
for col in ["first_purchase_date", "last_purchase_date"]:
    if col in customers.columns:
        customers[col] = pd.to_datetime(customers[col], errors="coerce")

st.sidebar.markdown("## Customer Intelligence")
st.sidebar.caption("BDS-34 | Probabilistic CLV and Next-Best-Action")

pages = [
    "Executive Overview",
    "Cohort Dynamics",
    "Customer Action Studio",
    "Customer 360",
    "Action Segments",
    "Next-Best-Action",
    "Campaign Simulator",
    "Model Evaluation",
]
page = st.sidebar.radio("Workspace", pages, index=0)

st.sidebar.divider()
st.sidebar.markdown("**Model context**")
st.sidebar.caption("Purchase model: BG/NBD")
st.sidebar.caption("Monetary model: Gamma-Gamma")
st.sidebar.caption("Inactivity model: Weibull survival")
st.sidebar.caption("CLV interval: Monte Carlo prediction")
st.sidebar.caption("Campaign data: synthetic simulation")

horizon = 90
clv_col = f"clv_expected_{horizon}d"


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def money(x):
    return f"£{x:,.2f}"


if page == "Executive Overview":
    hero("Executive Customer Intelligence", "A decision workspace for customer value, inactivity risk, cohort behaviour, and prescriptive actions.")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers", f"{len(customers):,}")
    k2.metric("Clean transactions", f"{len(tx):,}")
    k3.metric("90-day expected CLV", money(customers[clv_col].mean()))
    k4.metric("High-risk customers", f"{(customers['inactivity_prob_90d'] >= 0.60).sum():,}")
    k5.metric("Average P(Alive)", f"{customers['p_alive'].mean()*100:.1f}%")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Action segment distribution")
        seg = customers["action_segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        fig = px.bar(seg.sort_values("Customers"), x="Customers", y="Segment", orientation="h")
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Value and inactivity risk")
        sample = customers.sample(min(1800, len(customers)), random_state=42)
        fig = px.scatter(sample, x="inactivity_prob_90d", y=clv_col, color="action_segment", hover_data=["customer_id","p_alive","frequency"], labels={"inactivity_prob_90d":"90-day inactivity probability",clv_col:"Expected 90-day CLV (£)"})
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10), legend_title_text="Segment")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decision summary")
    summary = customers.groupby("nba_recommended_action").agg(Customers=("customer_id","size"), Expected_Incremental_Value=("nba_expected_incremental_value","sum"), Contact_Cost=("nba_estimated_cost","sum")).reset_index().sort_values("Expected_Incremental_Value", ascending=False)
    summary["Net Expected Value"] = summary["Expected_Incremental_Value"] - summary["Contact_Cost"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

elif page == "Cohort Dynamics":
    hero("Cohort and Retention Dynamics", "Explore acquisition cohorts, retention decay, and revenue progression over time.")
    if retention is None:
        st.info("Cohort retention artifacts are unavailable. Run the pipeline first.")
    else:
        tab1, tab2, tab3 = st.tabs(["Retention Matrix", "Retention Curves", "Revenue Progression"])
        with tab1:
            periods = st.slider("Periods to display", 3, min(12, retention.shape[1]), min(12, retention.shape[1]))
            view = (retention.iloc[:, :periods] * 100).round(1)
            fig = px.imshow(view, text_auto=True, aspect="auto", labels={"x":"Months since acquisition","y":"Acquisition cohort","color":"Retention (%)"})
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            selected = st.multiselect("Cohorts", list(retention.index.astype(str)), default=list(retention.index.astype(str))[:6])
            fig = go.Figure()
            for cohort in selected:
                row = retention.loc[cohort].dropna() * 100
                fig.add_trace(go.Scatter(x=list(range(len(row))), y=row, mode="lines+markers", name=str(cohort)))
            fig.update_layout(height=460, xaxis_title="Months since acquisition", yaxis_title="Retention (%)")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            if cohort_summary is not None and "cohort_period" in cohort_summary.columns:
                max_period = st.slider("Maximum cohort period", 1, int(cohort_summary["cohort_period"].max()), min(12, int(cohort_summary["cohort_period"].max())))
                view = cohort_summary[cohort_summary["cohort_period"] <= max_period]
                fig = px.line(view, x="cohort_period", y="cumulative_clv", color="cohort_month_str", labels={"cohort_period":"Cohort month","cumulative_clv":"Cumulative revenue per customer (£)"})
                fig.update_layout(height=460)
                st.plotly_chart(fig, use_container_width=True)

elif page == "Customer Action Studio":
    hero("Customer 360 | Action Workspace", "A CRM-style customer record for reviewing behaviour, model predictions, commercial value, and the next recommended action.")

    # CRM shell styling. The analytical models below are unchanged; this section only
    # changes how a single customer is selected, scored, and presented.
    st.markdown("""
    <style>
      .crm-header {background:#ffffff;border:1px solid #e5e7eb;border-radius:14px;padding:18px 20px;margin-bottom:14px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
      .crm-name {font-size:1.35rem;font-weight:700;color:#111827;margin:0}
      .crm-meta {color:#6b7280;font-size:.86rem;margin-top:4px}
      .crm-chip {display:inline-block;padding:4px 9px;border-radius:999px;background:#f3f4f6;color:#374151;font-size:.76rem;font-weight:600;margin-right:5px}
      .crm-card {background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;height:100%;box-shadow:0 1px 2px rgba(15,23,42,.03)}
      .crm-label {font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:#6b7280;font-weight:700}
      .crm-value {font-size:1.25rem;font-weight:700;color:#111827;margin-top:3px}
      .crm-sub {font-size:.8rem;color:#6b7280;margin-top:2px}
      .crm-action {background:#f8fafc;border:1px solid #dbe3ef;border-radius:12px;padding:16px 18px}
      .crm-timeline {border-left:2px solid #e5e7eb;padding-left:16px;margin-left:5px}
      .crm-event {margin-bottom:13px}
      .crm-event-date {font-size:.75rem;color:#6b7280}
      .crm-event-main {font-weight:600;color:#1f2937}
    </style>
    """, unsafe_allow_html=True)

    model_dir = BASE / "models"
    required_models = [model_dir / "bg_nbd_model.pkl", model_dir / "gamma_gamma_model.pkl", model_dir / "survival_model.pkl"]
    missing_models = [str(x.name) for x in required_models if not x.exists()]
    if missing_models:
        st.warning("Fitted model artifacts are missing: " + ", ".join(missing_models))
        st.code("python scripts/run_pipeline.py", language="bash")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def load_scoring_models():
        from src.config.config import load_config
        from src.models.purchase_model import PurchaseModelBGNBD
        from src.models.monetary_model import MonetaryModelGammaGamma
        from src.models.inactivity_model import InactivitySurvivalModel
        from src.clv.clv_calculator import ProbabilisticCLVCalculator
        from src.segmentation.segmenter import ActionSegmenter
        from src.nba.nba_engine import NBAEngine
        cfg = load_config()
        bgf = PurchaseModelBGNBD(cfg); bgf.load_model()
        ggf = MonetaryModelGammaGamma(cfg); ggf.load_model()
        survival = InactivitySurvivalModel(cfg); survival.load_model()
        return cfg, bgf, ggf, survival, ProbabilisticCLVCalculator(cfg), ActionSegmenter(cfg), NBAEngine(cfg)

    cfg, bgf, ggf, survival, clv_calc, segmenter, nba_engine = load_scoring_models()

    # Persistent customer selection makes the page feel like a CRM record rather than a form.
    customer_ids = customers.customer_id.astype(str).tolist()
    default_id = st.session_state.get("crm_customer_id", customer_ids[0])
    if default_id not in customer_ids:
        default_id = customer_ids[0]
    s1, s2 = st.columns([2.5, 1])
    with s1:
        source_id = st.selectbox("Find customer", customer_ids, index=customer_ids.index(default_id), key="crm_customer_selector")
    st.session_state["crm_customer_id"] = source_id
    base = customers[customers.customer_id.astype(str) == source_id].iloc[0]

    customer_tx = tx[tx.customer_id.astype(str) == source_id].copy() if "customer_id" in tx.columns else pd.DataFrame()
    customer_tx = customer_tx.sort_values("transaction_date", ascending=False) if "transaction_date" in customer_tx.columns else customer_tx
    country = str(base.get("country", "Unknown"))
    segment = str(base.get("action_segment", "Unassigned"))
    action = str(base.get("nba_recommended_action", "Review"))
    channel = str(base.get("nba_channel", "-") )
    risk = float(base.get("inactivity_prob_90d", 0.0))
    risk_label = "High" if risk >= .60 else ("Medium" if risk >= .30 else "Low")
    first_date = pd.to_datetime(base.get("first_purchase_date"), errors="coerce")
    last_date = pd.to_datetime(base.get("last_purchase_date"), errors="coerce")
    joined = first_date.strftime("%d %b %Y") if pd.notna(first_date) else "—"
    last_seen = last_date.strftime("%d %b %Y") if pd.notna(last_date) else "—"

    st.markdown(f"""
    <div class="crm-header">
      <div class="crm-name">Customer {source_id}</div>
      <div class="crm-meta">{country} &nbsp; · &nbsp; Customer since {joined} &nbsp; · &nbsp; Last purchase {last_seen}</div>
      <div style="margin-top:10px"><span class="crm-chip">{segment}</span><span class="crm-chip">{risk_label} inactivity risk</span><span class="crm-chip">{action}</span></div>
    </div>
    """, unsafe_allow_html=True)

    tab_overview, tab_score, tab_history = st.tabs(["Overview", "Model Score", "Purchase History"])

    with tab_overview:
        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f'<div class="crm-card"><div class="crm-label">90-day CLV</div><div class="crm-value">{money(base[clv_col])}</div><div class="crm-sub">Expected customer value</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="crm-card"><div class="crm-label">P(Alive)</div><div class="crm-value">{base["p_alive"]*100:.1f}%</div><div class="crm-sub">Purchase relationship active</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="crm-card"><div class="crm-label">Inactivity risk</div><div class="crm-value">{risk*100:.1f}%</div><div class="crm-sub">90-day probability</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="crm-card"><div class="crm-label">Expected action value</div><div class="crm-value">{money(base["nba_expected_incremental_value"])}</div><div class="crm-sub">Before contact cost</div></div>', unsafe_allow_html=True)

        left, right = st.columns([1.15, .85])
        with left:
            st.subheader("Customer profile")
            profile = pd.DataFrame({
                "Attribute": ["Customer ID", "Country", "Repeat purchases", "Total revenue", "Average order value", "Days since last purchase", "Customer tenure"],
                "Value": [source_id, country, f'{int(base["frequency"]):,}', money(base["total_revenue"]), money(base["average_order_value"]), f'{base["days_since_last_purchase"]:.0f} days', f'{base["customer_tenure_days"]:.0f} days']
            })
            st.dataframe(profile, use_container_width=True, hide_index=True)
        with right:
            st.subheader("Recommended action")
            st.markdown(f'<div class="crm-action"><div class="crm-label">Next-best-action</div><div style="font-size:1.15rem;font-weight:700;margin:5px 0 7px">{action}</div><div class="crm-sub">Preferred channel: {channel}</div><div style="margin-top:10px;color:#374151;font-size:.9rem">{base.get("nba_reason", "No explanation available.")}</div></div>', unsafe_allow_html=True)

        st.subheader("Recent activity")
        if customer_tx.empty:
            st.info("No transaction history is available for this customer.")
        else:
            events = customer_tx.head(6)
            timeline = '<div class="crm-timeline">'
            for _, r in events.iterrows():
                date = pd.to_datetime(r.get("transaction_date"), errors="coerce")
                date_text = date.strftime("%d %b %Y") if pd.notna(date) else "Unknown date"
                amount = r.get("total_price", r.get("price", r.get("monetary_value", 0)))
                try: amount_text = money(float(amount))
                except Exception: amount_text = "—"
                desc = str(r.get("description", "Purchase"))[:90]
                timeline += f'<div class="crm-event"><div class="crm-event-date">{date_text}</div><div class="crm-event-main">Purchase · {amount_text}</div><div class="crm-sub">{desc}</div></div>'
            timeline += '</div>'
            st.markdown(timeline, unsafe_allow_html=True)

    with tab_score:
        st.caption("Edit observed behaviour to run a temporary what-if score. The source customer record is not modified.")
        with st.form("single_customer_form"):
            a,b,c,d = st.columns(4)
            frequency = a.number_input("Repeat purchases", min_value=0, max_value=500, value=int(base["frequency"]), step=1)
            recency = b.number_input("Recency since first purchase (days)", min_value=0.0, max_value=5000.0, value=float(base["recency_days"]), step=1.0)
            tenure = c.number_input("Customer tenure (days)", min_value=1.0, max_value=5000.0, value=max(1.0,float(base["customer_tenure_days"])), step=1.0)
            days_since = d.number_input("Days since last purchase", min_value=0.0, max_value=5000.0, value=float(base["days_since_last_purchase"]), step=1.0)
            e,f,g = st.columns(3)
            monetary = e.number_input("Average monetary value (£)", min_value=0.01, max_value=50000.0, value=max(0.01,float(base["monetary_value"])), step=1.0)
            aov = f.number_input("Average order value (£)", min_value=0.01, max_value=50000.0, value=max(0.01,float(base["average_order_value"])), step=1.0)
            revenue = g.number_input("Total historical revenue (£)", min_value=0.01, max_value=1000000.0, value=max(0.01,float(base["total_revenue"])), step=10.0)
            submitted = st.form_submit_button("Score Customer", type="primary", use_container_width=True)

        if submitted:
            candidate = base.to_frame().T.copy()
            candidate["frequency"] = int(frequency)
            candidate["recency_days"] = float(recency)
            candidate["customer_tenure_days"] = float(max(tenure, recency, days_since + 1.0))
            candidate["days_since_last_purchase"] = float(days_since)
            candidate["monetary_value"] = float(monetary)
            candidate["average_order_value"] = float(aov)
            candidate["total_revenue"] = float(revenue)
            candidate["customer_id"] = source_id
            candidate["p_alive"] = bgf.predict_p_alive(candidate).values
            candidate["exp_avg_monetary"] = ggf.predict_expected_average_spend(candidate).values
            inactivity = survival.predict_inactivity_probabilities(candidate)
            candidate = pd.concat([candidate, inactivity], axis=1)
            candidate = clv_calc.compute_clv(candidate, bgf, ggf, n_simulation_samples=500)
            population = customers.copy()
            replace_idx = population.index[population.customer_id.astype(str) == source_id]
            if len(replace_idx):
                for col in candidate.columns:
                    if col in population.columns: population.loc[replace_idx[0], col] = candidate.iloc[0][col]
            else:
                population = pd.concat([population, candidate], ignore_index=True)
            population = segmenter.segment_customers(population)
            scored = nba_engine.generate_recommendations(population[population.customer_id.astype(str) == source_id].copy())
            row = scored.iloc[0]
            st.session_state["crm_scored_row"] = row.to_dict()
            st.success("Customer score refreshed.")

        row_data = st.session_state.get("crm_scored_row")
        if row_data:
            row = pd.Series(row_data)
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Expected CLV (90d)", money(row[clv_col]))
            k2.metric("P(Alive)", f"{row['p_alive']*100:.1f}%")
            k3.metric("Inactivity risk", f"{row['inactivity_prob_90d']*100:.1f}%")
            k4.metric("Expected purchases", f"{row[f'exp_purchases_{cfg.clv.default_horizon_days}d']:.2f}")
            l,r = st.columns([1.1,.9])
            with l:
                st.subheader("Model output")
                st.dataframe(pd.DataFrame({"Metric":["P(Alive)","Expected purchases (90d)","Expected average spend","90-day CLV","CLV lower interval","CLV upper interval","90-day inactivity probability","Risk tier"],"Value":[f"{row['p_alive']:.4f}",f"{row[f'exp_purchases_{cfg.clv.default_horizon_days}d']:.3f}",money(row['exp_avg_monetary']),money(row[clv_col]),money(row['clv_lower_80pct']),money(row['clv_upper_80pct']),f"{row['inactivity_prob_90d']:.2%}",str(row['inactivity_risk_tier'])]}), use_container_width=True, hide_index=True)
            with r:
                st.subheader("Decision")
                st.markdown(f'<div class="crm-action"><div class="crm-label">Action segment</div><div class="crm-value">{row["action_segment"]}</div><div style="margin-top:12px" class="crm-label">Next-best-action</div><div style="font-weight:700;margin-top:3px">{row["nba_recommended_action"]}</div><div class="crm-sub">Channel: {row["nba_channel"]}</div><div style="margin-top:10px">Expected incremental value: <b>{money(row["nba_expected_incremental_value"])}</b></div><div class="crm-sub">{row["nba_reason"]}</div></div>', unsafe_allow_html=True)
            st.download_button("Export customer score", scored.to_csv(index=False).encode("utf-8"), f"customer_{source_id}_score.csv", "text/csv")
        else:
            st.info("Run Score Customer to generate a fresh model score for this customer.")

    with tab_history:
        st.subheader("Purchase history")
        if customer_tx.empty:
            st.info("No purchase records available.")
        else:
            display_cols = [c for c in ["transaction_date","invoice_no","description","quantity","unit_price","total_price"] if c in customer_tx.columns]
            st.dataframe(customer_tx[display_cols].head(100), use_container_width=True, hide_index=True)

elif page == "Customer 360":
    hero("Customer 360 Explorer", "Inspect one customer’s observed behaviour, probabilistic value, inactivity risk, and recommended action.")
    c1, c2, c3 = st.columns(3)
    segment_filter = c1.selectbox("Segment", ["All"] + sorted(customers.action_segment.dropna().unique().tolist()))
    country_filter = c2.selectbox("Country", ["All"] + sorted(customers.country.dropna().astype(str).unique().tolist()))
    search = c3.text_input("Customer ID contains", "")
    filtered = customers.copy()
    if segment_filter != "All": filtered = filtered[filtered.action_segment == segment_filter]
    if country_filter != "All": filtered = filtered[filtered.country.astype(str) == country_filter]
    if search.strip(): filtered = filtered[filtered.customer_id.astype(str).str.contains(search.strip(), case=False, na=False)]
    if filtered.empty:
        st.warning("No customers match the selected filters.")
    else:
        ids = filtered.customer_id.astype(str).tolist()
        selected_id = st.selectbox("Customer", ids)
        row = customers[customers.customer_id.astype(str) == selected_id].iloc[0]
        a,b,c,e = st.columns(4)
        a.metric("Expected CLV (90d)", money(row[clv_col]))
        b.metric("P(Alive)", f"{row['p_alive']*100:.1f}%")
        c.metric("Inactivity risk (90d)", f"{row['inactivity_prob_90d']*100:.1f}%")
        e.metric("Uncertainty spread", f"{row['clv_uncertainty_spread']:.2f}")
        st.markdown(f"<div class='decision'><b>Recommended action:</b> {row['nba_recommended_action']} &nbsp; | &nbsp; <b>Channel:</b> {row['nba_channel']}<br><span class='small'>{row['nba_reason']}</span></div>", unsafe_allow_html=True)
        st.divider()
        left, right = st.columns([1,1])
        with left:
            st.subheader("Customer profile")
            profile = pd.DataFrame({"Metric":["Country","Cohort","Tenure (days)","Days since purchase","Repeat purchases","Total revenue","Average order value","Segment"],"Value":[row.get("country",""),row.get("cohort_month",""),round(row["customer_tenure_days"],1),round(row["days_since_last_purchase"],1),int(row["frequency"]),money(row["total_revenue"]),money(row["average_order_value"]),row["action_segment"]]})
            st.dataframe(profile, use_container_width=True, hide_index=True)
        with right:
            cust_tx = tx[tx.customer_id.astype(str) == selected_id].sort_values("transaction_date", ascending=False)
            st.subheader("Recent transactions")
            cols = [c for c in ["invoice_id","transaction_date","stock_code","description","quantity","unit_price","revenue"] if c in cust_tx.columns]
            st.dataframe(cust_tx[cols].head(30), use_container_width=True, hide_index=True)

elif page == "Action Segments":
    hero("Action Segments", "Compare operational segments using transparent, model-informed rules rather than opaque clustering labels.")
    seg = customers.groupby("action_segment").agg(Customers=("customer_id","size"), Mean_CLV=(clv_col,"mean"), Mean_Risk=("inactivity_prob_90d","mean"), Mean_PAlive=("p_alive","mean"), Revenue=("total_revenue","sum")).reset_index()
    st.dataframe(seg.sort_values("Customers", ascending=False), use_container_width=True, hide_index=True)
    st.subheader("Segment value versus risk")
    fig = px.scatter(seg, x="Mean_Risk", y="Mean_CLV", size="Customers", text="action_segment", labels={"Mean_Risk":"Mean 90-day inactivity probability","Mean_CLV":"Mean 90-day CLV (£)"})
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Customer-level segment audit")
    selected = st.selectbox("Segment to inspect", sorted(customers.action_segment.unique()))
    view = customers[customers.action_segment == selected].sort_values(clv_col, ascending=False)
    st.dataframe(view[["customer_id","action_segment",clv_col,"p_alive","inactivity_prob_90d","nba_recommended_action","nba_channel"]].head(100), use_container_width=True, hide_index=True)

elif page == "Next-Best-Action":
    hero("Next-Best-Action Workbench", "Filter the decision queue by urgency, action, segment, and expected incremental value.")
    c1,c2,c3 = st.columns(3)
    priority = c1.multiselect("Priority", sorted(customers.nba_priority.dropna().unique()), default=sorted(customers.nba_priority.dropna().unique()))
    action = c2.multiselect("Action", sorted(customers.nba_recommended_action.dropna().unique()), default=sorted(customers.nba_recommended_action.dropna().unique()))
    top_n = c3.slider("Rows", 10, 200, 50, 10)
    queue = customers[customers.nba_priority.isin(priority) & customers.nba_recommended_action.isin(action)].copy()
    queue = queue.sort_values(["nba_priority","nba_expected_incremental_value"], ascending=[True,False]).head(top_n)
    st.metric("Customers in queue", f"{len(queue):,}")
    cols = ["customer_id","action_segment","nba_recommended_action","nba_priority","nba_channel",clv_col,"nba_expected_incremental_value","nba_estimated_cost","nba_offer","nba_reason"]
    st.dataframe(queue[cols], use_container_width=True, hide_index=True)
    st.download_button("Export current action queue", queue[cols].to_csv(index=False).encode("utf-8"), "nba_action_queue.csv", "text/csv")

elif page == "Campaign Simulator":
    hero("Campaign Scenario Simulator", "Run transparent what-if calculations. Response rates and uplift assumptions are synthetic planning inputs, not observed campaign outcomes.")
    from src.simulation.campaign_simulator import ScenarioSimulator
    c1,c2 = st.columns([1,1.4])
    segments = sorted(customers.action_segment.unique())
    with c1:
        target = st.selectbox("Target segment", segments)
        available = int((customers.action_segment == target).sum())
        size = st.slider("Campaign size", 10, max(10, available), min(500, max(10, available)), 10)
        action = st.selectbox("Objective", ["RETENTION","UPSELL","LOYALTY_REWARD","REACTIVATION","WIN_BACK"])
        channel = st.selectbox("Channel", ["email","sms","direct_mail","push"])
        response = st.slider("Assumed response rate", 1.0, 40.0, 15.0, 0.5) / 100
        aov = st.slider("Incremental spend per response (£)", 10.0, 200.0, 65.0, 5.0)
        discount = st.slider("Discount", 0.0, 30.0, 12.0, 1.0) / 100
        fixed = st.number_input("Fixed setup cost (£)", 0.0, 10000.0, 150.0, 25.0)
    with c2:
        result = ScenarioSimulator().simulate_campaign(customers, target, action, size, channel, response, aov, discount, fixed)
        r1,r2,r3 = st.columns(3)
        r1.metric("Expected responders", f"{result['expected_responders']:,}")
        r2.metric("Campaign cost", money(result['total_campaign_cost_gbp']))
        r3.metric("Projected ROI", f"{result['roi_percent']:.1f}%")
        st.metric("Expected net value", money(result['expected_net_value_gbp']))
        st.caption(f"80% simulated range for net value: {money(result['net_value_80pct_range'][0])} to {money(result['net_value_80pct_range'][1])}")
        chart = pd.DataFrame({"Component":["Fixed cost","Contact cost","Discount cost","Gross incremental revenue"],"Amount (£)":[result['fixed_cost_gbp'],result['channel_cost_gbp'],result['discount_cost_gbp'],result['gross_incremental_revenue_gbp']]})
        st.plotly_chart(px.bar(chart,x="Component",y="Amount (£)"), use_container_width=True)

elif page == "Model Evaluation":
    hero("Model Evaluation & Monitoring", "Evidence-first model governance: holdout performance, uncertainty calibration, risk calibration, segment stability, data validation, and drift.")

    validation_audit = d.get("validation_audit")
    bench = pd.DataFrame(evaluation.get("benchmark_results", []))
    cov = evaluation.get("interval_coverage", {}) or {}
    cal = evaluation.get("calibration", {}) or {}
    stability = evaluation.get("segment_stability", {}) or {}
    sparse = pd.DataFrame(evaluation.get("sparse_history_sensitivity", []))
    overall = monitoring.get("overall_health", {}) or {}

    # Governance summary
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Holdout customers", f"{len(bench):,}" if not bench.empty else "—")
    g2.metric("80% interval coverage", f"{cov.get('empirical_coverage_pct', 0):.1f}%" if cov else "—")
    g3.metric("Inactivity Brier", f"{cal.get('brier_score', 0):.4f}" if cal else "—")
    g4.metric("Segment agreement", f"{stability.get('agreement_rate_pct', 0):.1f}%" if stability else "—")
    g5.metric("Monitoring", overall.get("overall_status", "PENDING"))

    tabs = st.tabs(["Performance", "Uncertainty", "Risk Calibration", "Segment Stability", "Data Validation", "Drift & Monitoring"])

    with tabs[0]:
        st.subheader("Holdout benchmark")
        st.caption("Predictions are evaluated against the future holdout revenue window. Lower MAE/RMSE is better; higher Spearman indicates stronger customer-value ranking association.")
        if bench.empty:
            st.info("Holdout evaluation report is not available. Run the full pipeline to generate it.")
        else:
            # Evaluation reports use presentation-friendly column names such as
            # "MAE (£)" and "RMSE (£)".  Normalise them for charting so the page
            # remains compatible with both current and older reports.
            bench_plot = bench.rename(columns={
                "MAE (£)": "MAE",
                "RMSE (£)": "RMSE",
            }).copy()
            metric_cols = [c for c in ["Model", "MAE", "RMSE", "Spearman Rank Corr"] if c in bench_plot.columns]
            st.dataframe(bench_plot[metric_cols], use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(bench_plot, x="Model", y="MAE", title="MAE — lower is better", text_auto=".0f")
                fig.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(bench_plot, x="Model", y="Spearman Rank Corr", title="Spearman rank correlation — higher is better", text_auto=".3f")
                fig.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            if "Probabilistic CLV" in bench.get("Model", pd.Series(dtype=str)).astype(str).values:
                p = bench[bench["Model"].astype(str) == "Probabilistic CLV"].iloc[0]
                st.info(f"Interpretation: the probabilistic CLV model should be read across all benchmark metrics. Its ranking correlation and error metrics can tell different stories; the dashboard does not collapse them into a single score.")

        st.subheader("Sparse-history sensitivity")
        if sparse.empty:
            st.info("Sparse-history analysis is not available in the current report.")
        else:
            st.dataframe(sparse, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("CLV prediction interval coverage")
        if not cov:
            st.info("Interval coverage is not available. Run the evaluation pipeline.")
        else:
            target = float(cov.get("nominal_target", .80)) * 100
            actual = float(cov.get("empirical_coverage_pct", 0))
            c1, c2, c3 = st.columns(3)
            c1.metric("Nominal coverage", f"{target:.0f}%")
            c2.metric("Observed coverage", f"{actual:.2f}%")
            c3.metric("Coverage gap", f"{actual-target:+.2f} pp")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=actual, number={"suffix":"%"}, gauge={"axis":{"range":[0,100]},"threshold":{"line":{"width":4},"value":target}}))
            fig.update_layout(height=320, margin=dict(l=20,r=20,t=30,b=20), title="Empirical vs nominal interval coverage")
            st.plotly_chart(fig, use_container_width=True)
            if abs(actual - target) > 5:
                st.warning(cov.get("status", "Coverage is materially below the nominal target."))
            else:
                st.success(cov.get("status", "Coverage is within tolerance."))
            c = pd.DataFrame({"Component":["Inside interval","Below lower bound","Above upper bound"],"Percent":[actual,float(cov.get("below_lower_bound_pct",0)),float(cov.get("above_upper_bound_pct",0))]})
            st.plotly_chart(px.bar(c,x="Component",y="Percent",text_auto=".1f",title="Outcome placement relative to interval"),use_container_width=True)

    with tabs[2]:
        st.subheader("Inactivity probability calibration")
        if not cal:
            st.info("Calibration report is not available.")
        else:
            c1,c2 = st.columns([.7,1.3])
            c1.metric("Brier score", f"{cal.get('brier_score',0):.4f}")
            c1.caption("Lower is better. Interpret together with event prevalence and the reliability table.")
            bins = pd.DataFrame(cal.get("calibration_bins", []))
            if not bins.empty:
                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=bins["mean_predicted_prob"],y=bins["observed_inactivity_rate"],mode="lines+markers",name="Observed"))
                    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Perfect calibration",line=dict(dash="dash")))
                    fig.update_layout(xaxis_title="Mean predicted probability",yaxis_title="Observed inactivity rate",height=360)
                    st.plotly_chart(fig,use_container_width=True)
                st.dataframe(bins,use_container_width=True,hide_index=True)

    with tabs[3]:
        st.subheader("Segment stability and transitions")
        if not stability:
            st.info("Segment stability is not present in the evaluation report. Re-run the pipeline.")
        else:
            c1,c2 = st.columns(2)
            c1.metric("Agreement rate",f"{stability.get('agreement_rate_pct',0):.2f}%")
            c2.metric("Evaluated customers",f"{stability.get('evaluated_customers',0):,}")
            matrix=pd.DataFrame(stability.get("transition_matrix", []))
            if not matrix.empty:
                st.dataframe(matrix,use_container_width=True,hide_index=True)
                idx_col=matrix.columns[0]
                heat=matrix.set_index(idx_col)
                fig=px.imshow(heat,text_auto=True,aspect="auto",labels={"x":"Target segment","y":"Baseline segment","color":"Share"},title="Row-normalized segment transition matrix")
                st.plotly_chart(fig,use_container_width=True)
            st.caption("Segment agreement is a stability diagnostic, not a model-quality score by itself.")

    with tabs[4]:
        st.subheader("Data validation status")
        if validation_audit is None:
            st.warning("The validation contract is packaged, but no generated audit_trail.json is present in this project snapshot.")
            st.code("python -m src.validation.validator", language="bash")
            st.caption("This is intentionally shown as pending rather than presenting fabricated validation results.")
        else:
            status = validation_audit.get("validation_status", "REVIEW")
            if status == "PASS":
                st.success(f"Validation status: {status}")
            else:
                st.warning(f"Validation status: {status}")
            a,b,c,d = st.columns(4)
            a.metric("Raw rows",f"{validation_audit.get('initial_row_count',0):,}")
            b.metric("Clean rows",f"{validation_audit.get('final_clean_row_count',0):,}")
            c.metric("Retention",f"{validation_audit.get('retention_rate_pct',0):.2f}%")
            d.metric("Duplicate rows",f"{validation_audit.get('metrics',{}).get('duplicate_rows',0):,}")
            vm = validation_audit.get("metrics",{})
            checks=pd.DataFrame({"Check":["Missing customer IDs","Cancellations","Non-positive quantity","Non-positive price","Quantity outliers","Price outliers","Invalid dates","Exact duplicates"],"Rows":[vm.get("missing_customer_rows",0),vm.get("cancelled_rows",0),vm.get("return_rows",0),vm.get("zero_price_rows",0),vm.get("outlier_quantity_rows",0),vm.get("outlier_price_rows",0),vm.get("invalid_date_rows",0),vm.get("duplicate_rows",0)]})
            st.dataframe(checks,use_container_width=True,hide_index=True)

    with tabs[5]:
        st.subheader("Production-style drift monitoring")
        if not monitoring:
            st.warning("No monitoring report is available. Run the full pipeline to generate drift diagnostics.")
        else:
            max_psi=float(overall.get("max_psi",0))
            c1,c2,c3=st.columns(3)
            c1.metric("Maximum PSI",f"{max_psi:.4f}")
            c2.metric("Overall status",overall.get("overall_status","UNKNOWN"))
            c3.metric("PSI threshold", "0.10 / 0.25")
            rows=[]
            for key,val in monitoring.items():
                if isinstance(val,dict) and "psi_value" in val:
                    rows.append({"Feature":key.replace("psi_",""),"PSI":float(val["psi_value"]),"Status":val.get("status","")})
            if rows:
                drift_df=pd.DataFrame(rows).sort_values("PSI",ascending=False)
                fig=px.bar(drift_df,x="PSI",y="Feature",orientation="h",color="Status",title="Population Stability Index")
                fig.add_vline(x=.10,line_dash="dash",annotation_text="Moderate threshold")
                fig.add_vline(x=.25,line_dash="dot",annotation_text="Alert threshold")
                fig.update_layout(height=420)
                st.plotly_chart(fig,use_container_width=True)
                st.dataframe(drift_df,use_container_width=True,hide_index=True)
            seg_drift=pd.DataFrame(monitoring.get("segment_population_drift", []))
            if not seg_drift.empty:
                st.subheader("Segment population drift")
                fig=px.bar(seg_drift,x="segment",y="drift_pct_points",title="Target minus baseline segment share (percentage points)")
                st.plotly_chart(fig,use_container_width=True)

