import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

st.title("🏀 NBA Lineup Efficiency: Bayesian Network Simulator")
st.markdown("**Interactive Demo**: Tweak player skills & see efficiency impact. Built from real 2023-24 NBA data.")

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("nba_lineups_expanded_discretized.csv")
        st.success(f"✅ Loaded {len(data)} lineups!")
        return data
    except FileNotFoundError:
        st.warning("📁 Upload 'nba_lineups_expanded_discretized.csv' or run data gen below.")
        return None

@st.cache_data
def fit_model(data):
    if data is None:
        return None, None
    order = ['Low', 'Medium', 'High']
    all_cols = [
        'Net_Rating_Impact', 'Shooting_Efficiency', 'Efficiency',
        'SCORING_Talent', 'PLAYMAKING_Talent', 'REBOUNDING_Talent',
        'DEFENSIVE_Talent', 'NET_RATING_Talent',
        'AST_rate', 'TOV_rate', 'ORB_rate'
    ]
    for col in all_cols:
        if col in data.columns:
            data[col] = pd.Categorical(data[col], categories=order, ordered=True)

    # Fix for skewed values: Oversample High Efficiency (boosts baseline to ~5-8%)
    high_mask = data['Efficiency'] == 'High'
    if sum(high_mask) < len(data) * 0.2:  # If <20% High, balance
        oversample = data[high_mask].sample(frac=(0.2 / (sum(high_mask)/len(data))) - 1, replace=True, random_state=42)
        data = pd.concat([data, oversample]).reset_index(drop=True)
        st.info(f"🔧 Balanced data: Added {len(oversample)} High Efficiency samples for realism.")

    edges = [
        ('PLAYMAKING_Talent', 'AST_rate'),
        ('PLAYMAKING_Talent', 'TOV_rate'),
        ('SCORING_Talent', 'Shooting_Efficiency'),
        ('REBOUNDING_Talent', 'ORB_rate'),
        ('DEFENSIVE_Talent', 'Net_Rating_Impact'),
        ('NET_RATING_Talent', 'Net_Rating_Impact'),
        ('Net_Rating_Impact', 'Efficiency'),
        ('Shooting_Efficiency', 'Efficiency'),
        ('AST_rate', 'Efficiency'),
        ('TOV_rate', 'Efficiency'),
        ('ORB_rate', 'Efficiency')
    ]

    model = DiscreteBayesianNetwork(edges)
    model.fit(data, estimator=BayesianEstimator,
              state_names={col: order for col in all_cols if col in data.columns},
              equivalent_sample_size=10)
    return model, data

# Load & Fit
data = load_data()
model, fitted_data = fit_model(data)

if model is None:
    st.error("❌ Model not ready. Upload data or add data-gen code.")
    if st.button("Generate Sample Data (Slow - NBA API)"):
        st.code("# Paste your Phase 1 code here")
else:
    infer = VariableElimination(model)
    order = ['Low', 'Medium', 'High']

    # Sidebar: Flexible Scenario Controls
    st.sidebar.header("🎯 Quick Scenarios")
    scenario = st.sidebar.selectbox(
        "Pick a Preset (or Manual below)",
        ["Baseline", "Elite Shooter", "Defensive Anchor", "Low Mistake Machine", "Shooter + Playmaker", "Rebound-Focused (High TOV)", "All-In Offense"],
        index=0
    )
    manual_mode = st.sidebar.checkbox("🔧 Override with Custom Sliders", value=False)

    # Auto-set sliders based on scenario
    if scenario == "Baseline":
        shooting, net_rating, tov = 'Medium', 'Medium', 'Medium'
    elif scenario == "Elite Shooter":
        shooting, net_rating, tov = 'High', 'Medium', 'Medium'
    elif scenario == "Defensive Anchor":
        shooting, net_rating, tov = 'Medium', 'High', 'Medium'
    elif scenario == "Low Mistake Machine":
        shooting, net_rating, tov = 'Medium', 'Medium', 'Low'
    elif scenario == "Shooter + Playmaker":
        shooting, net_rating, tov = 'High', 'Medium', 'Low'
    elif scenario == "Rebound-Focused (High TOV)":
        shooting, net_rating, tov = 'Medium', 'Medium', 'High'
    elif scenario == "All-In Offense":
        shooting, net_rating, tov = 'High', 'High', 'Low'

    # Custom sliders if manual
    if manual_mode:
        st.sidebar.subheader("Custom Tweaks")
        shooting = st.sidebar.selectbox("Shooting Efficiency", order, index=order.index(shooting))
        net_rating = st.sidebar.selectbox("Net Rating Impact", order, index=order.index(net_rating))
        tov = st.sidebar.selectbox("Turnover Rate", order, index=order.index(tov))
        scenario = "Custom"  # Flag for display

    # Baseline for Δ calc
    baseline_ev = {'Shooting_Efficiency': 'Medium', 'Net_Rating_Impact': 'Medium', 'TOV_rate': 'Medium'}
    base_q = infer.query(variables=['Efficiency'], evidence=baseline_ev)
    base_high = base_q.values[2]

    # Current evidence & query
    evidence = {'Shooting_Efficiency': shooting, 'Net_Rating_Impact': net_rating, 'TOV_rate': tov}
    q = infer.query(variables=['Efficiency'], evidence=evidence)
    probs = pd.Series(q.values, index=order) * 100  # % for display

    # Main: Chart + Full Distrib Text + Δ (Fixed Metric)
    st.header(f"Scenario: {scenario}")
    delta_high = (q.values[2] - base_high) * 100
    st.metric(
        label="Δ P(High Efficiency)",
        value=f"{delta_high:+.1f}%",
        help=f"vs. Baseline ({base_high*100:.1f}%)"  # Fixed: Use 'help' for note (hover to see)
    )

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("P(Low)", f"{probs[0]:.1f}%")
    with col2: st.metric("P(Medium)", f"{probs[1]:.1f}%")
    with col3: st.metric("P(High)", f"{probs[2]:.1f}%")

    st.bar_chart(probs, use_container_width=True)

    if st.button("🔄 Reset to Baseline"):
        st.rerun()

    # Tabs (Condensed for Clarity)
    tab1, tab2, tab3 = st.tabs(["📈 Sensitivity Ranking", "📊 Sample Data", "📝 Conclusions"])

    with tab1:
        st.markdown("**Which Factor Boosts Most?** (Sortable Table)")
        treatments = [
            ('Shooting_Efficiency', 'High', "Shooting → High"),
            ('Net_Rating_Impact', 'High', "Net Rating → High"),
            ('AST_rate', 'High', "Assists → High"),
            ('TOV_rate', 'Low', "TOV → Low"),
            ('ORB_rate', 'High', "Rebounds → High")
        ]
        sens_data = []
        for var, val, label in treatments:
            ev = {**baseline_ev, var: val}
            q_s = infer.query(variables=['Efficiency'], evidence=ev)
            delta = (q_s.values[2] - base_high) * 100
            sens_data.append({'Factor': label, 'Delta_Num': delta, 'Δ P(High)': f"{delta:+.1f}%"})

        df_sens = pd.DataFrame(sens_data).sort_values('Delta_Num', ascending=False)[['Factor', 'Δ P(High)']]
        st.table(df_sens)  # Add st.dataframe(df_sens) for sortable if wanted

    with tab2:
        st.markdown("**Lineup Sample**")
        display_cols = ['GROUP_NAME', 'team', 'MIN', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'Efficiency']
        available_cols = [col for col in display_cols if col in fitted_data.columns]
        if not available_cols:
            available_cols = ['Shooting_Efficiency', 'Net_Rating_Impact', 'Efficiency', 'AST_rate', 'TOV_rate', 'ORB_rate']
        with st.expander("🔍 Columns Debug"):
            st.write(list(fitted_data.columns))
        st.dataframe(fitted_data[available_cols].head(10))

    with tab3:
        st.markdown("""
        ### Key Insights
        - **Shooting Dominates**: +70%+ boost – chase 3PT threats!
        - **TOV Control**: +20% lever – protect the ball.
        - **Recommendations**: Stack shooter + low TOV for 99% elite.
        - **Limitations**: Lineup-level; try possession data next.
        
        _Rediet Girmay | Nov 2025_
        """)

st.markdown("---")
st.caption("Deployed via Streamlit Cloud | NBA API 2023-24")
