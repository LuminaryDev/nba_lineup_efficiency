import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

# Page config for wide, clean layout
st.set_page_config(layout="wide", page_title="NBA Lineup Optimizer", page_icon="🏀")

# Custom CSS: Light, calm professional theme (soft grays, clean lines)
st.markdown("""
<style>
    .main-header {color: #2c3e50; font-size: 2.2em; text-align: center; font-weight: 300; margin-bottom: 0.5em;}
    .subtle-metric {background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px; margin: 4px 0;}
    .sidebar .sidebar-content {background-color: #ffffff; border-right: 1px solid #e9ecef;}
    .stMetric > label {font-size: 0.9em; color: #6c757d;}
    .stMetric > div > div > div {font-size: 1.4em; font-weight: 400;}
    section[data-testid="stHorizontalBlock"] {gap: 1rem;}
</style>
""", unsafe_allow_html=True)

# Hero Header: Minimal & Professional
st.markdown('<h1 class="main-header">NBA Lineup Optimizer</h1>', unsafe_allow_html=True)
st.markdown("Bayesian insights from 2023-24 data. Simulate substitutions, measure efficiency. " +
            "<i>By Rediet Girmay | GSE/0945-17</i>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("nba_lineups_expanded_discretized.csv")
        st.success(f"Loaded {len(data):,} lineups.")
        return data
    except FileNotFoundError:
        st.warning("Upload 'nba_lineups_expanded_discretized.csv'.")
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

    # Balance for realistic baselines
    high_mask = data['Efficiency'] == 'High'
    if sum(high_mask) < len(data) * 0.2:
        frac = (0.2 / (sum(high_mask)/len(data))) - 1
        oversample = data[high_mask].sample(frac=frac, replace=True, random_state=42)
        data = pd.concat([data, oversample]).reset_index(drop=True)

    edges = [
        ('PLAYMAKING_Talent', 'AST_rate'), ('PLAYMAKING_Talent', 'TOV_rate'),
        ('SCORING_Talent', 'Shooting_Efficiency'), ('REBOUNDING_Talent', 'ORB_rate'),
        ('DEFENSIVE_Talent', 'Net_Rating_Impact'), ('NET_RATING_Talent', 'Net_Rating_Impact'),
        ('Net_Rating_Impact', 'Efficiency'), ('Shooting_Efficiency', 'Efficiency'),
        ('AST_rate', 'Efficiency'), ('TOV_rate', 'Efficiency'), ('ORB_rate', 'Efficiency')
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
    st.error("Load data to proceed.")
else:
    infer = VariableElimination(model)
    order = ['Low', 'Medium', 'High']
    neutral_colors = ['#adb5bd', '#6c757d', '#495057']  # Calm grays

    # Sidebar: Streamlined & Light
    with st.sidebar:
        st.header("Scenario Selector")
        scenario = st.selectbox(
            "Select Preset:",
            ["Baseline", "Elite Shooter", "Defensive Focus", 
             "Low Turnover", "Shooter + Low TO", 
             "High Turnover Risk", "Full Elite"],
            index=0
        )
        manual_mode = st.checkbox("Custom Adjustments", value=False)

        # Presets
        presets = {
            "Baseline": ('Medium', 'Medium', 'Medium'),
            "Elite Shooter": ('High', 'Medium', 'Medium'),
            "Defensive Focus": ('Medium', 'High', 'Medium'),
            "Low Turnover": ('Medium', 'Medium', 'Low'),
            "Shooter + Low TO": ('High', 'Medium', 'Low'),
            "High Turnover Risk": ('Medium', 'Medium', 'High'),
            "Full Elite": ('High', 'High', 'Low')
        }
        shooting, net_rating, tov = presets[scenario]

        if manual_mode:
            shooting = st.selectbox("Shooting Efficiency", order, index=order.index(shooting))
            net_rating = st.selectbox("Net Rating Impact", order, index=order.index(net_rating))
            tov = st.selectbox("Turnover Rate", order, index=order.index(tov))
            scenario = "Custom"

    # Baseline
    baseline_ev = {'Shooting_Efficiency': 'Medium', 'Net_Rating_Impact': 'Medium', 'TOV_rate': 'Medium'}
    base_q = infer.query(variables=['Efficiency'], evidence=baseline_ev)
    base_high = base_q.values[2]
    base_probs = pd.Series(base_q.values, index=order) * 100

    # Current Query
    evidence = {'Shooting_Efficiency': shooting, 'Net_Rating_Impact': net_rating, 'TOV_rate': tov}
    q = infer.query(variables=['Efficiency'], evidence=evidence)
    probs = pd.Series(q.values, index=order) * 100

    # Main: Clean Split-View
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Scenario Analysis")
        delta_high = (q.values[2] - base_high) * 100
        st.metric("P(High Efficiency) Change", f"{delta_high:+.1f}%")

        # Clean Metrics (Neutral, Spaced)
        for lvl in order:
            idx = order.index(lvl)
            st.metric(lvl, f"{probs[idx]:.1f}%", delta_color="off")

    with col_right:
        st.subheader("Baseline")
        for lvl in order:
            idx = order.index(lvl)
            st.metric(lvl, f"{base_probs[idx]:.1f}%", delta_color="off")

    # Subtle Charts
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        plt.figure(figsize=(6, 4))
        probs.plot(kind='bar', color=neutral_colors, ax=plt.gca(), width=0.6)
        plt.title("")
        plt.ylabel("Probability (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    with col_chart2:
        plt.figure(figsize=(6, 4))
        base_probs.plot(kind='bar', color=neutral_colors, ax=plt.gca(), width=0.6)
        plt.title("")
        plt.ylabel("Probability (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    # Reset
    st.button("Reset to Baseline", type="secondary", help="Revert to default.")

    # Tabs: Minimalist
    tab1, tab2, tab3 = st.tabs(["Factor Impact", "Data Overview", "Insights"])

    with tab1:
        st.markdown("**Sensitivity Analysis**")
        treatments = [
            ('Shooting_Efficiency', 'High', "Shooting Efficiency"),
            ('TOV_rate', 'Low', "Turnover Rate"),
            ('Net_Rating_Impact', 'High', "Net Rating Impact"),
            ('ORB_rate', 'High', "Rebound Rate"),
            ('AST_rate', 'High', "Assist Rate")
        ]
        sens_data = []
        for var, val, label in treatments:
            ev = {**baseline_ev, var: val}
            q_s = infer.query(variables=['Efficiency'], evidence=ev)
            delta = (q_s.values[2] - base_high) * 100
            sens_data.append({'Factor': label, 'Delta_Num': delta, 'Change': f"{delta:+.1f}%"})

        df_sens = pd.DataFrame(sens_data).sort_values('Delta_Num', ascending=False)[['Factor', 'Change']]
        st.dataframe(df_sens, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**Sample Lineups**")
        display_cols = ['GROUP_NAME', 'team', 'MIN', 'PLUS_MINUS', 'Efficiency']
        available_cols = [col for col in display_cols if col in fitted_data.columns]
        if not available_cols:
            available_cols = ['Efficiency', 'Shooting_Efficiency', 'Net_Rating_Impact']
        st.dataframe(fitted_data[available_cols].head(5), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("""
        **Key Takeaways**
        - Prioritize shooting efficiency for maximum impact.
        - Minimize turnovers to sustain performance.
        - Defensive rating supports but does not drive alone.
        
        _Professional Edition | November 2025_
        """)

# Footer: Understated
st.markdown("---")
st.markdown('<p style="text-align: center; color: #6c757d; font-size: 0.9em;">NBA Analytics | Optimize Strategically</p>', unsafe_allow_html=True)
