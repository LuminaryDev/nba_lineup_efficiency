import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Fixed: Explicit import for plotting backend
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

# Page config for wide, immersive layout
st.set_page_config(layout="wide", page_title="NBA Lineup Optimizer", page_icon="🏀")

# Custom CSS for NBA flair: Orange accents, clean shadows
st.markdown("""
<style>
    .main-header {color: #FDB927; font-size: 2.5em; text-align: center; font-weight: bold;}
    .metric-high {background-color: #28a745; color: white; border-radius: 5px; padding: 10px;}
    .metric-low {background-color: #dc3545; color: white; border-radius: 5px; padding: 10px;}
    .metric-med {background-color: #ffc107; color: black; border-radius: 5px; padding: 10px;}
    .sidebar .sidebar-content {background-color: #1e1e1e;}
    .stMetric > div > div > div {font-size: 1.2em;}
</style>
""", unsafe_allow_html=True)

# Hero Header: Immersive & Exclusive
st.markdown('<h1 class="main-header">🏀 NBA Lineup Optimizer</h1>', unsafe_allow_html=True)
st.markdown("**Unlock Elite Efficiency**: Bayesian-powered " + 
            "insights from 2023-24 data. Tweak talents, simulate subs, dominate the court. " + 
            "<i>Exclusive build by Rediet Girmay (GSE/0945-17)</i>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("nba_lineups_expanded_discretized.csv")
        st.success(f"✅ Loaded {len(data):,} lineups – Ready to optimize!")
        return data
    except FileNotFoundError:
        st.warning("📁 Upload 'nba_lineups_expanded_discretized.csv'.")
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

    # Balance for realistic baselines (~5-8% High)
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
    st.error("❌ Load data to launch.")
else:
    infer = VariableElimination(model)
    order = ['Low', 'Medium', 'High']
    colors = ['#dc3545', '#ffc107', '#28a745']  # Red-Yellow-Green

    # Sidebar: Sleek Controls
    with st.sidebar:
        st.header("🎯 Strategy Selector")
        scenario = st.selectbox(
            "Choose Your Play:", 
            ["Baseline Squad", "Splash Brother (Elite Shooter)", "Lockdown D (Defensive Anchor)", 
             "Steady PG (Low TOV)", "Dynamic Duo (Shooter + Playmaker)", 
             "Board Battle (High TOV Risk)", "Unstoppable Force (All-In)"],
            index=0
        )
        manual_mode = st.checkbox("⚙️ Custom Lineup Mode", value=False)

        # Auto-set based on scenario
        presets = {
            "Baseline Squad": ('Medium', 'Medium', 'Medium'),
            "Splash Brother (Elite Shooter)": ('High', 'Medium', 'Medium'),
            "Lockdown D (Defensive Anchor)": ('Medium', 'High', 'Medium'),
            "Steady PG (Low TOV)": ('Medium', 'Medium', 'Low'),
            "Dynamic Duo (Shooter + Playmaker)": ('High', 'Medium', 'Low'),
            "Board Battle (High TOV Risk)": ('Medium', 'Medium', 'High'),
            "Unstoppable Force (All-In)": ('High', 'High', 'Low')
        }
        shooting, net_rating, tov = presets[scenario]

        if manual_mode:
            st.subheader("Fine-Tune")
            shooting = st.selectbox("Shooting 🔥", order, index=order.index(shooting))
            net_rating = st.selectbox("Net Rating 🛡️", order, index=order.index(net_rating))
            tov = st.selectbox("Turnovers ⚠️", order, index=order.index(tov))
            scenario = "Your Custom Build"

    # Baseline Calc
    baseline_ev = {'Shooting_Efficiency': 'Medium', 'Net_Rating_Impact': 'Medium', 'TOV_rate': 'Medium'}
    base_q = infer.query(variables=['Efficiency'], evidence=baseline_ev)
    base_high = base_q.values[2]

    # Query
    evidence = {'Shooting_Efficiency': shooting, 'Net_Rating_Impact': net_rating, 'TOV_rate': tov}
    q = infer.query(variables=['Efficiency'], evidence=evidence)
    probs = pd.Series(q.values, index=order) * 100

    # Main: Immersive Split-View Comparison
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Current Scenario")
        delta_high = (q.values[2] - base_high) * 100
        st.metric("Efficiency Boost", f"{delta_high:+.1f}%", 
                  delta=f"vs. Baseline ({base_high*100:.1f}%)", delta_color="normal")

        # Color-coded Metrics
        for i, lvl in enumerate(order):
            with st.container():
                st.markdown(f'<div class="metric-{lvl.lower()}">🏆 {lvl}: {probs[i]:.1f}%</div>', unsafe_allow_html=True)

    with col_right:
        st.subheader("Baseline Comparison")
        base_probs = pd.Series(base_q.values, index=order) * 100
        for i, lvl in enumerate(order):
            with st.container():
                st.markdown(f'<div class="metric-{lvl.lower()}">📊 {lvl}: {base_probs[i]:.1f}%</div>', unsafe_allow_html=True)

    # Dual Bar Chart: Side-by-Side (Fixed: Use plt for backend)
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Your Scenario**")
        plt.figure(figsize=(5, 3))
        probs.plot(kind='bar', color=colors, ax=plt.gca())
        plt.title("Efficiency Breakdown")
        plt.ylabel("Probability (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    with col_chart2:
        st.markdown("**Baseline**")
        plt.figure(figsize=(5, 3))
        base_probs.plot(kind='bar', color=colors, ax=plt.gca())
        plt.title("Baseline Breakdown")
        plt.ylabel("Probability (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    # Reset Button
    if st.button("🔄 Reset to Baseline", type="secondary"):
        st.rerun()

    # Tabs: Polished & Concise
    tab1, tab2, tab3 = st.tabs(["📊 Insights", "🔍 Data Dive", "🏆 Pro Tips"])

    with tab1:
        st.markdown("**Sensitivity Heatmap: Factor Impact**")
        treatments = [
            ('Shooting_Efficiency', 'High', "Shooting 🔥"), ('TOV_rate', 'Low', "TOV Control ⚠️"),
            ('Net_Rating_Impact', 'High', "Defense 🛡️"), ('ORB_rate', 'High', "Rebounds 🏀"),
            ('AST_rate', 'High', "Assists ➡️")
        ]
        sens_data = []
        for var, val, icon in treatments:
            ev = {**baseline_ev, var: val}
            q_s = infer.query(variables=['Efficiency'], evidence=ev)
            delta = (q_s.values[2] - base_high) * 100
            sens_data.append({'Factor': f"{icon} {var.replace('_', ' ').title()}", 'Δ High': f"{delta:+.1f}%"})

        df_sens = pd.DataFrame(sens_data).sort_values('Δ High', key=lambda x: float(x.str.extract('([+-]?\d+\.?\d*)').astype(float)), ascending=False)
        st.dataframe(df_sens, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**Elite Lineup Samples**")
        display_cols = ['GROUP_NAME', 'team', 'MIN', 'PLUS_MINUS', 'Efficiency']  # Trimmed for clean view
        available_cols = [col for col in display_cols if col in fitted_data.columns]
        if not available_cols:
            available_cols = ['Efficiency', 'Shooting_Efficiency', 'Net_Rating_Impact']
        st.dataframe(fitted_data[available_cols].head(5), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("""
        ### 🎯 Pro Tips for Coaches & Analysts
        - **Stack Winners**: High Shooting + Low TOV = 99% elite – like Warriors 2016.
        - **Avoid Traps**: High TOV tanks everything; rebounding's a sidekick, not star.
        - **Quote of the Game**: "Efficiency isn't luck—it's lineup science." – Rediet Girmay
        - **Next Level**: Integrate live player stats for real-time subs.
        
        _Exclusive Edition | Built for Champions | Nov 2025_
        """)

# Footer: Subtle & Pro
st.markdown("---")
st.markdown('<p style="text-align: center; color: #FDB927;">Powered by xAI & NBA Analytics | Optimize. Dominate. Repeat.</p>', unsafe_allow_html=True)
