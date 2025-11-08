import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

st.title("🏀 NBA Lineup Efficiency: Bayesian Network Simulator")
st.markdown("**Interactive Demo**: Tweak player skills & see efficiency impact. Built from real 2023-24 NBA data.")

@st.cache_data
def load_data():
    # Try loading your processed CSV (upload via Streamlit if needed)
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
        data[col] = pd.Categorical(data[col], categories=order, ordered=True)

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

    model = BayesianNetwork(edges)
    model.fit(data, estimator=BayesianEstimator,
              state_names={col: order for col in all_cols},
              equivalent_sample_size=10)
    return model, data

# Load & Fit
data = load_data()
model, data = fit_model(data)

if model is None:
    st.error("❌ Model not ready. Upload data or add data-gen code.")
    # Optional: Add your Phase 1 data-gen here if no CSV (nba_api works in cloud)
    if st.button("Generate Sample Data (Slow - NBA API)"):
        st.code("# Paste your Phase 1 code here")
else:
    infer = VariableElimination(model)
    order = ['Low', 'Medium', 'High']

    # Interactive Controls (Phase 4 Scenarios)
    st.header("🎯 Test Lineup Tweaks")
    col1, col2, col3 = st.columns(3)
    with col1:
        shooting = st.selectbox("Shooting Efficiency", order, index=1)
    with col2:
        net_rating = st.selectbox("Net Rating Impact", order, index=1)
    with col3:
        tov = st.selectbox("Turnover Rate", order, index=1)

    evidence = {'Shooting_Efficiency': shooting, 'Net_Rating_Impact': net_rating, 'TOV_rate': tov}
    q = infer.query(variables=['Efficiency'], evidence=evidence)

    st.metric("P(High Efficiency)", f"{q.values[2]:.1%}", delta=None)
    probs = pd.Series(q.values, index=order)
    st.bar_chart(probs * 100, use_container_width=True)  # % scale

    # Pre-Built Scenarios (Phase 4.1-4.3)
    st.subheader("Quick Tests")
    if st.button("🏹 Elite 3PT Shooter (High Shooting)"):
        evidence = {'Shooting_Efficiency': 'High', 'Net_Rating_Impact': 'Medium'}
        q = infer.query(variables=['Efficiency'], evidence=evidence)
        st.metric("P(High)", f"{q.values[2]:.1%}")
        st.bar_chart(pd.Series(q.values, index=order) * 100)

    if st.button("🛡️ Elite Defender (High Net Rating)"):
        evidence = {'Net_Rating_Impact': 'High', 'Shooting_Efficiency': 'Medium'}
        q = infer.query(variables=['Efficiency'], evidence=evidence)
        st.metric("P(High)", f"{q.values[2]:.1%}")
        st.bar_chart(pd.Series(q.values, index=order) * 100)

    # Tabs for Deeper Views (Phases 2-5)
    tab1, tab2, tab3 = st.tabs(["📈 Sensitivity Ranking", "📊 Sample Data", "📝 Conclusions"])

    with tab1:
        st.markdown("**Phase 4.4: Which Factor Boosts Efficiency Most?**")
        baseline_ev = {'Shooting_Efficiency': 'Medium', 'Net_Rating_Impact': 'Medium'}
        base_q = infer.query(variables=['Efficiency'], evidence=baseline_ev)
        base_high = base_q.values[2]
        st.info(f"Baseline P(High): {base_high:.1%}")

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
            delta = q_s.values[2] - base_high
            sens_data.append({'Factor': label, 'Δ P(High)': f"{delta:+.1%}"})

        df_sens = pd.DataFrame(sens_data).sort_values('Δ P(High)', key=lambda x: x.str.rstrip('%').astype('float'), ascending=False)
        st.table(df_sens)

    with tab2:
        st.markdown("**Phase 1-2: Real NBA Lineup Sample**")
        st.dataframe(data[['GROUP_NAME', 'team', 'MIN', 'PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'Efficiency']].head(10))

    with tab3:
        st.markdown("""
        ### Key Insights (Phase 6)
        - **Shooting Dominates**: +64% boost to high efficiency – prioritize 3PT threats!
        - **Turnover Control**: Next biggest lever (+16%).
        - **Recommendations**: Shooter + Playmaker = Elite lineup.
        - **Limitations**: Lineup-level only; add possession data next.
        
        _Built by Rediet Girmay | Oct 2025_
        """)

# Footer
st.markdown("---")
st.caption("Deployed via Streamlit Cloud | Source: NBA API 2023-24")
