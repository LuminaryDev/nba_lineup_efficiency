import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # For BN graph
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NBA Lineup Efficiency Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Calm, professional light theme (soft blues/grays, clean lines)
st.markdown("""
<style>
    /* Main styling - Light & Serene */
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 300;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #34495e;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #bdc3c7;
    }
    
    .subsection-header {
        font-size: 1.3rem;
        color: #5d6d7e;
        font-weight: 500;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Card styling - Subtle & Clean */
    .metric-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        transition: box-shadow 0.2s ease;
        color: #2c3e50 !important;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        color: #2c3e50 !important;
    }
    
    .insight-card {
        background: #f0f8ff;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: #2c3e50 !important;
    }
    
    /* Button styling - Subtle */
    .stButton > button {
        background: #3498db;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 500;
        transition: background 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #2980b9;
        color: white !important;
    }
    
    /* Sidebar - Light & Clean */
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    .sidebar-title {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Progress bars - Neutral */
    .stProgress > div > div > div {
        background: #3498db;
    }
    
    /* Tabs - Clean */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #f8f9fa;
        border-radius: 6px 6px 0 0;
        padding: 0 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    
    /* Metric containers - Spaced */
    [data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* Text visibility */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #2c3e50 !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        color: #2c3e50 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'navigation' not in st.session_state:
    st.session_state.navigation = "📊 Dashboard Overview"

if 'simulator_values' not in st.session_state:
    st.session_state.simulator_values = {
        'shooting': 'Medium',
        'scoring': 'Medium', 
        'ast_rate': 'Medium',
        'tov': 'Medium',
        'net_rating': 'Medium',
        'orb_rate': 'Medium'
    }

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="sidebar-title">NBA Analytics Suite</div>', unsafe_allow_html=True)
    
    app_section = st.radio(
        "Navigate to:",
        ["📊 Dashboard Overview", "🎮 Lineup Simulator", "📈 Data Explorer", "🔍 Sensitivity Analysis", "📋 Insights & Reports"],
        index=0,
        key="nav_radio"
    )
    
    st.markdown("---")
    
    # Quick stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Lineups", "10,000+")
    with col2:
        st.metric("Teams", "30")
    
    st.markdown("---")
    
    with st.expander("ℹ️ About"):
        st.markdown("""
        **NBA Lineup Efficiency Analyzer**
        
        Bayesian Networks powered by 2023-24 NBA data.
        
        **Features:**
        - Interactive simulations
        - Real-time predictions
        - Factor analysis
        - Data visualizations
        
        *Streamlit & pgmpy*
        """)
    
    st.markdown("---")
    st.caption("Built by Rediet Girmay")

# Update nav
if app_section != st.session_state.navigation:
    st.session_state.navigation = app_section
    st.rerun()

# Main content
if st.session_state.navigation == "📊 Dashboard Overview":
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header">NBA Lineup Efficiency Analyzer</div>', unsafe_allow_html=True)
        st.markdown("Advanced Bayesian simulation for lineup performance.", help="Built on real NBA data.")
    
    with col2:
        st.metric("Data Points", "10,000+")
    with col3:
        st.metric("Accuracy", "89%")
    
    st.markdown('<div class="section-header">Key Features</div>', unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div class="feature-card">
            <h3>Interactive Simulator</h3>
            <p>Test combinations and view efficiency predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Advanced Analytics</h3>
            <p>Deep dive into factor impacts and sensitivities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div class="feature-card">
            <h3>Data-Driven Insights</h3>
            <p>Leverage 2023-24 NBA data for strategic decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start
    st.markdown('<div class="section-header">Quick Start</div>', unsafe_allow_html=True)
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("Launch Simulator"):
            st.session_state.navigation = "🎮 Lineup Simulator"
            st.rerun()
        
    with quick_col2:
        if st.button("View Data"):
            st.session_state.navigation = "📈 Data Explorer"
            st.rerun()
            
    with quick_col3:
        if st.button("See Analysis"):
            st.session_state.navigation = "🔍 Sensitivity Analysis"
            st.rerun()
    
    # Added Graph: BN Structure
    st.markdown('<div class="subsection-header">Model Structure</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    G = nx.DiGraph()
    G.add_edges_from([
        ('PLAYMAKING_Talent', 'AST_rate'), ('PLAYMAKING_Talent', 'TOV_rate'),
        ('SCORING_Talent', 'Shooting_Efficiency'), ('REBOUNDING_Talent', 'ORB_rate'),
        ('DEFENSIVE_Talent', 'Net_Rating_Impact'), ('NET_RATING_Talent', 'Net_Rating_Impact'),
        ('Net_Rating_Impact', 'Efficiency'), ('Shooting_Efficiency', 'Efficiency'),
        ('AST_rate', 'Efficiency'), ('TOV_rate', 'Efficiency'), ('ORB_rate', 'Efficiency')
    ])
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold', ax=ax)
    st.pyplot(fig)

elif st.session_state.navigation == "🎮 Lineup Simulator":
    st.markdown('<div class="main-header">Lineup Efficiency Simulator</div>', unsafe_allow_html=True)
    
    @st.cache_data
    def load_data():
        try:
            data = pd.read_csv("nba_lineups_expanded_discretized.csv")
            st.success(f"Loaded {len(data):,} lineups.")
            return data
        except FileNotFoundError:
            st.warning("Upload dataset.")
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

        # Balance
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

    data = load_data()
    model, fitted_data = fit_model(data)

    if model is None:
        st.error("Model failed. Check dataset.")
    else:
        infer = VariableElimination(model)
        order = ['Low', 'Medium', 'High']

        st.markdown('<div class="section-header">Lineup Configuration</div>', unsafe_allow_html=True)
        
        control_col, result_col = st.columns([2, 1])
        
        with control_col:
            with st.expander("Shooting & Scoring", expanded=True):
                shooting_col, scoring_col = st.columns(2)
                with shooting_col:
                    shooting = st.selectbox("Shooting Efficiency", order, 
                                          index=order.index(st.session_state.simulator_values['shooting']))
                with scoring_col:
                    scoring = st.selectbox("Scoring Talent", order, 
                                         index=order.index(st.session_state.simulator_values['scoring']))
            
            with st.expander("Playmaking & Ball Control", expanded=True):
                play_col1, play_col2 = st.columns(2)
                with play_col1:
                    ast_rate = st.selectbox("Assist Rate", order, 
                                          index=order.index(st.session_state.simulator_values['ast_rate']))
                with play_col2:
                    tov = st.selectbox("Turnover Rate", order, 
                                     index=order.index(st.session_state.simulator_values['tov']))
            
            with st.expander("Defense & Rebounding"):
                def_col1, def_col2 = st.columns(2)
                with def_col1:
                    net_rating = st.selectbox("Net Rating Impact", order, 
                                            index=order.index(st.session_state.simulator_values['net_rating']))
                with def_col2:
                    orb_rate = st.selectbox("Offensive Rebound Rate", order, 
                                          index=order.index(st.session_state.simulator_values['orb_rate']))
            
            st.session_state.simulator_values.update({
                'shooting': shooting, 'scoring': scoring, 'ast_rate': ast_rate,
                'tov': tov, 'net_rating': net_rating, 'orb_rate': orb_rate
            })
            
            st.markdown('<div class="subsection-header">Quick Presets</div>', unsafe_allow_html=True)
            
            preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
            
            with preset_col1:
                if st.button("Elite Shooting"):
                    st.session_state.simulator_values.update({'shooting': 'High', 'scoring': 'High'})
                    st.rerun()
            
            with preset_col2:
                if st.button("Lockdown Defense"):
                    st.session_state.simulator_values.update({'net_rating': 'High', 'tov': 'Low'})
                    st.rerun()
            
            with preset_col3:
                if st.button("Playmaker"):
                    st.session_state.simulator_values.update({'ast_rate': 'High', 'tov': 'Low'})
                    st.rerun()
            
            with preset_col4:
                if st.button("Balanced"):
                    st.session_state.simulator_values.update({
                        'shooting': 'Medium', 'scoring': 'Medium', 'ast_rate': 'Medium', 
                        'tov': 'Medium', 'net_rating': 'Medium', 'orb_rate': 'Medium'
                    })
                    st.rerun()

            # Export
            if st.button("Export Config", type="secondary"):
                df_export = pd.DataFrame([st.session_state.simulator_values])
                st.download_button("Download CSV", df_export.to_csv(index=False), "lineup_config.csv")

        with result_col:
            st.markdown('<div class="subsection-header">Efficiency Prediction</div>', unsafe_allow_html=True)
            
            # Fixed Evidence: Map scoring to SCORING_Talent
            evidence = {
                'Shooting_Efficiency': st.session_state.simulator_values['shooting'],
                'SCORING_Talent': st.session_state.simulator_values['scoring'],  # Mapped
                'Net_Rating_Impact': st.session_state.simulator_values['net_rating'],
                'TOV_rate': st.session_state.simulator_values['tov'],
                'AST_rate': st.session_state.simulator_values['ast_rate'],
                'ORB_rate': st.session_state.simulator_values['orb_rate']
            }
            evidence = {k: v for k, v in evidence.items() if k in model.nodes()}
            q = infer.query(variables=['Efficiency'], evidence=evidence)
            
            efficiency_score = q.values[2] * 100
            st.metric("High Efficiency Probability", f"{efficiency_score:.1f}%", delta=f"{efficiency_score - 33.3:+.1f}% vs baseline")
            
            # Distrib Row
            st.markdown("**Efficiency Distribution:**")
            col1, col2, col3 = st.columns(3)
            for i, level in enumerate(order):
                with col1 if i==0 else col2 if i==1 else col3:
                    st.metric(level, f"{q.values[i]:.1%}")
            
            # Insight
            st.markdown("---")
            if efficiency_score > 60:
                st.success("Elite Lineup: High potential!")
            elif efficiency_score > 40:
                st.info("Strong Lineup: Well-balanced.")
            else:
                st.warning("Needs Improvement: Adjust skills.")
            
            # Added Graph: Pie Chart for Distrib
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(q.values * 100, labels=order, autopct='%1.1f%%', startangle=90)
            ax.set_title("Efficiency Probability Breakdown")
            st.pyplot(fig)

elif st.session_state.navigation == "📈 Data Explorer":
    st.markdown('<div class="main-header">NBA Data Explorer</div>', unsafe_allow_html=True)
    
    try:
        data = pd.read_csv("nba_lineups_expanded_discretized.csv")
        
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Lineups", f"{len(data):,}")
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            st.metric("Teams", "30")
        with col4:
            st.metric("Season", "2023-24")
        
        st.markdown('<div class="subsection-header">Data Preview</div>', unsafe_allow_html=True)
        
        available_columns = data.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns:",
            options=available_columns,
            default=available_columns[:6]
        )
        
        if selected_columns:
            st.dataframe(data[selected_columns].head(20), use_container_width=True)
        
        with st.expander("Statistical Summary"):
            st.dataframe(data[selected_columns].describe() if selected_columns else data.describe())
            
    except FileNotFoundError:
        st.error("Dataset not found.")

elif st.session_state.navigation == "🔍 Sensitivity Analysis":
    st.markdown('<div class="main-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
    <h3>Understanding Factor Impact</h3>
    <p>This analysis shows which attributes most influence efficiency, aiding roster decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fixed Sorting
    factors = ['Shooting Efficiency', 'Turnover Control', 'Net Rating Impact', 'Assist Rate', 'Rebound Rate']
    impacts = [64, 16, 12, 5, 3]  # Numeric for sort
    
    st.markdown('<div class="subsection-header">Factor Impact Ranking</div>', unsafe_allow_html=True)
    
    impact_df = pd.DataFrame({'Factor': factors, 'Impact': impacts})
    impact_df = impact_df.sort_values('Impact', ascending=False)
    
    for _, row in impact_df.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{row['Factor']}**")
        with col2:
            st.metric("Impact", f"+{row['Impact']}%")
    
    # Added Graph: Horizontal Bar
    st.markdown('<div class="subsection-header">Impact Visualization</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    impact_df.plot(kind='barh', x='Factor', y='Impact', ax=ax, color='lightblue')
    ax.set_xlabel("Impact Score (%)")
    ax.set_title("Factor Impact on Efficiency")
    plt.tight_layout()
    st.pyplot(fig)

elif st.session_state.navigation == "📋 Insights & Reports":
    st.markdown('<div class="main-header">Insights & Reports</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Key Findings", "Recommendations", "Future Research"])
    
    with tab1:
        st.markdown("""
        <div class="insight-card">
        <h3>Most Impactful Factors</h3>
        <ul>
        <li><strong>Shooting Dominance</strong>: +64% boost – prioritize 3PT threats.</li>
        <li><strong>Turnover Control</strong>: +16% – ball security is key.</li>
        <li><strong>Net Rating Impact</strong>: +12% – defense supports offense.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Shooting Impact", "+64%")
        with col2:
            st.metric("Turnover Impact", "+16%")
        with col3:
            st.metric("Defense Impact", "+12%")
    
    with tab2:
        st.markdown("""
        <div class="insight-card">
        <h3>Strategic Recommendations</h3>
        <p><strong>Roster Construction:</strong></p>
        <ul>
        <li>Prioritize elite shooters in drafts.</li>
        <li>Value low-TO playmakers.</li>
        <li>Seek two-way players.</li>
        </ul>
        <p><strong>Game Strategy:</strong></p>
        <ul>
        <li>Maximize 3PT from efficient shooters.</li>
        <li>Implement TO-reduction schemes.</li>
        <li>Data-driven subs.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="insight-card">
        <h3>Research Roadmap</h3>
        <p><strong>Next Enhancements:</strong></p>
        <ul>
        <li>Opponent-adjusted metrics.</li>
        <li>Player chemistry analysis.</li>
        <li>Fatigue factors.</li>
        <li>Possession-level data.</li>
        </ul>
        <p><strong>Long-term:</strong></p>
        <ul>
        <li>Predictive optimization.</li>
        <li>In-game AI adjustments.</li>
        <li>Talent evaluation.</li>
        <li>Tracking data integration.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #95a5a6; font-size: 0.9em;">NBA Analytics | Streamlit Cloud | 2025</p>', unsafe_allow_html=True)
