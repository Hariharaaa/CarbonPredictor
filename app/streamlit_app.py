"""
Corporate Carbon Footprint Predictor — Streamlit Dashboard.

A modern, dark-themed prediction interface that takes company financial
and operational inputs and outputs:

- Predicted Scope 1 & 2 emissions (tCO₂e)
- Confidence intervals
- SHAP waterfall explanation
- Feature importance chart
- Model methodology info

Launch::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from src.predict import CarbonPredictor
from src.synthetic_data import SECTOR_PROFILES, INDUSTRY_MAP, COUNTRY_DATA

# ══════════════════════════════════════════════
#  Page config & custom styling
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="Carbon Footprint Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Premium Minimalist Dark Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 0%, #1a1a24 0%, #000000 70%);
        color: #f5f5f7;
    }
    
    .main .block-container {
        padding-top: 3rem;
        max-width: 1100px;
    }

    /* Minimal Glassmorphism Cards */
    .metric-card {
        background: rgba(30, 30, 35, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0.5rem 0;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .metric-sublabel {
        font-size: 0.8rem;
        color: #6e6e73;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Clean Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #f5f5f7;
        margin: 2.5rem 0 1.5rem 0;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-header::before {
        content: "";
        display: block;
        width: 4px;
        height: 16px;
        background: #0071e3;
        border-radius: 2px;
    }

    /* Refined Confidence bar */
    .confidence-bar {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 999px;
        height: 6px;
        overflow: hidden;
        margin: 0.5rem 0;
        flex: 1;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 1s cubic-bezier(0.22, 1, 0.36, 1);
    }

    /* Premium Main Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f5f5f7;
        text-align: center;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }
    .subtitle {
        text-align: center;
        color: #86868b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3.5rem;
        letter-spacing: -0.01em;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    /* Sleek Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 15, 18, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #f5f5f7;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: -0.01em;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        color: #f5f5f7 !important;
    }
    .stNumberInput > div > div > input:focus, .stSelectbox > div > div > div:focus {
        border-color: #0071e3 !important;
        box-shadow: 0 0 0 1px #0071e3 !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  Load model (cached)
# ══════════════════════════════════════════════

@st.cache_resource
def load_predictor() -> CarbonPredictor:
    """Load the prediction model (cached across reruns)."""
    return CarbonPredictor()


# ══════════════════════════════════════════════
#  Sidebar inputs
# ══════════════════════════════════════════════

def render_sidebar() -> dict:
    """Render sidebar input controls."""
    st.sidebar.markdown("## 🏢 Company Profile")

    sectors = sorted(SECTOR_PROFILES.keys())
    sector = st.sidebar.selectbox(
        "Sector",
        sectors,
        index=sectors.index("Industrials") if "Industrials" in sectors else 0,
        help="GICS sector classification",
    )

    industries = INDUSTRY_MAP.get(sector, ["General"])
    industry = st.sidebar.selectbox(
        "Industry",
        industries,
        help="Sub-industry classification",
    )

    countries = sorted(set(c["country"] for c in COUNTRY_DATA))
    country = st.sidebar.selectbox(
        "Country",
        countries,
        index=countries.index("USA") if "USA" in countries else 0,
        help="ISO 3166-1 alpha-3 code",
    )

    st.sidebar.markdown("## 💰 Financial Data")

    revenue = st.sidebar.number_input(
        "Annual Revenue (USD)",
        min_value=1_000,
        max_value=500_000_000_000,
        value=5_000_000_000,
        step=100_000_000,
        format="%d",
        help="Total annual revenue in USD",
    )

    total_assets = st.sidebar.number_input(
        "Total Assets (USD)",
        min_value=1_000,
        max_value=1_000_000_000_000,
        value=10_000_000_000,
        step=500_000_000,
        format="%d",
        help="Total assets on balance sheet",
    )

    market_cap = st.sidebar.number_input(
        "Market Cap (USD)",
        min_value=1_000,
        max_value=3_000_000_000_000,
        value=15_000_000_000,
        step=500_000_000,
        format="%d",
        help="Market capitalisation",
    )

    employee_count = st.sidebar.number_input(
        "Employee Count",
        min_value=10,
        max_value=3_000_000,
        value=25_000,
        step=1000,
        format="%d",
        help="Number of full-time employees",
    )

    # Country macro data lookup
    country_info = next(
        (c for c in COUNTRY_DATA if c["country"] == country),
        {"gdp_per_capita": 50000, "energy_intensity": 4.5},
    )

    return {
        "revenue": revenue,
        "total_assets": total_assets,
        "market_cap": market_cap,
        "employee_count": int(employee_count),
        "sector": sector,
        "industry": industry,
        "country": country,
        "gdp_per_capita": country_info["gdp_per_capita"],
        "energy_intensity": country_info["energy_intensity"],
    }


# ══════════════════════════════════════════════
#  Main content
# ══════════════════════════════════════════════

def render_header() -> None:
    """Render the main title."""
    st.markdown('<div class="main-title">🌍 Corporate Carbon Footprint Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">ML-powered Scope 1 & 2 GHG emissions estimation</div>',
        unsafe_allow_html=True,
    )


def render_prediction_cards(result) -> None:
    """Render the main prediction metric cards."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Scope 1 Emissions</div>
            <div class="metric-value">{result.scope1:,.0f}</div>
            <div class="metric-sublabel">tCO₂e (direct)</div>
            <div class="metric-sublabel">
                Range: {result.scope1_interval[0]:,.0f} – {result.scope1_interval[1]:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Scope 2 Emissions</div>
            <div class="metric-value">{result.scope2:,.0f}</div>
            <div class="metric-sublabel">tCO₂e (indirect energy)</div>
            <div class="metric-sublabel">
                Range: {result.scope2_interval[0]:,.0f} – {result.scope2_interval[1]:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Emissions</div>
            <div class="metric-value">{result.total:,.0f}</div>
            <div class="metric-sublabel">tCO₂e (Scope 1 + 2)</div>
            <div class="metric-sublabel">
                Confidence: {result.confidence_score * 100:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_confidence_bar(confidence: float) -> None:
    """Render a visual confidence indicator."""
    pct = confidence * 100
    color = (
        "#0071e3" if pct >= 70 else  /* Apple Blue */
        "#f56300" if pct >= 40 else  /* Subtle Orange */
        "#e30000"                    /* Subtle Red */
    )
    st.markdown(f"""
    <div class="section-header">Prediction Confidence</div>
    <div style="display: flex; align-items: center; gap: 1rem;">
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {pct}%; background: {color};"></div>
        </div>
        <span style="color: {color}; font-weight: 500; font-size: 1rem;">{pct:.0f}%</span>
    </div>
    """, unsafe_allow_html=True)


def render_shap_explanation(predictor: CarbonPredictor, features: dict) -> None:
    """Render SHAP waterfall for the prediction."""
    st.markdown('<div class="section-header">🔍 Prediction Explanation (SHAP)</div>', unsafe_allow_html=True)

    try:
        import shap
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        col1, col2 = st.columns(2)

        for i, target in enumerate(cfg.TARGETS):
            with [col1, col2][i]:
                explanation = predictor.get_shap_explanation(features, target)
                if explanation is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(explanation, show=False, max_display=10)
                    plt.title(f"{target.replace('_', ' ').title()}", fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info(f"SHAP not available for {target}")

    except ImportError:
        st.warning("SHAP library not installed — install with `pip install shap`")
    except Exception as exc:
        st.warning(f"SHAP visualization error: {exc}")


def render_feature_importance(predictor: CarbonPredictor) -> None:
    """Render feature importance bar chart from the model."""
    st.markdown('<div class="section-header">📊 Feature Importance</div>', unsafe_allow_html=True)

    for target in cfg.TARGETS:
        if target not in predictor.models:
            continue

        model = predictor.models[target]

        # Extract feature importance
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)

        if importance is None:
            continue

        feat_names = predictor.feature_names.get(target, [f"f{i}" for i in range(len(importance))])
        if len(feat_names) != len(importance):
            feat_names = [f"f{i}" for i in range(len(importance))]

        imp_df = pd.DataFrame({
            "feature": feat_names,
            "importance": importance,
        }).sort_values("importance", ascending=True).tail(15)

        fig = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker=dict(
                color=imp_df["importance"],
                colorscale="Viridis",
            ),
        ))
        fig.update_layout(
            title=f"Top Features — {target.replace('_', ' ').title()}",
            template="plotly_dark",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_emissions_breakdown(result) -> None:
    """Donut chart showing Scope 1 vs Scope 2 breakdown."""
    fig = go.Figure(go.Pie(
        values=[result.scope1, result.scope2],
        labels=["Scope 1 (Direct)", "Scope 2 (Indirect Energy)"],
        hole=0.6,
        marker=dict(colors=["#00c896", "#00b4d8"]),
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig.update_layout(
        title="Emissions Breakdown",
        template="plotly_dark",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        annotations=[dict(
            text=f"<b>{result.total:,.0f}</b><br>tCO₂e",
            x=0.5, y=0.5, font_size=16, showarrow=False,
            font_color="white",
        )],
    )
    return fig


def render_model_info(predictor: CarbonPredictor) -> None:
    """Render model metadata tab."""
    st.markdown('<div class="section-header">🤖 Model Information</div>', unsafe_allow_html=True)

    for target in cfg.TARGETS:
        meta = predictor.meta.get(target, {})
        if meta:
            col1, col2, col3 = st.columns(3)
            col1.metric("Model", meta.get("model_name", "N/A"))
            col2.metric("R² Score", f"{meta.get('r2', 0):.4f}")
            col3.metric("Target", target.replace("_", " ").title())


def render_methodology() -> None:
    """Render methodology information."""
    with st.expander("📖 Methodology", expanded=False):
        st.markdown("""
        ### Data Sources
        - **Emissions**: CDP Open Data (Scope 1 & 2 disclosure)
        - **Financials**: Yahoo Finance (revenue, assets, market cap, employees)
        - **Macro**: World Bank (GDP per capita, energy intensity)

        ### Feature Engineering
        - Log transforms on skewed financial variables
        - Financial ratios (asset/employee, revenue/employee, capital intensity)
        - Country-level macro indicators
        - Sector & industry encoding with group-level aggregation features

        ### Model
        - Best-performing model selected from: Linear Regression, Random Forest,
          Gradient Boosting, XGBoost, CatBoost, LightGBM
        - Hyperparameter tuning via RandomizedSearchCV
        - Sector-aware GroupKFold cross-validation
        - Target: log-transformed emissions (expm1 for final output)

        ### Confidence Interval
        - Based on model R² and prediction variance
        - Width proportional to prediction uncertainty

        ### Limitations
        - Predictions are estimates — actual emissions depend on many factors
          not captured in financial data (fuel mix, process type, etc.)
        - Model trained on voluntary CDP disclosures (selection bias possible)
        - Scope 3 emissions are not covered
        """)


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════

def main() -> None:
    """Main Streamlit application."""
    render_header()

    # Load predictor
    predictor = load_predictor()

    if not predictor.is_ready:
        st.error(
            "⚠️ No trained models found. Run the training pipeline first:\n\n"
            "```bash\n"
            "python -m src.synthetic_data\n"
            "python -m src.ingestion --synthetic\n"
            "python -m src.preprocessing\n"
            "python -m src.feature_engineering\n"
            "python -m src.train\n"
            "```"
        )
        st.stop()

    # Sidebar inputs
    inputs = render_sidebar()

    # Generate prediction
    result = predictor.predict(**inputs)

    # Main content
    render_prediction_cards(result)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence bar
    render_confidence_bar(result.confidence_score)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout for breakdown + feature importance
    col_left, col_right = st.columns([1, 2])

    with col_left:
        fig = render_emissions_breakdown(result)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        render_feature_importance(predictor)

    st.markdown("<br>", unsafe_allow_html=True)

    # SHAP explanation
    render_shap_explanation(predictor, result.feature_values)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model info
    render_model_info(predictor)

    # Methodology
    render_methodology()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem;'>"
        "Corporate Carbon Footprint Predictor v1.0 · Built with Streamlit & scikit-learn"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
