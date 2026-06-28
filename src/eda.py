"""
Exploratory Data Analysis module — publication-quality Plotly charts.

Generates interactive HTML charts and static PNGs for reports.

All charts are saved to ``reports/figures/`` and can also be rendered
inline in Jupyter via the companion notebook ``01_EDA.ipynb``.

Usage::

    python -m src.eda                  # reads clean.csv, writes figures
    python -m src.eda --input path.csv # custom input
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import get_logger, load_csv

logger = get_logger(__name__)

# Consistent theme
_TEMPLATE = "plotly_dark"
_COLOR_SEQ = px.colors.qualitative.Vivid
_CONTINUOUS = "Viridis"


def _save(fig: go.Figure, name: str) -> None:
    """Save a figure as interactive HTML and static PNG."""
    html_path = cfg.FIGURES_DIR / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        png_path = cfg.FIGURES_DIR / f"{name}.png"
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
    except Exception:
        logger.warning("PNG export failed for %s (kaleido missing?)", name)
    logger.info("Saved figure: %s", name)


# ══════════════════════════════════════════════
#  Individual chart generators
# ══════════════════════════════════════════════

def plot_emissions_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram + KDE of Scope 1 and Scope 2 emissions (log scale)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Scope 1 Emissions (log)", "Scope 2 Emissions (log)"],
    )
    for i, target in enumerate(cfg.TARGETS, 1):
        log_col = f"log_{target}"
        if log_col not in df.columns:
            log_col = target
        fig.add_trace(
            go.Histogram(
                x=df[log_col].dropna(),
                nbinsx=50,
                marker_color=_COLOR_SEQ[i - 1],
                opacity=0.75,
                name=target,
            ),
            row=1, col=i,
        )
    fig.update_layout(
        title="Distribution of GHG Emissions",
        template=_TEMPLATE,
        showlegend=False,
        height=500,
    )
    _save(fig, "emissions_distribution")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of numeric feature correlations."""
    numeric = df.select_dtypes(include=[np.number])
    # Keep only meaningful columns (drop missingness flags)
    cols = [c for c in numeric.columns if not c.endswith("_missing")]
    corr = numeric[cols].corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=_CONTINUOUS,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template=_TEMPLATE,
        height=800,
        width=900,
    )
    _save(fig, "correlation_heatmap")
    return fig


def plot_sector_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of mean Scope 1 & 2 by sector."""
    if "sector" not in df.columns:
        logger.warning("No 'sector' column – skipping sector comparison")
        return go.Figure()

    agg = (
        df.groupby("sector")[cfg.TARGETS]
        .median()
        .reset_index()
        .sort_values(cfg.TARGET_SCOPE1, ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=agg["sector"], x=agg[cfg.TARGET_SCOPE1],
        name="Scope 1", orientation="h",
        marker_color=_COLOR_SEQ[0],
    ))
    fig.add_trace(go.Bar(
        y=agg["sector"], x=agg[cfg.TARGET_SCOPE2],
        name="Scope 2", orientation="h",
        marker_color=_COLOR_SEQ[1],
    ))
    fig.update_layout(
        title="Median Emissions by Sector",
        barmode="group",
        template=_TEMPLATE,
        height=600,
        xaxis_title="Emissions (tCO₂e)",
        yaxis_title="Sector",
    )
    _save(fig, "sector_comparison")
    return fig


def plot_revenue_vs_emissions(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: Revenue vs Scope 1, colored by sector."""
    fig = px.scatter(
        df,
        x="revenue",
        y=cfg.TARGET_SCOPE1,
        color="sector" if "sector" in df.columns else None,
        log_x=True,
        log_y=True,
        opacity=0.6,
        color_discrete_sequence=_COLOR_SEQ,
        title="Revenue vs Scope 1 Emissions",
        labels={"revenue": "Revenue ($)", cfg.TARGET_SCOPE1: "Scope 1 (tCO₂e)"},
        template=_TEMPLATE,
        height=600,
    )
    _save(fig, "revenue_vs_scope1")
    return fig


def plot_assets_vs_emissions(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: Total Assets vs Scope 2, colored by sector."""
    fig = px.scatter(
        df,
        x="total_assets",
        y=cfg.TARGET_SCOPE2,
        color="sector" if "sector" in df.columns else None,
        log_x=True,
        log_y=True,
        opacity=0.6,
        color_discrete_sequence=_COLOR_SEQ,
        title="Total Assets vs Scope 2 Emissions",
        labels={"total_assets": "Total Assets ($)", cfg.TARGET_SCOPE2: "Scope 2 (tCO₂e)"},
        template=_TEMPLATE,
        height=600,
    )
    _save(fig, "assets_vs_scope2")
    return fig


def plot_boxplots_by_sector(df: pd.DataFrame) -> go.Figure:
    """Box plots of emissions by sector."""
    if "sector" not in df.columns:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Scope 1", "Scope 2"])

    for i, target in enumerate(cfg.TARGETS, 1):
        for j, sector in enumerate(sorted(df["sector"].unique())):
            subset = df[df["sector"] == sector][target].dropna()
            fig.add_trace(
                go.Box(
                    y=np.log1p(subset),
                    name=sector,
                    marker_color=_COLOR_SEQ[j % len(_COLOR_SEQ)],
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )

    fig.update_layout(
        title="Emissions Distribution by Sector (log scale)",
        template=_TEMPLATE,
        height=600,
    )
    _save(fig, "boxplots_by_sector")
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Heatmap showing missingness pattern across all columns."""
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        logger.info("No missing values to plot")
        # Create a simple informational figure
        fig = go.Figure()
        fig.add_annotation(text="No missing values detected", showarrow=False,
                           font=dict(size=20))
        fig.update_layout(title="Missing Values", template=_TEMPLATE)
        _save(fig, "missing_values")
        return fig

    fig = go.Figure(go.Bar(
        x=missing_pct.values,
        y=missing_pct.index.tolist(),
        orientation="h",
        marker_color=_COLOR_SEQ[3],
        text=[f"{v:.1f}%" for v in missing_pct.values],
        textposition="auto",
    ))
    fig.update_layout(
        title="Missing Values by Column (%)",
        template=_TEMPLATE,
        height=max(400, len(missing_pct) * 30),
        xaxis_title="Missing (%)",
        yaxis_title="Column",
    )
    _save(fig, "missing_values")
    return fig


def plot_feature_importance_preview(df: pd.DataFrame) -> go.Figure:
    """Mutual information scores as a quick feature-importance preview.

    Uses sklearn's ``mutual_info_regression`` — no model required.
    """
    from sklearn.feature_selection import mutual_info_regression

    target = cfg.TARGET_SCOPE1
    if target not in df.columns:
        return go.Figure()

    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in cfg.TARGETS + [f"log_{t}" for t in cfg.TARGETS]
                 if c in df.columns],
        errors="ignore",
    )
    # Drop columns with zero variance or all NaN
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.std() > 0]

    valid = numeric.dropna()
    if len(valid) < 20:
        logger.warning("Too few valid rows for MI computation")
        return go.Figure()

    y = df.loc[valid.index, target]
    mi = mutual_info_regression(valid, y, random_state=cfg.RANDOM_STATE)
    mi_series = pd.Series(mi, index=valid.columns).sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=mi_series.values,
        y=mi_series.index.tolist(),
        orientation="h",
        marker_color=_COLOR_SEQ[2],
    ))
    fig.update_layout(
        title="Feature Importance Preview (Mutual Information → Scope 1)",
        template=_TEMPLATE,
        height=max(400, len(mi_series) * 25),
        xaxis_title="Mutual Information Score",
    )
    _save(fig, "feature_importance_preview")
    return fig


def plot_pairplot(df: pd.DataFrame) -> go.Figure:
    """Scatter matrix of top numeric features vs Scope 1."""
    top_cols = ["log_revenue", "log_assets", "log_employees", "log_market_cap"]
    available = [c for c in top_cols if c in df.columns]
    if not available or cfg.TARGET_SCOPE1 not in df.columns:
        return go.Figure()

    plot_cols = available + [cfg.TARGET_SCOPE1]
    sample = df[plot_cols].dropna()
    if len(sample) > 300:
        sample = sample.sample(300, random_state=cfg.RANDOM_STATE)

    fig = px.scatter_matrix(
        sample,
        dimensions=plot_cols,
        color=None,
        opacity=0.5,
        title="Pair Plot — Key Features",
        template=_TEMPLATE,
        height=900,
        width=900,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
    _save(fig, "pairplot")
    return fig


# ══════════════════════════════════════════════
#  Run all
# ══════════════════════════════════════════════

def run_eda(input_path: Optional[Path] = None) -> None:
    """Generate all EDA charts.

    Args:
        input_path: Path to clean CSV. Defaults to ``config.CLEAN_FILE``.
    """
    input_path = input_path or cfg.CLEAN_FILE
    df = load_csv(input_path)
    logger.info("── Running EDA on %d rows ──", len(df))

    plot_emissions_distribution(df)
    plot_correlation_heatmap(df)
    plot_sector_comparison(df)
    plot_revenue_vs_emissions(df)
    plot_assets_vs_emissions(df)
    plot_boxplots_by_sector(df)
    plot_missing_values(df)
    plot_feature_importance_preview(df)
    plot_pairplot(df)

    logger.info("✓ All EDA figures saved to %s", cfg.FIGURES_DIR)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run EDA charts")
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()
    run_eda(Path(args.input) if args.input else None)


if __name__ == "__main__":
    main()
