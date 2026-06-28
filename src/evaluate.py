"""
Model evaluation module for the Corporate Carbon Footprint Predictor.

Computes R², RMSE, MAE, MAPE on train/test splits and generates
publication-quality diagnostic plots:

- Model comparison bar charts
- Actual vs Predicted scatter
- Residual plots
- Cross-validation box plots
- Sector-stratified performance breakdown

Usage::

    python -m src.evaluate   # evaluates best models on test sets
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import get_logger, save_csv, load_csv

logger = get_logger(__name__)

_TEMPLATE = "plotly_dark"
_COLORS = px.colors.qualitative.Vivid


# ══════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        prefix: Optional prefix for metric keys.

    Returns:
        Dict with r2, rmse, mae, mape.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Avoid MAPE division-by-zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    else:
        mape = np.nan

    return {
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}mape": float(mape),
    }


# ══════════════════════════════════════════════
#  Diagnostic plots
# ══════════════════════════════════════════════

def _save(fig: go.Figure, name: str) -> None:
    """Save figure as HTML + PNG."""
    html_path = cfg.FIGURES_DIR / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(cfg.FIGURES_DIR / f"{name}.png"), width=1200, height=700, scale=2)
    except Exception:
        pass
    logger.info("Saved: %s", name)


def plot_model_comparison(comparison_path: Path) -> go.Figure:
    """Bar chart comparing all models on R², RMSE, MAE, MAPE."""
    df = load_csv(comparison_path)
    target = comparison_path.stem.replace("model_comparison_", "")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["R²", "RMSE", "MAE", "MAPE (%)"],
    )
    metrics = [("R²", 1, 1), ("RMSE", 1, 2), ("MAE", 2, 1), ("MAPE", 2, 2)]
    for metric, row, col in metrics:
        if metric not in df.columns:
            continue
        sorted_df = df.sort_values(metric, ascending=(metric != "R²"))
        fig.add_trace(
            go.Bar(
                x=sorted_df["model"],
                y=sorted_df[metric],
                marker_color=_COLORS[:len(sorted_df)],
                name=metric,
                showlegend=False,
                text=[f"{v:.3f}" for v in sorted_df[metric]],
                textposition="auto",
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title=f"Model Comparison — {target}",
        template=_TEMPLATE,
        height=700,
    )
    _save(fig, f"model_comparison_{target}")
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    model_name: str,
) -> go.Figure:
    """Scatter: actual vs predicted with ideal line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(size=5, opacity=0.6, color=_COLORS[0]),
        name="Predictions",
    ))
    # Ideal line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines",
        line=dict(dash="dash", color="white", width=1),
        name="Ideal",
    ))
    fig.update_layout(
        title=f"Actual vs Predicted — {model_name} ({target})",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template=_TEMPLATE,
        height=600,
    )
    _save(fig, f"actual_vs_predicted_{target}")
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    model_name: str,
) -> go.Figure:
    """Residual distribution and scatter."""
    residuals = y_true - y_pred

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuals vs Predicted", "Residual Distribution"],
    )
    fig.add_trace(
        go.Scatter(
            x=y_pred, y=residuals,
            mode="markers",
            marker=dict(size=4, opacity=0.5, color=_COLORS[1]),
            name="Residuals",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=1)

    fig.add_trace(
        go.Histogram(
            x=residuals, nbinsx=40,
            marker_color=_COLORS[2],
            name="Distribution",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Residual Analysis — {model_name} ({target})",
        template=_TEMPLATE,
        height=500,
        showlegend=False,
    )
    _save(fig, f"residuals_{target}")
    return fig


def plot_sector_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sectors: np.ndarray,
    target: str,
) -> go.Figure:
    """R² breakdown by sector."""
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "sector": sectors,
    })

    rows = []
    for sector in sorted(df["sector"].unique()):
        mask = df["sector"] == sector
        if mask.sum() < 5:
            continue
        metrics = compute_metrics(df.loc[mask, "y_true"], df.loc[mask, "y_pred"])
        rows.append({"sector": sector, **metrics})

    if not rows:
        return go.Figure()

    perf = pd.DataFrame(rows).sort_values("r2", ascending=True)

    fig = go.Figure(go.Bar(
        y=perf["sector"],
        x=perf["r2"],
        orientation="h",
        marker_color=_COLORS[:len(perf)],
        text=[f"{v:.3f}" for v in perf["r2"]],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"R² by Sector — {target}",
        xaxis_title="R²",
        template=_TEMPLATE,
        height=500,
    )
    _save(fig, f"sector_performance_{target}")

    # Save table
    save_csv(perf, cfg.REPORTS_DIR / f"sector_performance_{target}.csv")
    return fig


# ══════════════════════════════════════════════
#  Full evaluation pipeline
# ══════════════════════════════════════════════

def run_evaluation(targets: Optional[List[str]] = None) -> None:
    """Evaluate best models and generate all diagnostic plots.

    Args:
        targets: Targets to evaluate (default: both scopes).
    """
    targets = targets or cfg.TARGETS
    logger.info("── Evaluation pipeline ──")

    for target in targets:
        logger.info("━━━ Evaluating %s ━━━", target)

        # Load best model
        model_path = cfg.MODELS_DIR / f"best_{target}.joblib"
        meta_path = cfg.MODELS_DIR / f"best_{target}_meta.joblib"
        if not model_path.exists():
            logger.warning("No trained model for %s – skipping", target)
            continue

        model = joblib.load(model_path)
        meta = joblib.load(meta_path) if meta_path.exists() else {}
        model_name = meta.get("model_name", "Unknown")

        # Load test set
        test_path = cfg.PROCESSED_DIR / f"test_{target}.csv"
        if not test_path.exists():
            logger.warning("No test set for %s – skipping", target)
            continue
        test_df = load_csv(test_path)

        log_target = f"log_{target}"
        y_test = test_df[log_target].values
        sectors = test_df["sector"].values if "sector" in test_df.columns else None

        feature_cols = [c for c in test_df.columns
                        if c not in [log_target, "sector", target]]
        X_test = test_df[feature_cols]

        # Predict
        y_pred = model.predict(X_test)

        # Metrics on original scale
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        metrics = compute_metrics(y_test_orig, y_pred_orig)
        logger.info(
            "  %s | R²=%.4f  RMSE=%.1f  MAE=%.1f  MAPE=%.1f%%",
            model_name, metrics["r2"], metrics["rmse"], metrics["mae"], metrics["mape"],
        )

        # Plots
        plot_actual_vs_predicted(y_test_orig, y_pred_orig, target, model_name)
        plot_residuals(y_test_orig, y_pred_orig, target, model_name)

        if sectors is not None:
            plot_sector_performance(y_test_orig, y_pred_orig, sectors, target)

        # Model comparison chart
        comp_path = cfg.REPORTS_DIR / f"model_comparison_{target}.csv"
        if comp_path.exists():
            plot_model_comparison(comp_path)

    logger.info("✓ Evaluation complete — plots in %s", cfg.FIGURES_DIR)


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
