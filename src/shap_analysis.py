"""
SHAP-based model explainability for the Corporate Carbon Footprint Predictor.

Generates:
    - Summary (beeswarm) plot
    - Global bar importance plot
    - Waterfall plot (single prediction)
    - Force plot (single prediction)
    - Dependence plots (top features)

All plots are saved to ``reports/shap/``.

Usage::

    python -m src.shap_analysis
    python -m src.shap_analysis --target scope1_emissions --sample-idx 0
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import get_logger, load_csv

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════
#  SHAP computation
# ══════════════════════════════════════════════

def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str = "",
    max_samples: int = 200,
) -> shap.Explanation:
    """Compute SHAP values using the appropriate explainer.

    - Tree models → ``TreeExplainer`` (fast, exact)
    - Linear models → ``LinearExplainer``
    - Fallback → ``KernelExplainer`` (slow but universal)

    Args:
        model: Fitted scikit-learn-compatible model.
        X: Feature DataFrame.
        model_name: Model class name (for explainer selection).
        max_samples: Max samples for SHAP computation (speed).

    Returns:
        :class:`shap.Explanation` object.
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=cfg.RANDOM_STATE)
        logger.info("Subsampled to %d rows for SHAP", max_samples)

    model_type = type(model).__name__.lower()
    tree_types = {"randomforest", "gradientboosting", "xgb", "lgbm", "catboost"}

    if any(t in model_type for t in tree_types):
        logger.info("Using TreeExplainer for %s", type(model).__name__)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    elif "linear" in model_type:
        logger.info("Using LinearExplainer for %s", type(model).__name__)
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer(X)
    else:
        logger.info("Using KernelExplainer for %s (may be slow)", type(model).__name__)
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 50))
        sv = explainer.shap_values(X)
        shap_values = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value,
            data=X.values,
            feature_names=X.columns.tolist(),
        )

    return shap_values


# ══════════════════════════════════════════════
#  Plot generators
# ══════════════════════════════════════════════

def plot_summary(
    shap_values: shap.Explanation,
    target: str,
) -> None:
    """Beeswarm summary plot showing feature impact distribution."""
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.title(f"SHAP Summary — {target}", fontsize=14, pad=20)
    plt.tight_layout()
    path = cfg.SHAP_DIR / f"summary_{target}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def plot_bar_importance(
    shap_values: shap.Explanation,
    target: str,
) -> None:
    """Global feature importance bar chart (mean |SHAP|)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.title(f"Global Feature Importance — {target}", fontsize=14, pad=20)
    plt.tight_layout()
    path = cfg.SHAP_DIR / f"bar_importance_{target}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def plot_waterfall(
    shap_values: shap.Explanation,
    target: str,
    sample_idx: int = 0,
) -> None:
    """Waterfall plot explaining a single prediction."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values[sample_idx], show=False, max_display=15)
    plt.title(f"Waterfall — Sample {sample_idx} ({target})", fontsize=14, pad=20)
    plt.tight_layout()
    path = cfg.SHAP_DIR / f"waterfall_{target}_sample{sample_idx}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)


def plot_force(
    shap_values: shap.Explanation,
    target: str,
    sample_idx: int = 0,
) -> None:
    """Force plot for a single prediction (saved as HTML)."""
    try:
        force = shap.plots.force(
            shap_values[sample_idx],
            matplotlib=False,
        )
        path = cfg.SHAP_DIR / f"force_{target}_sample{sample_idx}.html"
        shap.save_html(str(path), force)
        logger.info("Saved: %s", path)
    except Exception as exc:
        logger.warning("Force plot failed: %s", exc)


def plot_dependence(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    target: str,
    top_n: int = 4,
) -> None:
    """Dependence plots for top-N most important features."""
    # Rank features by mean absolute SHAP
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    feature_names = (
        shap_values.feature_names
        if hasattr(shap_values, "feature_names") and shap_values.feature_names
        else X.columns.tolist()
    )
    top_features = [
        feature_names[i]
        for i in np.argsort(mean_abs)[::-1][:top_n]
    ]

    for feat in top_features:
        fig, ax = plt.subplots(figsize=(8, 6))
        feat_idx = feature_names.index(feat)
        shap.plots.scatter(
            shap_values[:, feat_idx],
            show=False,
        )
        plt.title(f"Dependence: {feat} ({target})", fontsize=12)
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        path = cfg.SHAP_DIR / f"dependence_{target}_{safe_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", path)


# ══════════════════════════════════════════════
#  Full pipeline
# ══════════════════════════════════════════════

def run_shap_analysis(
    targets: Optional[List[str]] = None,
    sample_idx: int = 0,
) -> None:
    """Generate all SHAP plots for the best model per target.

    Args:
        targets: Target columns to explain.
        sample_idx: Index of sample for single-prediction plots.
    """
    targets = targets or cfg.TARGETS
    logger.info("── SHAP Analysis ──")

    for target in targets:
        logger.info("━━━ Explaining %s ━━━", target)

        model_path = cfg.MODELS_DIR / f"best_{target}.joblib"
        meta_path = cfg.MODELS_DIR / f"best_{target}_meta.joblib"
        if not model_path.exists():
            logger.warning("No model for %s – skipping", target)
            continue

        model = joblib.load(model_path)
        meta = joblib.load(meta_path) if meta_path.exists() else {}

        # Load test set
        test_path = cfg.PROCESSED_DIR / f"test_{target}.csv"
        if not test_path.exists():
            logger.warning("No test data for %s – skipping", target)
            continue
        test_df = load_csv(test_path)

        log_target = f"log_{target}"
        feature_cols = [
            c for c in test_df.columns
            if c not in [log_target, "sector", target]
        ]
        X_test = test_df[feature_cols]

        # Compute SHAP values
        shap_values = compute_shap_values(
            model, X_test,
            model_name=meta.get("model_name", ""),
        )

        # Generate all plots
        plot_summary(shap_values, target)
        plot_bar_importance(shap_values, target)
        plot_waterfall(shap_values, target, sample_idx=sample_idx)
        plot_force(shap_values, target, sample_idx=sample_idx)
        plot_dependence(shap_values, X_test, target)

        # Save SHAP values for Streamlit
        shap_path = cfg.MODELS_DIR / f"shap_values_{target}.joblib"
        joblib.dump(shap_values, shap_path)
        logger.info("SHAP values saved → %s", shap_path)

    logger.info("✓ SHAP analysis complete — plots in %s", cfg.SHAP_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    targets = [args.target] if args.target else None
    run_shap_analysis(targets=targets, sample_idx=args.sample_idx)


if __name__ == "__main__":
    main()
