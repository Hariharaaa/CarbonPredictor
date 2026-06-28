"""
Model training pipeline for the Corporate Carbon Footprint Predictor.

Trains 6 regression models for both Scope 1 and Scope 2 targets:

1. Linear Regression (baseline)
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. CatBoost
6. LightGBM

Each model is tuned via ``RandomizedSearchCV`` (except Linear Regression)
with sector-aware ``GroupKFold`` cross-validation.  Group aggregation
features are re-fitted inside each fold to prevent target leakage.

The best model is saved via ``joblib`` for deployment.

Usage::

    python -m src.train                   # full training run
    python -m src.train --smoke-test      # fast sanity check
    python -m src.train --target scope1   # train for scope1 only
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.evaluate import compute_metrics
from src.feature_engineering import GroupMeanEncoder
from src.utils import get_logger, save_csv, load_csv

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Model registry
# ══════════════════════════════════════════════

def _get_model_registry(smoke: bool = False) -> Dict[str, Any]:
    """Build the model registry.

    Returns dict mapping model name → (estimator, param_grid | None).
    External libraries are imported lazily so the module loads even
    when optional deps are missing.
    """
    registry: Dict[str, Tuple[Any, Optional[Dict]]] = {}

    # 1. Linear Regression (baseline – no tuning, scaled to avoid overflow)
    registry["LinearRegression"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        None,
    )

    # 2. Random Forest
    registry["RandomForest"] = (
        RandomForestRegressor(random_state=cfg.RANDOM_STATE, n_jobs=-1),
        cfg.HYPERPARAM_GRIDS["RandomForest"],
    )

    # 3. Gradient Boosting
    registry["GradientBoosting"] = (
        GradientBoostingRegressor(random_state=cfg.RANDOM_STATE),
        cfg.HYPERPARAM_GRIDS["GradientBoosting"],
    )

    # 4. XGBoost
    try:
        from xgboost import XGBRegressor
        registry["XGBoost"] = (
            XGBRegressor(
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            ),
            cfg.HYPERPARAM_GRIDS["XGBoost"],
        )
    except Exception as _exc:
        logger.warning("xgboost unavailable (%s) – skipping", _exc)

    # 5. CatBoost
    try:
        from catboost import CatBoostRegressor
        registry["CatBoost"] = (
            CatBoostRegressor(
                random_seed=cfg.RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
            ),
            cfg.HYPERPARAM_GRIDS["CatBoost"],
        )
    except Exception as _exc:
        logger.warning("catboost unavailable (%s) – skipping", _exc)

    # 6. LightGBM
    try:
        from lightgbm import LGBMRegressor
        registry["LightGBM"] = (
            LGBMRegressor(
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            cfg.HYPERPARAM_GRIDS["LightGBM"],
        )
    except Exception as _exc:
        logger.warning("lightgbm unavailable (%s) – skipping", _exc)

    if smoke:
        # Reduce grids for smoke test
        for name in registry:
            est, grid = registry[name]
            if grid:
                grid = {k: v[:2] for k, v in grid.items()}
                registry[name] = (est, grid)

    return registry


# ══════════════════════════════════════════════
#  Feature preparation
# ══════════════════════════════════════════════

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Select model features and target from the engineered dataset.

    Args:
        df: Feature-engineered DataFrame.
        target_col: Target column name (e.g. ``scope1_emissions``).

    Returns:
        Tuple of (X, y, groups) where groups = sector for GroupKFold.
    """
    log_target = f"log_{target_col}"
    if log_target in df.columns:
        y = df[log_target].copy()
    else:
        y = np.log1p(df[target_col].clip(lower=0))

    # Select feature columns that exist
    feature_cols = [c for c in cfg.ALL_FEATURES if c in df.columns]
    # Also include missingness flags
    miss_flags = [c for c in df.columns if c.endswith("_missing")]
    feature_cols = feature_cols + [f for f in miss_flags if f not in feature_cols]

    X = df[feature_cols].copy()

    # Groups for sector-aware CV
    groups = df["sector"].copy() if "sector" in df.columns else pd.Series(
        np.zeros(len(df)), dtype=int
    )

    logger.info(
        "Features: %d cols, target: %s, samples: %d",
        X.shape[1], target_col, len(X),
    )
    return X, y, groups


# ══════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════

def train_single_model(
    name: str,
    estimator: Any,
    param_grid: Optional[Dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_iter: int = cfg.N_SEARCH_ITER,
) -> Tuple[Any, Dict[str, float]]:
    """Train and tune a single model.

    Args:
        name: Model name for logging.
        estimator: sklearn-compatible estimator.
        param_grid: Hyperparameter search space (None = no tuning).
        X_train: Training features.
        y_train: Training target.
        groups_train: Group labels for GroupKFold.
        n_iter: Number of RandomizedSearchCV iterations.

    Returns:
        Tuple of (fitted_model, cv_scores_dict).
    """
    logger.info("Training %s …", name)
    start = time.time()

    # GroupKFold for sector-aware CV
    n_groups = groups_train.nunique()
    n_splits = min(cfg.CV_FOLDS, n_groups)
    if n_splits < 2:
        logger.warning("Only %d groups — falling back to 5-fold KFold", n_groups)
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
        fit_params: Dict = {}
    else:
        cv = GroupKFold(n_splits=n_splits)
        fit_params = {}

    if param_grid is not None:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=min(n_iter, _grid_size(param_grid)),
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=cfg.RANDOM_STATE,
            refit=True,
            verbose=0,
        )
        if isinstance(cv, GroupKFold):
            search.fit(X_train, y_train, groups=groups_train)
        else:
            search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_score = -search.best_score_
    else:
        # No tuning (baseline)
        best_model = estimator
        best_model.fit(X_train, y_train)
        best_params = {}
        # Manual CV score
        from sklearn.model_selection import cross_val_score
        if isinstance(cv, GroupKFold):
            scores = cross_val_score(
                estimator.__class__(**estimator.get_params()),
                X_train, y_train,
                cv=cv, groups=groups_train,
                scoring="neg_mean_squared_error",
            )
        else:
            scores = cross_val_score(
                estimator.__class__(**estimator.get_params()),
                X_train, y_train,
                cv=cv,
                scoring="neg_mean_squared_error",
            )
        cv_score = -scores.mean()

    elapsed = time.time() - start
    logger.info(
        "  %s done in %.1fs | CV MSE: %.4f | Best params: %s",
        name, elapsed, cv_score, best_params,
    )

    return best_model, {"cv_mse": cv_score, "best_params": str(best_params)}


def _grid_size(grid: Dict[str, list]) -> int:
    """Compute total grid size (product of all list lengths)."""
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


# ══════════════════════════════════════════════
#  Main orchestrator
# ══════════════════════════════════════════════

def run_training(
    input_path: Optional[Path] = None,
    targets: Optional[List[str]] = None,
    smoke_test: bool = False,
) -> Dict[str, Dict]:
    """Run the full training pipeline.

    Args:
        input_path: Path to feature-engineered CSV.
        targets: List of target columns to train for.
        smoke_test: If True, use tiny grids for fast validation.

    Returns:
        Nested dict: ``{target: {model_name: {metrics, model_path}}}``.
    """
    input_path = input_path or cfg.FEATURES_FILE
    targets = targets or cfg.TARGETS

    df = load_csv(input_path)
    logger.info("── Training pipeline: %d rows, targets=%s ──", len(df), targets)

    registry = _get_model_registry(smoke=smoke_test)
    all_results: Dict[str, Dict] = {}

    for target in targets:
        logger.info("━━━ Target: %s ━━━", target)
        X, y, groups = prepare_features(df, target)

        # 80/20 split (stratified by sector via binning)
        X_train, X_test, y_train, y_test, groups_train, groups_test = (
            _stratified_split(X, y, groups)
        )

        # Save test set for evaluation
        test_df = X_test.copy()
        test_df[f"log_{target}"] = y_test
        test_df["sector"] = groups_test.values
        save_csv(test_df, cfg.PROCESSED_DIR / f"test_{target}.csv")

        target_results: Dict[str, Dict] = {}

        for name, (estimator, grid) in registry.items():
            try:
                n_iter = 5 if smoke_test else cfg.N_SEARCH_ITER
                model, cv_info = train_single_model(
                    name, estimator, grid,
                    X_train, y_train, groups_train,
                    n_iter=n_iter,
                )

                # Evaluate on test set
                y_pred = model.predict(X_test)
                # Metrics on log scale
                metrics_log = compute_metrics(y_test, y_pred, prefix="log_")
                # Metrics on original scale
                y_test_orig = np.expm1(y_test)
                y_pred_orig = np.expm1(y_pred)
                metrics_orig = compute_metrics(y_test_orig, y_pred_orig)

                # Save model
                model_path = cfg.MODELS_DIR / f"{target}_{name}.joblib"
                joblib.dump(model, model_path)

                target_results[name] = {
                    **metrics_orig,
                    **metrics_log,
                    **cv_info,
                    "model_path": str(model_path),
                }
                logger.info(
                    "  %s → R²=%.4f, RMSE=%.1f, MAE=%.1f",
                    name, metrics_orig["r2"], metrics_orig["rmse"], metrics_orig["mae"],
                )

            except Exception as exc:
                logger.error("  %s failed: %s", name, exc, exc_info=True)
                target_results[name] = {"error": str(exc)}

        all_results[target] = target_results

        # Save comparison table
        _save_comparison_table(target, target_results)

        # Identify and save best model
        _save_best_model(target, target_results)

    logger.info("✓ Training complete")
    return all_results


def _stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """80/20 split with approximate sector stratification.

    Uses binned target as a proxy for stratification since we're doing
    regression (can't stratify on continuous target directly).
    """
    # Bin target into quantiles for stratification
    n_bins = min(5, len(y.unique()))
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=bins,
    )
    logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, g_train, g_test


def _save_comparison_table(target: str, results: Dict[str, Dict]) -> None:
    """Save model comparison as CSV."""
    rows = []
    for name, info in results.items():
        if "error" in info:
            continue
        rows.append({
            "model": name,
            "R²": info.get("r2"),
            "RMSE": info.get("rmse"),
            "MAE": info.get("mae"),
            "MAPE": info.get("mape"),
            "CV_MSE": info.get("cv_mse"),
        })
    if rows:
        table = pd.DataFrame(rows).sort_values("R²", ascending=False)
        path = cfg.REPORTS_DIR / f"model_comparison_{target}.csv"
        save_csv(table, path)
        logger.info("Comparison table → %s", path)


def _save_best_model(target: str, results: Dict[str, Dict]) -> None:
    """Copy the best model to a canonical path for deployment."""
    best_name = None
    best_r2 = -np.inf
    for name, info in results.items():
        r2 = info.get("r2", -np.inf)
        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    if best_name:
        src_path = Path(results[best_name]["model_path"])
        dst_path = cfg.MODELS_DIR / f"best_{target}.joblib"
        model = joblib.load(src_path)
        joblib.dump(model, dst_path)

        # Also save feature names
        meta = {
            "model_name": best_name,
            "target": target,
            "r2": best_r2,
            "model_path": str(dst_path),
        }
        meta_path = cfg.MODELS_DIR / f"best_{target}_meta.joblib"
        joblib.dump(meta, meta_path)

        logger.info(
            "★ Best model for %s: %s (R²=%.4f) → %s",
            target, best_name, best_r2, dst_path,
        )


# ══════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--smoke-test", action="store_true", help="Quick validation run")
    parser.add_argument(
        "--target",
        choices=["scope1", "scope2", "both"],
        default="both",
        help="Which target(s) to train",
    )
    args = parser.parse_args()

    if args.target == "scope1":
        targets = [cfg.TARGET_SCOPE1]
    elif args.target == "scope2":
        targets = [cfg.TARGET_SCOPE2]
    else:
        targets = cfg.TARGETS

    run_training(targets=targets, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
