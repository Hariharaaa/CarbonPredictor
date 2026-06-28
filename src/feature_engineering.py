"""
Feature engineering for the Corporate Carbon Footprint Predictor.

Creates domain-informed features from clean data.  All group-level
aggregation features (sector means, country means) are wrapped in a
sklearn-compatible transformer so they are computed **inside the
training fold only**, preventing target leakage.

Usage::

    python -m src.feature_engineering   # reads clean.csv → features.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import get_logger, save_csv, load_csv, safe_divide, safe_log1p

logger = get_logger(__name__)


# ══════════════════════════════════════════════
#  Deterministic (non-leaking) features
# ══════════════════════════════════════════════

def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial ratio features.

    These ratios capture operational efficiency and capital structure —
    known drivers of emissions intensity.

    Args:
        df: DataFrame with raw financial columns.

    Returns:
        DataFrame with new ratio columns appended.
    """
    df = df.copy()

    df["asset_per_employee"] = safe_divide(df["total_assets"], df["employee_count"])
    df["revenue_per_employee"] = safe_divide(df["revenue"], df["employee_count"])
    df["asset_to_revenue"] = safe_divide(df["total_assets"], df["revenue"])
    df["employee_density"] = safe_divide(df["employee_count"], df["total_assets"])
    df["market_cap_to_revenue"] = safe_divide(df["market_cap"], df["revenue"])
    df["capital_intensity"] = safe_divide(df["total_assets"], df["revenue"])

    logger.info("Created %d ratio features", len(cfg.RATIO_FEATURES))
    return df


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure log-transformed features exist.

    Preprocessing may have already created them; this is idempotent.
    """
    df = df.copy()
    mapping = {
        "revenue": "log_revenue",
        "total_assets": "log_assets",
        "employee_count": "log_employees",
        "market_cap": "log_market_cap",
    }
    for raw, log_col in mapping.items():
        if raw in df.columns and log_col not in df.columns:
            df[log_col] = safe_log1p(df[raw])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode sector, industry, and country.

    Uses ``OrdinalEncoder`` with ``handle_unknown='use_encoded_value'``
    so unseen categories at inference time map to ``-1``.
    """
    df = df.copy()
    cat_map = {
        "sector": "sector_encoded",
        "industry": "industry_encoded",
        "country": "country_encoded",
    }

    for raw_col, enc_col in cat_map.items():
        if raw_col not in df.columns:
            continue
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[enc_col] = encoder.fit_transform(
            df[[raw_col]].fillna("Unknown")
        ).astype(int)
        logger.info(
            "Encoded %s → %s (%d categories)",
            raw_col, enc_col, df[raw_col].nunique(),
        )
    return df


# ══════════════════════════════════════════════
#  Leakage-safe group aggregation transformer
# ══════════════════════════════════════════════

class GroupMeanEncoder(BaseEstimator, TransformerMixin):
    """Compute group-level mean of log-target on training data only.

    Fits on training fold, transforms both train and test.  Unknown
    groups at inference time receive the global mean.

    Parameters:
        group_col: Column to group by (e.g. ``"sector"``).
        target_cols: Targets to compute means for.
        prefix: Prefix for output column names.
    """

    def __init__(
        self,
        group_col: str,
        target_cols: List[str],
        prefix: str = "",
    ) -> None:
        self.group_col = group_col
        self.target_cols = target_cols
        self.prefix = prefix
        self.group_means_: Dict[str, Dict] = {}
        self.global_means_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Compute group means from training data.

        Args:
            X: DataFrame containing ``group_col`` and ``target_cols``.
        """
        for t in self.target_cols:
            log_t = f"log_{t}"
            col = log_t if log_t in X.columns else t
            if col not in X.columns:
                continue
            self.global_means_[t] = X[col].mean()
            self.group_means_[t] = X.groupby(self.group_col)[col].mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply precomputed group means.

        Args:
            X: DataFrame to transform.

        Returns:
            DataFrame with new group-mean columns.
        """
        X = X.copy()
        for t in self.target_cols:
            if t not in self.group_means_:
                continue
            out_col = f"{self.prefix}mean_log_{t.replace('_emissions', '')}"
            X[out_col] = (
                X[self.group_col]
                .map(self.group_means_[t])
                .fillna(self.global_means_[t])
            )
        return X

    def get_feature_names_out(self) -> List[str]:
        """Return output column names."""
        return [
            f"{self.prefix}mean_log_{t.replace('_emissions', '')}"
            for t in self.target_cols
            if t in self.group_means_
        ]


# ══════════════════════════════════════════════
#  Full pipeline
# ══════════════════════════════════════════════

def run_feature_engineering(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    fit_group_means: bool = True,
) -> pd.DataFrame:
    """Run the complete feature engineering pipeline.

    When ``fit_group_means=True`` (default for initial processing),
    group aggregation features are computed on the full dataset.
    During model training, :class:`GroupMeanEncoder` is instead fitted
    inside each cross-validation fold to prevent leakage.

    Args:
        input_path: Clean CSV input.
        output_path: Engineered features output.
        fit_group_means: Whether to compute group-agg features here.

    Returns:
        Feature-engineered DataFrame.
    """
    input_path = input_path or cfg.CLEAN_FILE
    output_path = output_path or cfg.FEATURES_FILE

    df = load_csv(input_path)
    logger.info("── Feature engineering on %d rows ──", len(df))

    df = create_log_features(df)
    df = create_ratio_features(df)
    df = encode_categoricals(df)

    if fit_group_means:
        # For EDA / exploration only — during training we refit per fold
        sector_enc = GroupMeanEncoder("sector", cfg.TARGETS, prefix="sector_")
        country_enc = GroupMeanEncoder("country", cfg.TARGETS, prefix="country_")
        df = sector_enc.fit_transform(df)
        df = country_enc.fit_transform(df)
        logger.info(
            "Group-agg features added (NOTE: refit per CV fold during training)"
        )

    save_csv(df, output_path)
    logger.info("✓ Features saved: %d rows × %d cols", *df.shape)
    return df


def main() -> None:
    run_feature_engineering()


if __name__ == "__main__":
    main()
