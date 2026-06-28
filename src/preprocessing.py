"""
Data preprocessing pipeline for the Corporate Carbon Footprint Predictor.

Transforms raw merged data into a clean, analysis-ready dataset by applying:

1. Duplicate removal (company_name × reporting_year composite key)
2. Company name normalisation
3. Sector / country standardisation
4. Missing-value imputation (median numeric, mode categorical)
5. Outlier detection & winsorisation
6. Log transforms on skewed numerics
7. Missingness indicator flags

Usage::

    python -m src.preprocessing          # reads merged_raw.csv → clean.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import (
    get_logger,
    save_csv,
    load_csv,
    clip_outliers,
    safe_log1p,
    validate_columns,
)

logger = get_logger(__name__)

# Suffixes to strip during company-name normalisation
_COMPANY_SUFFIXES = re.compile(
    r"\b(inc\.?|ltd\.?|corp\.?|plc\.?|co\.?|llc|sa|ag|nv|se|"
    r"group|holdings|international|global|solutions|systems|"
    r"partners|industries)\b",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════

def run_preprocessing(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute the full cleaning pipeline.

    Args:
        input_path: Merged raw CSV (default: ``config.MERGED_RAW_FILE``).
        output_path: Destination for clean CSV (default: ``config.CLEAN_FILE``).

    Returns:
        Cleaned DataFrame.
    """
    input_path = input_path or cfg.MERGED_RAW_FILE
    output_path = output_path or cfg.CLEAN_FILE

    df = load_csv(input_path)
    logger.info("── Preprocessing %d rows ──", len(df))

    df = remove_duplicates(df)
    df = normalise_company_names(df)
    df = validate_sectors(df)
    df = normalise_countries(df)
    df = add_missingness_flags(df)
    df = impute_missing(df)
    df = detect_and_clip_outliers(df)
    df = apply_log_transforms(df)

    save_csv(df, output_path)
    logger.info("✓ Clean dataset: %d rows × %d cols → %s", *df.shape, output_path)
    return df


# ══════════════════════════════════════════════
#  Pipeline steps
# ══════════════════════════════════════════════

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (company_name, reporting_year) rows.

    Keeps the row with more non-null values when duplicates exist.
    """
    before = len(df)
    # Sort by non-null count descending so first occurrence is most complete
    df["_nnull"] = df.notna().sum(axis=1)
    df = df.sort_values("_nnull", ascending=False)
    df = df.drop_duplicates(subset=["company_name", "reporting_year"], keep="first")
    df = df.drop(columns=["_nnull"])
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows", removed)
    return df.reset_index(drop=True)


def normalise_company_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip legal suffixes, and collapse whitespace."""
    if "company_name" not in df.columns:
        return df

    df["company_name_raw"] = df["company_name"]  # keep original
    df["company_name"] = (
        df["company_name"]
        .str.lower()
        .str.replace(_COMPANY_SUFFIXES, "", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    n_changed = (df["company_name"] != df["company_name_raw"].str.lower()).sum()
    logger.info("Normalised %d company names", n_changed)
    return df


def validate_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """Map sector values to GICS taxonomy; flag unknowns.

    Unknown sectors are replaced with the mode (most frequent sector).
    """
    if "sector" not in df.columns:
        return df

    valid = set(s.lower() for s in cfg.VALID_SECTORS)
    mapping = {s.lower(): s for s in cfg.VALID_SECTORS}

    df["sector"] = df["sector"].str.strip()
    df["_sector_lower"] = df["sector"].str.lower()

    unknown_mask = ~df["_sector_lower"].isin(valid) & df["_sector_lower"].notna()
    n_unknown = unknown_mask.sum()
    if n_unknown:
        logger.warning("%d rows have unrecognised sectors – replacing with mode", n_unknown)
        mode_sector = df.loc[~unknown_mask, "sector"].mode().iloc[0]
        df.loc[unknown_mask, "sector"] = mode_sector

    # Standardise casing
    df["sector"] = df["_sector_lower"].map(mapping).fillna(df["sector"])
    df = df.drop(columns=["_sector_lower"])
    return df


def normalise_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise country codes to ISO 3166-1 alpha-3.

    Attempts to resolve full country names to alpha-3 codes.
    """
    if "country" not in df.columns:
        return df

    # Try pycountry for name→code resolution
    try:
        import pycountry

        def _to_alpha3(val: str) -> str:
            if pd.isna(val):
                return val
            val = val.strip()
            if len(val) == 3 and val.isalpha():
                return val.upper()
            if len(val) == 2 and val.isalpha():
                try:
                    return pycountry.countries.get(alpha_2=val.upper()).alpha_3
                except (AttributeError, LookupError):
                    pass
            try:
                return pycountry.countries.lookup(val).alpha_3
            except LookupError:
                return val.upper()[:3]

        df["country"] = df["country"].apply(_to_alpha3)
    except ImportError:
        logger.warning("pycountry not installed – skipping country normalisation")
        df["country"] = df["country"].str.upper().str.strip()

    logger.info("Countries normalised: %d unique", df["country"].nunique())
    return df


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags for originally-missing numeric columns.

    These flags capture informative missingness (e.g. a company that
    doesn't report assets may systematically differ in emissions).
    """
    numeric_cols = cfg.NUMERIC_RAW + cfg.TARGETS
    for col in numeric_cols:
        if col in df.columns:
            flag_col = f"{col}_missing"
            df[flag_col] = df[col].isna().astype(int)
            n_miss = df[flag_col].sum()
            if n_miss:
                logger.info("  %s: %d missing (%.1f%%)", col, n_miss, 100 * n_miss / len(df))
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values.

    - Numeric columns → median
    - Categorical columns → mode

    Imputation is applied **after** missingness flags are created so
    that the model can distinguish imputed from observed values.
    """
    # Numeric imputation
    numeric_cols = [c for c in cfg.NUMERIC_RAW + cfg.TARGETS if c in df.columns]
    for col in numeric_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("  Imputed %s: %d values → median %.2f", col, n_miss, median_val)

    # Categorical imputation
    for col in cfg.CATEGORICAL_RAW:
        if col not in df.columns:
            continue
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            logger.info("  Imputed %s: %d values → mode '%s'", col, n_miss, mode_val)

    return df


def detect_and_clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise extreme values at 1st / 99th percentiles.

    Only applied to financial and emissions columns.
    """
    cols_to_clip = [c for c in cfg.NUMERIC_RAW + cfg.TARGETS if c in df.columns]
    df = clip_outliers(df, cols_to_clip, lower_quantile=0.01, upper_quantile=0.99)
    logger.info("Outliers clipped for %d columns", len(cols_to_clip))
    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to skewed financial columns.

    Creates ``log_*`` columns alongside the originals.
    """
    transform_map = {
        "revenue": "log_revenue",
        "total_assets": "log_assets",
        "employee_count": "log_employees",
        "market_cap": "log_market_cap",
    }
    for raw_col, log_col in transform_map.items():
        if raw_col in df.columns:
            df[log_col] = safe_log1p(df[raw_col])
            logger.info("  Created %s (skew: %.2f → %.2f)",
                        log_col,
                        df[raw_col].skew(),
                        df[log_col].skew())

    # Log-transform targets as well (models predict log-emissions)
    for target in cfg.TARGETS:
        if target in df.columns:
            log_target = f"log_{target}"
            df[log_target] = safe_log1p(df[target])

    return df


# ══════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════

def main() -> None:
    """CLI entry point."""
    run_preprocessing()


if __name__ == "__main__":
    main()
