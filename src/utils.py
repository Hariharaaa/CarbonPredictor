"""
Shared utility functions used across all pipeline stages.

Provides:
    - Structured logging setup
    - Safe file I/O with validation
    - Common data transformations
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a consistently-formatted logger.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level string.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ──────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Save a DataFrame to CSV, creating parent dirs if needed.

    Args:
        df: DataFrame to persist.
        path: Target file path.
        index: Whether to write the row index.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger = get_logger(__name__)
    logger.info("Saved %s rows → %s", len(df), path)


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV with validation that the file exists.

    Args:
        path: Source file path.
        **kwargs: Forwarded to :func:`pandas.read_csv`.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")
    df = pd.read_csv(path, **kwargs)
    logger = get_logger(__name__)
    logger.info("Loaded %s rows from %s", len(df), path)
    return df


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def safe_log1p(series: pd.Series) -> pd.Series:
    """Apply ``log1p`` after clipping negative values to zero.

    Args:
        series: Numeric pandas Series.

    Returns:
        Transformed Series.
    """
    return np.log1p(series.clip(lower=0))


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill: float = 0.0,
) -> pd.Series:
    """Element-wise division that replaces inf / NaN with *fill*.

    Args:
        numerator: Numerator series.
        denominator: Denominator series.
        fill: Replacement value for invalid results.

    Returns:
        Result Series with no infinities.
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(fill).replace([np.inf, -np.inf], fill)


def clip_outliers(
    df: pd.DataFrame,
    columns: Sequence[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Winsorise columns at the given quantiles.

    Args:
        df: Input DataFrame (modified in-place).
        columns: Columns to clip.
        lower_quantile: Lower bound quantile.
        upper_quantile: Upper bound quantile.

    Returns:
        DataFrame with clipped columns.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower_quantile)
        hi = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lo, hi)
    return df


def validate_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    context: str = "",
) -> None:
    """Raise if any required columns are missing.

    Args:
        df: DataFrame to check.
        required: List of expected column names.
        context: Label for error messages.

    Raises:
        ValueError: If any column is absent.
    """
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{context}] Missing columns: {sorted(missing)}"
        )


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with downcasted dtypes.
    """
    for col in df.select_dtypes(include=["int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df
