"""
Prediction interface for the Corporate Carbon Footprint Predictor.

Provides a clean API for loading the best model and generating
predictions from raw input features.  Used by the Streamlit app
and any downstream consumers.

Usage::

    from src.predict import CarbonPredictor
    predictor = CarbonPredictor()
    result = predictor.predict(
        revenue=5e9, total_assets=10e9, market_cap=15e9,
        employee_count=20000, sector="Energy",
        industry="Oil & Gas Refining", country="USA",
    )
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import get_logger, safe_log1p, safe_divide

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Container for a single prediction output."""

    scope1: float
    scope2: float
    total: float
    confidence_score: float
    scope1_interval: tuple[float, float]
    scope2_interval: tuple[float, float]
    feature_values: Dict[str, float]


class CarbonPredictor:
    """Production prediction interface.

    Loads the best saved model for each target and provides
    a ``predict()`` method that accepts raw company features.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = models_dir or cfg.MODELS_DIR
        self.models: Dict[str, Any] = {}
        self.meta: Dict[str, Dict] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load best models for both targets."""
        for target in cfg.TARGETS:
            model_path = self.models_dir / f"best_{target}.joblib"
            meta_path = self.models_dir / f"best_{target}_meta.joblib"

            if not model_path.exists():
                logger.warning("Model not found: %s", model_path)
                continue

            self.models[target] = joblib.load(model_path)
            self.meta[target] = (
                joblib.load(meta_path) if meta_path.exists() else {}
            )

            # Infer feature names from test set
            test_path = cfg.PROCESSED_DIR / f"test_{target}.csv"
            if test_path.exists():
                test_df = pd.read_csv(test_path, nrows=1)
                log_target = f"log_{target}"
                self.feature_names[target] = [
                    c for c in test_df.columns
                    if c not in [log_target, "sector", target]
                ]

            logger.info(
                "Loaded model for %s: %s",
                target, self.meta[target].get("model_name", "unknown"),
            )

    @property
    def is_ready(self) -> bool:
        """Check if models are loaded."""
        return len(self.models) > 0

    def predict(
        self,
        revenue: float,
        total_assets: float,
        market_cap: float,
        employee_count: int,
        sector: str,
        industry: str,
        country: str,
        gdp_per_capita: float = 50000.0,
        energy_intensity: float = 4.5,
    ) -> PredictionResult:
        """Generate emissions prediction from raw company features.

        Args:
            revenue: Annual revenue in USD.
            total_assets: Total assets in USD.
            market_cap: Market capitalisation in USD.
            employee_count: Number of full-time employees.
            sector: GICS sector name.
            industry: Industry sub-classification.
            country: ISO 3166-1 alpha-3 country code.
            gdp_per_capita: Country GDP per capita (USD).
            energy_intensity: Country energy use per capita (kg oil eq).

        Returns:
            :class:`PredictionResult` with predictions and confidence.
        """
        # Build feature vector
        features = self._build_features(
            revenue, total_assets, market_cap, employee_count,
            sector, industry, country, gdp_per_capita, energy_intensity,
        )

        results = {}
        for target in cfg.TARGETS:
            if target not in self.models:
                results[target] = 0.0
                continue

            model = self.models[target]
            feat_names = self.feature_names.get(target, list(features.keys()))

            # Create DataFrame with aligned columns
            X = pd.DataFrame([features])
            # Add any missing columns with 0
            for col in feat_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[feat_names]

            log_pred = model.predict(X)[0]
            results[target] = float(np.expm1(log_pred))

        scope1 = max(results.get(cfg.TARGET_SCOPE1, 0), 0)
        scope2 = max(results.get(cfg.TARGET_SCOPE2, 0), 0)

        # Confidence estimation via residual-based interval
        confidence, s1_interval, s2_interval = self._estimate_confidence(
            scope1, scope2
        )

        return PredictionResult(
            scope1=round(scope1, 2),
            scope2=round(scope2, 2),
            total=round(scope1 + scope2, 2),
            confidence_score=confidence,
            scope1_interval=s1_interval,
            scope2_interval=s2_interval,
            feature_values=features,
        )

    def _build_features(
        self,
        revenue: float,
        total_assets: float,
        market_cap: float,
        employee_count: int,
        sector: str,
        industry: str,
        country: str,
        gdp_per_capita: float,
        energy_intensity: float,
    ) -> Dict[str, float]:
        """Transform raw inputs into model features."""
        emp = max(employee_count, 1)
        rev = max(revenue, 1)
        assets = max(total_assets, 1)

        features: Dict[str, float] = {
            # Log features
            "log_revenue": float(np.log1p(rev)),
            "log_assets": float(np.log1p(assets)),
            "log_employees": float(np.log1p(emp)),
            "log_market_cap": float(np.log1p(max(market_cap, 1))),
            # Ratio features
            "asset_per_employee": assets / emp,
            "revenue_per_employee": rev / emp,
            "asset_to_revenue": assets / rev,
            "employee_density": emp / assets,
            "market_cap_to_revenue": max(market_cap, 1) / rev,
            "capital_intensity": assets / rev,
            # Country features
            "gdp_per_capita": gdp_per_capita,
            "energy_intensity": energy_intensity,
            # Encoded categoricals (simple hash-based for inference)
            "sector_encoded": hash(sector) % 100,
            "industry_encoded": hash(industry) % 200,
            "country_encoded": hash(country) % 50,
            # Group agg features (use sector-level defaults)
            "sector_mean_log_scope1": self._sector_default(sector, "scope1"),
            "sector_mean_log_scope2": self._sector_default(sector, "scope2"),
            "country_mean_log_scope1": 10.0,
            "country_mean_log_scope2": 9.0,
            # Missingness flags (no missing data at inference time)
            "revenue_missing": 0,
            "total_assets_missing": 0,
            "market_cap_missing": 0,
            "employee_count_missing": 0,
            "scope1_emissions_missing": 0,
            "scope2_emissions_missing": 0,
        }
        return features

    def _sector_default(self, sector: str, scope: str) -> float:
        """Return a reasonable sector-level emission mean."""
        from src.synthetic_data import SECTOR_PROFILES
        profile = SECTOR_PROFILES.get(sector, {})
        factor_key = "s1_factor" if scope == "scope1" else "s2_factor"
        factor = profile.get(factor_key, 100.0)
        return float(np.log1p(factor * 50))  # Approximate log mean

    def _estimate_confidence(
        self,
        scope1: float,
        scope2: float,
    ) -> tuple[float, tuple[float, float], tuple[float, float]]:
        """Estimate prediction confidence via heuristic interval.

        Uses the model's R² as a proxy for prediction reliability.
        In production, this would use conformal prediction or
        quantile regression.
        """
        # Use best R² as confidence proxy
        r2_vals = [
            self.meta.get(t, {}).get("r2", 0.5)
            for t in cfg.TARGETS
        ]
        avg_r2 = np.mean(r2_vals) if r2_vals else 0.5
        confidence = min(max(avg_r2, 0.0), 1.0)

        # Interval width inversely proportional to confidence
        spread = 1.0 - confidence + 0.1  # minimum 10% spread
        s1_lo = max(scope1 * (1 - spread), 0)
        s1_hi = scope1 * (1 + spread)
        s2_lo = max(scope2 * (1 - spread), 0)
        s2_hi = scope2 * (1 + spread)

        return (
            round(confidence, 3),
            (round(s1_lo, 2), round(s1_hi, 2)),
            (round(s2_lo, 2), round(s2_hi, 2)),
        )

    def get_shap_explanation(
        self,
        features: Dict[str, float],
        target: str,
    ):
        """Compute SHAP values for a single prediction.

        Args:
            features: Feature dict from :meth:`predict`.
            target: Target column name.

        Returns:
            :class:`shap.Explanation` or None if unavailable.
        """
        if target not in self.models:
            return None

        try:
            import shap
            model = self.models[target]
            feat_names = self.feature_names.get(target, list(features.keys()))
            X = pd.DataFrame([features])
            for col in feat_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[feat_names]

            model_type = type(model).__name__.lower()
            if any(t in model_type for t in ["forest", "boosting", "xgb", "lgbm", "catboost"]):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X)

            shap_values = explainer(X)
            return shap_values[0]
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)
            return None
