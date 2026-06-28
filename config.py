"""
Central configuration for the Corporate Carbon Footprint Predictor.

All project-wide constants, paths, feature lists, and model hyperparameter
grids are defined here to avoid magic numbers scattered across modules.
"""

from pathlib import Path
from typing import Dict, List, Any

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_DIR = REPORTS_DIR / "shap"
APP_DIR = PROJECT_ROOT / "app"

# Ensure directories exist at import time
for _d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR,
           REPORTS_DIR, FIGURES_DIR, SHAP_DIR, APP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Data files
# ──────────────────────────────────────────────
RAW_EMISSIONS_FILE = RAW_DIR / "cdp_emissions.csv"
RAW_FINANCIALS_FILE = RAW_DIR / "yahoo_financials.csv"
RAW_WORLDBANK_FILE = EXTERNAL_DIR / "worldbank_indicators.csv"
MERGED_RAW_FILE = PROCESSED_DIR / "merged_raw.csv"
CLEAN_FILE = PROCESSED_DIR / "clean.csv"
FEATURES_FILE = PROCESSED_DIR / "features.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

# ──────────────────────────────────────────────
# Random seed
# ──────────────────────────────────────────────
RANDOM_STATE: int = 42

# ──────────────────────────────────────────────
# Targets
# ──────────────────────────────────────────────
TARGET_SCOPE1: str = "scope1_emissions"
TARGET_SCOPE2: str = "scope2_emissions"
TARGETS: List[str] = [TARGET_SCOPE1, TARGET_SCOPE2]

# ──────────────────────────────────────────────
# Identifiers (never used as features)
# ──────────────────────────────────────────────
ID_COLS: List[str] = [
    "company_name",
    "ticker",
    "reporting_year",
]

# ──────────────────────────────────────────────
# Raw feature columns (before engineering)
# ──────────────────────────────────────────────
NUMERIC_RAW: List[str] = [
    "revenue",
    "total_assets",
    "market_cap",
    "employee_count",
]

CATEGORICAL_RAW: List[str] = [
    "sector",
    "industry",
    "country",
]

# ──────────────────────────────────────────────
# Engineered features
# ──────────────────────────────────────────────
LOG_FEATURES: List[str] = [
    "log_revenue",
    "log_assets",
    "log_employees",
    "log_market_cap",
]

RATIO_FEATURES: List[str] = [
    "asset_per_employee",
    "revenue_per_employee",
    "asset_to_revenue",
    "employee_density",
    "market_cap_to_revenue",
    "capital_intensity",
]

COUNTRY_FEATURES: List[str] = [
    "gdp_per_capita",
    "energy_intensity",
]

# Group-aggregation features (computed inside train fold only)
GROUP_AGG_FEATURES: List[str] = [
    "sector_mean_log_scope1",
    "sector_mean_log_scope2",
    "country_mean_log_scope1",
    "country_mean_log_scope2",
]

ENCODED_FEATURES: List[str] = [
    "sector_encoded",
    "industry_encoded",
    "country_encoded",
]

# Final feature set for modelling
ALL_NUMERIC_FEATURES: List[str] = (
    LOG_FEATURES + RATIO_FEATURES + COUNTRY_FEATURES + GROUP_AGG_FEATURES
)
ALL_FEATURES: List[str] = ALL_NUMERIC_FEATURES + ENCODED_FEATURES

# ──────────────────────────────────────────────
# Sectors (GICS Level-1 approximation)
# ──────────────────────────────────────────────
VALID_SECTORS: List[str] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]

# ──────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# ──────────────────────────────────────────────
# Model hyper-parameter grids (for RandomizedSearchCV)
# ──────────────────────────────────────────────
HYPERPARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "RandomForest": {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [6, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [3, 5, 7, 10, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    },
    "CatBoost": {
        "iterations": [200, 400, 600],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7],
        "bagging_temperature": [0, 0.5, 1.0],
    },
}

# Number of iterations for RandomizedSearchCV
N_SEARCH_ITER: int = 30

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
