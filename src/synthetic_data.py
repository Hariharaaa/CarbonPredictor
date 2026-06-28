"""
Synthetic data generator for development and demonstration.

Produces ~500 realistic company records with emissions, financial data,
and country metadata.  Distributions are calibrated against publicly
available CDP / EPA sector-level benchmarks so that downstream models
learn plausible relationships rather than pure noise.

Usage::

    python -m src.synthetic_data          # writes to data/raw/
    python -m src.synthetic_data --n 1000 # custom count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running as ``python -m src.synthetic_data`` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from src.utils import get_logger, save_csv

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Sector profiles (calibrated to public CDP data)
# ──────────────────────────────────────────────
# Each profile: (revenue_mu_log, revenue_sigma_log,
#                assets_mu_log, assets_sigma_log,
#                employees_mu_log, employees_sigma_log,
#                scope1_factor, scope2_factor)
# Factors: median tonnes CO2e per $M revenue (order-of-magnitude)

SECTOR_PROFILES: Dict[str, Dict[str, float]] = {
    "Energy": {
        "rev_mu": 9.5, "rev_sig": 1.2,
        "ast_mu": 10.0, "ast_sig": 1.1,
        "emp_mu": 9.0, "emp_sig": 0.9,
        "s1_factor": 850.0, "s2_factor": 120.0,
        "mcap_mu": 10.0, "mcap_sig": 1.3,
    },
    "Materials": {
        "rev_mu": 8.8, "rev_sig": 1.1,
        "ast_mu": 9.3, "ast_sig": 1.0,
        "emp_mu": 8.5, "emp_sig": 0.8,
        "s1_factor": 600.0, "s2_factor": 180.0,
        "mcap_mu": 9.2, "mcap_sig": 1.2,
    },
    "Industrials": {
        "rev_mu": 8.5, "rev_sig": 1.0,
        "ast_mu": 9.0, "ast_sig": 1.0,
        "emp_mu": 9.2, "emp_sig": 0.9,
        "s1_factor": 200.0, "s2_factor": 100.0,
        "mcap_mu": 9.0, "mcap_sig": 1.1,
    },
    "Consumer Discretionary": {
        "rev_mu": 8.3, "rev_sig": 1.1,
        "ast_mu": 8.8, "ast_sig": 1.0,
        "emp_mu": 9.5, "emp_sig": 1.0,
        "s1_factor": 50.0, "s2_factor": 80.0,
        "mcap_mu": 9.0, "mcap_sig": 1.2,
    },
    "Consumer Staples": {
        "rev_mu": 8.8, "rev_sig": 1.0,
        "ast_mu": 9.0, "ast_sig": 0.9,
        "emp_mu": 9.6, "emp_sig": 0.8,
        "s1_factor": 80.0, "s2_factor": 60.0,
        "mcap_mu": 9.3, "mcap_sig": 1.0,
    },
    "Health Care": {
        "rev_mu": 8.5, "rev_sig": 1.2,
        "ast_mu": 9.0, "ast_sig": 1.1,
        "emp_mu": 9.3, "emp_sig": 1.0,
        "s1_factor": 30.0, "s2_factor": 50.0,
        "mcap_mu": 9.5, "mcap_sig": 1.3,
    },
    "Financials": {
        "rev_mu": 8.8, "rev_sig": 1.3,
        "ast_mu": 11.0, "ast_sig": 1.5,
        "emp_mu": 9.5, "emp_sig": 1.0,
        "s1_factor": 5.0, "s2_factor": 30.0,
        "mcap_mu": 10.0, "mcap_sig": 1.4,
    },
    "Information Technology": {
        "rev_mu": 8.5, "rev_sig": 1.3,
        "ast_mu": 9.0, "ast_sig": 1.2,
        "emp_mu": 9.0, "emp_sig": 1.1,
        "s1_factor": 10.0, "s2_factor": 60.0,
        "mcap_mu": 9.8, "mcap_sig": 1.5,
    },
    "Communication Services": {
        "rev_mu": 8.5, "rev_sig": 1.2,
        "ast_mu": 9.2, "ast_sig": 1.1,
        "emp_mu": 9.0, "emp_sig": 0.9,
        "s1_factor": 15.0, "s2_factor": 55.0,
        "mcap_mu": 9.5, "mcap_sig": 1.3,
    },
    "Utilities": {
        "rev_mu": 8.5, "rev_sig": 0.9,
        "ast_mu": 9.8, "ast_sig": 0.8,
        "emp_mu": 8.5, "emp_sig": 0.7,
        "s1_factor": 1200.0, "s2_factor": 50.0,
        "mcap_mu": 9.3, "mcap_sig": 0.9,
    },
    "Real Estate": {
        "rev_mu": 7.8, "rev_sig": 1.0,
        "ast_mu": 9.5, "ast_sig": 1.0,
        "emp_mu": 7.5, "emp_sig": 0.8,
        "s1_factor": 20.0, "s2_factor": 70.0,
        "mcap_mu": 9.0, "mcap_sig": 1.1,
    },
}

# Industry sub-classifications per sector
INDUSTRY_MAP: Dict[str, List[str]] = {
    "Energy": [
        "Oil & Gas Exploration", "Oil & Gas Refining",
        "Renewable Energy", "Coal Mining",
    ],
    "Materials": [
        "Chemicals", "Steel & Metals", "Paper & Forestry",
        "Construction Materials",
    ],
    "Industrials": [
        "Aerospace & Defence", "Machinery", "Transportation",
        "Building Products",
    ],
    "Consumer Discretionary": [
        "Automobiles", "Retail", "Textiles & Apparel",
        "Hotels & Leisure",
    ],
    "Consumer Staples": [
        "Food & Beverages", "Household Products",
        "Tobacco", "Personal Care",
    ],
    "Health Care": [
        "Pharmaceuticals", "Biotechnology",
        "Medical Devices", "Healthcare Services",
    ],
    "Financials": [
        "Banks", "Insurance", "Asset Management",
        "Diversified Financial Services",
    ],
    "Information Technology": [
        "Software", "Hardware", "Semiconductors", "IT Services",
    ],
    "Communication Services": [
        "Telecommunications", "Media & Entertainment",
        "Internet Services", "Advertising",
    ],
    "Utilities": [
        "Electric Utilities", "Gas Utilities",
        "Water Utilities", "Multi-Utilities",
    ],
    "Real Estate": [
        "REITs", "Real Estate Development",
        "Property Management", "Real Estate Services",
    ],
}

# Country pool with GDP-per-capita and energy intensity proxies
COUNTRY_DATA: List[Dict[str, float | str]] = [
    {"country": "USA", "gdp_per_capita": 76330, "energy_intensity": 4.58},
    {"country": "GBR", "gdp_per_capita": 46510, "energy_intensity": 2.88},
    {"country": "DEU", "gdp_per_capita": 51380, "energy_intensity": 3.31},
    {"country": "FRA", "gdp_per_capita": 43520, "energy_intensity": 3.58},
    {"country": "JPN", "gdp_per_capita": 39290, "energy_intensity": 3.41},
    {"country": "CHN", "gdp_per_capita": 12720, "energy_intensity": 8.91},
    {"country": "IND", "gdp_per_capita": 2410, "energy_intensity": 4.73},
    {"country": "BRA", "gdp_per_capita": 8920, "energy_intensity": 3.89},
    {"country": "CAN", "gdp_per_capita": 52960, "energy_intensity": 6.81},
    {"country": "AUS", "gdp_per_capita": 64490, "energy_intensity": 5.32},
    {"country": "KOR", "gdp_per_capita": 34990, "energy_intensity": 5.01},
    {"country": "ZAF", "gdp_per_capita": 6010, "energy_intensity": 7.22},
    {"country": "MEX", "gdp_per_capita": 10820, "energy_intensity": 3.41},
    {"country": "NOR", "gdp_per_capita": 82830, "energy_intensity": 4.91},
    {"country": "SAU", "gdp_per_capita": 27610, "energy_intensity": 6.52},
]

# Weights for country sampling (larger economies get more companies)
COUNTRY_WEIGHTS = np.array([
    0.25, 0.08, 0.07, 0.06, 0.08,
    0.12, 0.08, 0.04, 0.04, 0.03,
    0.04, 0.02, 0.03, 0.02, 0.04,
])
COUNTRY_WEIGHTS = COUNTRY_WEIGHTS / COUNTRY_WEIGHTS.sum()

# Ticker-like prefixes per sector for company name generation
_SECTOR_PREFIXES: Dict[str, List[str]] = {
    "Energy": ["Petro", "Ener", "Fuel", "Solar", "Geo", "Hydro"],
    "Materials": ["Chem", "Steel", "Poly", "Timber", "Mine", "Alloy"],
    "Industrials": ["Aero", "Mech", "Trans", "Build", "Dynamo", "Forge"],
    "Consumer Discretionary": ["Auto", "Retail", "Luxe", "Travel", "Style"],
    "Consumer Staples": ["Nutri", "Fresh", "Harvest", "Pure", "Globe"],
    "Health Care": ["Pharma", "Bio", "Medi", "Vita", "Gene", "Cure"],
    "Financials": ["Capital", "Trust", "Apex", "Vanguard", "Pinnacle"],
    "Information Technology": ["Tech", "Cyber", "Quantum", "Nexus", "Digi"],
    "Communication Services": ["Tele", "Media", "Signal", "Stream", "Link"],
    "Utilities": ["Power", "Grid", "Volt", "Aqua", "Flux"],
    "Real Estate": ["Realty", "Metro", "Urban", "Haven", "Horizon"],
}

_SUFFIXES = [
    "Corp", "Inc", "Group", "Holdings", "Global",
    "International", "Solutions", "Systems", "Partners", "Industries",
]


def _generate_company_name(sector: str, idx: int, rng: np.random.Generator) -> str:
    """Create a plausible company name."""
    prefix = rng.choice(_SECTOR_PREFIXES.get(sector, ["Acme"]))
    suffix = rng.choice(_SUFFIXES)
    return f"{prefix}{idx:04d} {suffix}"


def _generate_ticker(name: str, rng: np.random.Generator) -> str:
    """Derive a 3–5 char ticker from company name."""
    base = name.split()[0][:4].upper()
    return base + rng.choice(list("XYZQW"))


def generate_synthetic_dataset(
    n: int = 500,
    seed: int = cfg.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic emissions + financials dataset.

    The data models real-world structure:
    - Revenue, assets, market cap drawn from log-normal distributions
      (sector-specific parameters calibrated to public CDP data)
    - Scope 1 & 2 emissions correlated with revenue & sector via
      multiplicative emission factors + noise
    - Country macro data appended from a fixed lookup table

    Args:
        n: Number of company records to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (company_df, country_df).

        * ``company_df`` has columns matching the raw ingestion schema.
        * ``country_df`` has per-country macro indicators.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic company records …", n)

    sectors = list(SECTOR_PROFILES.keys())
    sector_weights = np.ones(len(sectors)) / len(sectors)

    records: List[Dict] = []
    used_names: set = set()

    for i in range(n):
        sector = rng.choice(sectors, p=sector_weights)
        prof = SECTOR_PROFILES[sector]

        # Company identity
        name = _generate_company_name(sector, i, rng)
        while name in used_names:
            name = _generate_company_name(sector, i + 1000, rng)
        used_names.add(name)

        ticker = _generate_ticker(name, rng)

        industry = rng.choice(INDUSTRY_MAP[sector])

        # Country
        cidx = rng.choice(len(COUNTRY_DATA), p=COUNTRY_WEIGHTS)
        country_info = COUNTRY_DATA[cidx]

        # Financials (log-normal)
        revenue = np.exp(rng.normal(prof["rev_mu"], prof["rev_sig"]))
        total_assets = np.exp(rng.normal(prof["ast_mu"], prof["ast_sig"]))
        market_cap = np.exp(rng.normal(prof["mcap_mu"], prof["mcap_sig"]))
        employee_count = int(np.exp(rng.normal(prof["emp_mu"], prof["emp_sig"])))
        employee_count = max(employee_count, 50)

        # Revenue in millions
        revenue_m = revenue / 1e6

        # Emissions = factor × revenue_M × noise × country adjustment
        country_energy_adj = country_info["energy_intensity"] / 4.5  # normalised
        noise_s1 = rng.lognormal(0, 0.4)
        noise_s2 = rng.lognormal(0, 0.35)

        scope1 = max(
            prof["s1_factor"] * revenue_m * noise_s1 * country_energy_adj, 1.0
        )
        scope2 = max(
            prof["s2_factor"] * revenue_m * noise_s2 * country_energy_adj, 1.0
        )

        # Reporting year (2020–2023)
        year = int(rng.choice([2020, 2021, 2022, 2023]))

        records.append({
            "company_name": name,
            "ticker": ticker,
            "reporting_year": year,
            "sector": sector,
            "industry": industry,
            "country": country_info["country"],
            "revenue": round(revenue, 2),
            "total_assets": round(total_assets, 2),
            "market_cap": round(market_cap, 2),
            "employee_count": employee_count,
            cfg.TARGET_SCOPE1: round(scope1, 2),
            cfg.TARGET_SCOPE2: round(scope2, 2),
        })

    company_df = pd.DataFrame(records)

    # Inject ~5 % missingness for realism
    _inject_missingness(company_df, rng, frac=0.05)

    # Country macro table
    country_df = pd.DataFrame(COUNTRY_DATA)

    logger.info(
        "Generated dataset: %d records, %d sectors, %d countries",
        len(company_df),
        company_df["sector"].nunique(),
        company_df["country"].nunique(),
    )

    return company_df, country_df


def _inject_missingness(
    df: pd.DataFrame,
    rng: np.random.Generator,
    frac: float = 0.05,
) -> None:
    """Randomly null-out *frac* of cells in numeric/categorical columns.

    Modifies *df* in place.
    """
    cols_to_corrupt = [
        "revenue", "total_assets", "market_cap", "employee_count",
        "industry",
    ]
    for col in cols_to_corrupt:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan
    logger.info("Injected ~%.0f%% missingness into %s", frac * 100, cols_to_corrupt)


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

def main() -> None:
    """Generate synthetic data and write to ``data/raw/``."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic CDP-like emissions dataset"
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Number of company records (default: 500)",
    )
    args = parser.parse_args()

    company_df, country_df = generate_synthetic_dataset(n=args.n)

    save_csv(company_df, cfg.RAW_EMISSIONS_FILE)
    save_csv(country_df, cfg.RAW_WORLDBANK_FILE)
    logger.info("✓ Synthetic data written to %s", cfg.RAW_DIR)


if __name__ == "__main__":
    main()
