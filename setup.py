"""Minimal setup.py for editable installs (pip install -e .)."""

from setuptools import setup, find_packages

setup(
    name="corporate-carbon-predictor",
    version="1.0.0",
    description=(
        "ML system predicting corporate Scope 1 & 2 GHG emissions "
        "from publicly available financial and operational data."
    ),
    author="Harihara",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "catboost>=1.2",
        "lightgbm>=4.0",
        "shap>=0.43",
        "plotly>=5.18",
        "streamlit>=1.30",
        "joblib>=1.3",
    ],
)
