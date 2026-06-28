# Installation Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 1 GB |
| CPU | 2 cores | 4+ cores |
| GPU | Not required | Optional (XGBoost GPU) |

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CorporateCarbonPredictor.git
cd CorporateCarbonPredictor
```

### 2. Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Package in Development Mode

```bash
pip install -e .
```

### 5. Verify Installation

```bash
python -c "import sklearn, xgboost, catboost, lightgbm, shap, streamlit; print('✓ All packages installed')"
```

## Running the Full Pipeline

```bash
# Step 1: Generate synthetic training data
python -m src.synthetic_data

# Step 2: Run data ingestion
python -m src.ingestion --synthetic

# Step 3: Preprocess and clean
python -m src.preprocessing

# Step 4: Generate EDA charts
python -m src.eda

# Step 5: Engineer features
python -m src.feature_engineering

# Step 6: Train models
python -m src.train

# Step 7: Evaluate models
python -m src.evaluate

# Step 8: Generate SHAP explanations
python -m src.shap_analysis

# Step 9: Launch dashboard
streamlit run app/streamlit_app.py
```

## Troubleshooting

### CatBoost installation fails on Apple Silicon

```bash
pip install catboost --no-binary :all:
```

### LightGBM requires libomp on macOS

```bash
brew install libomp
pip install lightgbm
```

### Kaleido (for static image export) fails

```bash
pip install kaleido==0.2.1
```

### Port conflict for Streamlit

```bash
streamlit run app/streamlit_app.py --server.port 8502
```
