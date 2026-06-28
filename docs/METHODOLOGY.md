# Methodology

## 1. Problem Statement

Estimate a company's annual Scope 1 (direct) and Scope 2 (indirect energy-related) greenhouse gas emissions in tonnes of CO₂ equivalent (tCO₂e), using publicly available financial and operational data.

This is framed as a **supervised regression** problem where:
- **Inputs**: Revenue, total assets, market capitalisation, employee count, sector, industry, country, and derived features
- **Outputs**: Predicted Scope 1 and Scope 2 emissions (continuous, positive values)

## 2. Data Sources

### 2.1 CDP Open Data
The Carbon Disclosure Project collects self-reported emissions data from ~13,000 companies globally. We use:
- Scope 1 emissions (direct from owned/controlled sources)
- Scope 2 emissions (indirect from purchased energy)
- Company identifiers and sector classification

### 2.2 Yahoo Finance
Financial fundamentals for publicly traded companies:
- Annual revenue, total assets, market capitalisation
- Employee count
- Sector and industry classification

### 2.3 World Bank
Country-level macroeconomic indicators:
- GDP per capita (proxy for economic development)
- Energy intensity (energy use per unit GDP — proxy for grid carbon intensity)

### 2.4 Synthetic Data
For development and demonstration, a calibrated synthetic data generator produces ~500 company records with:
- Log-normal distributions for financials (parameters calibrated to public CDP sector-level statistics)
- Emission factors scaled by sector and country energy intensity
- ~5% random missingness for realistic imputation testing

## 3. Preprocessing

| Step | Method | Rationale |
|---|---|---|
| Deduplication | (company_name, year) composite key | Prevent information leakage |
| Name normalisation | Lowercase, strip legal suffixes | Enable cross-source merging |
| Sector validation | Map to GICS taxonomy | Standardise classification |
| Country normalisation | ISO 3166-1 alpha-3 | Consistent geographic encoding |
| Missingness flags | Binary indicators before imputation | Capture informative missingness |
| Numeric imputation | Median | Robust to outliers |
| Categorical imputation | Mode | Most common category fill |
| Outlier clipping | Winsorisation at 1st/99th percentile | Reduce extreme value influence |
| Log transforms | `log1p(x)` | Reduce right-skew in financials and emissions |

## 4. Feature Engineering

### 4.1 Log Transforms
Financial data and emissions are heavily right-skewed. `log1p` compression:
- Stabilises variance
- Linearises multiplicative relationships
- Reduces outlier influence

### 4.2 Financial Ratios
| Feature | Formula | ESG Interpretation |
|---|---|---|
| `asset_per_employee` | Assets / Employees | Capital intensity per worker |
| `revenue_per_employee` | Revenue / Employees | Labour productivity |
| `asset_to_revenue` | Assets / Revenue | Asset intensity |
| `employee_density` | Employees / Assets | Labour/capital ratio |
| `capital_intensity` | Assets / Revenue | Physical capital requirements |

### 4.3 Group Aggregation Features
Sector-level and country-level mean log-emissions capture the "typical" emissions profile for a given segment.

> **Leakage Prevention**: These features are computed using a `GroupMeanEncoder` transformer that fits exclusively on training data within each cross-validation fold. At inference time, pre-computed group means from the full training set are used.

### 4.4 Categorical Encoding
Ordinal encoding maps sector, industry, and country strings to integers. Unknown categories at inference time receive a sentinel value (-1).

## 5. Model Selection

### 5.1 Candidate Models

| Model | Strengths | Weaknesses |
|---|---|---|
| Linear Regression | Interpretable, fast | Cannot capture non-linear relationships |
| Random Forest | Robust, handles outliers | Can overfit with many trees |
| Gradient Boosting | Strong performance | Slower training |
| XGBoost | Regularisation, speed | Requires careful tuning |
| CatBoost | Handles categoricals natively | Slower initial training |
| LightGBM | Fast, memory-efficient | Sensitive to overfitting on small data |

### 5.2 Hyperparameter Tuning
- **Method**: `RandomizedSearchCV` with 30 iterations
- **CV Strategy**: `GroupKFold` with sectors as groups (prevents same-sector data in both train and validation)
- **Scoring**: Negative mean squared error

### 5.3 Model Selection Criteria
The model with the highest **R²** on the held-out test set is selected as the production model. In case of ties, RMSE is used as a tiebreaker.

## 6. Validation Strategy

- **Hold-out**: 80% train / 20% test, stratified by binned target
- **Cross-validation**: 5-fold `GroupKFold` (sector-aware)
- **Metrics**: R², RMSE, MAE, MAPE
- **Sector-stratified evaluation**: Per-sector R² to identify weak segments

## 7. Interpretability

SHAP (SHapley Additive exPlanations) provides:
- **Global**: Which features drive emissions predictions on average
- **Local**: Feature contributions for individual company predictions
- **Interaction**: How pairs of features jointly influence predictions

`TreeExplainer` is used for tree-based models (exact, polynomial-time computation).

## 8. Limitations

1. **Self-selection bias**: CDP data is voluntarily reported; non-reporters may differ systematically
2. **Temporal lag**: Financial data and emissions are from different reporting periods
3. **Scope 3 excluded**: Value chain emissions are not modelled
4. **Sector granularity**: GICS level-1 sectors may be too coarse for some industries
5. **Country aggregation**: National-level energy intensity is a rough proxy for actual grid mix
6. **Synthetic data**: Development uses synthetic data; real-world accuracy requires real CDP data
