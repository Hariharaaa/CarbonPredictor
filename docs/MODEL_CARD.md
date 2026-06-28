# Model Card — Corporate Carbon Footprint Predictor

*Following the [Google Model Card](https://arxiv.org/abs/1810.03993) framework.*

---

## Model Details

| Field | Value |
|---|---|
| **Developer** | Harihara |
| **Model date** | 2026 |
| **Model version** | 1.0.0 |
| **Model type** | Supervised regression (gradient-boosted trees) |
| **Framework** | scikit-learn / XGBoost / CatBoost / LightGBM |
| **Input** | Company financial & operational features (13 engineered features) |
| **Output** | Scope 1 & Scope 2 GHG emissions (tCO₂e) |
| **License** | MIT |

## Intended Use

### Primary Use Cases
- **ESG screening**: Estimate emissions for companies that do not publicly disclose
- **Portfolio analysis**: Carbon footprint assessment across investment portfolios
- **Due diligence**: Initial emissions estimate before detailed verification
- **Research**: Academic study of the relationship between financials and emissions

### Out-of-Scope Use Cases
- **Regulatory compliance**: This model should NOT be used for official emissions reporting
- **Carbon credit verification**: Predictions are estimates, not measurements
- **Individual facility assessment**: Model operates at the company level only

## Training Data

### Sources
- CDP Open Data Portal (company-level emissions disclosures)
- Yahoo Finance (financial fundamentals)
- World Bank (country-level macro indicators)

### Composition
- ~500 companies across 11 GICS sectors and 15 countries
- Reporting years: 2020–2023
- Scope 1 and Scope 2 emissions as target variables

### Known Biases
- **Self-selection**: Only companies that voluntarily report to CDP are represented
- **Size bias**: Larger companies are over-represented in CDP data
- **Geographic bias**: Developed economies are over-represented
- **Sector coverage**: Some sectors (e.g., Financials) have lower emission variance

## Evaluation

### Metrics
| Metric | Description |
|---|---|
| R² | Proportion of variance explained (higher is better) |
| RMSE | Root mean squared error in tCO₂e (lower is better) |
| MAE | Mean absolute error in tCO₂e (lower is better) |
| MAPE | Mean absolute percentage error (lower is better) |

### Validation Strategy
- 80/20 train-test split (stratified by binned target)
- 5-fold GroupKFold cross-validation (sector-aware)
- Sector-stratified performance breakdown

## Ethical Considerations

### Potential Harms
- **Greenwashing**: Companies could use low predictions to claim environmental responsibility
- **Misallocation**: Investment decisions based on inaccurate estimates could misallocate capital
- **Anchoring**: Predictions could anchor stakeholder expectations at incorrect levels

### Mitigations
- Confidence intervals are provided with every prediction
- Model limitations are clearly documented
- The system is positioned as a screening tool, not a measurement system

## Limitations and Recommendations

1. Predictions are **estimates** based on statistical relationships, not physical measurement
2. Accuracy varies significantly by sector — Energy and Utilities tend to be more predictable
3. Always verify predictions against actual disclosure data when available
4. Do not use for Scope 3 estimation — the model does not cover value chain emissions
5. Re-train periodically as new CDP data becomes available

## Caveats and Recommendations

Users should:
- Treat predictions as order-of-magnitude estimates
- Report confidence intervals alongside point predictions
- Validate against known disclosures where possible
- Understand that the model reflects historical patterns, not future commitments
