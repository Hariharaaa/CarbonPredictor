# Future Improvements

## Short-Term (v1.1)

### 🎯 Real CDP Data Integration
- Obtain actual CDP Open Data exports and validate model against real-world emissions
- Calibrate synthetic data generator against observed distributions
- Benchmark model accuracy against published CDP sector averages

### 📊 Enhanced Feature Engineering
- **Temporal features**: Year-over-year revenue growth, asset growth trends
- **Interaction features**: Sector × country interactions, revenue × employee interactions
- **Text features**: NLP on company descriptions from Yahoo Finance for additional signal
- **Polynomial features**: Quadratic terms for key financial ratios

### 🔧 Model Improvements
- **Optuna integration**: Bayesian hyperparameter optimization replacing RandomizedSearchCV
- **Stacking ensemble**: Meta-learner combining predictions from all 6 base models
- **Quantile regression**: Direct prediction of confidence intervals instead of heuristic estimation
- **Target encoding**: Replace ordinal encoding with smooth target encoding (with proper CV)

## Medium-Term (v2.0)

### 🌐 Scope 3 Estimation
- Extend model to predict Scope 3 (value chain) emissions
- Use industry-level input-output tables for upstream/downstream estimation
- Integrate supply chain databases for key sectors

### ⏳ Time-Series Modelling
- Predict emissions trajectories, not just point estimates
- Use LSTM or Temporal Fusion Transformer for multi-step forecasting
- Track company-level decarbonisation trends over time

### 📄 NLP from Sustainability Reports
- Extract emission-relevant information from annual reports and sustainability disclosures
- Use sentence transformers for document embedding
- Fine-tune language models on ESG-specific text

### 🔌 API Deployment
- FastAPI REST endpoint for programmatic access
- Batch prediction endpoint for portfolio-level analysis
- Authentication and rate limiting
- Docker containerisation

## Long-Term (v3.0)

### 🌍 Global Coverage
- Expand to cover private companies using alternative data sources
- Integrate with government registries (EU ETS, EPA GHGRP)
- Support regional regulatory frameworks (CSRD, SEC climate disclosure)

### 🤖 Active Learning
- Identify companies where model uncertainty is highest
- Prioritise data collection for these companies
- Human-in-the-loop validation workflow

### 📱 Mobile Dashboard
- React Native or Flutter mobile app for ESG analysts
- Push notifications for portfolio emissions changes
- Offline prediction capability

### 🏗️ MLOps Pipeline
- Automated retraining with new CDP data releases
- Model drift monitoring (data drift + concept drift)
- A/B testing for model versions
- Feature store for reusable feature computation
- ML experiment tracking (MLflow / Weights & Biases)

## Research Directions

1. **Causal inference**: Move beyond correlation to identify causal drivers of emissions
2. **Transfer learning**: Pre-train on large cross-sector datasets, fine-tune on specific sectors
3. **Graph neural networks**: Model supply chain relationships as graphs
4. **Conformal prediction**: Statistically rigorous prediction intervals
5. **Fairness auditing**: Ensure model doesn't systematically underestimate for specific regions/sectors
