# Architecture

## System Architecture

```mermaid
graph TB
    subgraph Data Sources
        CDP["CDP Open Data<br/>(Emissions)"]
        YF["Yahoo Finance<br/>(Financials)"]
        WB["World Bank<br/>(Macro Indicators)"]
        SYN["Synthetic Generator<br/>(Fallback)"]
    end

    subgraph Ingestion Layer
        ING["ingestion.py<br/>━━━━━━━━━━━━━<br/>CDPIngestion<br/>YahooFinanceIngestion<br/>WorldBankIngestion<br/>merge_all_sources()"]
    end

    subgraph Processing Pipeline
        PRE["preprocessing.py<br/>━━━━━━━━━━━━━<br/>Dedup → Normalise →<br/>Validate → Impute →<br/>Clip → Log Transform"]
        FE["feature_engineering.py<br/>━━━━━━━━━━━━━<br/>Ratios + Logs +<br/>Encodings + GroupMeans"]
    end

    subgraph Model Training
        TR["train.py<br/>━━━━━━━━━━━━━<br/>6 Models ×<br/>RandomizedSearchCV ×<br/>GroupKFold CV"]
        EV["evaluate.py<br/>━━━━━━━━━━━━━<br/>R² RMSE MAE MAPE<br/>Diagnostic Plots"]
        SH["shap_analysis.py<br/>━━━━━━━━━━━━━<br/>TreeExplainer<br/>Summary + Waterfall +<br/>Force + Dependence"]
    end

    subgraph Deployment
        PR["predict.py<br/>CarbonPredictor"]
        ST["Streamlit App<br/>streamlit_app.py"]
    end

    CDP --> ING
    YF --> ING
    WB --> ING
    SYN -.-> ING
    ING --> PRE
    PRE --> FE
    FE --> TR
    TR --> EV
    TR --> SH
    TR --> PR
    PR --> ST
    EV --> ST
    SH --> ST
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input
        R["Raw CSV / API"]
    end

    subgraph Transform
        M["Merged<br/>merged_raw.csv"]
        C["Cleaned<br/>clean.csv"]
        F["Features<br/>features.csv"]
    end

    subgraph Split
        TR["Train Set<br/>(80%)"]
        TE["Test Set<br/>(20%)"]
    end

    subgraph Output
        MOD["Best Model<br/>(.joblib)"]
        REP["Reports<br/>(CSV + HTML)"]
        SHA["SHAP Values<br/>(.joblib)"]
    end

    R --> M --> C --> F
    F --> TR
    F --> TE
    TR --> MOD
    TE --> REP
    MOD --> SHA
```

## Module Dependencies

```mermaid
graph LR
    CFG["config.py"] --> UTL["utils.py"]
    CFG --> ING["ingestion.py"]
    CFG --> SYN["synthetic_data.py"]
    CFG --> PRE["preprocessing.py"]
    CFG --> FE["feature_engineering.py"]
    CFG --> TR["train.py"]
    CFG --> EV["evaluate.py"]
    CFG --> SH["shap_analysis.py"]
    CFG --> PR["predict.py"]
    CFG --> ST["streamlit_app.py"]

    UTL --> ING
    UTL --> PRE
    UTL --> FE
    UTL --> EV

    SYN --> ING
    EV --> TR
    FE --> TR
    PR --> ST
    SYN --> PR
    SYN --> ST

    style CFG fill:#1a1a2e
    style ST fill:#16213e
```

## Key Design Decisions

### 1. Leakage-Safe Group Aggregation
Group-level features (`sector_mean_log_scope1`, etc.) are implemented as a sklearn `TransformerMixin` that fits only on training data within each CV fold.

### 2. Dual-Mode Ingestion
Every data source has a primary (API) and fallback (CSV) path, plus a final fallback to synthetic data. This ensures the pipeline is always runnable.

### 3. Log-Space Modelling
Models predict `log1p(emissions)` rather than raw emissions. This:
- Handles the heavy right-skew in emissions data
- Naturally produces positive predictions after `expm1`
- Stabilises gradient-based optimization

### 4. Best-Model-Only Deployment
All 6 models are trained and compared, but only the best (by R²) is loaded by the Streamlit app. This keeps the deployment footprint minimal.
