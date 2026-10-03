# Electricity Load Forecasting with Machine Learning

An end-to-end Python workflow for forecasting half-hourly electricity demand in Great Britain using public NESO data.

This project demonstrates practical skills in data preparation, time-aware feature engineering, baseline modelling, machine-learning model comparison, robustness analysis, testing, and reproducible execution.

## Project highlights

- Processes half-hourly National Demand (`ND`) measurements in MW.
- Handles UK daylight-saving-time changes using timezone-aware timestamps.
- Creates calendar, lag, and bank-holiday features.
- Compares persistence baselines with Ridge Regression, Random Forest, XGBoost, and LightGBM.
- Evaluates models with RMSE, MAE, and MAPE.
- Uses chronological train, validation, and final-test splits to avoid information leakage.
- Includes feature ablation, training-period sensitivity, distribution-shift analysis, and fixed-origin performance monitoring.
- Includes automated regression tests for the data and modelling pipeline.

## Data

The workflow uses the public [NESO Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data). Bank-holiday dates are taken from the public [GOV.UK Bank Holidays API](https://www.gov.uk/bank-holidays.json).

Raw demand files are intentionally not included in this repository. Download the annual files for 2021–2025 and place them in `data/raw/` using these names:

```text
data/raw/demanddata_2021.csv
data/raw/demanddata_2022.csv
data/raw/demanddata_2023.csv
data/raw/demanddata_2024.csv
data/raw/demanddata_2025.csv
```

## Repository structure

```text
.
├── data/
│   └── external/          # Versioned public holiday data
├── notebooks/             # Analysis notebook
├── src/                   # Data, feature, modelling, and analysis modules
├── tests/                 # Automated tests
├── requirements.txt       # Python dependencies
└── run_pipeline.py        # Reproducible pipeline entry point
```

## Setup

Python 3.11 or newer is recommended. From PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the pipeline

Preview the planned steps:

```powershell
python run_pipeline.py --dry-run
```

Run the complete workflow after adding the source data:

```powershell
python run_pipeline.py
```

To skip the slower robustness analysis:

```powershell
python run_pipeline.py --skip-robustness
```

## Reproducibility and evaluation

Model-selection decisions use earlier chronological periods. The final test period is kept separate from feature selection and hyperparameter selection, preserving a realistic forecasting evaluation and reducing the risk of temporal data leakage.

## Licence and attribution

This project uses publicly available data from NESO and GOV.UK. Please follow the terms of the original data sources when reusing the data or results.
