"""Evaluate final-year, seasonal and rolling-origin forecast robustness."""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_ml_models import (
    calculate_metrics,
    load_feature_data,
    get_models,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_complete_year_coverage,
)
from baseline_models import fit_arima_p10, rolling_arima_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Europe/London"

RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

BASELINE_PREDICTIONS_FILE = RESULTS_TABLES_DIR / "baseline_predictions_2025.csv"
ML_PREDICTIONS_FILE = RESULTS_TABLES_DIR / "ml_model_predictions.csv"

OUTPUT_MONTHLY_FILE = RESULTS_TABLES_DIR / "monthly_model_performance.csv"
OUTPUT_SEASONAL_FILE = RESULTS_TABLES_DIR / "seasonal_model_performance.csv"

FEATURE_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_demand_data_2021_2025.csv"
)

OUTPUT_ROLLING_YEARLY_FILE = (
    RESULTS_TABLES_DIR
    / "rolling_origin_yearly_performance.csv"
)
# Map readable model labels to the saved prediction columns used for the 2025
# monthly and seasonal error analysis.
BASELINE_MODEL_COLUMNS = {
    "lag_48_baseline": "lag_48",
    "lag_336_baseline": "lag_336",
    "ARIMA(3,1,0) selected": "arima_prediction",
}
ML_MODEL_COLUMNS = {
    "Ridge Regression": "ridge_regression_prediction",
    "Random Forest": "random_forest_prediction",
    "XGBoost": "xgboost_prediction",
    "LightGBM": "lightgbm_prediction",
}
MODEL_COLUMNS = {**BASELINE_MODEL_COLUMNS, **ML_MODEL_COLUMNS}
ROLLING_TEST_YEARS = (2022, 2023, 2024, 2025)
ROLLING_ARIMA_ORDER = 2
ROLLING_ARIMA_MODEL_NAME = "ARIMA(2,1,0) fixed"
ROLLING_ML_CONFIGURATIONS = {
    "Ridge Regression": "Base alpha=1.0 with training-only standardisation",
    "Random Forest": "Base n_estimators=100, max_depth=None, random_state=42",
    "XGBoost": "Base n_estimators=200, max_depth=6, random_state=42",
    "LightGBM": "Base n_estimators=200, num_leaves=31, random_state=42",
}


def load_prediction_data() -> pd.DataFrame:
    """
    Load baseline and machine learning prediction files,
    then merge them.
    """
    if not BASELINE_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {BASELINE_PREDICTIONS_FILE}"
        )

    if not ML_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {ML_PREDICTIONS_FILE}"
        )

    # Load predictions produced by the official baseline and ML stages rather
    # than refitting models for the final-year subgroup analysis.
    baseline_df = pd.read_csv(BASELINE_PREDICTIONS_FILE)
    ml_df = pd.read_csv(ML_PREDICTIONS_FILE)

    required_baseline_columns = {"DATETIME", "ND"} | set(
        BASELINE_MODEL_COLUMNS.values()
    )
    required_ml_columns = {"DATETIME", "ND"} | set(
        ML_MODEL_COLUMNS.values()
    )

    missing_baseline_columns = required_baseline_columns - set(baseline_df.columns)
    missing_ml_columns = required_ml_columns - set(ml_df.columns)

    if missing_baseline_columns:
        raise ValueError(
            "Missing baseline prediction columns: "
            f"{sorted(missing_baseline_columns)}"
        )
    if missing_ml_columns:
        raise ValueError(
            f"Missing machine-learning prediction columns: {sorted(missing_ml_columns)}"
        )

    baseline_df["DATETIME"] = (
        pd.to_datetime(
            baseline_df["DATETIME"],
            format="mixed",
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    ml_df["DATETIME"] = (
        pd.to_datetime(
            ml_df["DATETIME"],
            format="mixed",
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    for dataset_name, prediction_df in {
        "baseline predictions": baseline_df,
        "machine-learning predictions": ml_df,
    }.items():
        invalid_datetime_count = int(prediction_df["DATETIME"].isna().sum())
        if invalid_datetime_count:
            raise ValueError(
                f"{dataset_name} contain {invalid_datetime_count} invalid DATETIME values."
            )

        duplicate_datetime_count = int(prediction_df["DATETIME"].duplicated().sum())
        if duplicate_datetime_count:
            raise ValueError(
                f"{dataset_name} contain {duplicate_datetime_count} duplicate DATETIME values."
            )

        prediction_columns = (
            BASELINE_MODEL_COLUMNS.values()
            if dataset_name == "baseline predictions"
            else ML_MODEL_COLUMNS.values()
        )
        numeric_columns = ["ND", *prediction_columns]
        parsed_numeric = {
            column: pd.to_numeric(prediction_df[column], errors="coerce")
            for column in numeric_columns
        }
        invalid_numeric_counts = {
            column: int(values.isna().sum())
            for column, values in parsed_numeric.items()
        }
        if any(invalid_numeric_counts.values()):
            raise ValueError(
                f"{dataset_name} contain invalid numeric values: "
                f"{invalid_numeric_counts}."
            )
        for column, values in parsed_numeric.items():
            prediction_df[column] = values.astype(float)

        non_finite_counts = {
            column: int((~np.isfinite(prediction_df[column])).sum())
            for column in numeric_columns
        }
        if any(non_finite_counts.values()):
            raise ValueError(
                f"{dataset_name} contain non-finite numeric values: "
                f"{non_finite_counts}."
            )
        if (prediction_df["ND"] <= 0).any():
            raise ValueError(
                f"{dataset_name} contain non-positive actual demand values."
            )

    if len(baseline_df) != len(ml_df):
        raise ValueError(
            "Prediction files have different row counts: "
            f"baseline={len(baseline_df)}, machine_learning={len(ml_df)}."
        )

    baseline_df = baseline_df[
        ["DATETIME", "ND", *BASELINE_MODEL_COLUMNS.values()]
    ].rename(columns={"ND": "baseline_ND"})

    # Require a one-to-one timestamp match before comparing model errors.
    merged_df = pd.merge(
        ml_df,
        baseline_df,
        on="DATETIME",
        how="inner",
        validate="one_to_one"
    )

    if len(merged_df) != len(ml_df):
        raise ValueError(
            "Prediction timestamps do not match exactly between the two files."
        )

    # Both files must contain the identical target series; otherwise the model
    # comparison would mix predictions made for different observations.
    if not np.array_equal(merged_df["ND"], merged_df["baseline_ND"]):
        raise ValueError(
            "Actual ND values do not match between the two prediction files."
        )

    merged_df = (
        merged_df
        .drop(columns="baseline_ND")
        .sort_values("DATETIME")
        .reset_index(drop=True)
    )
    validate_complete_year_coverage(
        merged_df,
        2025,
        "Merged robustness prediction set",
    )

    merged_df["month"] = merged_df["DATETIME"].dt.month
    merged_df["month_name"] = merged_df["DATETIME"].dt.month_name()

    return merged_df


def get_season(month: int) -> str:
    """
    Convert month number into UK-style seasons.
    """
    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    else:
        return "Autumn"


def evaluate_by_group(
    df: pd.DataFrame,
    group_column: str
) -> pd.DataFrame:
    """
    Evaluate each model by month or season.
    """
    results = []

    # Every model is scored on the same rows within each month or season.
    for group_value, group_df in df.groupby(group_column):

        y_true = group_df["ND"]

        for model_name, prediction_column in MODEL_COLUMNS.items():

            y_pred = group_df[prediction_column]

            metrics = calculate_metrics(
                y_true,
                y_pred
            )

            results.append({
                group_column: group_value,
                "Model": model_name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "Rows": len(group_df),
            })

    return pd.DataFrame(results)

def evaluate_rolling_origin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate model performance across different test years
    using an expanding-window rolling-origin strategy.
    """

    df = df.sort_values("DATETIME").reset_index(drop=True)
    results = []

    # Move the origin forward one year at a time and use only earlier years for
    # the corresponding expanding-window training set.
    for test_year in ROLLING_TEST_YEARS:

        print(f"\nRolling-origin evaluation: Test year {test_year}")

        train_df = df[
            df["DATETIME"].dt.year < test_year
        ].copy()

        test_df = df[
            df["DATETIME"].dt.year == test_year
        ].copy()

        if train_df.empty:
            raise ValueError(
                f"Rolling-origin training set is empty before {test_year}."
            )
        validate_complete_year_coverage(
            test_df,
            test_year,
            "Rolling-origin test set",
        )

        training_years = sorted(train_df["DATETIME"].dt.year.unique())
        training_label = f"{training_years[0]}-{training_years[-1]}"

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN]

        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]

        # Persistence baselines provide a no-training reference for each year.
        baseline_models = {
            "lag_48_baseline": (
                test_df["lag_48"],
                "Observed demand 48 elapsed periods earlier",
            ),
            "lag_336_baseline": (
                test_df["lag_336"],
                "Observed demand 336 elapsed periods earlier",
            ),
        }

        # Use a fixed ARIMA(2,1,0) base configuration here so that 2024
        # validation choices cannot leak into earlier robustness years.
        arima_coefficients = fit_arima_p10(
            train_df[TARGET_COLUMN],
            ROLLING_ARIMA_ORDER,
        )
        baseline_models[ROLLING_ARIMA_MODEL_NAME] = (
            rolling_arima_forecast(
                df[TARGET_COLUMN].to_numpy(),
                test_df.index.to_numpy(),
                arima_coefficients,
                ROLLING_ARIMA_ORDER,
            ),
            "Fixed ARIMA(2,1,0); no future-year order selection",
        )

        for model_name, (y_pred, configuration) in baseline_models.items():

            metrics = calculate_metrics(
                y_test,
                y_pred
            )

            results.append({
                "Test_Year": test_year,
                "Model": model_name,
                "Model_Type": (
                    "Fixed statistical model"
                    if model_name == ROLLING_ARIMA_MODEL_NAME
                    else "Persistence baseline"
                ),
                "Training_Years": (
                    training_label
                    if model_name == ROLLING_ARIMA_MODEL_NAME
                    else "Not applicable"
                ),
                "Training_Rows": (
                    len(train_df)
                    if model_name == ROLLING_ARIMA_MODEL_NAME
                    else 0
                ),
                "Test_Rows": len(test_df),
                "Configuration": configuration,
                "Future_Year_Validation_Used": False,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
            })

        # Recreate the same fixed base configurations at every origin so future
        # validation choices cannot influence comparisons with earlier years.
        models = get_models()

        for model_name, model in models.items():

            print(f"Training: {model_name}")

            model.fit(
                X_train,
                y_train
            )

            y_pred = model.predict(
                X_test
            )

            metrics = calculate_metrics(
                y_test,
                y_pred
            )

            results.append({
                "Test_Year": test_year,
                "Model": model_name,
                "Model_Type": "Machine-learning base configuration",
                "Training_Years": training_label,
                "Training_Rows": len(train_df),
                "Test_Rows": len(test_df),
                "Configuration": ROLLING_ML_CONFIGURATIONS[model_name],
                "Future_Year_Validation_Used": False,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
            })

    return pd.DataFrame(results)
def plot_monthly_mape(
    monthly_results: pd.DataFrame
):
    """
    Plot monthly MAPE trend for all models.
    """
    plt.figure(figsize=(12, 6))

    for model_name in monthly_results["Model"].unique():

        model_df = monthly_results[
            monthly_results["Model"] == model_name
        ].copy()

        model_df = model_df.sort_values("month")

        plt.plot(
            model_df["month"],
            model_df["MAPE"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Month")
    plt.ylabel("MAPE (%)")
    plt.title("Monthly MAPE Trend by Model in 2025")
    plt.xticks(range(1, 13))
    plt.legend()
    plt.tight_layout()

    output_file = (
        RESULTS_FIGURES_DIR
        / "monthly_mape_trend.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(f"Saved: {output_file}")


def plot_seasonal_mape(
    seasonal_results: pd.DataFrame
):
    """
    Plot seasonal MAPE comparison.
    """
    season_order = [
        "Winter",
        "Spring",
        "Summer",
        "Autumn"
    ]

    pivot_df = seasonal_results.pivot(
        index="season",
        columns="Model",
        values="MAPE"
    )

    pivot_df = pivot_df.reindex(
        season_order
    )

    ax = pivot_df.plot(
        kind="bar",
        figsize=(12, 6)
    )

    ax.set_xlabel("Season")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(
        "Seasonal MAPE Comparison by Model in 2025"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_file = (
        RESULTS_FIGURES_DIR
        / "seasonal_mape_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(f"Saved: {output_file}")

def plot_rolling_origin_mape(
    yearly_results: pd.DataFrame
):
    """
    Plot MAPE across rolling-origin test years.
    """

    plt.figure(figsize=(12, 6))

    for model_name in yearly_results["Model"].unique():

        model_df = yearly_results[
            yearly_results["Model"] == model_name
        ].copy()

        model_df = model_df.sort_values(
            "Test_Year"
        )

        plt.plot(
            model_df["Test_Year"],
            model_df["MAPE"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Test Year")
    plt.ylabel("MAPE (%)")

    plt.title(
        "Rolling-Origin MAPE Across Test Years"
    )

    plt.xticks(
        [2022, 2023, 2024, 2025]
    )

    plt.legend()
    plt.tight_layout()

    output_file = (
        RESULTS_FIGURES_DIR
        / "rolling_origin_mape_by_year.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(f"Saved: {output_file}")

def main():
    RESULTS_TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # First analyse monthly and seasonal variation inside the untouched 2025
    # prediction set.
    df = load_prediction_data()

    df["season"] = df["month"].apply(
        get_season
    )

    print("Prediction data loaded.")
    print(f"Rows: {len(df)}")

    print(
        f"Date range: "
        f"{df['DATETIME'].min()} "
        f"to {df['DATETIME'].max()}"
    )

    monthly_results = evaluate_by_group(
        df,
        "month"
    )

    seasonal_results = evaluate_by_group(
        df,
        "season"
    )

    monthly_results.to_csv(
        OUTPUT_MONTHLY_FILE,
        index=False
    )

    seasonal_results.to_csv(
        OUTPUT_SEASONAL_FILE,
        index=False
    )

    print("\nMonthly model performance:")
    print(monthly_results)

    print("\nSeasonal model performance:")
    print(seasonal_results)

    plot_monthly_mape(
        monthly_results
    )

    plot_seasonal_mape(
        seasonal_results
    )
    print("\nStarting rolling-origin robustness analysis...")

    # Then refit expanding-window base models separately for each test year.
    feature_df = load_feature_data(
        FEATURE_DATA_FILE
    )

    rolling_results = evaluate_rolling_origin(
        feature_df
    )

    rolling_results.to_csv(
        OUTPUT_ROLLING_YEARLY_FILE,
        index=False
    )

    print("\nRolling-origin yearly performance:")
    print(rolling_results)

    plot_rolling_origin_mape(
        rolling_results
    )
    print("\nRobustness analysis completed.")

    print(
        f"Monthly results saved to: "
        f"{OUTPUT_MONTHLY_FILE}"
    )

    print(
        f"Seasonal results saved to: "
        f"{OUTPUT_SEASONAL_FILE}"
    )


if __name__ == "__main__":
    main()
