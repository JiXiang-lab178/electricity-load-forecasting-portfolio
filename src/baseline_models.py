"""Evaluate persistence and rolling 48-step-ahead ARIMA baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Europe/London"
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "features_demand_data_2021_2025.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_RESULTS_FILE = RESULTS_DIR / "baseline_results.csv"
OUTPUT_PREDICTIONS_FILE = RESULTS_DIR / "baseline_predictions_2025.csv"
OUTPUT_ARIMA_VALIDATION_FILE = RESULTS_DIR / "arima_validation_results.csv"

ARIMA_CANDIDATE_ORDERS = (1, 2, 3)
FORECAST_HORIZON_PERIODS = 48
EXPECTED_TEST_YEAR = 2025
LAG_COLUMNS = {
    "lag_48": 48,
    "lag_336": 336,
}


def load_feature_data(input_file: Path) -> pd.DataFrame:
    """Load and fail-fast validate the chronological baseline input."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Input file is empty: {input_file}") from error
    if df.empty:
        raise ValueError(f"Input file contains no observations: {input_file}")

    required_columns = ["DATETIME", "ND", *LAG_COLUMNS]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required baseline columns: {missing_columns}")

    parsed_datetime = pd.to_datetime(
        df["DATETIME"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    numeric_columns = ["ND", *LAG_COLUMNS]
    parsed_numeric = {
        column: pd.to_numeric(df[column], errors="coerce")
        for column in numeric_columns
    }

    invalid_counts = {
        "DATETIME": int(parsed_datetime.isna().sum()),
        **{
            column: int(values.isna().sum())
            for column, values in parsed_numeric.items()
        },
    }
    if any(invalid_counts.values()):
        raise ValueError(
            f"Invalid required baseline values after parsing: {invalid_counts}"
        )

    df["DATETIME"] = parsed_datetime.dt.tz_convert(LOCAL_TIMEZONE)
    for column, values in parsed_numeric.items():
        df[column] = values.astype(float)

    non_finite_counts = {
        column: int((~np.isfinite(df[column])).sum())
        for column in numeric_columns
    }
    non_positive_counts = {
        column: int((df[column] <= 0).sum())
        for column in numeric_columns
    }
    if any(non_finite_counts.values()) or any(non_positive_counts.values()):
        raise ValueError(
            "Invalid baseline demand values: "
            f"non_finite={non_finite_counts}, "
            f"non_positive={non_positive_counts}."
        )

    df = df.sort_values("DATETIME").reset_index(drop=True)

    duplicate_timestamps = int(df["DATETIME"].duplicated().sum())
    utc_timestamps = df["DATETIME"].dt.tz_convert("UTC")
    invalid_intervals = int(
        utc_timestamps.diff().dropna().ne(pd.Timedelta(minutes=30)).sum()
    )
    if duplicate_timestamps or invalid_intervals:
        raise ValueError(
            "Baseline input is not a unique continuous half-hourly series: "
            f"duplicates={duplicate_timestamps}, "
            f"invalid_intervals={invalid_intervals}."
        )

    # Recompute each persistence lag from ND so stale or misaligned feature
    # columns cannot make a baseline appear artificially accurate.
    for lag_column, lag_periods in LAG_COLUMNS.items():
        expected_values = df["ND"].shift(lag_periods)
        comparable_rows = expected_values.notna()
        if not np.allclose(
            df.loc[comparable_rows, lag_column],
            expected_values.loc[comparable_rows],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{lag_column} is inconsistent with an ND shift of "
                f"{lag_periods} elapsed periods."
            )

    return df

def create_test_set(df: pd.DataFrame) -> pd.DataFrame:
    """Return the complete untouched 2025 half-hourly test year."""
    test_df = (
        df[df["DATETIME"].dt.year == EXPECTED_TEST_YEAR]
        .copy()
        .sort_values("DATETIME")
        .reset_index(drop=False)
        .rename(columns={"index": "SOURCE_INDEX"})
    )

    if test_df.empty:
        raise ValueError(f"No {EXPECTED_TEST_YEAR} data found in the dataset.")

    expected_timestamps = pd.date_range(
        start=pd.Timestamp(f"{EXPECTED_TEST_YEAR}-01-01", tz=LOCAL_TIMEZONE),
        end=pd.Timestamp(f"{EXPECTED_TEST_YEAR + 1}-01-01", tz=LOCAL_TIMEZONE),
        freq="30min",
        inclusive="left",
    )
    observed_timestamps = pd.DatetimeIndex(test_df["DATETIME"])
    if not observed_timestamps.equals(expected_timestamps):
        missing_timestamps = expected_timestamps.difference(observed_timestamps)
        extra_timestamps = observed_timestamps.difference(expected_timestamps)
        raise ValueError(
            f"{EXPECTED_TEST_YEAR} test coverage is incomplete: "
            f"expected_rows={len(expected_timestamps)}, "
            f"observed_rows={len(observed_timestamps)}, "
            f"missing_examples={missing_timestamps[:3].tolist()}, "
            f"extra_examples={extra_timestamps[:3].tolist()}."
        )

    return test_df


def fit_arima_p10(demand, autoregressive_order: int) -> np.ndarray:
    """Fit ARIMA(p,1,0) by conditional least squares.

    First differences are modelled as an AR(p) process with an intercept.
    The returned vector contains the intercept followed by p AR coefficients.
    """
    if autoregressive_order < 1:
        raise ValueError("ARIMA autoregressive order must be at least 1.")

    demand_array = np.asarray(demand, dtype=float)
    if demand_array.ndim != 1 or demand_array.size == 0:
        raise ValueError("ARIMA training demand must be a non-empty 1D series.")
    if not np.isfinite(demand_array).all():
        raise ValueError("ARIMA training demand contains non-finite values.")
    if (demand_array <= 0).any():
        raise ValueError("ARIMA training demand contains non-positive values.")

    # ARIMA(p,1,0) is fitted as an AR(p) model on first differences.
    differences = np.diff(demand_array)
    if len(differences) <= autoregressive_order:
        raise ValueError("Not enough observations to fit the requested ARIMA order.")

    target = differences[autoregressive_order:]
    lagged_columns = [
        differences[
            autoregressive_order - lag:
            len(differences) - lag
        ]
        for lag in range(1, autoregressive_order + 1)
    ]
    design_matrix = np.column_stack([
        np.ones(len(target)),
        *lagged_columns,
    ])
    coefficients, *_ = np.linalg.lstsq(design_matrix, target, rcond=None)
    validate_ar_stability(coefficients, autoregressive_order)
    return coefficients


def validate_ar_stability(
    coefficients,
    autoregressive_order: int,
) -> float:
    """Require the AR process on first differences to be stationary.

    For ``1 - phi_1 z - ... - phi_p z^p``, every root must lie outside
    the unit circle. The minimum root modulus is returned for reporting or tests.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if autoregressive_order < 1:
        raise ValueError("ARIMA autoregressive order must be at least 1.")
    if coefficients.shape != (autoregressive_order + 1,):
        raise ValueError("ARIMA coefficient count does not match its order.")
    if not np.isfinite(coefficients).all():
        raise ValueError("ARIMA coefficients contain non-finite values.")

    autoregressive_coefficients = coefficients[1:]
    polynomial = np.r_[-autoregressive_coefficients[::-1], 1.0]
    roots = np.roots(polynomial)
    minimum_root_modulus = float(np.min(np.abs(roots)))
    if minimum_root_modulus <= 1.0 + 1e-10:
        raise ValueError(
            "ARIMA differenced process is not stationary: "
            f"minimum_AR_root_modulus={minimum_root_modulus:.6f}."
        )
    return minimum_root_modulus


def rolling_arima_forecast(
    demand,
    target_indices,
    coefficients,
    autoregressive_order: int,
    horizon_periods: int = FORECAST_HORIZON_PERIODS,
) -> np.ndarray:
    """Generate leakage-safe recursive forecasts from each rolling origin.

    For target index t, the most recent level used is t - horizon_periods.
    Actual observations between that origin and t are never read.
    """
    demand_array = np.asarray(demand, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)
    coefficients = np.asarray(coefficients, dtype=float)

    if demand_array.ndim != 1 or demand_array.size == 0:
        raise ValueError("ARIMA demand must be a non-empty 1D series.")
    if not np.isfinite(demand_array).all():
        raise ValueError("ARIMA demand contains non-finite values.")
    if target_indices.ndim != 1:
        raise ValueError("ARIMA target indices must be one-dimensional.")
    if len(coefficients) != autoregressive_order + 1:
        raise ValueError("ARIMA coefficient count does not match its order.")
    validate_ar_stability(coefficients, autoregressive_order)
    if horizon_periods < 1:
        raise ValueError("Forecast horizon must be positive.")

    origin_indices = target_indices - horizon_periods
    if origin_indices.size == 0:
        return np.array([], dtype=float)
    if origin_indices.min() < autoregressive_order:
        raise ValueError("Insufficient pre-origin history for ARIMA forecasting.")
    if target_indices.max() >= len(demand_array):
        raise ValueError("ARIMA target index is outside the demand series.")

    differences = np.diff(demand_array)
    levels = demand_array[origin_indices].astype(float, copy=True)
    lag_state = np.column_stack([
        differences[origin_indices - lag]
        for lag in range(1, autoregressive_order + 1)
    ])

    intercept = coefficients[0]
    autoregressive_coefficients = coefficients[1:]

    # Forecast all 48 steps recursively; predicted differences, rather than
    # unavailable observations after the origin, update the AR state.
    for _ in range(horizon_periods):
        predicted_difference = (
            intercept + lag_state @ autoregressive_coefficients
        )
        levels += predicted_difference
        if autoregressive_order > 1:
            lag_state[:, 1:] = lag_state[:, :-1].copy()
        lag_state[:, 0] = predicted_difference

    if not np.isfinite(levels).all():
        raise ValueError("ARIMA produced non-finite forecasts.")
    if (levels <= 0).any():
        raise ValueError("ARIMA produced non-positive electricity-demand forecasts.")
    return levels


def select_arima_order(development_df: pd.DataFrame):
    """Select p on 2024 without accepting or accessing the 2025 test set."""
    required_columns = ["DATETIME", "ND"]
    missing_columns = [
        column for column in required_columns
        if column not in development_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing ARIMA development columns: {missing_columns}"
        )

    development_df = (
        development_df.sort_values("DATETIME")
        .reset_index(drop=True)
        .copy()
    )
    if (development_df["DATETIME"].dt.year >= 2025).any():
        raise ValueError("ARIMA development data must not contain 2025 rows.")

    training_df = development_df[
        development_df["DATETIME"].dt.year < 2024
    ]
    validation_df = development_df[
        development_df["DATETIME"].dt.year == 2024
    ]
    if training_df.empty or validation_df.empty:
        raise ValueError("ARIMA training or validation data is empty.")

    validation_rows = []
    demand = development_df["ND"].to_numpy()
    validation_indices = validation_df.index.to_numpy()

    # Every candidate is fitted before 2024 and ranked on the same 2024
    # validation year, leaving the 2025 test year untouched.
    for autoregressive_order in ARIMA_CANDIDATE_ORDERS:
        coefficients = fit_arima_p10(
            training_df["ND"],
            autoregressive_order,
        )
        minimum_root_modulus = validate_ar_stability(
            coefficients,
            autoregressive_order,
        )
        predictions = rolling_arima_forecast(
            demand,
            validation_indices,
            coefficients,
            autoregressive_order,
        )
        metrics = calculate_metrics(validation_df["ND"], predictions)
        validation_rows.append({
            "Order": f"ARIMA({autoregressive_order},1,0)",
            "Autoregressive_Order": autoregressive_order,
            "Forecast_Horizon_Periods": FORECAST_HORIZON_PERIODS,
            "Coefficients": json.dumps(coefficients.tolist()),
            "Minimum_AR_Root_Modulus": minimum_root_modulus,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
        })

        print(
            f"ARIMA({autoregressive_order},1,0) validation: "
            f"MAPE={metrics['MAPE']:.4f}%"
        )

    validation_results = pd.DataFrame(validation_rows).sort_values(
        ["MAPE", "RMSE", "Autoregressive_Order"]
    ).reset_index(drop=True)
    validation_results["Selected"] = False
    validation_results.loc[0, "Selected"] = True
    selected_order = int(validation_results.loc[0, "Autoregressive_Order"])
    return selected_order, validation_results

def calculate_metrics(y_true, y_pred) -> dict:
    """Calculate RMSE, MAE, and MAPE under a positive-demand contract."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Metric inputs must be one-dimensional.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Metric inputs must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Metric input lengths differ: true={len(y_true)}, pred={len(y_pred)}."
        )
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metric inputs contain non-finite values.")
    if (y_true <= 0).any():
        raise ValueError("MAPE requires strictly positive actual demand values.")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }


def evaluate_baselines(
    test_df: pd.DataFrame,
    arima_predictions,
    selected_order: int,
    final_coefficients,
) -> pd.DataFrame:
    """Evaluate two persistence baselines and the selected ARIMA baseline."""
    # y_true is the real National Demand of 2025.
    y_true = test_df["ND"]
    arima_predictions = np.asarray(arima_predictions, dtype=float)
    if len(arima_predictions) != len(test_df):
        raise ValueError(
            "ARIMA prediction count does not match the test set: "
            f"predictions={len(arima_predictions)}, rows={len(test_df)}."
        )
    minimum_root_modulus = validate_ar_stability(
        final_coefficients,
        selected_order,
    )

    baseline_models = {
        # Use demand from 48 half-hour periods (24 elapsed hours) earlier.
        "lag_48_baseline": test_df["lag_48"],
        # Use demand from 336 half-hour periods (168 elapsed hours) earlier.
        "lag_336_baseline": test_df["lag_336"],
        "ARIMA": arima_predictions,
    }
    # Create an empty list to store the three baseline results.
    results = []

    for model_name, y_pred in baseline_models.items():
        metrics = calculate_metrics(y_true, y_pred)
        results.append({
            "Model": model_name,
            "Selected_Order": (
                f"ARIMA({selected_order},1,0)"
                if model_name == "ARIMA"
                else "Not applicable"
            ),
            "Forecast_Horizon_Periods": FORECAST_HORIZON_PERIODS,
            "Final_Coefficients": (
                json.dumps(np.asarray(final_coefficients).tolist())
                if model_name == "ARIMA"
                else "Not applicable"
            ),
            "Minimum_AR_Root_Modulus": (
                minimum_root_modulus
                if model_name == "ARIMA"
                else "Not applicable"
            ),
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"]
        })
    # Convert results into DataFrame.
    results_df = pd.DataFrame(results)
    return results_df


def save_dataframe_atomically(df: pd.DataFrame, output_file: Path) -> None:
    """Write a result CSV only after its temporary file is complete."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_suffix(".csv.tmp")
    try:
        df.to_csv(temporary_output, index=False)
        temporary_output.replace(output_file)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def main() -> None:
    # Create results directory.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_data(INPUT_FILE)
    test_df = create_test_set(df)

    # ARIMA order selection is physically isolated from the 2025 test rows.
    development_df = (
        df[df["DATETIME"].dt.year < 2025]
        .copy()
        .reset_index(drop=True)
    )
    selected_order, validation_results = select_arima_order(development_df)
    save_dataframe_atomically(
        validation_results,
        OUTPUT_ARIMA_VALIDATION_FILE,
    )

    # Refit only the selected order on all development years before the single
    # final evaluation on 2025.
    final_training_df = df[df["DATETIME"].dt.year < 2025]
    final_coefficients = fit_arima_p10(
        final_training_df["ND"],
        selected_order,
    )
    arima_predictions = rolling_arima_forecast(
        df["ND"].to_numpy(),
        test_df["SOURCE_INDEX"].to_numpy(),
        final_coefficients,
        selected_order,
    )

    print("Feature data loaded.")
    print(f"Total rows: {len(df)}")
    print(f"Full date range: {df['DATETIME'].min()} to {df['DATETIME'].max()}")
    print(f"Test rows: {len(test_df)}")
    print(f"Test date range: {test_df['DATETIME'].min()} to {test_df['DATETIME'].max()}")

    results_df = evaluate_baselines(
        test_df,
        arima_predictions,
        selected_order,
        final_coefficients,
    )
    save_dataframe_atomically(results_df, OUTPUT_RESULTS_FILE)

    prediction_df = test_df[["DATETIME", "ND", "lag_48", "lag_336"]].copy()
    prediction_df["arima_prediction"] = arima_predictions
    if not np.isfinite(
        prediction_df[["ND", "lag_48", "lag_336", "arima_prediction"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("Baseline prediction output contains non-finite values.")
    save_dataframe_atomically(prediction_df, OUTPUT_PREDICTIONS_FILE)

    print("\nBaseline evaluation completed.")
    print(results_df)
    print(f"\nResults saved to: {OUTPUT_RESULTS_FILE}")
    print(f"ARIMA validation saved to: {OUTPUT_ARIMA_VALIDATION_FILE}")
    print(f"Predictions saved to: {OUTPUT_PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
