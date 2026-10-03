"""Tune, fit and compare final machine-learning demand forecasts.

Models use 2021-2023 for parameter fitting, 2024 for hyperparameter validation,
and 2025 as an untouched final test set.
"""

import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from feature_config import FINAL_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Europe/London"
INPUT_FILE = (PROJECT_ROOT / "data" / "processed" / "features_demand_data_2021_2025.csv")
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_RESULTS_FILE = (RESULTS_DIR / "ml_model_results.csv")
OUTPUT_VALIDATION_FILE = (RESULTS_DIR / "ml_validation_results.csv")
# Keep row-level 2025 predictions for visualisation and paired model comparison.
OUTPUT_PREDICTIONS_FILE = (RESULTS_DIR / "ml_model_predictions.csv")
OUTPUT_COMPARISON_FILE = (RESULTS_DIR / "ml_model_comparison_bootstrap.csv")

EXPECTED_VALIDATION_YEAR = 2024
EXPECTED_TEST_YEAR = 2025
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

# These eight variables were selected by the 2023 feature-ablation experiment
# and are imported from the same frozen configuration used by that experiment.
# The rolling means remain in the feature dataset so their contribution can be
# measured, but they are not used by the final models because they increased
# validation error relative to the lag-calendar-holiday configuration.
FEATURE_COLUMNS = list(FINAL_FEATURE_COLUMNS)
TARGET_COLUMN = "ND"

def load_feature_data(input_file: Path) -> pd.DataFrame:
    """Load model features and fail fast on malformed chronological data."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Input file is empty: {input_file}") from error
    if df.empty:
        raise ValueError(f"Input file contains no observations: {input_file}")

    required_columns = ["DATETIME", TARGET_COLUMN, *FEATURE_COLUMNS]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required model columns: {missing_columns}")

    parsed_datetime = pd.to_datetime(
        df["DATETIME"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    numeric_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
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
            f"Invalid required model values after parsing: {invalid_counts}"
        )

    df["DATETIME"] = parsed_datetime.dt.tz_convert(LOCAL_TIMEZONE)
    for column, values in parsed_numeric.items():
        df[column] = values.astype(float)

    non_finite_counts = {
        column: int((~np.isfinite(df[column])).sum())
        for column in numeric_columns
    }
    if any(non_finite_counts.values()):
        raise ValueError(
            f"Non-finite required model values: {non_finite_counts}"
        )
    if (df[TARGET_COLUMN] <= 0).any():
        invalid_target_count = int((df[TARGET_COLUMN] <= 0).sum())
        raise ValueError(
            "Model target demand must be strictly positive: "
            f"invalid_rows={invalid_target_count}."
        )

    df = df.sort_values("DATETIME").reset_index(drop=True)

    duplicate_timestamps = int(df["DATETIME"].duplicated().sum())
    utc_timestamps = df["DATETIME"].dt.tz_convert("UTC")
    invalid_intervals = int(
        utc_timestamps.diff().dropna().ne(pd.Timedelta(minutes=30)).sum()
    )
    if duplicate_timestamps or invalid_intervals:
        raise ValueError(
            "Model input is not a unique continuous half-hourly series: "
            f"duplicates={duplicate_timestamps}, "
            f"invalid_intervals={invalid_intervals}."
        )

    return df


def validate_complete_year_coverage(
    df: pd.DataFrame,
    year: int,
    dataset_label: str,
) -> None:
    """Require every elapsed half-hour in one local calendar year."""
    observed_timestamps = pd.DatetimeIndex(
        df.loc[df["DATETIME"].dt.year == year, "DATETIME"].sort_values()
    )
    expected_timestamps = pd.date_range(
        start=pd.Timestamp(f"{year}-01-01", tz=LOCAL_TIMEZONE),
        end=pd.Timestamp(f"{year + 1}-01-01", tz=LOCAL_TIMEZONE),
        freq="30min",
        inclusive="left",
    )
    if not observed_timestamps.equals(expected_timestamps):
        missing_timestamps = expected_timestamps.difference(observed_timestamps)
        extra_timestamps = observed_timestamps.difference(expected_timestamps)
        raise ValueError(
            f"{dataset_label} coverage is incomplete for {year}: "
            f"expected_rows={len(expected_timestamps)}, "
            f"observed_rows={len(observed_timestamps)}, "
            f"missing_examples={missing_timestamps[:3].tolist()}, "
            f"extra_examples={extra_timestamps[:3].tolist()}."
        )


def train_validation_test_split_by_year(df: pd.DataFrame):
    """
    Use 2021-2023 for parameter fitting, 2024 for validation, and 2025
    for the final test. The final model is refitted on 2021-2024 only
    after its hyperparameters have been selected.
    """
    train_df = df[df["DATETIME"].dt.year < 2024].copy()
    validation_df = df[df["DATETIME"].dt.year == 2024].copy()
    final_train_df = df[df["DATETIME"].dt.year < 2025].copy()
    test_df = df[df["DATETIME"].dt.year == 2025].copy()

    if train_df.empty:
        raise ValueError("Training set is empty.")
    if validation_df.empty:
        raise ValueError("Validation set is empty.")
    if test_df.empty:
        raise ValueError("Test set is empty.")

    return train_df, validation_df, final_train_df, test_df


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


def moving_block_bootstrap_mean_ci(
    values,
    block_length: int = BOOTSTRAP_BLOCK_DAYS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Return a moving-block bootstrap interval for serial daily values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty 1D array.")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap values contain non-finite observations.")
    if block_length < 1 or n_resamples < 2:
        raise ValueError("Bootstrap block length and resample count are invalid.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap confidence level must be between zero and one.")

    # Resample consecutive days in blocks so the interval does not assume that
    # neighbouring forecast errors are independent.
    block_length = min(block_length, len(values))
    blocks_per_resample = int(np.ceil(len(values) / block_length))
    maximum_start = len(values) - block_length
    rng = np.random.default_rng(random_state)
    starts = rng.integers(
        0,
        maximum_start + 1,
        size=(n_resamples, blocks_per_resample),
    )
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(n_resamples, -1)
    sampled_means = values[indices[:, :len(values)]].mean(axis=1)

    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        sampled_means,
        [tail_probability, 1.0 - tail_probability],
    )
    return float(lower), float(upper)


def calculate_daily_mape(
    datetimes,
    y_true,
    y_pred,
) -> pd.Series:
    """Aggregate half-hourly absolute percentage errors to local-day MAPE."""
    calculate_metrics(y_true, y_pred)
    datetime_series = pd.Series(datetimes).reset_index(drop=True)
    if datetime_series.isna().any():
        raise ValueError("Daily MAPE datetimes contain missing values.")

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    daily_frame = pd.DataFrame({
        "Local_Date": datetime_series.dt.date,
        "Absolute_Percentage_Error": (
            np.abs((y_true_array - y_pred_array) / y_true_array) * 100.0
        ),
    })
    return daily_frame.groupby("Local_Date")["Absolute_Percentage_Error"].mean()


def compare_model_predictions(
    predictions_df: pd.DataFrame,
    model_prediction_columns: dict[str, str],
) -> pd.DataFrame:
    """Make pre-specified paired comparisons of every final ML model."""
    required_columns = {"DATETIME", TARGET_COLUMN, *model_prediction_columns.values()}
    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns for model comparison: {sorted(missing_columns)}"
        )

    overall_mape = {}
    daily_mape = {}
    for model_name, prediction_column in model_prediction_columns.items():
        overall_mape[model_name] = calculate_metrics(
            predictions_df[TARGET_COLUMN],
            predictions_df[prediction_column],
        )["MAPE"]
        daily_mape[model_name] = calculate_daily_mape(
            predictions_df["DATETIME"],
            predictions_df[TARGET_COLUMN],
            predictions_df[prediction_column],
        )

    # Pair models on the same local days before bootstrapping their MAPE
    # difference; this controls for easy and difficult evaluation days.
    point_estimate_order = sorted(overall_mape, key=overall_mape.get)
    comparison_rows = []
    for model_a, model_b in combinations(point_estimate_order, 2):
        aligned_errors = pd.concat(
            [daily_mape[model_a], daily_mape[model_b]],
            axis=1,
            keys=[model_a, model_b],
            join="inner",
        ).dropna()
        daily_delta = (
            aligned_errors[model_a] - aligned_errors[model_b]
        ).to_numpy()
        ci_lower, ci_upper = moving_block_bootstrap_mean_ci(daily_delta)
        ci_includes_zero = ci_lower <= 0.0 <= ci_upper

        if ci_includes_zero:
            interpretation = (
                "Statistically comparable at the 95% level "
                "(paired interval includes zero)."
            )
        elif ci_upper < 0.0:
            interpretation = (
                "Model A has lower mean daily MAPE "
                "(paired 95% interval excludes zero)."
            )
        else:
            interpretation = (
                "Model B has lower mean daily MAPE "
                "(paired 95% interval excludes zero)."
            )

        comparison_rows.append({
            "Model_A": model_a,
            "Model_B": model_b,
            "Overall_MAPE_A": overall_mape[model_a],
            "Overall_MAPE_B": overall_mape[model_b],
            "Mean_Daily_MAPE_Difference_A_Minus_B": float(daily_delta.mean()),
            "Difference_CI_Lower": ci_lower,
            "Difference_CI_Upper": ci_upper,
            "CI_Includes_Zero": ci_includes_zero,
            "Bootstrap_Block_Days": BOOTSTRAP_BLOCK_DAYS,
            "Bootstrap_Resamples": BOOTSTRAP_RESAMPLES,
            "Interpretation": interpretation,
        })

    return pd.DataFrame(comparison_rows)


def get_models(model_parameters=None):
    """
    Define machine learning models and optionally apply selected parameters.
    """
    models = {
        # Scaling is fitted inside the pipeline, so Ridge never learns
        # preprocessing statistics from validation or test observations.
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators = 100,
            random_state = 42,
            n_jobs =-1,
        ),



        "XGBoost": XGBRegressor(
            n_estimators = 200,
            learning_rate = 0.05,
            max_depth = 6,
            subsample = 0.8,
            colsample_bytree = 0.8,
            objective = "reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    }

    if model_parameters:
        unknown_models = set(model_parameters) - set(models)
        if unknown_models:
            raise ValueError(f"Unknown models in parameter configuration: {unknown_models}")

        for model_name, parameters in model_parameters.items():
            models[model_name].set_params(**parameters)

    return models


def get_parameter_grids():
    """
    Define a small, reproducible hyperparameter search space for each model.
    """
    return {
        "Ridge Regression": {
            "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20]
        },
        "XGBoost": {
            "n_estimators": [200, 400],
            "max_depth": [4, 6]
        },
        "LightGBM": {
            "n_estimators": [200, 400],
            "num_leaves": [31, 63]
        }
    }


def tune_models(X_train, y_train, X_validation, y_validation):
    """
    Select each model's hyperparameters using only the 2024 validation set.
    The 2025 test set is deliberately not accepted by this function.
    """
    models = get_models()
    parameter_grids = get_parameter_grids()
    validation_rows = []
    best_parameters = {}
    best_candidate_numbers = {}

    # All candidates are fitted on 2021-2023 and ranked on 2024 only. This
    # function deliberately has no access to the 2025 final test set.
    for model_name, base_model in models.items():
        best_score = None

        for candidate_number, parameters in enumerate(
            ParameterGrid(parameter_grids[model_name]),
            start=1
        ):
            candidate_model = clone(base_model)
            candidate_model.set_params(**parameters)
            candidate_model.fit(X_train, y_train)
            validation_prediction = candidate_model.predict(X_validation)
            metrics = calculate_metrics(y_validation, validation_prediction)

            validation_rows.append({
                "Model": model_name,
                "Candidate": candidate_number,
                "Parameters": json.dumps(parameters, sort_keys=True),
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"]
            })

            candidate_score = (metrics["MAPE"], metrics["RMSE"])
            if best_score is None or candidate_score < best_score:
                best_score = candidate_score
                best_parameters[model_name] = parameters.copy()
                best_candidate_numbers[model_name] = candidate_number

            print(
                f"Validation: {model_name}, candidate {candidate_number}, "
                f"MAPE={metrics['MAPE']:.4f}%"
            )

    validation_results = pd.DataFrame(validation_rows)
    validation_results["Selected"] = False

    for model_name, candidate_number in best_candidate_numbers.items():
        selected_mask = (
            (validation_results["Model"] == model_name)
            & (validation_results["Candidate"] == candidate_number)
        )
        validation_results.loc[selected_mask, "Selected"] = True

    validation_results = validation_results.sort_values(
        ["Model", "MAPE", "RMSE"]
    ).reset_index(drop=True)

    return best_parameters, validation_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Create chronological partitions before constructing any model matrices.
    df = load_feature_data(INPUT_FILE)
    train_df, validation_df, final_train_df, test_df = (
        train_validation_test_split_by_year(df)
    )
    validate_complete_year_coverage(
        validation_df,
        EXPECTED_VALIDATION_YEAR,
        "Hyperparameter-validation set",
    )
    validate_complete_year_coverage(
        test_df,
        EXPECTED_TEST_YEAR,
        "Final test set",
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]
    X_final_train = final_train_df[FEATURE_COLUMNS]
    y_final_train = final_train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("Feature data loaded.")
    print(f"Total rows: {len(df)}")
    print(f"Parameter-training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Final-training rows: {len(final_train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Parameter-training date range: {train_df['DATETIME'].min()} to {train_df['DATETIME'].max()}")
    print(f"Validation date range: {validation_df['DATETIME'].min()} to {validation_df['DATETIME'].max()}")
    print(f"Final-training date range: {final_train_df['DATETIME'].min()} to {final_train_df['DATETIME'].max()}")
    print(f"Test date range: {test_df['DATETIME'].min()} to {test_df['DATETIME'].max()}")

    # Hyperparameter selection ends after the 2024 validation results.
    best_parameters, validation_results_df = tune_models(
        X_train,
        y_train,
        X_validation,
        y_validation
    )
    validation_results_df.to_csv(OUTPUT_VALIDATION_FILE, index=False)

    selected_validation_results = (
        validation_results_df[validation_results_df["Selected"]]
        .set_index("Model")
    )

    models = get_models(best_parameters)

    results = []
    predictions_df = test_df[["DATETIME", TARGET_COLUMN]].copy()
    model_prediction_columns = {}

    for model_name, model in models.items():
        # Refit the selected configuration on 2021-2024 before testing once on
        # the untouched 2025 observations.
        model.fit(X_final_train, y_final_train)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        validation_metrics = selected_validation_results.loc[model_name]

        results.append({
            "Model": model_name,
            "Selected_Parameters": json.dumps(
                best_parameters[model_name],
                sort_keys=True
            ),
            "Validation_MAPE": validation_metrics["MAPE"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"]
        })

        safe_model_name = (model_name.lower().replace(" ", "_"))
        prediction_column = f"{safe_model_name}_prediction"
        predictions_df[prediction_column] = y_pred
        model_prediction_columns[model_name] = prediction_column

        model_file = MODELS_DIR / f"{safe_model_name}.joblib"
        # Persist the exact final estimator used to generate the reported test
        # predictions and feature-importance outputs.
        joblib.dump(model, model_file)

        print(f"{model_name} completed.")
        print(f"Selected parameters: {best_parameters[model_name]}")
        print(f"Validation MAPE: {validation_metrics['MAPE']:.2f}%")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"Model saved to: {model_file}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("MAPE").reset_index(drop=True)
    comparison_df = compare_model_predictions(
        predictions_df,
        model_prediction_columns,
    )

    # Save aggregate metrics, row-level predictions and paired uncertainty as
    # separate reproducible outputs.
    results_df.to_csv(OUTPUT_RESULTS_FILE, index=False)
    predictions_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
    comparison_df.to_csv(OUTPUT_COMPARISON_FILE, index=False)

    print("\nMachine learning model evaluation completed.")
    print(results_df)
    print("\nPaired 7-day moving-block bootstrap comparisons:")
    print(comparison_df.to_string(index=False))
    print(f"\nValidation search saved to: {OUTPUT_VALIDATION_FILE}")
    print(f"\nResults saved to: {OUTPUT_RESULTS_FILE}")
    print(f"Predictions saved to: {OUTPUT_PREDICTIONS_FILE}")
    print(f"Model comparisons saved to: {OUTPUT_COMPARISON_FILE}")


if __name__ == "__main__":
    main()
