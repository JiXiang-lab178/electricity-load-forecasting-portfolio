"""Run leakage-safe feature ablation and candidate refinement.

The experiment fits on 2021-2022 and performs feature selection on 2023.
The separate 2024 hyperparameter-validation year and untouched 2025 final
test year are never accepted by the modelling function.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from feature_config import (
    ALL_CANDIDATE_FEATURES as SHARED_ALL_CANDIDATE_FEATURES,
    CALENDAR_FEATURES as SHARED_CALENDAR_FEATURES,
    FINAL_FEATURE_COLUMNS,
    FINAL_FEATURE_SET_NAME,
    HOLIDAY_FEATURES as SHARED_HOLIDAY_FEATURES,
    LAG_FEATURES as SHARED_LAG_FEATURES,
    ROLLING_FEATURES as SHARED_ROLLING_FEATURES,
)
from train_ml_models import (
    INPUT_FILE,
    LOCAL_TIMEZONE,
    TARGET_COLUMN,
    calculate_metrics,
    get_models,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_TABLE = RESULTS_DIR / "feature_ablation_results.csv"
OUTPUT_FIGURE = FIGURES_DIR / "feature_ablation_mape.png"

FEATURE_SELECTION_TRAINING_YEARS = (2021, 2022)
FEATURE_SELECTION_VALIDATION_YEAR = 2023
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

# List aliases are retained for readable DataFrame column selection and for the
# public interface used by the project's tests. Their source of truth is the
# immutable shared configuration in feature_config.py.
LAG_FEATURES = list(SHARED_LAG_FEATURES)
ROLLING_FEATURES = list(SHARED_ROLLING_FEATURES)
CALENDAR_FEATURES = list(SHARED_CALENDAR_FEATURES)
HOLIDAY_FEATURES = list(SHARED_HOLIDAY_FEATURES)
ALL_CANDIDATE_FEATURES = list(SHARED_ALL_CANDIDATE_FEATURES)

REFERENCE_FEATURE_SET_NAME = "All candidate features"


def _without(features: list[str], removed_features: list[str]) -> list[str]:
    """Return a feature list with one pre-specified group removed."""
    removed = set(removed_features)
    return [feature for feature in features if feature not in removed]


# The first seven configurations form a symmetric leave-one-group-out study
# around one explicit reference. The remaining configurations refine the
# reduced lag-calendar candidate or provide build-up diagnostics.
FEATURE_SET_DEFINITIONS = {
    REFERENCE_FEATURE_SET_NAME: {
        "features": ALL_CANDIDATE_FEATURES,
        "experiment_type": "Reference",
        "removed_group": "Not applicable",
        "eligible_for_selection": False,
    },
    "Without daily lag": {
        "features": _without(ALL_CANDIDATE_FEATURES, [LAG_FEATURES[0]]),
        "experiment_type": "Leave-one-group-out",
        "removed_group": "Daily lag",
        "eligible_for_selection": False,
    },
    "Without weekly lag": {
        "features": _without(ALL_CANDIDATE_FEATURES, [LAG_FEATURES[1]]),
        "experiment_type": "Leave-one-group-out",
        "removed_group": "Weekly lag",
        "eligible_for_selection": False,
    },
    "Selected + both GB holiday flags": {
        "features": LAG_FEATURES + CALENDAR_FEATURES + HOLIDAY_FEATURES,
        "experiment_type": "Leave-one-group-out / candidate refinement",
        "removed_group": "Rolling features",
        "eligible_for_selection": True,
    },
    "Without calendar features": {
        "features": _without(ALL_CANDIDATE_FEATURES, CALENDAR_FEATURES),
        "experiment_type": "Leave-one-group-out",
        "removed_group": "Calendar features",
        "eligible_for_selection": False,
    },
    "Without England/Wales holiday": {
        "features": _without(ALL_CANDIDATE_FEATURES, [HOLIDAY_FEATURES[0]]),
        "experiment_type": "Leave-one-group-out",
        "removed_group": "England/Wales holiday",
        "eligible_for_selection": False,
    },
    "Without Scotland holiday": {
        "features": _without(ALL_CANDIDATE_FEATURES, [HOLIDAY_FEATURES[1]]),
        "experiment_type": "Leave-one-group-out",
        "removed_group": "Scotland holiday",
        "eligible_for_selection": False,
    },
    "Lag + calendar features": {
        "features": LAG_FEATURES + CALENDAR_FEATURES,
        "experiment_type": "Candidate refinement",
        "removed_group": "Rolling and holiday features",
        "eligible_for_selection": True,
    },
    FINAL_FEATURE_SET_NAME: {
        "features": list(FINAL_FEATURE_COLUMNS),
        "experiment_type": "Candidate refinement",
        "removed_group": "Rolling and Scotland-holiday features",
        "eligible_for_selection": True,
    },
    "Selected + Scotland holiday": {
        "features": LAG_FEATURES + CALENDAR_FEATURES + [HOLIDAY_FEATURES[1]],
        "experiment_type": "Candidate refinement",
        "removed_group": "Rolling and England/Wales-holiday features",
        "eligible_for_selection": True,
    },
    "Lag features only": {
        "features": LAG_FEATURES,
        "experiment_type": "Build-up diagnostic",
        "removed_group": "Not applicable",
        "eligible_for_selection": False,
    },
    "Calendar features only": {
        "features": CALENDAR_FEATURES,
        "experiment_type": "Build-up diagnostic",
        "removed_group": "Not applicable",
        "eligible_for_selection": False,
    },
}
FEATURE_SETS = {
    name: list(definition["features"])
    for name, definition in FEATURE_SET_DEFINITIONS.items()
}


def load_ablation_data(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load and fail-fast validate the chronological feature dataset."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Input file is empty: {input_file}") from error
    if df.empty:
        raise ValueError(f"Input file contains no observations: {input_file}")

    required_columns = ["DATETIME", TARGET_COLUMN, *ALL_CANDIDATE_FEATURES]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required ablation columns: {missing_columns}")

    parsed_datetime = pd.to_datetime(
        df["DATETIME"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    numeric_columns = [TARGET_COLUMN, *ALL_CANDIDATE_FEATURES]
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
            f"Invalid required ablation values after parsing: {invalid_counts}"
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
            "Ablation data contains non-finite numeric values: "
            f"{non_finite_counts}."
        )

    positive_columns = [
        TARGET_COLUMN,
        *LAG_FEATURES,
        *ROLLING_FEATURES,
    ]
    non_positive_counts = {
        column: int((df[column] <= 0).sum())
        for column in positive_columns
    }
    if any(non_positive_counts.values()):
        raise ValueError(
            "Demand-derived ablation values must be positive: "
            f"{non_positive_counts}."
        )

    df = df.sort_values("DATETIME").reset_index(drop=True)
    duplicate_timestamps = int(df["DATETIME"].duplicated().sum())
    invalid_intervals = int(
        df["DATETIME"]
        .dt.tz_convert("UTC")
        .diff()
        .dropna()
        .ne(pd.Timedelta(minutes=30))
        .sum()
    )
    if duplicate_timestamps or invalid_intervals:
        raise ValueError(
            "Ablation input is not a unique continuous half-hourly series: "
            f"duplicates={duplicate_timestamps}, "
            f"invalid_intervals={invalid_intervals}."
        )

    # Reconstruct critical calendar and demand-history features independently
    # before any candidate is allowed into the experiment.
    _validate_calendar_features(df)
    _validate_demand_history_features(df)
    return df


def _validate_calendar_features(df: pd.DataFrame) -> None:
    """Validate calendar ranges and their agreement with local timestamps."""
    integer_columns = [
        "settlement_period",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        *HOLIDAY_FEATURES,
    ]
    non_integer_counts = {
        column: int((df[column] != np.floor(df[column])).sum())
        for column in integer_columns
    }
    if any(non_integer_counts.values()):
        raise ValueError(
            f"Calendar features must be integer-valued: {non_integer_counts}."
        )

    expected_calendar = {
        "day_of_week": df["DATETIME"].dt.dayofweek,
        "month": df["DATETIME"].dt.month,
        "day_of_year": df["DATETIME"].dt.dayofyear,
        "is_weekend": df["DATETIME"].dt.dayofweek.isin([5, 6]).astype(int),
    }
    for column, expected_values in expected_calendar.items():
        if not np.array_equal(df[column].to_numpy(), expected_values.to_numpy()):
            raise ValueError(f"{column} is inconsistent with DATETIME.")

    if not df["settlement_period"].between(1, 50).all():
        raise ValueError("settlement_period must be between 1 and 50.")
    for column in ["is_weekend", *HOLIDAY_FEATURES]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be binary.")


def _validate_demand_history_features(df: pd.DataFrame) -> None:
    """Check that every internally comparable history feature is aligned."""
    lag_periods = {
        "lag_48": 48,
        "lag_336": 336,
    }
    for feature, periods in lag_periods.items():
        expected_values = df[TARGET_COLUMN].shift(periods)
        comparable = expected_values.notna()
        if not np.allclose(
            df.loc[comparable, feature],
            expected_values.loc[comparable],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{feature} is inconsistent with an ND shift of {periods}."
            )

    for feature, window in {
        "rolling_mean_48": 48,
        "rolling_mean_336": 336,
    }.items():
        expected_values = (
            df[TARGET_COLUMN]
            .shift(48)
            .rolling(window=window, min_periods=window)
            .mean()
        )
        comparable = expected_values.notna()
        if not np.allclose(
            df.loc[comparable, feature],
            expected_values.loc[comparable],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{feature} is inconsistent with a window ending at t-48."
            )


def split_feature_selection_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use 2021-2022 for fitting and the complete 2023 year for selection."""
    if "DATETIME" not in df.columns:
        raise ValueError("Feature-selection data must contain DATETIME.")
    if not pd.api.types.is_datetime64_any_dtype(df["DATETIME"]):
        raise ValueError("Feature-selection DATETIME must already be parsed.")

    training_df = (
        df[df["DATETIME"].dt.year.isin(FEATURE_SELECTION_TRAINING_YEARS)]
        .copy()
        .sort_values("DATETIME")
        .reset_index(drop=True)
    )
    validation_df = (
        df[df["DATETIME"].dt.year == FEATURE_SELECTION_VALIDATION_YEAR]
        .copy()
        .sort_values("DATETIME")
        .reset_index(drop=True)
    )

    observed_training_years = set(training_df["DATETIME"].dt.year.unique())
    if observed_training_years != set(FEATURE_SELECTION_TRAINING_YEARS):
        raise ValueError(
            "Feature-ablation training years are incomplete: "
            f"observed={sorted(observed_training_years)}."
        )
    if validation_df.empty:
        raise ValueError("Feature-ablation validation set is empty.")

    expected_validation_timestamps = pd.date_range(
        start=pd.Timestamp(
            f"{FEATURE_SELECTION_VALIDATION_YEAR}-01-01",
            tz=LOCAL_TIMEZONE,
        ),
        end=pd.Timestamp(
            f"{FEATURE_SELECTION_VALIDATION_YEAR + 1}-01-01",
            tz=LOCAL_TIMEZONE,
        ),
        freq="30min",
        inclusive="left",
    )
    observed_validation_timestamps = pd.DatetimeIndex(validation_df["DATETIME"])
    if not observed_validation_timestamps.equals(expected_validation_timestamps):
        missing_timestamps = expected_validation_timestamps.difference(
            observed_validation_timestamps
        )
        extra_timestamps = observed_validation_timestamps.difference(
            expected_validation_timestamps
        )
        raise ValueError(
            f"{FEATURE_SELECTION_VALIDATION_YEAR} feature-selection coverage "
            "is incomplete: "
            f"expected_rows={len(expected_validation_timestamps)}, "
            f"observed_rows={len(observed_validation_timestamps)}, "
            f"missing_examples={missing_timestamps[:3].tolist()}, "
            f"extra_examples={extra_timestamps[:3].tolist()}."
        )
    if training_df["DATETIME"].max() >= validation_df["DATETIME"].min():
        raise ValueError("Feature-selection training and validation overlap.")

    return training_df, validation_df


def moving_block_bootstrap_mean_ci(
    values,
    block_length: int = BOOTSTRAP_BLOCK_DAYS,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Return a moving-block bootstrap interval for a serial daily metric."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty 1D array.")
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap values contain non-finite observations.")
    if block_length < 1 or n_resamples < 2:
        raise ValueError("Bootstrap block length and resample count are invalid.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap confidence level must be between zero and one.")

    # Resample consecutive seven-day blocks to retain short-range dependence
    # between neighbouring daily forecast errors.
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


def _validated_metrics(y_true, y_pred) -> dict[str, float]:
    """Validate prediction arrays before using the shared metric definition."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Ablation metric inputs must be one-dimensional.")
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Ablation metric inputs are empty or have different shapes.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Ablation metric inputs contain non-finite values.")
    if (y_true <= 0).any():
        raise ValueError("Ablation MAPE requires strictly positive actual demand.")
    metrics = calculate_metrics(y_true, y_pred)
    return {key: float(value) for key, value in metrics.items()}


def _daily_mape(y_true, y_pred, timestamps: pd.Series) -> np.ndarray:
    """Aggregate absolute percentage error into equal 48-period UTC days."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true) * 100.0
    utc_days = timestamps.dt.tz_convert("UTC").dt.floor("D").to_numpy()
    daily_mape = (
        pd.DataFrame({
            "UTC_DAY": utc_days,
            "APE": absolute_percentage_error,
        })
        .groupby("UTC_DAY", sort=True)["APE"]
        .mean()
        .to_numpy()
    )
    if len(daily_mape) != 365:
        raise ValueError(
            "Feature-selection uncertainty requires 365 complete UTC days; "
            f"observed={len(daily_mape)}."
        )
    return daily_mape


def _model_parameters_for_reporting(model) -> str:
    """Return deterministic JSON describing the fixed LightGBM configuration."""
    return json.dumps(
        model.get_params(deep=False),
        sort_keys=True,
        default=str,
    )


def run_feature_ablation(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit one fixed LightGBM configuration for every feature definition."""
    required_columns = ["DATETIME", TARGET_COLUMN, *ALL_CANDIDATE_FEATURES]
    for label, frame in {
        "training": training_df,
        "validation": validation_df,
    }.items():
        missing_columns = [
            column for column in required_columns if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing {label} ablation columns: {missing_columns}"
            )
        if frame.empty:
            raise ValueError(f"Feature-ablation {label} data is empty.")

    training_years = set(training_df["DATETIME"].dt.year.unique())
    validation_years = set(validation_df["DATETIME"].dt.year.unique())
    if training_years != set(FEATURE_SELECTION_TRAINING_YEARS):
        raise ValueError(f"Unexpected ablation training years: {training_years}.")
    if validation_years != {FEATURE_SELECTION_VALIDATION_YEAR}:
        raise ValueError(f"Unexpected ablation validation years: {validation_years}.")

    # Keep one fixed model configuration across candidates so performance
    # differences can be attributed to the feature sets themselves.
    base_model = get_models()["LightGBM"]
    model_parameters = _model_parameters_for_reporting(base_model)
    y_validation = validation_df[TARGET_COLUMN].to_numpy(dtype=float)
    results = []
    daily_errors = {}

    for experiment_order, (feature_set_name, definition) in enumerate(
        FEATURE_SET_DEFINITIONS.items(),
        start=1,
    ):
        feature_columns = list(definition["features"])
        model = clone(base_model)
        model.fit(training_df[feature_columns], training_df[TARGET_COLUMN])
        predictions = model.predict(validation_df[feature_columns])
        metrics = _validated_metrics(y_validation, predictions)
        daily_error = _daily_mape(
            y_validation,
            predictions,
            validation_df["DATETIME"],
        )
        daily_errors[feature_set_name] = daily_error
        mape_ci_lower, mape_ci_upper = moving_block_bootstrap_mean_ci(
            daily_error
        )

        results.append({
            "Experiment_Order": experiment_order,
            "Feature_Set": feature_set_name,
            "Experiment_Type": definition["experiment_type"],
            "Removed_Group": definition["removed_group"],
            "Eligible_For_Selection": definition["eligible_for_selection"],
            "Feature_Count": len(feature_columns),
            "Features": ", ".join(feature_columns),
            "Training_Years": "2021-2022",
            "Training_Start": training_df["DATETIME"].min().isoformat(),
            "Training_End": training_df["DATETIME"].max().isoformat(),
            "Training_Rows": len(training_df),
            "Validation_Year": FEATURE_SELECTION_VALIDATION_YEAR,
            "Validation_Start": validation_df["DATETIME"].min().isoformat(),
            "Validation_End": validation_df["DATETIME"].max().isoformat(),
            "Validation_Rows": len(validation_df),
            "Model": "LightGBM (fixed parameters)",
            "Model_Parameters": model_parameters,
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "MAPE_CI_Lower": mape_ci_lower,
            "MAPE_CI_Upper": mape_ci_upper,
        })

        print(
            f"{feature_set_name}: {len(feature_columns)} features, "
            f"MAPE={metrics['MAPE']:.4f}% "
            f"(95% block-bootstrap CI "
            f"{mape_ci_lower:.4f}-{mape_ci_upper:.4f}%)"
        )

    results_df = pd.DataFrame(results)
    result_index = results_df.set_index("Feature_Set")
    reference_daily_error = daily_errors[REFERENCE_FEATURE_SET_NAME]

    # Paired daily differences are more informative than comparing two
    # independent confidence intervals for the candidate MAPEs.
    delta_reference_values = []
    delta_reference_lower_values = []
    delta_reference_upper_values = []
    for feature_set_name in results_df["Feature_Set"]:
        daily_delta = daily_errors[feature_set_name] - reference_daily_error
        lower, upper = moving_block_bootstrap_mean_ci(daily_delta)
        delta_reference_values.append(float(daily_delta.mean()))
        delta_reference_lower_values.append(lower)
        delta_reference_upper_values.append(upper)

    results_df["Delta_MAPE_vs_All_Features"] = delta_reference_values
    results_df["Delta_vs_All_CI_Lower"] = delta_reference_lower_values
    results_df["Delta_vs_All_CI_Upper"] = delta_reference_upper_values

    eligible_rows = results_df[results_df["Eligible_For_Selection"]].copy()
    eligible_rows = eligible_rows.sort_values(
        ["MAPE", "RMSE", "Feature_Count", "Feature_Set"]
    )
    best_candidate_name = str(eligible_rows.iloc[0]["Feature_Set"])
    best_candidate_daily_error = daily_errors[best_candidate_name]

    results_df["Candidate_Rank"] = pd.Series(
        pd.NA,
        index=results_df.index,
        dtype="Int64",
    )
    candidate_ranks = eligible_rows["MAPE"].rank(method="min").astype(int)
    results_df.loc[eligible_rows.index, "Candidate_Rank"] = candidate_ranks
    results_df["Delta_MAPE_vs_Best_Candidate"] = np.nan
    results_df["Delta_vs_Best_CI_Lower"] = np.nan
    results_df["Delta_vs_Best_CI_Upper"] = np.nan
    results_df["Statistically_Tied_With_Best"] = False

    tied_candidate_names = []
    for row_index, row in eligible_rows.iterrows():
        feature_set_name = str(row["Feature_Set"])
        daily_delta = daily_errors[feature_set_name] - best_candidate_daily_error
        lower, upper = moving_block_bootstrap_mean_ci(daily_delta)
        statistically_tied = lower <= 0.0 <= upper
        results_df.loc[row_index, "Delta_MAPE_vs_Best_Candidate"] = float(
            daily_delta.mean()
        )
        results_df.loc[row_index, "Delta_vs_Best_CI_Lower"] = lower
        results_df.loc[row_index, "Delta_vs_Best_CI_Upper"] = upper
        results_df.loc[row_index, "Statistically_Tied_With_Best"] = (
            statistically_tied
        )
        if statistically_tied:
            tied_candidate_names.append(feature_set_name)

    # Apply a parsimonious selection rule: among candidates statistically tied
    # with the best, prefer fewer inputs before using MAPE and RMSE as tie-breaks.
    selected_name = min(
        tied_candidate_names,
        key=lambda name: (
            int(result_index.loc[name, "Feature_Count"]),
            float(result_index.loc[name, "MAPE"]),
            float(result_index.loc[name, "RMSE"]),
            name,
        ),
    )
    results_df["Selected"] = results_df["Feature_Set"].eq(selected_name)
    results_df["Matches_Frozen_Final_Config"] = results_df["Features"].eq(
        ", ".join(FINAL_FEATURE_COLUMNS)
    )
    results_df["Selection_Rationale"] = ""
    selected_index = results_df.index[results_df["Selected"]][0]
    results_df.loc[selected_index, "Selection_Rationale"] = (
        "Lowest feature count among candidate configurations whose paired "
        "7-day moving-block bootstrap interval includes the best candidate; "
        "MAPE and RMSE break remaining ties."
    )

    return results_df.sort_values("Experiment_Order").reset_index(drop=True)


def validate_final_feature_selection(results_df: pd.DataFrame) -> None:
    """Ensure the statistically selected result matches the frozen final config."""
    selected_rows = results_df[results_df["Selected"]]
    if len(selected_rows) != 1:
        raise ValueError(
            "Feature ablation must select exactly one final configuration."
        )
    selected_row = selected_rows.iloc[0]
    selected_name = str(selected_row["Feature_Set"])
    selected_features = FEATURE_SETS[selected_name]
    if (
        selected_name != FINAL_FEATURE_SET_NAME
        or selected_features != list(FINAL_FEATURE_COLUMNS)
    ):
        raise ValueError(
            "Robust feature selection no longer matches the frozen final-model "
            "configuration. Review the experiment before training the final model: "
            f"selected={selected_name}."
        )


def save_dataframe_atomically(df: pd.DataFrame, output_file: Path) -> None:
    """Replace a CSV only after its temporary file is complete."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(f"{output_file.name}.tmp")
    try:
        df.to_csv(temporary_output, index=False)
        temporary_output.replace(output_file)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def save_ablation_figure(
    results_df: pd.DataFrame,
    output_file: Path = OUTPUT_FIGURE,
) -> None:
    """Save a comparison plot with moving-block MAPE intervals."""
    plot_df = results_df.sort_values("MAPE", ascending=True).reset_index(drop=True)
    colours = []
    for _, row in plot_df.iterrows():
        if row["Selected"]:
            colours.append("#2a9d8f")
        elif row["Eligible_For_Selection"]:
            colours.append("#457b9d")
        else:
            colours.append("#8d99ae")

    fig, ax = plt.subplots(figsize=(11, 7.5))
    bars = ax.barh(plot_df["Feature_Set"], plot_df["MAPE"], color=colours)
    lower_errors = plot_df["MAPE"] - plot_df["MAPE_CI_Lower"]
    upper_errors = plot_df["MAPE_CI_Upper"] - plot_df["MAPE"]
    ax.errorbar(
        plot_df["MAPE"],
        np.arange(len(plot_df)),
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="#1d3557",
        elinewidth=1.0,
        capsize=2.5,
    )
    ax.bar_label(bars, fmt="%.3f%%", padding=4)
    ax.set_xlabel("2023 validation MAPE (%) with 95% moving-block intervals")
    ax.set_ylabel("Feature configuration")
    ax.set_title("LightGBM feature ablation and candidate refinement")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(f"{output_file.name}.tmp")
    try:
        fig.savefig(
            temporary_output,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        temporary_output.replace(output_file)
    finally:
        plt.close(fig)
        if temporary_output.exists():
            temporary_output.unlink()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Feature selection ends in 2023; later validation and test years are not
    # accepted by the ablation functions.
    df = load_ablation_data()
    training_df, validation_df = split_feature_selection_data(df)
    results_df = run_feature_ablation(training_df, validation_df)
    validate_final_feature_selection(results_df)
    save_dataframe_atomically(results_df, OUTPUT_TABLE)
    save_ablation_figure(results_df)

    selected_row = results_df.loc[results_df["Selected"]].iloc[0]
    print("\nFeature-ablation experiment completed.")
    print(f"Selected feature set: {selected_row['Feature_Set']}")
    print(
        f"Validation MAPE: {selected_row['MAPE']:.4f}% "
        f"(95% CI {selected_row['MAPE_CI_Lower']:.4f}-"
        f"{selected_row['MAPE_CI_Upper']:.4f}%)"
    )
    print(f"Selection rule: {selected_row['Selection_Rationale']}")
    print(f"Results saved to: {OUTPUT_TABLE}")
    print(f"Figure saved to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
