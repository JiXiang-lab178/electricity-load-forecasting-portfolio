"""Create validated, leakage-safe features for 48-step-ahead forecasting."""
# It postpones the evaluation of type annotations and makes type hints more flexible.
from __future__ import annotations
# Use the JSON module to load the external bank-holiday snapshot.
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The London timezone automatically represents GMT in winter and BST in summer.
LOCAL_TIMEZONE = "Europe/London"

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "clean_demand_data_2021_2025.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "features_demand_data_2021_2025.csv"
)
# gb_bank_holidays_2021_2025.json stores bank-holiday dates for Great Britain from 2021 to 2025 and is used to create holiday forecasting features.
BANK_HOLIDAY_FILE = (
    PROJECT_ROOT / "data" / "external" / "gb_bank_holidays_2021_2025.json"
)
# Feature-engineering constants.
FORECAST_HORIZON_PERIODS = 48
# The daily lag uses the demand observed 48 periods, or 24 elapsed hours, earlier.
DAILY_LAG_PERIODS = 48
# The weekly lag uses demand from 336 half-hour periods
# (168 elapsed hours) earlier.
WEEKLY_LAG_PERIODS = 336
# These constants define rolling windows containing 48 and 336 observations respectively.
DAILY_ROLLING_WINDOW = 48
WEEKLY_ROLLING_WINDOW = 336

REQUIRED_INPUT_COLUMNS = ["DATETIME", "SETTLEMENT_PERIOD", "ND"]
# This list defines every new feature that the script is expected to create.
ENGINEERED_FEATURE_COLUMNS = [
    "lag_48",
    "lag_336",
    "rolling_mean_48",
    "rolling_mean_336",
    "settlement_period",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_bank_holiday_england_wales",
    "is_bank_holiday_scotland",
]

# This function checks whether a DataFrame contains every required column.
def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {context}: {missing_columns}"
        )

# This function loads the processed CSV and independently validates its critical fields.
# Revalidation is necessary because a CSV may be edited or replaced between pipeline stages.
def load_clean_data(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file)
    # Converts pandas' empty-file exception into a clearer project-level error.
    except pd.errors.EmptyDataError as error:
        # from error preserves the original exception for debugging.
        raise ValueError(f"Input file is empty: {input_file}") from error
    # This check catches a CSV that has column headers but no data rows.
    if df.empty:
        raise ValueError(f"Input file contains no observations: {input_file}")
    # Verifies that DATETIME, SETTLEMENT_PERIOD and ND are present.
    _require_columns(df, REQUIRED_INPUT_COLUMNS, "clean feature input")

    parsed_datetime = pd.to_datetime(
        df["DATETIME"],
    # format = "mixed" allows pandas to parse multiple timestamp formats.
        format="mixed",
        errors="coerce",
    # Take all timestamps onto one unambiguous UTC timeline.
        utc=True,
    )
    numeric_demand = pd.to_numeric(df["ND"], errors="coerce")
    numeric_period = pd.to_numeric(df["SETTLEMENT_PERIOD"], errors="coerce")
    # Counting invalid values.
    invalid_datetime = int(parsed_datetime.isna().sum())
    invalid_demand = int(numeric_demand.isna().sum())
    invalid_period = int(numeric_period.isna().sum())
    # This condition becomes true when any invalid-value count is greater than zero.
    if invalid_datetime or invalid_demand or invalid_period:
        raise ValueError(
            "Critical values are invalid after parsing: "
            f"DATETIME={invalid_datetime}, ND={invalid_demand}, "
            f"SETTLEMENT_PERIOD={invalid_period}."
        )
    # This ensures that all demand values are finite numbers.
    if np.isinf(numeric_demand).any():
        raise ValueError("Non-finite ND values found.")
    # Reject zero or negative National Demand values.
    if (numeric_demand <= 0).any():
        raise ValueError("ND contains non-positive values.")
    # Count values such as 2.5 that are not valid integer periods.
    non_integer_periods = int((numeric_period % 1 != 0).sum())
    # This counts settlement periods outside the preliminary valid range of 1 to 50.
    out_of_range_periods = int((~numeric_period.between(1, 50)).sum())
    if non_integer_periods or out_of_range_periods:
        raise ValueError(
            "Invalid SETTLEMENT_PERIOD values found: "
            f"non_integer={non_integer_periods}, "
            f"outside_1_to_50={out_of_range_periods}."
        )
    # This converts UTC timestamps to London local time without changing the actual instant.
    df["DATETIME"] = parsed_datetime.dt.tz_convert(LOCAL_TIMEZONE)
    df["ND"] = numeric_demand.astype(float)
    df["SETTLEMENT_PERIOD"] = numeric_period.astype("int64")
    df = df.sort_values("DATETIME").reset_index(drop=True)
    # Check whether duplicate timestamps exist.
    duplicate_timestamps = int(df["DATETIME"].duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(
            f"Duplicate DATETIME values found: {duplicate_timestamps}"
        )
    # Converts London timestamps to UTC for an unambiguous continuity check.
    utc_timestamps = df["DATETIME"].dt.tz_convert("UTC")
    non_continuous_intervals = int(
        utc_timestamps.diff().dropna().ne(pd.Timedelta(minutes=30)).sum()
    )
    # Explains that UTC timestamps must be continuous at 30-minute intervals before row-based features are created.
    if non_continuous_intervals:
        raise ValueError(
            "DATETIME must be continuous at 30-minute UTC intervals before "
            f"row-based features are created: {non_continuous_intervals} "
            "invalid intervals."
        )

    return df

# Define a function that creates calendar-based time features.
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, REQUIRED_INPUT_COLUMNS, "time-feature input")
    # Create calendar variables representing intraday, weekly and annual demand patterns.
    df["settlement_period"] = df["SETTLEMENT_PERIOD"].astype("int64")
    df["day_of_week"] = df["DATETIME"].dt.dayofweek
    df["month"] = df["DATETIME"].dt.month
    df["day_of_year"] = df["DATETIME"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    return df

# Defines the bank-holiday feature function, using the configured holiday file by default.
def create_bank_holiday_features(
    df: pd.DataFrame,
    holiday_file: Path = BANK_HOLIDAY_FILE,
) -> pd.DataFrame:
    """Create separate GB regional holiday flags from the GOV.UK snapshot.

    Northern Ireland is excluded because the target, National Demand (ND),
    covers Great Britain rather than the whole United Kingdom.
    """
    if not holiday_file.exists():
        raise FileNotFoundError(f"Bank-holiday file not found: {holiday_file}")

    try:
        with holiday_file.open("r", encoding="utf-8") as file:
            holiday_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Bank-holiday file is not valid JSON: {holiday_file}"
        ) from error
    # Defines the three required JSON keys as a set.
    required_keys = {
        "coverage_years",
        "england-and-wales",
        "scotland",
    }
    # Uses set difference to identify missing keys.
    missing_keys = required_keys - set(holiday_data)
    if missing_keys:
        raise ValueError(
            f"Bank-holiday data is missing keys: {sorted(missing_keys)}"
        )
    # Uses a set comprehension to convert coverage years to unique integers.
    try:
        coverage_years = {int(year) for year in holiday_data["coverage_years"]}
    except (TypeError, ValueError) as error:
        raise ValueError("Bank-holiday coverage_years must contain years.") from error
    # Extracts all distinct years present in the demand dataset.
    observed_years = set(df["DATETIME"].dt.year.astype(int).unique())
    # Finds observed years not covered by the holiday file.
    uncovered_years = observed_years - coverage_years
    if uncovered_years:
        raise ValueError(
            "Bank-holiday data does not cover years: "
            f"{sorted(uncovered_years)}"
        )
    # Extracts the local calendar date from each London timestamp.
    local_dates = df["DATETIME"].dt.date
    for division, feature_name in {
        "england-and-wales": "is_bank_holiday_england_wales",
        "scotland": "is_bank_holiday_scotland",
    }.items():
        division_dates = holiday_data[division]
        if not isinstance(division_dates, list) or not division_dates:
            raise ValueError(
                f"Bank-holiday division must be a non-empty list: {division}"
            )
    #   Parses holiday dates using the explicit YYYY-MM-DD format; invalid values become NaT.
        parsed_dates = pd.to_datetime(
            division_dates,
            format="%Y-%m-%d",
            errors="coerce",
        )
        if parsed_dates.isna().any():
            raise ValueError(f"Invalid bank-holiday date in division: {division}")
        if parsed_dates.duplicated().any():
            raise ValueError(f"Duplicate bank-holiday date in division: {division}")

        unexpected_years = set(parsed_dates.year.astype(int)) - coverage_years
        if unexpected_years:
            raise ValueError(
                f"Bank-holiday dates in {division} fall outside coverage_years: "
                f"{sorted(unexpected_years)}"
            )

        holiday_dates = set(parsed_dates.date)
        df[feature_name] = local_dates.isin(holiday_dates).astype("int8")

    return df

# Defines the function that creates historical demand features.
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["ND"], "lag-feature input")
    # Shift demand by 48 rows so each row receives demand from 24 elapsed hours earlier.
    df["lag_48"] = df["ND"].shift(DAILY_LAG_PERIODS)

    # Shifts demand by 336 rows to obtain demand from 168 elapsed hours earlier.
    df["lag_336"] = df["ND"].shift(WEEKLY_LAG_PERIODS)

    # Shifts demand by 48 periods so only values available at the forecast origin remain eligible.
    demand_available_at_origin = df["ND"].shift(FORECAST_HORIZON_PERIODS)
    df["rolling_mean_48"] = demand_available_at_origin.rolling(
        window=DAILY_ROLLING_WINDOW,
        min_periods=DAILY_ROLLING_WINDOW,
    ).mean()
    # Calculates mean demand across the latest 336 half-hour observations available at the forecast.
    df["rolling_mean_336"] = demand_available_at_origin.rolling(
        window=WEEKLY_ROLLING_WINDOW,
        min_periods=WEEKLY_ROLLING_WINDOW,
    ).mean()
    return df


def validate_feature_output(df: pd.DataFrame) -> None:
    """Validate the modelling contract after all features are created."""
    required_columns = ["DATETIME", "ND", *ENGINEERED_FEATURE_COLUMNS]
    _require_columns(df, required_columns, "feature output")
    if df.empty:
        raise ValueError("Feature output contains no observations.")

    numeric_columns = ["ND", *ENGINEERED_FEATURE_COLUMNS]
    missing_values = int(df[numeric_columns].isna().sum().sum())
    non_finite_values = int(
        (~np.isfinite(df[numeric_columns].to_numpy(dtype=float))).sum()
    )
    if missing_values or non_finite_values:
        raise ValueError(
            "Feature output contains invalid modelling values: "
            f"missing={missing_values}, non_finite={non_finite_values}."
        )
    if (df["ND"] <= 0).any():
        raise ValueError("Feature output contains non-positive ND values.")

    duplicate_timestamps = int(df["DATETIME"].duplicated().sum())
    utc_timestamps = df["DATETIME"].dt.tz_convert("UTC")
    invalid_intervals = int(
        utc_timestamps.diff().dropna().ne(pd.Timedelta(minutes=30)).sum()
    )
    if duplicate_timestamps or invalid_intervals:
        raise ValueError(
            "Feature output is not a unique continuous half-hourly series: "
            f"duplicates={duplicate_timestamps}, "
            f"invalid_intervals={invalid_intervals}."
        )

    range_contracts = {
        "settlement_period": (1, 50),
        "day_of_week": (0, 6),
        "month": (1, 12),
        "day_of_year": (1, 366),
    }
    for column, (minimum, maximum) in range_contracts.items():
        if not df[column].between(minimum, maximum).all():
            raise ValueError(
                f"Feature {column} contains values outside "
                f"{minimum} to {maximum}."
            )

    for flag_column in [
        "is_weekend",
        "is_bank_holiday_england_wales",
        "is_bank_holiday_scotland",
    ]:
        if not df[flag_column].isin([0, 1]).all():
            raise ValueError(f"Feature {flag_column} must be binary.")


def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove only the predictable leading rows created by history features."""
    required_columns = ["ND", *ENGINEERED_FEATURE_COLUMNS]
    _require_columns(df, required_columns, "feature-cleaning input")

    expected_leading_rows = max(
        DAILY_LAG_PERIODS,
        WEEKLY_LAG_PERIODS,
        FORECAST_HORIZON_PERIODS + DAILY_ROLLING_WINDOW - 1,
        FORECAST_HORIZON_PERIODS + WEEKLY_ROLLING_WINDOW - 1,
    )
    if len(df) <= expected_leading_rows:
        raise ValueError(
            "Not enough observations to create all history features: "
            f"rows={len(df)}, required_more_than={expected_leading_rows}."
        )

    missing_feature_rows = np.flatnonzero(
        df[ENGINEERED_FEATURE_COLUMNS].isna().any(axis=1).to_numpy()
    )
    expected_missing_rows = np.arange(expected_leading_rows)
    if not np.array_equal(missing_feature_rows, expected_missing_rows):
        raise ValueError(
            "Missing feature values are not limited to the expected leading "
            f"{expected_leading_rows} rows."
        )

    df = df.iloc[expected_leading_rows:].copy().reset_index(drop=True)
    validate_feature_output(df)
    return df


def save_feature_data(df: pd.DataFrame, output_file: Path) -> None:
    """Write the feature dataset atomically to avoid a partial official CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_suffix(".csv.tmp")
    try:
        df.to_csv(temporary_output, index=False)
        temporary_output.replace(output_file)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def main() -> None:
    df = load_clean_data(INPUT_FILE)

    print("Clean data loaded.")
    print(f"Original rows: {len(df):,}")
    print(
        f"Original data range: {df['DATETIME'].min()} to "
        f"{df['DATETIME'].max()}"
    )

    original_rows = len(df)
    df = create_time_features(df)
    df = create_bank_holiday_features(df)
    df = create_lag_features(df)
    df = clean_feature_data(df)
    save_feature_data(df, OUTPUT_FILE)

    print("\nFeature engineering completed.")
    print(f"Rows after feature creation: {len(df):,}")
    print(f"Predictable leading rows removed: {original_rows - len(df):,}")
    print(
        f"Feature date range: {df['DATETIME'].min()} to "
        f"{df['DATETIME'].max()}"
    )
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
