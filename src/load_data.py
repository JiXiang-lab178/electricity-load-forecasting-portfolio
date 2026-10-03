"""Load and validate the five annual NESO demand files and create DST-safe timestamps.
The module checks file names, schemas, settlement-period structure and critical
values before writing the combined processed dataset atomically.
"""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Europe/London"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "clean_demand_data_2021_2025.csv"

# Define the exact annual source-file contract before reading any raw data.
EXPECTED_YEARS = tuple(range(2021, 2026))
EXPECTED_RAW_FILES = tuple(
    f"demanddata_{year}.csv"
    for year in EXPECTED_YEARS
)
REQUIRED_SOURCE_COLUMNS = {
    "SETTLEMENT_DATE",
    "SETTLEMENT_PERIOD",
    "ND",
}


def load_all_csv_files(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load the five expected annual CSV files and validate each schema.
    """
    discovered_files = {
        file.name: file
        for file in raw_data_dir.glob("*.csv")
    }
    if not discovered_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    missing_files = sorted(set(EXPECTED_RAW_FILES) - set(discovered_files))
    unexpected_files = sorted(set(discovered_files) - set(EXPECTED_RAW_FILES))
    if missing_files:
        raise FileNotFoundError(
            f"Missing expected raw data files: {missing_files}"
        )
    if unexpected_files:
        raise ValueError(
            "Unexpected CSV files found in the raw data folder: "
            f"{unexpected_files}"
        )

    dataframes = []
    for filename in EXPECTED_RAW_FILES:
        file = discovered_files[filename]
        print(f"Processing {file.name}")
        df = pd.read_csv(file)

        # Standardise headers before validating each annual file's schema.
        df.columns = df.columns.str.strip().str.upper()
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            raise ValueError(
                f"Duplicate columns after standardisation in {file.name}: "
                f"{duplicate_columns}"
            )

        missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {file.name}: {missing_columns}"
            )
        if df.empty:
            raise ValueError(f"Raw data file is empty: {file.name}")

        df["SOURCE_FILE"] = file.name
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def validate_settlement_day_structure(df: pd.DataFrame) -> None:
    """Require every date to contain one complete 46, 48 or 50-period sequence."""
    duplicate_periods = df.duplicated(
        ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
        keep=False,
    )
    if duplicate_periods.any():
        examples = (
            df.loc[
                duplicate_periods,
                ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
            ]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate settlement date/period combinations found: "
            f"{examples}"
        )

    # The row count, unique values, minimum, maximum and arithmetic sum verify
    # that each day contains one complete sequence from 1 to N.
    daily_structure = (
        df.groupby("SETTLEMENT_DATE")["SETTLEMENT_PERIOD"]
        .agg(["size", "nunique", "min", "max", "sum"])
    )
    invalid_counts = ~daily_structure["size"].isin([46, 48, 50])
    expected_sums = (
        daily_structure["size"]
        * (daily_structure["size"] + 1)
        / 2
    )
    invalid_sequences = (
        (daily_structure["nunique"] != daily_structure["size"])
        | (daily_structure["min"] != 1)
        | (daily_structure["max"] != daily_structure["size"])
        | (daily_structure["sum"] != expected_sums)
    )
    invalid_days = daily_structure[invalid_counts | invalid_sequences]
    if not invalid_days.empty:
        raise ValueError(
            "Invalid daily settlement-period structure. Example dates: "
            f"{invalid_days.head(5).index.strftime('%Y-%m-%d').tolist()}"
        )


def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    It converts SETTLEMENT_DATE and SETTLEMENT_PERIOD into one timezone-aware DATETIME column in the Europe/London timezone.
    """
    required_columns = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Settlement periods must be valid before they are converted into elapsed
    # half-hours from local midnight.
    df["SETTLEMENT_PERIOD"] = pd.to_numeric(df["SETTLEMENT_PERIOD"], errors="coerce")

    missing_period_count = int(df["SETTLEMENT_PERIOD"].isna().sum())
    if missing_period_count:
        raise ValueError(
            f"Missing or non-numeric settlement periods: {missing_period_count}"
        )

    invalid_period = (
        ~df["SETTLEMENT_PERIOD"].between(1, 50)
        | (df["SETTLEMENT_PERIOD"] % 1 != 0)
    )
    if invalid_period.any():
        raise ValueError("SETTLEMENT_PERIOD must be an integer between 1 and 50.")

    # Parse known source formats in a deterministic order, using the mixed
    # fallback only for values that remain unresolved.
    date_text = df["SETTLEMENT_DATE"].astype(str).str.strip()
    parsed_date = pd.to_datetime(
        date_text,
        format="%Y-%m-%d",
        errors="coerce")
    missing = parsed_date.isna()
    parsed_date.loc[missing] = pd.to_datetime(
        date_text.loc[missing],
        format="%d/%m/%Y",
        errors="coerce"
    )

    missing = parsed_date.isna()
    parsed_date.loc[missing] = pd.to_datetime(
        date_text.loc[missing],
        format="%d-%b-%Y",
        errors="coerce"
    )

    missing = parsed_date.isna()
    parsed_date.loc[missing] = pd.to_datetime(
        date_text.loc[missing],
        format="mixed",
        # Prefer day-month-year interpretation for any remaining ambiguous text.
        dayfirst=True,
        errors="coerce"
    )
    invalid_date_count = int(parsed_date.isna().sum())
    print(f"Invalid settlement dates: {invalid_date_count:,}")
    if invalid_date_count:
        raise ValueError(
            f"Invalid settlement dates found: {invalid_date_count}"
        )

    df["SETTLEMENT_DATE"] = parsed_date

    # A file-year cross-check detects silently misplaced or mislabelled inputs.
    if "SOURCE_FILE" in df.columns:
        expected_year_by_file = {
            filename: year
            for filename, year in zip(EXPECTED_RAW_FILES, EXPECTED_YEARS)
        }
        for source_file, source_df in df.groupby("SOURCE_FILE"):
            expected_year = expected_year_by_file.get(source_file)
            if expected_year is None:
                raise ValueError(f"Unexpected source file label: {source_file}")
            observed_years = set(
                source_df["SETTLEMENT_DATE"].dt.year.astype(int).unique()
            )
            if observed_years != {expected_year}:
                raise ValueError(
                    f"{source_file} contains settlement years "
                    f"{sorted(observed_years)}; expected {expected_year}."
                )

    validate_settlement_day_structure(df)

    # Adding elapsed periods to timezone-aware local midnight correctly creates
    # the UK's 46-period spring and 50-period autumn transition days.
    local_midnight = df["SETTLEMENT_DATE"].dt.tz_localize(LOCAL_TIMEZONE)

    df["DATETIME"] = local_midnight + pd.to_timedelta(
        (df["SETTLEMENT_PERIOD"] - 1) * 30,
        unit="minutes"
    )

    return df


def clean_demand_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert demand values to numeric, validate timestamps and demand,
    and sort records chronologically.
    """
    if "ND" not in df.columns:
        raise ValueError("Missing target column: ND")
    if "DATETIME" not in df.columns:
        raise ValueError("Missing DATETIME column")

    df["ND"] = pd.to_numeric(df["ND"], errors="coerce")

    rows_before = len(df)
    missing_datetime = df["DATETIME"].isna().sum()
    missing_demand = df["ND"].isna().sum()

    if missing_datetime or missing_demand:
        raise ValueError(
            "Critical values are missing after parsing: "
            f"DATETIME={missing_datetime}, ND={missing_demand}."
        )

    non_finite_demand = int((~np.isfinite(df["ND"])).sum())
    non_positive_demand = int((df["ND"] <= 0).sum())
    if non_finite_demand:
        raise ValueError(f"Non-finite ND values found: {non_finite_demand}")
    if non_positive_demand:
        raise ValueError(f"Non-positive ND values found: {non_positive_demand}")

    df = df.sort_values("DATETIME").reset_index(drop=True)

    duplicate_timestamps = int(df["DATETIME"].duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(
            f"Duplicate DATETIME values found: {duplicate_timestamps}"
        )

    # UTC continuity is the unambiguous half-hour contract across DST changes.
    utc_timestamps = df["DATETIME"].dt.tz_convert("UTC")
    non_continuous_intervals = int(
        utc_timestamps.diff().dropna().ne(pd.Timedelta(minutes=30)).sum()
    )
    if non_continuous_intervals:
        raise ValueError(
            "DATETIME is not continuous at 30-minute intervals: "
            f"{non_continuous_intervals} invalid intervals."
        )

    rows_removed = rows_before - len(df)

    print(f"Missing datetime values: {missing_datetime:,}")
    print(f"Missing demand values: {missing_demand:,}")
    print(f"Rows removed during cleaning: {rows_removed:,}")

    return df


def validate_project_coverage(df: pd.DataFrame) -> None:
    """Validate complete daily coverage for the intended 2021-2025 project."""
    observed_years = set(df["DATETIME"].dt.year.astype(int).unique())
    expected_years = set(EXPECTED_YEARS)
    if observed_years != expected_years:
        raise ValueError(
            f"Observed years {sorted(observed_years)} do not match "
            f"expected years {sorted(expected_years)}."
        )

    expected_dates = set(
        pd.date_range(
            f"{EXPECTED_YEARS[0]}-01-01",
            f"{EXPECTED_YEARS[-1]}-12-31",
            freq="D",
        ).date
    )
    observed_dates = set(df["SETTLEMENT_DATE"].dt.date)
    missing_dates = sorted(expected_dates - observed_dates)
    extra_dates = sorted(observed_dates - expected_dates)
    if missing_dates or extra_dates:
        raise ValueError(
            "Project date coverage is incomplete or out of range: "
            f"missing={missing_dates[:5]}, extra={extra_dates[:5]}."
        )

    daily_counts = df.groupby("SETTLEMENT_DATE").size()
    for year in EXPECTED_YEARS:
        year_counts = daily_counts[daily_counts.index.year == year]
        if int(year_counts.eq(46).sum()) != 1 or int(year_counts.eq(50).sum()) != 1:
            raise ValueError(
                f"Year {year} must contain exactly one 46-period day and "
                "one 50-period day."
            )


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_csv_files(RAW_DATA_DIR)
    df = create_datetime_column(df)
    df = clean_demand_data(df)
    validate_project_coverage(df)

    # Replace the official processed file only after a complete temporary CSV
    # has been written successfully.
    temporary_output = OUTPUT_FILE.with_suffix(".csv.tmp")
    try:
        df.to_csv(temporary_output, index=False)
        temporary_output.replace(OUTPUT_FILE)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    print("\nData loading completed.")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['DATETIME'].min()} to {df['DATETIME'].max()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
