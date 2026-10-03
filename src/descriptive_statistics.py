"""Create validated descriptive tables and figures from complete clean demand."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Europe/London"
DEVELOPMENT_START_YEAR = 2021
DEVELOPMENT_END_YEAR = 2024

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "clean_demand_data_2021_2025.csv"
)
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

TABLE_FILES = {
    "intraday": "intraday_average_demand.csv",
    "hourly": "hourly_average_demand.csv",
    "weekly": "weekly_average_demand.csv",
    "weekend": "weekend_average_demand.csv",
    "monthly": "monthly_average_demand.csv",
    "yearly": "yearly_average_demand.csv",
}

DAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def validate_complete_coverage(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> None:
    """Require every elapsed half-hour in the requested local-year range."""
    expected_timestamps = pd.date_range(
        start=pd.Timestamp(f"{start_year}-01-01", tz=LOCAL_TIMEZONE),
        end=pd.Timestamp(f"{end_year + 1}-01-01", tz=LOCAL_TIMEZONE),
        freq="30min",
        inclusive="left",
    )
    observed_timestamps = pd.DatetimeIndex(df["DATETIME"])

    if observed_timestamps.equals(expected_timestamps):
        return

    missing_timestamps = expected_timestamps.difference(observed_timestamps)
    extra_timestamps = observed_timestamps.difference(expected_timestamps)
    raise ValueError(
        "Development data does not have complete half-hourly coverage: "
        f"expected_rows={len(expected_timestamps)}, "
        f"observed_rows={len(observed_timestamps)}, "
        f"missing_examples={missing_timestamps[:3].tolist()}, "
        f"extra_examples={extra_timestamps[:3].tolist()}."
    )


def load_development_data(
    input_file: Path = DATA_PATH,
    start_year: int = DEVELOPMENT_START_YEAR,
    end_year: int = DEVELOPMENT_END_YEAR,
) -> pd.DataFrame:
    """Load complete clean demand and reserve 2025 as the final test period."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        df = pd.read_csv(input_file)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Input file is empty: {input_file}") from error

    if df.empty:
        raise ValueError(f"Input file contains no observations: {input_file}")

    required_columns = ["DATETIME", "ND"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required descriptive-statistics columns: {missing_columns}"
        )

    parsed_datetime = pd.to_datetime(
        df["DATETIME"],
        format="mixed",
        errors="coerce",
        utc=True,
    )
    numeric_demand = pd.to_numeric(df["ND"], errors="coerce")
    invalid_datetime = int(parsed_datetime.isna().sum())
    invalid_demand = int(numeric_demand.isna().sum())
    if invalid_datetime or invalid_demand:
        raise ValueError(
            "Invalid descriptive-statistics input values: "
            f"DATETIME={invalid_datetime}, ND={invalid_demand}."
        )

    non_finite_demand = int((~np.isfinite(numeric_demand)).sum())
    non_positive_demand = int((numeric_demand <= 0).sum())
    if non_finite_demand or non_positive_demand:
        raise ValueError(
            "Invalid demand values for descriptive statistics: "
            f"non_finite={non_finite_demand}, "
            f"non_positive={non_positive_demand}."
        )

    df["DATETIME"] = parsed_datetime.dt.tz_convert(LOCAL_TIMEZONE)
    df["ND"] = numeric_demand.astype(float)
    df = df.sort_values("DATETIME").reset_index(drop=True)

    duplicate_timestamps = int(df["DATETIME"].duplicated().sum())
    utc_timestamps = df["DATETIME"].dt.tz_convert("UTC")
    invalid_intervals = int(
        utc_timestamps.diff().dropna().ne(pd.Timedelta(minutes=30)).sum()
    )
    off_grid_timestamps = int(
        (
            ~df["DATETIME"].dt.minute.isin([0, 30])
            | df["DATETIME"].dt.second.ne(0)
            | df["DATETIME"].dt.microsecond.ne(0)
        ).sum()
    )
    if duplicate_timestamps or invalid_intervals or off_grid_timestamps:
        raise ValueError(
            "Descriptive-statistics input is not a valid half-hourly series: "
            f"duplicates={duplicate_timestamps}, "
            f"invalid_intervals={invalid_intervals}, "
            f"off_grid={off_grid_timestamps}."
        )

    # Descriptive evidence used to justify features is restricted to the
    # development period so the final test year cannot influence decisions.
    development_df = df[
        df["DATETIME"].dt.year.between(start_year, end_year)
    ].copy()
    validate_complete_coverage(development_df, start_year, end_year)
    return development_df.reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar fields from local timestamps, including DST-safe clock time."""
    if "DATETIME" not in df.columns:
        raise ValueError("Missing DATETIME column for calendar features.")
    if not isinstance(df["DATETIME"].dtype, pd.DatetimeTZDtype):
        raise ValueError("DATETIME must be timezone-aware.")

    df = df.copy()
    # Local clock fields describe human demand cycles, while the underlying
    # timezone-aware timestamps retain the 46/50-period DST days correctly.
    df["local_half_hour"] = (
        df["DATETIME"].dt.hour * 2
        + df["DATETIME"].dt.minute.floordiv(30)
        + 1
    ).astype("int8")
    df["hour"] = df["DATETIME"].dt.hour.astype("int8")
    df["day_of_week"] = df["DATETIME"].dt.dayofweek.astype("int8")
    df["month"] = df["DATETIME"].dt.month.astype("int8")
    df["day_of_year"] = df["DATETIME"].dt.dayofyear.astype("int16")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["year"] = df["DATETIME"].dt.year.astype("int16")
    return df


def calculate_grouped_statistics(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Calculate centre, spread, range, and sample size for each demand group."""
    if group_column not in df.columns:
        raise ValueError(f"Missing grouping column: {group_column}")
    if "ND" not in df.columns:
        raise ValueError("Missing ND column for grouped statistics.")

    # Report spread and sample size as well as the mean so each visual summary
    # has a transparent statistical definition behind it.
    return (
        df.groupby(group_column, sort=True, observed=True)["ND"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )


def create_statistics_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create all tables from one validated development-data snapshot."""
    intraday_stats = calculate_grouped_statistics(df, "local_half_hour")
    intraday_stats["time_of_day"] = intraday_stats["local_half_hour"].map(
        lambda period: (
            f"{(int(period) - 1) // 2:02d}:"
            f"{((int(period) - 1) % 2) * 30:02d}"
        )
    )

    hourly_stats = calculate_grouped_statistics(df, "hour")

    weekly_stats = calculate_grouped_statistics(df, "day_of_week")
    weekly_stats["day_name"] = weekly_stats["day_of_week"].map(DAY_NAMES)

    weekend_stats = calculate_grouped_statistics(df, "is_weekend")
    weekend_stats["day_type"] = weekend_stats["is_weekend"].map({
        0: "Weekday",
        1: "Weekend",
    })

    return {
        "intraday": intraday_stats,
        "hourly": hourly_stats,
        "weekly": weekly_stats,
        "weekend": weekend_stats,
        "monthly": calculate_grouped_statistics(df, "month"),
        "yearly": calculate_grouped_statistics(df, "year"),
    }


def save_statistics_tables(
    tables: dict[str, pd.DataFrame],
    table_dir: Path = TABLE_DIR,
) -> None:
    """Write every CSV atomically so interrupted runs cannot leave partial tables."""
    table_dir.mkdir(parents=True, exist_ok=True)
    for table_name, filename in TABLE_FILES.items():
        output_file = table_dir / filename
        temporary_output = output_file.with_suffix(".csv.tmp")
        try:
            tables[table_name].to_csv(temporary_output, index=False)
            temporary_output.replace(output_file)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()


def _save_figure(fig: plt.Figure, output_file: Path) -> None:
    """Save a PNG atomically and close its matplotlib figure."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_name(output_file.name + ".tmp")
    try:
        fig.tight_layout()
        fig.savefig(temporary_output, dpi=300, format="png")
        temporary_output.replace(output_file)
    finally:
        plt.close(fig)
        if temporary_output.exists():
            temporary_output.unlink()


def create_statistics_figures(
    tables: dict[str, pd.DataFrame],
    figure_dir: Path = FIGURE_DIR,
) -> None:
    """Create publication-ready figures from the saved statistical definitions."""
    title_suffix = "Development Data, 2021–2024"
    colour = "#2a6f97"

    intraday = tables["intraday"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        intraday["local_half_hour"],
        intraday["mean"],
        marker="o",
        markersize=3,
        color=colour,
    )
    tick_positions = list(range(1, 49, 4))
    label_by_period = intraday.set_index("local_half_hour")["time_of_day"]
    ax.set_xticks(
        tick_positions,
        [label_by_period.loc[position] for position in tick_positions],
        rotation=45,
    )
    ax.set_title(f"Average Electricity Demand by Local Time of Day\n{title_suffix}")
    ax.set_xlabel("Local Time of Day (Europe/London)")
    ax.set_ylabel("Average ND (MW)")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, figure_dir / "average_demand_by_time_of_day.png")

    hourly = tables["hourly"]
    minimum = hourly.loc[hourly["mean"].idxmin()]
    maximum = hourly.loc[hourly["mean"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["mean"], marker="o", color=colour)
    ax.scatter([minimum["hour"], maximum["hour"]], [minimum["mean"], maximum["mean"]])
    ax.annotate(
        f"Min: {int(minimum['hour']):02d}:00\n{minimum['mean']:.0f} MW",
        (minimum["hour"], minimum["mean"]),
        xytext=(10, 10),
        textcoords="offset points",
    )
    ax.annotate(
        f"Max: {int(maximum['hour']):02d}:00\n{maximum['mean']:.0f} MW",
        (maximum["hour"], maximum["mean"]),
        xytext=(10, -30),
        textcoords="offset points",
    )
    ax.set_title(f"Average Electricity Demand by Local Hour\n{title_suffix}")
    ax.set_xlabel("Local Hour (Europe/London)")
    ax.set_ylabel("Average ND (MW)")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)
    _save_figure(fig, figure_dir / "average_demand_by_hour.png")

    weekly = tables["weekly"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(weekly["day_name"], weekly["mean"], color=colour)
    ax.set_title(f"Average Electricity Demand by Day of Week\n{title_suffix}")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average ND (MW)")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, figure_dir / "average_demand_by_day_of_week.png")

    weekend = tables["weekend"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(weekend["day_type"], weekend["mean"], color=colour)
    ax.set_title(f"Average Demand: Weekday vs Weekend\n{title_suffix}")
    ax.set_xlabel("Day Type")
    ax.set_ylabel("Average ND (MW)")
    _save_figure(fig, figure_dir / "average_demand_weekday_vs_weekend.png")

    monthly = tables["monthly"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly["month"], monthly["mean"], marker="o", color=colour)
    ax.set_title(f"Average Electricity Demand by Month\n{title_suffix}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average ND (MW)")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3)
    _save_figure(fig, figure_dir / "average_demand_by_month.png")

    yearly = tables["yearly"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(yearly["year"], yearly["mean"], marker="o", color=colour)
    ax.set_title(f"Average Electricity Demand by Year\n{title_suffix}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average ND (MW)")
    ax.set_xticks(yearly["year"].astype(int))
    ax.grid(True, alpha=0.3)
    _save_figure(fig, figure_dir / "average_demand_by_year.png")


def main() -> None:
    # Build every table and figure from one validated 2021-2024 snapshot.
    development_df = load_development_data(DATA_PATH)
    development_df = add_calendar_features(development_df)
    tables = create_statistics_tables(development_df)
    save_statistics_tables(tables)
    create_statistics_figures(tables)

    print("\nDescriptive-statistics input:")
    print(f"Rows: {len(development_df):,}")
    print(
        f"Date range: {development_df['DATETIME'].min()} to "
        f"{development_df['DATETIME'].max()}"
    )
    print("Final 2025 test data excluded: yes")
    print("\nIntraday average demand:")
    print(tables["intraday"].head())
    print("\nWeekly average demand:")
    print(tables["weekly"])
    print("\nWeekend average demand:")
    print(tables["weekend"])
    print("\nMonthly average demand:")
    print(tables["monthly"])
    print("\nYearly average demand:")
    print(tables["yearly"])
    print("\nDescriptive statistics and figures saved successfully.")


if __name__ == "__main__":
    main()
