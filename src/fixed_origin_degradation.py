"""Evaluate frozen-model performance as time moves away from one origin.

Models are fitted once on 2021-2023 and evaluated on 2024 and 2025 without
refitting or future hyperparameter selection. Observed lag inputs are still
available at each prediction time, so this measures model ageing rather than a
literal two-year-ahead recursive forecast made at the end of 2023.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_ml_models import (
    FEATURE_COLUMNS,
    INPUT_FILE,
    TARGET_COLUMN,
    calculate_daily_mape,
    calculate_metrics,
    get_models,
    load_feature_data,
    moving_block_bootstrap_mean_ci,
    validate_complete_year_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DETAIL_FILE = TABLES_DIR / "fixed_origin_yearly_performance.csv"
OUTPUT_SUMMARY_FILE = TABLES_DIR / "fixed_origin_degradation_summary.csv"
OUTPUT_FIGURE_FILE = FIGURES_DIR / "fixed_origin_mape_degradation.png"

ORIGIN_YEAR = 2023
TEST_YEARS = (2024, 2025)


def split_fixed_origin_data(
    df: pd.DataFrame,
    origin_year: int = ORIGIN_YEAR,
    test_years: tuple[int, ...] = TEST_YEARS,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Create one frozen training set and later annual evaluation sets."""
    training_df = df[df["DATETIME"].dt.year <= origin_year].copy()
    test_frames = {
        year: df[df["DATETIME"].dt.year == year].copy()
        for year in test_years
    }

    if training_df.empty:
        raise ValueError("Fixed-origin training set is empty.")
    if training_df["DATETIME"].dt.year.max() != origin_year:
        raise ValueError(f"Training data does not reach origin year {origin_year}.")
    if any(year <= origin_year for year in test_years):
        raise ValueError("Every fixed-origin test year must be after the origin.")

    empty_years = [year for year, frame in test_frames.items() if frame.empty]
    if empty_years:
        raise ValueError(f"Fixed-origin test sets are empty for years: {empty_years}")

    return training_df, test_frames


def evaluate_fixed_origin(
    training_df: pd.DataFrame,
    test_frames: dict[int, pd.DataFrame],
    models: dict | None = None,
    origin_year: int = ORIGIN_YEAR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit each ML model exactly once, then score every later year."""
    if models is None:
        models = get_models()

    results = []
    daily_mape_by_model_year = {}
    training_years = sorted(training_df["DATETIME"].dt.year.unique())
    training_label = f"{training_years[0]}-{training_years[-1]}"

    # Persistence baselines do not learn parameters, but they help distinguish
    # model ageing from an evaluation year that is generally harder to predict.
    for test_year, test_df in sorted(test_frames.items()):
        for baseline_name, prediction_column in {
            "lag_48_baseline": "lag_48",
            "lag_336_baseline": "lag_336",
        }.items():
            predictions = test_df[prediction_column]
            metrics = calculate_metrics(
                test_df[TARGET_COLUMN],
                predictions,
            )
            daily_mape = calculate_daily_mape(
                test_df["DATETIME"],
                test_df[TARGET_COLUMN],
                predictions,
            )
            ci_lower, ci_upper = moving_block_bootstrap_mean_ci(daily_mape)
            daily_mape_by_model_year[(baseline_name, test_year)] = daily_mape
            results.append({
                "Origin_Year": origin_year,
                "Training_Years": "Not applicable",
                "Training_Rows": 0,
                "Test_Year": test_year,
                "Test_Rows": len(test_df),
                "Years_After_Origin": test_year - origin_year,
                "Model": baseline_name,
                "Model_Type": "Persistence baseline",
                "Configuration": "Observed demand lag; no fitted parameters",
                "Refitted_After_Origin": False,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "MAPE_CI_Lower": ci_lower,
                "MAPE_CI_Upper": ci_upper,
            })

    X_train = training_df[FEATURE_COLUMNS]
    y_train = training_df[TARGET_COLUMN]

    # Fit each model outside the test-year loop so the same frozen estimator is
    # evaluated in both 2024 and 2025 without refitting.
    for model_name, model in models.items():
        if verbose:
            print(f"Fitting once at {origin_year} origin: {model_name}")
        model.fit(X_train, y_train)

        for test_year, test_df in sorted(test_frames.items()):
            predictions = model.predict(test_df[FEATURE_COLUMNS])
            metrics = calculate_metrics(test_df[TARGET_COLUMN], predictions)
            daily_mape = calculate_daily_mape(
                test_df["DATETIME"],
                test_df[TARGET_COLUMN],
                predictions,
            )
            ci_lower, ci_upper = moving_block_bootstrap_mean_ci(daily_mape)
            daily_mape_by_model_year[(model_name, test_year)] = daily_mape
            results.append({
                "Origin_Year": origin_year,
                "Training_Years": training_label,
                "Training_Rows": len(training_df),
                "Test_Year": test_year,
                "Test_Rows": len(test_df),
                "Years_After_Origin": test_year - origin_year,
                "Model": model_name,
                "Model_Type": "Frozen machine-learning model",
                "Configuration": (
                    "Fixed base configuration from get_models(); "
                    "no 2024 hyperparameter tuning"
                ),
                "Refitted_After_Origin": False,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "MAPE_CI_Lower": ci_lower,
                "MAPE_CI_Upper": ci_upper,
            })

    results_df = pd.DataFrame(results).sort_values(
        ["Model_Type", "Model", "Test_Year"]
    ).reset_index(drop=True)
    first_mape = results_df.groupby("Model")["MAPE"].transform("first")
    results_df["MAPE_Change_From_First_Test"] = results_df["MAPE"] - first_mape
    results_df["MAPE_Relative_Change_Percent"] = (
        100.0 * results_df["MAPE_Change_From_First_Test"] / first_mape
    )
    results_df.attrs["daily_mape_by_model_year"] = daily_mape_by_model_year
    return results_df


def create_degradation_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise an observed two-year change without calling it a trend."""
    summary_rows = []
    daily_mape_by_model_year = results_df.attrs.get(
        "daily_mape_by_model_year",
        {},
    )
    for model_name, group in results_df.groupby("Model", sort=False):
        group = group.sort_values("Years_After_Origin")
        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        absolute_change = float(last_row["MAPE"] - first_row["MAPE"])

        first_key = (model_name, int(first_row["Test_Year"]))
        last_key = (model_name, int(last_row["Test_Year"]))
        change_ci_lower = np.nan
        change_ci_upper = np.nan
        ci_includes_zero = pd.NA

        if (
            first_key in daily_mape_by_model_year
            and last_key in daily_mape_by_model_year
        ):
            first_daily = daily_mape_by_model_year[first_key]
            last_daily = daily_mape_by_model_year[last_key]
            first_calendar = pd.DataFrame({
                "Calendar_Day": [date.strftime("%m-%d") for date in first_daily.index],
                "First_Daily_MAPE": first_daily.to_numpy(),
            })
            last_calendar = pd.DataFrame({
                "Calendar_Day": [date.strftime("%m-%d") for date in last_daily.index],
                "Last_Daily_MAPE": last_daily.to_numpy(),
            })
            # Pair identical calendar days before bootstrapping the change so
            # seasonal composition is matched across the two evaluation years.
            paired_days = first_calendar.merge(
                last_calendar,
                on="Calendar_Day",
                how="inner",
                validate="one_to_one",
            )
            daily_change = (
                paired_days["Last_Daily_MAPE"]
                - paired_days["First_Daily_MAPE"]
            )
            change_ci_lower, change_ci_upper = moving_block_bootstrap_mean_ci(
                daily_change
            )
            ci_includes_zero = (
                change_ci_lower <= 0.0 <= change_ci_upper
            )

        if absolute_change > 0:
            observed_change = "Higher error in last test year"
        elif absolute_change < 0:
            observed_change = "Lower error in last test year"
        else:
            observed_change = "No observed change"

        if ci_includes_zero is pd.NA:
            evidence_statement = "Change interval not estimated"
        elif ci_includes_zero:
            evidence_statement = (
                "Observed change is inconclusive at the 95% level"
            )
        elif change_ci_lower > 0:
            evidence_statement = (
                "Higher last-year error; paired 95% interval excludes zero"
            )
        else:
            evidence_statement = (
                "Lower last-year error; paired 95% interval excludes zero"
            )

        first_mape = float(first_row["MAPE"])
        relative_change = (
            100.0 * absolute_change / first_mape
            if first_mape != 0.0
            else (0.0 if absolute_change == 0.0 else np.nan)
        )

        summary_rows.append({
            "Origin_Year": int(first_row["Origin_Year"]),
            "Model": model_name,
            "Model_Type": first_row["Model_Type"],
            "First_Test_Year": int(first_row["Test_Year"]),
            "Last_Test_Year": int(last_row["Test_Year"]),
            "First_MAPE": first_mape,
            "Last_MAPE": float(last_row["MAPE"]),
            "Absolute_MAPE_Change": absolute_change,
            "Relative_MAPE_Change_Percent": relative_change,
            "Change_CI_Lower": change_ci_lower,
            "Change_CI_Upper": change_ci_upper,
            "Change_CI_Includes_Zero": ci_includes_zero,
            "Observed_Change": observed_change,
            "Evidence_Statement": evidence_statement,
            "Evidence_Scope": (
                "Two annual evaluations only; not a long-term trend"
            ),
        })

    return pd.DataFrame(summary_rows).sort_values(
        ["Model_Type", "Absolute_MAPE_Change"],
        ascending=[True, False],
    ).reset_index(drop=True)


def plot_fixed_origin_degradation(results_df: pd.DataFrame) -> None:
    """Plot annual MAPE for frozen models and persistence context."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name, group in results_df.groupby("Model", sort=False):
        group = group.sort_values("Test_Year")
        is_baseline = group["Model_Type"].iloc[0] == "Persistence baseline"
        ax.plot(
            group["Test_Year"],
            group["MAPE"],
            marker="o",
            linestyle="--" if is_baseline else "-",
            linewidth=1.5 if is_baseline else 2.0,
            label=model_name,
        )

    ax.set_xticks(list(TEST_YEARS))
    ax.set_xlabel("Test year (model frozen after 2023)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Observed fixed-origin MAPE change across two test years")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # The 2023 origin is fixed once, followed by two strictly later evaluations.
    df = load_feature_data(INPUT_FILE)
    training_df, test_frames = split_fixed_origin_data(df)
    for test_year, test_df in test_frames.items():
        validate_complete_year_coverage(
            test_df,
            test_year,
            "Fixed-origin test set",
        )
    results_df = evaluate_fixed_origin(training_df, test_frames)
    summary_df = create_degradation_summary(results_df)

    results_df.to_csv(OUTPUT_DETAIL_FILE, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)
    plot_fixed_origin_degradation(results_df)

    print("\nFixed-origin degradation experiment completed.")
    print(summary_df.to_string(index=False))
    print(f"\nDetailed results saved to: {OUTPUT_DETAIL_FILE}")
    print(f"Summary saved to: {OUTPUT_SUMMARY_FILE}")
    print(f"Figure saved to: {OUTPUT_FIGURE_FILE}")


if __name__ == "__main__":
    main()
