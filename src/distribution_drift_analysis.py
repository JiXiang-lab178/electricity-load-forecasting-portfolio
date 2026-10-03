"""Quantify temporal distribution drift in demand and model inputs.

The analysis compares the 2021-2023 training reference with 2024 and 2025,
then adds adjacent-year comparisons to show when changes occurred. It measures
distribution shift; it does not by itself prove that P(y | X) has changed.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from feature_config import FINAL_FEATURE_COLUMNS
from train_ml_models import INPUT_FILE, LOCAL_TIMEZONE, load_feature_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

OUTPUT_DETAIL_FILE = TABLES_DIR / "distribution_drift_results.csv"
OUTPUT_SUMMARY_FILE = TABLES_DIR / "distribution_drift_summary.csv"
OUTPUT_HEATMAP_FILE = FIGURES_DIR / "distribution_drift_psi_heatmap.png"
OUTPUT_ND_DISTRIBUTION_FILE = FIGURES_DIR / "nd_distribution_by_year.png"

CONTINUOUS_VARIABLES = ["ND", "lag_48", "lag_336", "day_of_year"]
CATEGORICAL_VARIABLES = [
    feature
    for feature in FINAL_FEATURE_COLUMNS
    if feature not in CONTINUOUS_VARIABLES
]
DRIFT_VARIABLES = CONTINUOUS_VARIABLES + CATEGORICAL_VARIABLES

# The first two comparisons answer whether deployment-period distributions have
# moved away from training. Adjacent years locate when any movement occurred.
COMPARISONS = [
    {
        "Comparison": "2021-2023 -> 2024",
        "Comparison_Type": "Training reference",
        "Reference_Years": (2021, 2022, 2023),
        "Current_Years": (2024,),
    },
    {
        "Comparison": "2021-2023 -> 2025",
        "Comparison_Type": "Training reference",
        "Reference_Years": (2021, 2022, 2023),
        "Current_Years": (2025,),
    },
    {
        "Comparison": "2021 -> 2022",
        "Comparison_Type": "Adjacent year",
        "Reference_Years": (2021,),
        "Current_Years": (2022,),
    },
    {
        "Comparison": "2022 -> 2023",
        "Comparison_Type": "Adjacent year",
        "Reference_Years": (2022,),
        "Current_Years": (2023,),
    },
    {
        "Comparison": "2023 -> 2024",
        "Comparison_Type": "Adjacent year",
        "Reference_Years": (2023,),
        "Current_Years": (2024,),
    },
    {
        "Comparison": "2024 -> 2025",
        "Comparison_Type": "Adjacent year",
        "Reference_Years": (2024,),
        "Current_Years": (2025,),
    },
]


def restrict_to_common_calendar_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Give every year the same set of complete local calendar days.

    The feature dataset starts part-way through January 2021 because lagged and
    rolling candidate variables require historical rows. Without this control,
    unequal seasonal coverage could be mistaken for drift.
    """
    coverage = pd.DataFrame({
        "Year": df["DATETIME"].dt.year,
        "Local_Date": df["DATETIME"].dt.date,
        "Calendar_Day": df["DATETIME"].dt.strftime("%m-%d"),
    })
    rows_per_date = coverage.groupby(
        ["Year", "Local_Date"],
    )["Local_Date"].transform("size")
    expected_rows_per_date = coverage["Local_Date"].map(
        lambda local_date: int(
            (
                pd.Timestamp(local_date + timedelta(days=1))
                .tz_localize(LOCAL_TIMEZONE)
                .tz_convert("UTC")
                - pd.Timestamp(local_date)
                .tz_localize(LOCAL_TIMEZONE)
                .tz_convert("UTC")
            )
            / pd.Timedelta(minutes=30)
        )
    )
    coverage["Complete_Day"] = rows_per_date.eq(expected_rows_per_date)

    calendar_days_by_year = [
        set(year_df.loc[year_df["Complete_Day"], "Calendar_Day"])
        for _, year_df in coverage.groupby("Year")
    ]
    if not calendar_days_by_year:
        raise ValueError("No years are available for common-coverage analysis.")

    common_calendar_days = set.intersection(*calendar_days_by_year)
    if not common_calendar_days:
        raise ValueError("No complete calendar days are common to every year.")

    common_mask = (
        coverage["Complete_Day"]
        & coverage["Calendar_Day"].isin(common_calendar_days)
    )
    return df.loc[common_mask].reset_index(drop=True)


def _probability_vectors(
    reference: np.ndarray,
    current: np.ndarray,
    variable_type: str,
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned probability vectors for PSI and JS calculations."""
    use_continuous_bins = (
        variable_type == "continuous" and np.unique(reference).size > 2
    )
    if use_continuous_bins:
        # Define continuous bins from reference quantiles only; the current
        # period is measured against, rather than allowed to redefine, them.
        edges = np.unique(
            np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
        ).astype(float)
        if edges.size > 2:
            edges[0] = -np.inf
            edges[-1] = np.inf
            reference_counts, _ = np.histogram(reference, bins=edges)
            current_counts, _ = np.histogram(current, bins=edges)
        else:
            use_continuous_bins = False

    if not use_continuous_bins:
        categories = pd.Index(reference).union(pd.Index(current))
        reference_counts = (
            pd.Series(reference).value_counts().reindex(categories, fill_value=0).to_numpy()
        )
        current_counts = (
            pd.Series(current).value_counts().reindex(categories, fill_value=0).to_numpy()
        )

    reference_probabilities = reference_counts / reference_counts.sum()
    current_probabilities = current_counts / current_counts.sum()
    return reference_probabilities, current_probabilities


def calculate_psi(
    reference,
    current,
    variable_type: str = "continuous",
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate Population Stability Index using reference-defined bins."""
    reference_array = np.asarray(reference)
    current_array = np.asarray(current)
    reference_probabilities, current_probabilities = _probability_vectors(
        reference_array,
        current_array,
        variable_type,
        bins,
    )
    reference_probabilities = np.clip(reference_probabilities, epsilon, None)
    current_probabilities = np.clip(current_probabilities, epsilon, None)
    return float(
        np.sum(
            (current_probabilities - reference_probabilities)
            * np.log(current_probabilities / reference_probabilities)
        )
    )


def classify_psi(psi: float) -> str:
    """Apply common heuristic bands; these are not hypothesis-test cut-offs."""
    if psi < 0.10:
        return "Stable"
    if psi < 0.25:
        return "Moderate shift"
    return "Large shift"


def analyse_variable(
    reference: pd.Series,
    current: pd.Series,
    variable: str,
    variable_type: str,
) -> dict:
    """Calculate complementary drift statistics for one variable."""
    reference_array = reference.dropna().to_numpy()
    current_array = current.dropna().to_numpy()
    if reference_array.size == 0 or current_array.size == 0:
        raise ValueError(f"Cannot analyse empty variable: {variable}")

    reference_probabilities, current_probabilities = _probability_vectors(
        reference_array,
        current_array,
        variable_type,
    )
    psi = calculate_psi(reference_array, current_array, variable_type)

    result = {
        "Variable": variable,
        "Variable_Type": variable_type,
        "Variable_Role": "Target" if variable == "ND" else "Model input",
        "Reference_Rows": reference_array.size,
        "Current_Rows": current_array.size,
        "Reference_Mean": np.nan,
        "Current_Mean": np.nan,
        "Mean_Change_Percent": np.nan,
        "PSI": psi,
        "PSI_Severity": classify_psi(psi),
        "KS_Statistic": np.nan,
        "KS_P_Value": np.nan,
        "Normalised_Wasserstein": np.nan,
        "Jensen_Shannon_Distance": np.nan,
    }

    # Continuous and categorical variables need different complementary
    # statistics, while PSI remains the common headline measure.
    if variable_type == "continuous":
        reference_mean = float(np.mean(reference_array))
        current_mean = float(np.mean(current_array))
        reference_std = float(np.std(reference_array, ddof=0))
        ks_result = ks_2samp(reference_array, current_array, method="asymp")

        result.update({
            "Reference_Mean": reference_mean,
            "Current_Mean": current_mean,
            "Mean_Change_Percent": (
                100.0 * (current_mean - reference_mean) / reference_mean
                if reference_mean != 0
                else np.nan
            ),
            "KS_Statistic": float(ks_result.statistic),
            "KS_P_Value": float(ks_result.pvalue),
            "Normalised_Wasserstein": (
                float(wasserstein_distance(reference_array, current_array) / reference_std)
                if reference_std > 0
                else np.nan
            ),
        })
    else:
        result["Jensen_Shannon_Distance"] = float(
            jensenshannon(
                reference_probabilities,
                current_probabilities,
                base=2,
            )
        )

    return result


def run_distribution_drift_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run training-reference and adjacent-year drift comparisons."""
    missing_columns = [column for column in DRIFT_VARIABLES if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing drift-analysis columns: {missing_columns}")

    results = []
    years = df["DATETIME"].dt.year

    for comparison in COMPARISONS:
        reference_df = df[years.isin(comparison["Reference_Years"])]
        current_df = df[years.isin(comparison["Current_Years"])]
        if reference_df.empty or current_df.empty:
            raise ValueError(f"Empty drift comparison: {comparison['Comparison']}")

        for variable in DRIFT_VARIABLES:
            variable_type = (
                "continuous" if variable in CONTINUOUS_VARIABLES else "categorical"
            )
            row = analyse_variable(
                reference_df[variable],
                current_df[variable],
                variable,
                variable_type,
            )
            row.update({
                "Comparison": comparison["Comparison"],
                "Comparison_Type": comparison["Comparison_Type"],
                "Reference_Years": ",".join(map(str, comparison["Reference_Years"])),
                "Current_Years": ",".join(map(str, comparison["Current_Years"])),
            })
            results.append(row)

    columns = [
        "Comparison",
        "Comparison_Type",
        "Reference_Years",
        "Current_Years",
        "Variable",
        "Variable_Type",
        "Variable_Role",
        "Reference_Rows",
        "Current_Rows",
        "Reference_Mean",
        "Current_Mean",
        "Mean_Change_Percent",
        "PSI",
        "PSI_Severity",
        "KS_Statistic",
        "KS_P_Value",
        "Normalised_Wasserstein",
        "Jensen_Shannon_Distance",
    ]
    return pd.DataFrame(results)[columns]


def create_drift_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise PSI classifications for each time comparison."""
    summary_rows = []
    for (comparison, comparison_type), group in results_df.groupby(
        ["Comparison", "Comparison_Type"],
        sort=False,
    ):
        largest_row = group.loc[group["PSI"].idxmax()]
        summary_rows.append({
            "Comparison": comparison,
            "Comparison_Type": comparison_type,
            "Variables_Monitored": len(group),
            "Stable_Variables": int(group["PSI_Severity"].eq("Stable").sum()),
            "Moderate_Shift_Variables": int(
                group["PSI_Severity"].eq("Moderate shift").sum()
            ),
            "Large_Shift_Variables": int(
                group["PSI_Severity"].eq("Large shift").sum()
            ),
            "Maximum_PSI": float(largest_row["PSI"]),
            "Largest_Shift_Variable": largest_row["Variable"],
        })
    return pd.DataFrame(summary_rows)


def plot_psi_heatmap(results_df: pd.DataFrame) -> None:
    """Plot PSI values for all variables and temporal comparisons."""
    comparison_order = [item["Comparison"] for item in COMPARISONS]
    pivot_df = results_df.pivot(index="Variable", columns="Comparison", values="PSI")
    pivot_df = pivot_df.reindex(index=DRIFT_VARIABLES, columns=comparison_order)

    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(pivot_df.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot_df.index)), pivot_df.index)
    ax.set_title("Distribution drift measured by Population Stability Index")

    for row_index in range(pivot_df.shape[0]):
        for column_index in range(pivot_df.shape[1]):
            value = pivot_df.iloc[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=8)

    colour_bar = fig.colorbar(image, ax=ax)
    colour_bar.set_label("PSI")
    fig.tight_layout()
    fig.savefig(OUTPUT_HEATMAP_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nd_distribution_by_year(df: pd.DataFrame) -> None:
    """Plot annual ND density histograms on common bins."""
    minimum, maximum = df["ND"].quantile([0.001, 0.999])
    bins = np.linspace(minimum, maximum, 60)

    fig, ax = plt.subplots(figsize=(11, 6))
    for year in sorted(df["DATETIME"].dt.year.unique()):
        values = df.loc[df["DATETIME"].dt.year == year, "ND"]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=str(year),
        )

    ax.set_xlabel("National Demand (MW)")
    ax.set_ylabel("Density")
    ax.set_title("Annual National Demand distributions")
    ax.legend(title="Year")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_ND_DISTRIBUTION_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_data(INPUT_FILE)
    # Equal calendar coverage prevents missing early-2021 history from being
    # misinterpreted as a change in the underlying distribution.
    df = restrict_to_common_calendar_coverage(df)
    results_df = run_distribution_drift_analysis(df)
    summary_df = create_drift_summary(results_df)

    results_df.to_csv(OUTPUT_DETAIL_FILE, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_FILE, index=False)
    plot_psi_heatmap(results_df)
    plot_nd_distribution_by_year(df)

    print("Distribution-drift analysis completed.")
    print(summary_df.to_string(index=False))
    print(f"\nDetailed metrics saved to: {OUTPUT_DETAIL_FILE}")
    print(f"Summary saved to: {OUTPUT_SUMMARY_FILE}")
    print(f"PSI heatmap saved to: {OUTPUT_HEATMAP_FILE}")
    print(f"Demand distribution figure saved to: {OUTPUT_ND_DISTRIBUTION_FILE}")


if __name__ == "__main__":
    main()
