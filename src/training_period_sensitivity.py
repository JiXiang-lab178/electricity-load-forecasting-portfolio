"""Measure the effect of excluding 2021 from model-fitting samples.

The primary experiment keeps 2021 in the fitting samples.  This independent
sensitivity analysis freezes the selected feature columns, repeats the same
2024 hyperparameter validation procedure, and compares that experiment with a
second experiment whose fitting samples start in 2022.  The 2024 validation
and 2025 final test years are unchanged in both scenarios.

The 2021 rows remain in the feature dataset while features are constructed.
This preserves historically available lag values for early 2022 forecasts;
the 2021 target rows are excluded only from model fitting in the alternative
scenario.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import train_ml_models as training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "features_demand_data_2021_2025.csv"
)
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_RESULTS_FILE = RESULTS_TABLES_DIR / "training_period_sensitivity.csv"
OUTPUT_FIGURE_FILE = RESULTS_FIGURES_DIR / "training_period_sensitivity_mape.png"

VALIDATION_YEAR = 2024
TEST_YEAR = 2025
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

SCENARIOS = {
    "Keep 2021": {
        "parameter_training_years": (2021, 2022, 2023),
        "final_training_years": (2021, 2022, 2023, 2024),
    },
    "Remove 2021": {
        "parameter_training_years": (2022, 2023),
        "final_training_years": (2022, 2023, 2024),
    },
}


def _rows_for_years(df: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    """Return feature rows whose local calendar year is in ``years``."""
    return df[df["DATETIME"].dt.year.isin(years)].copy()


def _validate_scenario_data(df: pd.DataFrame) -> None:
    """Validate the shared validation and test periods before fitting models."""
    for year, label in [
        (VALIDATION_YEAR, "Sensitivity validation data"),
        (TEST_YEAR, "Sensitivity test data"),
    ]:
        year_rows = _rows_for_years(df, (year,))
        if year_rows.empty:
            raise ValueError(f"No rows found for {label}: year={year}.")
        training.validate_complete_year_coverage(year_rows, year, label)


def _fit_scenario(
    df: pd.DataFrame,
    scenario_name: str,
    parameter_training_years: tuple[int, ...],
    final_training_years: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Tune, refit and evaluate every model for one training-period scenario."""
    parameter_training = _rows_for_years(df, parameter_training_years)
    validation = _rows_for_years(df, (VALIDATION_YEAR,))
    final_training = _rows_for_years(df, final_training_years)
    test = _rows_for_years(df, (TEST_YEAR,))

    if parameter_training.empty or final_training.empty:
        raise ValueError(f"Training data is empty for scenario: {scenario_name}.")

    feature_columns = list(training.FEATURE_COLUMNS)
    best_parameters, validation_results = training.tune_models(
        parameter_training[feature_columns],
        parameter_training[training.TARGET_COLUMN],
        validation[feature_columns],
        validation[training.TARGET_COLUMN],
    )

    selected_validation = validation_results[
        validation_results["Selected"]
    ].set_index("Model")
    daily_mapes: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []

    for model_name, parameters in best_parameters.items():
        model = training.get_models({model_name: parameters})[model_name]
        model.fit(
            final_training[feature_columns],
            final_training[training.TARGET_COLUMN],
        )
        predictions = model.predict(test[feature_columns])

        validation_metrics = selected_validation.loc[model_name]
        test_metrics = training.calculate_metrics(
            test[training.TARGET_COLUMN],
            predictions,
        )
        daily_mapes[model_name] = training.calculate_daily_mape(
            test["DATETIME"],
            test[training.TARGET_COLUMN],
            predictions,
        )

        rows.append({
            "Scenario": scenario_name,
            "Parameter_Training_Years": "-".join(
                str(year) for year in parameter_training_years
            ),
            "Final_Training_Years": "-".join(
                str(year) for year in final_training_years
            ),
            "Parameter_Training_Rows": len(parameter_training),
            "Final_Training_Rows": len(final_training),
            "Validation_Year": VALIDATION_YEAR,
            "Validation_Rows": len(validation),
            "Test_Year": TEST_YEAR,
            "Test_Rows": len(test),
            "Model": model_name,
            "Selected_Parameters": json.dumps(parameters, sort_keys=True),
            "Validation_RMSE": float(validation_metrics["RMSE"]),
            "Validation_MAE": float(validation_metrics["MAE"]),
            "Validation_MAPE": float(validation_metrics["MAPE"]),
            "Test_RMSE": float(test_metrics["RMSE"]),
            "Test_MAE": float(test_metrics["MAE"]),
            "Test_MAPE": float(test_metrics["MAPE"]),
            "Mean_Daily_MAPE": float(daily_mapes[model_name].mean()),
        })

    return pd.DataFrame(rows), daily_mapes


def _scenario_row(
    scenario_results: pd.DataFrame,
    scenario_name: str,
    model_name: str,
) -> pd.Series:
    """Return one model/scenario row and fail if it is not unique."""
    matches = scenario_results[
        (scenario_results["Scenario"] == scenario_name)
        & (scenario_results["Model"] == model_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one sensitivity result for "
            f"scenario={scenario_name}, model={model_name}; found={len(matches)}."
        )
    return matches.iloc[0]


def create_comparison_table(
    scenario_results: pd.DataFrame,
    daily_mapes: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    """Compare both scenarios and calculate paired daily bootstrap intervals."""
    model_order = list(training.get_models())
    rows: list[dict[str, object]] = []

    for model_name in model_order:
        keep = _scenario_row(scenario_results, "Keep 2021", model_name)
        removed = _scenario_row(scenario_results, "Remove 2021", model_name)

        aligned_daily = pd.concat(
            [
                daily_mapes["Keep 2021"][model_name],
                daily_mapes["Remove 2021"][model_name],
            ],
            axis=1,
            keys=["Keep_2021", "Remove_2021"],
            join="inner",
        ).dropna()
        if aligned_daily.empty:
            raise ValueError(f"No aligned daily errors for model: {model_name}.")

        daily_delta = (
            aligned_daily["Remove_2021"] - aligned_daily["Keep_2021"]
        ).to_numpy()
        ci_lower, ci_upper = training.moving_block_bootstrap_mean_ci(
            daily_delta,
            block_length=BOOTSTRAP_BLOCK_DAYS,
            n_resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
        )

        rows.append({
            "Model": model_name,
            "Parameter_Training_Rows_Keep_2021": int(keep["Parameter_Training_Rows"]),
            "Parameter_Training_Rows_Remove_2021": int(removed["Parameter_Training_Rows"]),
            "Final_Training_Rows_Keep_2021": int(keep["Final_Training_Rows"]),
            "Final_Training_Rows_Remove_2021": int(removed["Final_Training_Rows"]),
            "Selected_Parameters_Keep_2021": keep["Selected_Parameters"],
            "Selected_Parameters_Remove_2021": removed["Selected_Parameters"],
            "Validation_MAPE_Keep_2021": float(keep["Validation_MAPE"]),
            "Validation_MAPE_Remove_2021": float(removed["Validation_MAPE"]),
            "Validation_MAPE_Delta": float(
                removed["Validation_MAPE"] - keep["Validation_MAPE"]
            ),
            "Test_RMSE_Keep_2021": float(keep["Test_RMSE"]),
            "Test_RMSE_Remove_2021": float(removed["Test_RMSE"]),
            "Test_RMSE_Delta": float(removed["Test_RMSE"] - keep["Test_RMSE"]),
            "Test_MAE_Keep_2021": float(keep["Test_MAE"]),
            "Test_MAE_Remove_2021": float(removed["Test_MAE"]),
            "Test_MAE_Delta": float(removed["Test_MAE"] - keep["Test_MAE"]),
            "Test_MAPE_Keep_2021": float(keep["Test_MAPE"]),
            "Test_MAPE_Remove_2021": float(removed["Test_MAPE"]),
            "Test_MAPE_Delta": float(removed["Test_MAPE"] - keep["Test_MAPE"]),
            "Test_MAPE_Delta_Percent": float(
                100.0 * (removed["Test_MAPE"] - keep["Test_MAPE"])
                / keep["Test_MAPE"]
            ),
            "Mean_Daily_MAPE_Delta": float(daily_delta.mean()),
            "Mean_Daily_MAPE_Delta_CI_Lower": ci_lower,
            "Mean_Daily_MAPE_Delta_CI_Upper": ci_upper,
            "CI_Includes_Zero": bool(ci_lower <= 0.0 <= ci_upper),
            "Test_Rank_Keep_2021": 0,
            "Test_Rank_Remove_2021": 0,
        })

    comparison = pd.DataFrame(rows)
    comparison["Test_Rank_Keep_2021"] = (
        comparison["Test_MAPE_Keep_2021"].rank(method="min").astype(int)
    )
    comparison["Test_Rank_Remove_2021"] = (
        comparison["Test_MAPE_Remove_2021"].rank(method="min").astype(int)
    )
    return comparison


def save_results(comparison: pd.DataFrame) -> None:
    """Save the sensitivity table and a grouped 2025-MAPE figure."""
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_RESULTS_FILE, index=False)

    model_order = list(training.get_models())
    plot_data = comparison.set_index("Model").loc[model_order]
    axis = plot_data[["Test_MAPE_Keep_2021", "Test_MAPE_Remove_2021"]].plot(
        kind="bar",
        figsize=(10, 6),
        color=["#2a6f97", "#e07a5f"],
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("2025 Test MAPE (%)")
    axis.set_title("Sensitivity of 2025 Test MAPE to Excluding 2021 Training Data")
    axis.legend(["Keep 2021", "Remove 2021"], title="Training period")
    axis.tick_params(axis="x", rotation=25)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE_FILE, dpi=300)
    plt.close(figure)


def run_sensitivity(
    input_file: Path = INPUT_FILE,
) -> pd.DataFrame:
    """Run both scenarios without changing the primary result files."""
    data = training.load_feature_data(input_file)
    _validate_scenario_data(data)

    scenario_frames: list[pd.DataFrame] = []
    daily_mapes: dict[str, dict[str, pd.Series]] = {}
    for scenario_name, configuration in SCENARIOS.items():
        frame, scenario_daily_mapes = _fit_scenario(
            data,
            scenario_name,
            configuration["parameter_training_years"],
            configuration["final_training_years"],
        )
        scenario_frames.append(frame)
        daily_mapes[scenario_name] = scenario_daily_mapes

    scenario_results = pd.concat(scenario_frames, ignore_index=True)
    comparison = create_comparison_table(scenario_results, daily_mapes)
    save_results(comparison)
    return comparison


def main() -> None:
    comparison = run_sensitivity()
    print("\nTraining-period sensitivity results:")
    print(
        comparison[
            [
                "Model",
                "Validation_MAPE_Keep_2021",
                "Validation_MAPE_Remove_2021",
                "Test_MAPE_Keep_2021",
                "Test_MAPE_Remove_2021",
                "Test_MAPE_Delta",
                "Mean_Daily_MAPE_Delta_CI_Lower",
                "Mean_Daily_MAPE_Delta_CI_Upper",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved sensitivity table to: {OUTPUT_RESULTS_FILE}")
    print(f"Saved sensitivity figure to: {OUTPUT_FIGURE_FILE}")


if __name__ == "__main__":
    main()
