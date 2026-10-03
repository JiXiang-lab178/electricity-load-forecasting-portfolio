"""Create final model-comparison and example forecast figures."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Europe/London"

RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

BASELINE_RESULTS_FILE = RESULTS_TABLES_DIR / "baseline_results.csv"
ML_RESULTS_FILE = RESULTS_TABLES_DIR / "ml_model_results.csv"
ML_PREDICTIONS_FILE = RESULTS_TABLES_DIR / "ml_model_predictions.csv"
BASELINE_PREDICTIONS_FILE = (RESULTS_TABLES_DIR / "baseline_predictions_2025.csv")
OUTPUT_ALL_RESULTS_FILE = RESULTS_TABLES_DIR / "all_model_results.csv"


def load_results():
    """Combine baseline and ML metrics into one ranked result table."""
    # Both model families were evaluated on the same untouched 2025 test year.
    baseline_results = pd.read_csv(BASELINE_RESULTS_FILE)
    ml_results = pd.read_csv(ML_RESULTS_FILE)

    all_results = pd.concat([baseline_results, ml_results], ignore_index=True)
    all_results = all_results.sort_values("MAPE").reset_index(drop=True)

    return all_results


def plot_metric_comparison(all_results: pd.DataFrame, metric: str):
    """Save one comparison bar chart for a reported error metric."""
    plt.figure(figsize=(10, 6))
    plt.bar(all_results["Model"], all_results[metric])
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Model Comparison by {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_file = RESULTS_FIGURES_DIR / f"model_comparison_{metric.lower()}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")


def plot_actual_vs_best_prediction():
    """Plot one readable week of actual and LightGBM test demand."""
    predictions = pd.read_csv(ML_PREDICTIONS_FILE)
    # Restore local UK timestamps so the displayed daily cycle uses clock time.
    predictions["DATETIME"] = (
        pd.to_datetime(
            predictions["DATETIME"],
            format="mixed",
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(LOCAL_TIMEZONE)
    )

    # Use one fixed test week so short-term tracking remains readable.
    start_date = pd.Timestamp("2025-01-01", tz=LOCAL_TIMEZONE)
    end_date = pd.Timestamp("2025-01-08", tz=LOCAL_TIMEZONE)

    plot_df = predictions[
        (predictions["DATETIME"] >= start_date) &
        (predictions["DATETIME"] <= end_date)
    ].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df["DATETIME"], plot_df["ND"], label="Actual demand")
    plt.plot(plot_df["DATETIME"], plot_df["lightgbm_prediction"], label="LightGBM prediction")

    plt.xlabel("Datetime")
    plt.ylabel("National Demand (MW)")
    plt.title("Actual vs Predicted Electricity Demand: LightGBM")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_file = RESULTS_FIGURES_DIR / "actual_vs_predicted_lightgbm_one_week.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")


def main():
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = load_results()
    all_results.to_csv(OUTPUT_ALL_RESULTS_FILE, index=False)

    print("Combined model results:")
    print(all_results)
    print(f"\nSaved combined results to: {OUTPUT_ALL_RESULTS_FILE}")

    plot_metric_comparison(all_results, "RMSE")
    plot_metric_comparison(all_results, "MAE")
    plot_metric_comparison(all_results, "MAPE")

    def plot_actual_vs_baseline_predictions():
        """Plot the same test week for all three baseline forecasts."""
        predictions = pd.read_csv(BASELINE_PREDICTIONS_FILE)

        predictions["DATETIME"] = (
            pd.to_datetime(
                predictions["DATETIME"],
                format="mixed",
                errors="coerce",
                utc=True
            )
            .dt.tz_convert(LOCAL_TIMEZONE)
        )

        # Match the ML example window for a fair visual comparison.
        start_date = pd.Timestamp("2025-01-01", tz=LOCAL_TIMEZONE)
        end_date = pd.Timestamp("2025-01-08", tz=LOCAL_TIMEZONE)

        plot_df = predictions[
            (predictions["DATETIME"] >= start_date) &
            (predictions["DATETIME"] < end_date)
            ].copy()

        plt.figure(figsize=(12, 6))

        plt.plot(
            plot_df["DATETIME"],
            plot_df["ND"],
            label="Actual demand"
        )

        plt.plot(
            plot_df["DATETIME"],
            plot_df["lag_48"],
            label="lag_48 baseline"
        )

        plt.plot(
            plot_df["DATETIME"],
            plot_df["lag_336"],
            label="lag_336 baseline"
        )

        plt.plot(
            plot_df["DATETIME"],
            plot_df["arima_prediction"],
            label="ARIMA baseline"
        )

        plt.xlabel("Datetime")
        plt.ylabel("National Demand (MW)")
        plt.title("Actual vs Baseline Predicted Electricity Demand")

        plt.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        output_file = (
                RESULTS_FIGURES_DIR /
                "actual_vs_baseline_predictions_one_week.png"
        )

        plt.savefig(
            output_file,
            dpi=300
        )

        plt.close()

        print(f"Saved: {output_file}")

    plot_actual_vs_baseline_predictions()
    plot_actual_vs_best_prediction()

    print("\nVisualisation completed.")


if __name__ == "__main__":
    main()
