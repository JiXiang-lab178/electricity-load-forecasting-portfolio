"""Extract and visualise feature importance from fitted tree models."""

from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from train_ml_models import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

OUTPUT_IMPORTANCE_FILE = RESULTS_TABLES_DIR / "feature_importance.csv"


MODEL_FILES = {
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "XGBoost": MODELS_DIR / "xgboost.joblib",
    "LightGBM": MODELS_DIR / "lightgbm.joblib"
}


def load_models():
    """Load the three fitted tree models required for importance analysis."""
    models = {}

    # Fail explicitly if training has not produced every expected model file.
    for model_name, model_file in MODEL_FILES.items():
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        models[model_name] = joblib.load(model_file)

    return models


def extract_feature_importance(models):
    """Align each tree model's native importance values with final features."""
    all_importance = []

    for model_name, model in models.items():
        if not hasattr(model, "feature_importances_"):
            print(f"Skipping {model_name}: no feature_importances_ attribute.")
            continue

        # All fitted models use the same frozen FEATURE_COLUMNS order, so the
        # returned importance vector can be mapped directly to those names.
        importance_values = model.feature_importances_

        importance_df = pd.DataFrame({
            "Model": model_name,
            "Feature": FEATURE_COLUMNS,
            "Importance": importance_values
        })

        importance_df = importance_df.sort_values("Importance", ascending=False)

        all_importance.append(importance_df)

    return pd.concat(all_importance, ignore_index=True)


def plot_feature_importance(importance_df: pd.DataFrame):
    """Save one ranked horizontal importance plot for each tree model."""
    for model_name in importance_df["Model"].unique():
        model_df = importance_df[importance_df["Model"] == model_name].copy()
        model_df = model_df.sort_values("Importance", ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(model_df["Feature"], model_df["Importance"])
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance: {model_name}")
        plt.tight_layout()

        safe_model_name = model_name.lower().replace(" ", "_")
        output_file = RESULTS_FIGURES_DIR / f"feature_importance_{safe_model_name}.png"

        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Saved: {output_file}")


def main():
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Read the final fitted artefacts instead of retraining models in this stage.
    models = load_models()
    importance_df = extract_feature_importance(models)

    importance_df.to_csv(OUTPUT_IMPORTANCE_FILE, index=False)

    print("Feature importance results:")
    print(importance_df)

    plot_feature_importance(importance_df)

    print(f"\nFeature importance saved to: {OUTPUT_IMPORTANCE_FILE}")


if __name__ == "__main__":
    main()
