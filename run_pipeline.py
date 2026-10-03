"""Run the complete electricity-load forecasting workflow reproducibly."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Validate the complete five-year source snapshot before starting any stage.
REQUIRED_RAW_FILES = [
    PROJECT_ROOT / "data" / "raw" / f"demanddata_{year}.csv"
    for year in range(2021, 2026)
]

# The order follows data dependencies: each stage consumes validated artefacts
# produced by an earlier stage and writes the next set of reproducible outputs.
PIPELINE_STEPS = [
    ("Load and clean raw data", PROJECT_ROOT / "src" / "load_data.py"),
    ("Create forecasting features", PROJECT_ROOT / "src" / "create_features.py"),
    ("Generate descriptive statistics", PROJECT_ROOT / "src" / "descriptive_statistics.py"),
    ("Evaluate baseline models", PROJECT_ROOT / "src" / "baseline_models.py"),
    ("Run feature-ablation experiment", PROJECT_ROOT / "src" / "feature_ablation.py"),
    ("Analyse temporal distribution drift", PROJECT_ROOT / "src" / "distribution_drift_analysis.py"),
    ("Tune and train machine-learning models", PROJECT_ROOT / "src" / "train_ml_models.py"),
    ("Run training-period sensitivity analysis", PROJECT_ROOT / "src" / "training_period_sensitivity.py"),
    ("Calculate feature importance", PROJECT_ROOT / "src" / "feature_importance.py"),
    ("Create comparison figures", PROJECT_ROOT / "src" / "visualise_results.py"),
    ("Run fixed-origin degradation experiment", PROJECT_ROOT / "src" / "fixed_origin_degradation.py"),
    ("Run robustness analysis", PROJECT_ROOT / "src" / "robustness_analysis.py"),
]

ROBUSTNESS_SCRIPT_NAMES = {
    "fixed_origin_degradation.py",
    "robustness_analysis.py",
    "training_period_sensitivity.py",
}


def validate_project_inputs() -> None:
    """Fail early when required source files or pipeline scripts are missing."""
    missing_raw_files = [path for path in REQUIRED_RAW_FILES if not path.exists()]
    missing_scripts = [path for _, path in PIPELINE_STEPS if not path.exists()]

    if missing_raw_files:
        missing_list = "\n".join(f"  - {path}" for path in missing_raw_files)
        raise FileNotFoundError(
            "Required NESO raw data files are missing:\n"
            f"{missing_list}\n"
            "Download the annual files described in README.md before running the pipeline."
        )

    if missing_scripts:
        missing_list = "\n".join(f"  - {path}" for path in missing_scripts)
        raise FileNotFoundError(f"Pipeline scripts are missing:\n{missing_list}")


def run_command(label: str, command: list[str], dry_run: bool) -> None:
    """Display and optionally execute one pipeline command."""
    display_command = subprocess.list2cmdline(command)
    print(f"\n[{label}]\n{display_command}", flush=True)

    if not dry_run:
        # check=True stops the workflow immediately if one stage fails.
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_pipeline(
    *,
    dry_run: bool = False,
    skip_robustness: bool = False,
    skip_tests: bool = False,
) -> None:
    """Validate inputs, run tests, and execute the workflow in dependency order."""
    validate_project_inputs()

    # Run automated contracts before regenerating any official project output.
    if not skip_tests:
        run_command(
            "Automated tests",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
            ],
            dry_run,
        )

    # The optional shortcut removes the slower robustness and sensitivity
    # stages while preserving the dependency order of every remaining script.
    steps = [
        (label, script_path)
        for label, script_path in PIPELINE_STEPS
        if not (
            skip_robustness
            and script_path.name in ROBUSTNESS_SCRIPT_NAMES
        )
    ]

    for label, script_path in steps:
        run_command(
            label,
            [sys.executable, str(script_path)],
            dry_run,
        )

    if dry_run:
        print("\nDry run completed: no project outputs were changed.")
    else:
        print("\nFull forecasting pipeline completed successfully.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the COMP702 electricity-load forecasting pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and validate the planned commands without executing them.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help=(
            "Skip the slower fixed-origin, rolling-origin, and "
            "training-period sensitivity analyses."
        ),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Run the pipeline without executing the automated tests first.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    run_pipeline(
        dry_run=arguments.dry_run,
        skip_robustness=arguments.skip_robustness,
        skip_tests=arguments.skip_tests,
    )


if __name__ == "__main__":
    main()
