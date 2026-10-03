"""Regression tests for the core forecasting pipeline safeguards."""

from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import load_data  # noqa: E402
import baseline_models  # noqa: E402
import create_features  # noqa: E402
import descriptive_statistics  # noqa: E402
import distribution_drift_analysis  # noqa: E402
import feature_ablation  # noqa: E402
import feature_config  # noqa: E402
import fixed_origin_degradation  # noqa: E402
import robustness_analysis  # noqa: E402
import training_period_sensitivity  # noqa: E402
import run_pipeline  # noqa: E402
import train_ml_models  # noqa: E402


class SettlementDatetimeTests(unittest.TestCase):
    def test_dst_days_are_unique_and_continuous(self):
        spring_periods = pd.DataFrame({
            "SETTLEMENT_DATE": ["2025-03-30"] * 46,
            "SETTLEMENT_PERIOD": range(1, 47),
        })
        autumn_periods = pd.DataFrame({
            "SETTLEMENT_DATE": ["2025-10-26"] * 50,
            "SETTLEMENT_PERIOD": range(1, 51),
        })

        converted = load_data.create_datetime_column(
            pd.concat([spring_periods, autumn_periods], ignore_index=True)
        )

        for settlement_date, expected_rows in [
            ("2025-03-30", 46),
            ("2025-10-26", 50),
        ]:
            day = converted[
                converted["SETTLEMENT_DATE"] == pd.Timestamp(settlement_date)
            ]
            utc_timestamps = day["DATETIME"].dt.tz_convert("UTC")

            self.assertEqual(len(day), expected_rows)
            self.assertFalse(utc_timestamps.duplicated().any())
            self.assertTrue(
                utc_timestamps.diff().dropna().eq(pd.Timedelta(minutes=30)).all()
            )

        spring_period_3 = converted[
            (converted["SETTLEMENT_DATE"] == pd.Timestamp("2025-03-30"))
            & (converted["SETTLEMENT_PERIOD"] == 3)
        ]["DATETIME"].iloc[0]
        autumn_period_3 = converted[
            (converted["SETTLEMENT_DATE"] == pd.Timestamp("2025-10-26"))
            & (converted["SETTLEMENT_PERIOD"] == 3)
        ]["DATETIME"].iloc[0]
        autumn_period_5 = converted[
            (converted["SETTLEMENT_DATE"] == pd.Timestamp("2025-10-26"))
            & (converted["SETTLEMENT_PERIOD"] == 5)
        ]["DATETIME"].iloc[0]

        self.assertEqual(spring_period_3.hour, 2)
        self.assertEqual(autumn_period_3.hour, 1)
        self.assertEqual(autumn_period_5.hour, 1)
        self.assertNotEqual(autumn_period_3.utcoffset(), autumn_period_5.utcoffset())

    def test_incomplete_settlement_period_sequence_is_rejected(self):
        invalid_periods = list(range(1, 48)) + [49]
        frame = pd.DataFrame({
            "SETTLEMENT_DATE": ["2025-02-01"] * 48,
            "SETTLEMENT_PERIOD": invalid_periods,
        })

        with self.assertRaisesRegex(ValueError, "settlement-period structure"):
            load_data.create_datetime_column(frame)

    def test_cleaning_rejects_duplicate_timestamps(self):
        duplicate_time = pd.Timestamp("2025-01-01", tz="Europe/London")
        frame = pd.DataFrame({
            "DATETIME": [duplicate_time, duplicate_time],
            "ND": [100.0, 101.0],
        })

        with self.assertRaisesRegex(ValueError, "Duplicate DATETIME"):
            load_data.clean_demand_data(frame)

    def test_cleaning_rejects_non_finite_demand(self):
        frame = pd.DataFrame({
            "DATETIME": pd.date_range(
                "2025-01-01",
                periods=2,
                freq="30min",
                tz="Europe/London",
            ),
            "ND": [100.0, np.inf],
        })

        with self.assertRaisesRegex(ValueError, "Non-finite ND"):
            load_data.clean_demand_data(frame)


class RawFileValidationTests(unittest.TestCase):
    @staticmethod
    def write_expected_files(directory: Path) -> None:
        for year, filename in zip(
            load_data.EXPECTED_YEARS,
            load_data.EXPECTED_RAW_FILES,
        ):
            pd.DataFrame({
                "SETTLEMENT_DATE": [f"{year}-01-01"],
                "SETTLEMENT_PERIOD": [1],
                "ND": [100.0],
            }).to_csv(directory / filename, index=False)

    def test_unexpected_csv_file_is_rejected(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.write_expected_files(directory)
            pd.DataFrame({"value": [1]}).to_csv(
                directory / "backup.csv",
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "Unexpected CSV"):
                load_data.load_all_csv_files(directory)

    def test_each_source_file_must_have_required_columns(self):
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.write_expected_files(directory)
            pd.DataFrame({
                "SETTLEMENT_DATE": ["2023-01-01"],
                "SETTLEMENT_PERIOD": [1],
            }).to_csv(directory / "demanddata_2023.csv", index=False)

            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                load_data.load_all_csv_files(directory)


class ModellingTests(unittest.TestCase):
    @staticmethod
    def make_model_frame(rows: int = 20) -> pd.DataFrame:
        timestamps = pd.date_range(
            "2025-01-01",
            periods=rows,
            freq="30min",
            tz=train_ml_models.LOCAL_TIMEZONE,
        )
        frame = pd.DataFrame({
            column: np.arange(rows, dtype=float) + 1.0
            for column in train_ml_models.FEATURE_COLUMNS
        })
        frame["ND"] = np.arange(rows, dtype=float) + 1000.0
        frame["DATETIME"] = timestamps
        return frame

    def test_chronological_train_validation_test_split(self):
        timestamps = pd.to_datetime(
            [f"{year}-06-01 12:00:00" for year in range(2021, 2026)],
            utc=True,
        ).tz_convert(train_ml_models.LOCAL_TIMEZONE)
        frame = pd.DataFrame({"DATETIME": timestamps})

        train, validation, final_train, test = (
            train_ml_models.train_validation_test_split_by_year(frame)
        )

        self.assertEqual(set(train["DATETIME"].dt.year), {2021, 2022, 2023})
        self.assertEqual(set(validation["DATETIME"].dt.year), {2024})
        self.assertEqual(set(final_train["DATETIME"].dt.year), {2021, 2022, 2023, 2024})
        self.assertEqual(set(test["DATETIME"].dt.year), {2025})

    def test_metric_calculation(self):
        metrics = train_ml_models.calculate_metrics(
            np.array([100.0, 200.0]),
            np.array([110.0, 180.0]),
        )

        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(250.0))
        self.assertAlmostEqual(metrics["MAE"], 15.0)
        self.assertAlmostEqual(metrics["MAPE"], 10.0)

    def test_metric_calculation_rejects_non_positive_actuals(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            train_ml_models.calculate_metrics(
                np.array([100.0, 0.0]),
                np.array([100.0, 1.0]),
            )

    def test_model_loader_rejects_invalid_datetime_instead_of_dropping_it(self):
        frame = self.make_model_frame()
        frame["DATETIME"] = frame["DATETIME"].astype(object)
        frame.loc[5, "DATETIME"] = "invalid"
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "invalid_datetime.csv"
            frame.to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "DATETIME.*1"):
                train_ml_models.load_feature_data(input_file)

    def test_model_loader_rejects_a_time_gap(self):
        frame = self.make_model_frame().drop(index=5)
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "time_gap.csv"
            frame.to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "invalid_intervals=1"):
                train_ml_models.load_feature_data(input_file)

    def test_complete_year_validator_rejects_missing_half_hour(self):
        timestamps = pd.date_range(
            "2025-01-01",
            "2026-01-01",
            freq="30min",
            inclusive="left",
            tz=train_ml_models.LOCAL_TIMEZONE,
        ).delete(100)
        frame = pd.DataFrame({"DATETIME": timestamps})

        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            train_ml_models.validate_complete_year_coverage(
                frame,
                2025,
                "Test set",
            )

    def test_model_comparison_uses_paired_daily_errors(self):
        timestamps = pd.date_range(
            "2025-01-01",
            periods=14 * 48,
            freq="30min",
            tz=train_ml_models.LOCAL_TIMEZONE,
        )
        frame = pd.DataFrame({
            "DATETIME": timestamps,
            "ND": 100.0,
            "model_a_prediction": 101.0,
            "model_b_prediction": 102.0,
        })

        result = train_ml_models.compare_model_predictions(
            frame,
            {
                "Model A": "model_a_prediction",
                "Model B": "model_b_prediction",
            },
        ).iloc[0]

        self.assertEqual(result["Model_A"], "Model A")
        self.assertEqual(result["Model_B"], "Model B")
        self.assertAlmostEqual(
            result["Mean_Daily_MAPE_Difference_A_Minus_B"],
            -1.0,
        )
        self.assertFalse(result["CI_Includes_Zero"])

    def test_validation_search_cannot_receive_test_data(self):
        parameter_names = inspect.signature(
            train_ml_models.tune_models
        ).parameters

        self.assertEqual(
            list(parameter_names),
            ["X_train", "y_train", "X_validation", "y_validation"],
        )

    def test_parameter_grid_has_expected_candidates(self):
        grids = train_ml_models.get_parameter_grids()
        candidate_count = sum(
            len(list(train_ml_models.ParameterGrid(grid)))
            for grid in grids.values()
        )

        self.assertEqual(candidate_count, 17)


class ArimaBaselineTests(unittest.TestCase):
    @staticmethod
    def make_baseline_frame(rows: int = 400) -> pd.DataFrame:
        demand = np.arange(rows, dtype=float) + 1000.0
        lag_48 = np.concatenate([
            np.arange(48, dtype=float) + 900.0,
            demand[:-48],
        ])
        lag_336 = np.concatenate([
            np.arange(336, dtype=float) + 500.0,
            demand[:-336],
        ])
        return pd.DataFrame({
            "DATETIME": pd.date_range(
                "2025-01-01",
                periods=rows,
                freq="30min",
                tz=train_ml_models.LOCAL_TIMEZONE,
            ),
            "ND": demand,
            "lag_48": lag_48,
            "lag_336": lag_336,
        })

    def test_arima_p10_recovers_difference_dynamics(self):
        differences = [2.0]
        for _ in range(200):
            differences.append(1.5 + 0.6 * differences[-1])
        demand = np.cumsum([100.0, *differences])

        coefficients = baseline_models.fit_arima_p10(demand, 1)

        self.assertAlmostEqual(coefficients[0], 1.5, places=5)
        self.assertAlmostEqual(coefficients[1], 0.6, places=5)

    def test_arima_forecast_does_not_read_after_origin(self):
        demand = np.linspace(100.0, 300.0, 150)
        changed_future = demand.copy()
        target_index = 120
        origin_index = target_index - baseline_models.FORECAST_HORIZON_PERIODS
        changed_future[origin_index + 1:] += 10000.0
        coefficients = np.array([0.1, 0.5, 0.2])

        original_prediction = baseline_models.rolling_arima_forecast(
            demand,
            [target_index],
            coefficients,
            2,
        )
        changed_prediction = baseline_models.rolling_arima_forecast(
            changed_future,
            [target_index],
            coefficients,
            2,
        )

        np.testing.assert_allclose(original_prediction, changed_prediction)

    def test_arima_order_selection_rejects_test_year(self):
        frame = pd.DataFrame({
            "DATETIME": pd.to_datetime(
                ["2023-01-01", "2024-01-01", "2025-01-01"],
                utc=True,
            ).tz_convert(train_ml_models.LOCAL_TIMEZONE),
            "ND": [100.0, 101.0, 102.0],
        })

        with self.assertRaises(ValueError):
            baseline_models.select_arima_order(frame)

    def test_baseline_loader_rejects_non_finite_demand(self):
        frame = self.make_baseline_frame()
        frame.loc[10, "ND"] = np.inf
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "non_finite.csv"
            frame.to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "non_finite"):
                baseline_models.load_feature_data(input_file)

    def test_baseline_loader_rejects_inconsistent_lag(self):
        frame = self.make_baseline_frame()
        frame.loc[350, "lag_336"] += 1.0
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "incorrect_lag.csv"
            frame.to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "lag_336 is inconsistent"):
                baseline_models.load_feature_data(input_file)

    def test_baseline_loader_rejects_a_time_gap(self):
        frame = self.make_baseline_frame().drop(index=200)
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "time_gap.csv"
            frame.to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "invalid_intervals=1"):
                baseline_models.load_feature_data(input_file)

    def test_test_set_requires_complete_2025_coverage(self):
        frame = self.make_baseline_frame()

        with self.assertRaisesRegex(ValueError, "test coverage is incomplete"):
            baseline_models.create_test_set(frame)

    def test_metric_calculation_rejects_non_positive_actuals(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            baseline_models.calculate_metrics(
                np.array([100.0, 0.0]),
                np.array([100.0, 1.0]),
            )

    def test_unstable_ar_coefficients_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not stationary"):
            baseline_models.validate_ar_stability(
                np.array([0.0, 1.1]),
                autoregressive_order=1,
            )


class FeatureAblationTests(unittest.TestCase):
    @staticmethod
    def make_valid_ablation_frame(rows: int = 800) -> pd.DataFrame:
        timestamps = pd.date_range(
            "2025-01-01",
            periods=rows,
            freq="30min",
            tz=train_ml_models.LOCAL_TIMEZONE,
        )
        demand = pd.Series(np.arange(rows, dtype=float) + 1000.0)
        frame = pd.DataFrame({
            "DATETIME": timestamps,
            "ND": demand,
            "lag_48": demand.shift(48).fillna(900.0),
            "lag_336": demand.shift(336).fillna(700.0),
            "rolling_mean_48": (
                demand.shift(48).rolling(48, min_periods=48).mean().fillna(900.0)
            ),
            "rolling_mean_336": (
                demand.shift(48).rolling(336, min_periods=336).mean().fillna(700.0)
            ),
            "settlement_period": (np.arange(rows) % 48) + 1,
            "day_of_week": timestamps.dayofweek,
            "month": timestamps.month,
            "day_of_year": timestamps.dayofyear,
            "is_weekend": timestamps.dayofweek.isin([5, 6]).astype(int),
            "is_bank_holiday_england_wales": 0,
            "is_bank_holiday_scotland": 0,
        })
        return frame

    def test_feature_sets_are_non_empty_subsets_of_full_set(self):
        full_features = set(feature_ablation.ALL_CANDIDATE_FEATURES)

        self.assertEqual(
            len(feature_ablation.ALL_CANDIDATE_FEATURES),
            len(full_features),
        )
        for feature_set in feature_ablation.FEATURE_SETS.values():
            self.assertTrue(feature_set)
            self.assertTrue(set(feature_set).issubset(full_features))

    def test_final_model_uses_ablation_selected_feature_definition(self):
        self.assertEqual(
            train_ml_models.FEATURE_COLUMNS,
            list(feature_config.FINAL_FEATURE_COLUMNS),
        )
        self.assertEqual(
            feature_ablation.FEATURE_SETS[
                feature_config.FINAL_FEATURE_SET_NAME
            ],
            list(feature_config.FINAL_FEATURE_COLUMNS),
        )

    def test_leave_one_group_out_definitions_use_one_reference(self):
        full_features = set(feature_ablation.ALL_CANDIDATE_FEATURES)
        expected_removed_features = {
            "Without daily lag": {"lag_48"},
            "Without weekly lag": {"lag_336"},
            "Selected + both GB holiday flags": set(
                feature_ablation.ROLLING_FEATURES
            ),
            "Without calendar features": set(
                feature_ablation.CALENDAR_FEATURES
            ),
            "Without England/Wales holiday": {
                "is_bank_holiday_england_wales"
            },
            "Without Scotland holiday": {"is_bank_holiday_scotland"},
        }

        for feature_set_name, removed_features in expected_removed_features.items():
            observed_features = set(
                feature_ablation.FEATURE_SETS[feature_set_name]
            )
            self.assertEqual(full_features - observed_features, removed_features)

    def test_feature_selection_split_ends_before_hyperparameter_validation(self):
        validation_timestamps = pd.date_range(
            "2023-01-01",
            "2024-01-01",
            freq="30min",
            inclusive="left",
            tz=train_ml_models.LOCAL_TIMEZONE,
        )
        timestamps = pd.DatetimeIndex([
            pd.Timestamp("2021-06-01", tz=train_ml_models.LOCAL_TIMEZONE),
            pd.Timestamp("2022-06-01", tz=train_ml_models.LOCAL_TIMEZONE),
            *validation_timestamps,
            pd.Timestamp("2024-06-01", tz=train_ml_models.LOCAL_TIMEZONE),
            pd.Timestamp("2025-06-01", tz=train_ml_models.LOCAL_TIMEZONE),
        ])
        frame = pd.DataFrame({"DATETIME": timestamps})

        training, validation = feature_ablation.split_feature_selection_data(frame)

        self.assertEqual(set(training["DATETIME"].dt.year), {2021, 2022})
        self.assertEqual(set(validation["DATETIME"].dt.year), {2023})
        self.assertNotIn(2024, validation["DATETIME"].dt.year.to_list())
        self.assertNotIn(2025, validation["DATETIME"].dt.year.to_list())

    def test_feature_selection_split_rejects_incomplete_2023(self):
        validation_timestamps = pd.date_range(
            "2023-01-01",
            "2024-01-01",
            freq="30min",
            inclusive="left",
            tz=train_ml_models.LOCAL_TIMEZONE,
        ).delete(100)
        timestamps = pd.DatetimeIndex([
            pd.Timestamp("2021-06-01", tz=train_ml_models.LOCAL_TIMEZONE),
            pd.Timestamp("2022-06-01", tz=train_ml_models.LOCAL_TIMEZONE),
            *validation_timestamps,
        ])

        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            feature_ablation.split_feature_selection_data(
                pd.DataFrame({"DATETIME": timestamps})
            )

    def test_ablation_loader_rejects_non_finite_demand(self):
        frame = self.make_valid_ablation_frame()
        frame.loc[10, "ND"] = np.inf
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "non_finite.csv"
            frame.to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "non-finite"):
                feature_ablation.load_ablation_data(input_file)

    def test_ablation_loader_rejects_time_gap(self):
        frame = self.make_valid_ablation_frame().drop(index=200)
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "time_gap.csv"
            frame.to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "invalid_intervals=1"):
                feature_ablation.load_ablation_data(input_file)

    def test_moving_block_bootstrap_is_deterministic(self):
        daily_values = np.linspace(1.0, 4.0, 365)
        first_interval = feature_ablation.moving_block_bootstrap_mean_ci(
            daily_values,
            n_resamples=500,
        )
        second_interval = feature_ablation.moving_block_bootstrap_mean_ci(
            daily_values,
            n_resamples=500,
        )

        self.assertEqual(first_interval, second_interval)
        self.assertLess(first_interval[0], daily_values.mean())
        self.assertGreater(first_interval[1], daily_values.mean())

    def test_frozen_final_config_rejects_a_different_selection(self):
        results = pd.DataFrame({
            "Feature_Set": ["Lag + calendar features"],
            "Selected": [True],
        })

        with self.assertRaisesRegex(ValueError, "frozen final-model"):
            feature_ablation.validate_final_feature_selection(results)


class FeatureEngineeringValidationTests(unittest.TestCase):
    @staticmethod
    def make_valid_frame(rows: int = 800) -> pd.DataFrame:
        return pd.DataFrame({
            "DATETIME": pd.date_range(
                "2025-01-01",
                periods=rows,
                freq="30min",
                tz=create_features.LOCAL_TIMEZONE,
            ),
            "SETTLEMENT_PERIOD": (np.arange(rows) % 48) + 1,
            "ND": np.arange(rows, dtype=float) + 1000.0,
        })

    def test_loader_requires_settlement_period(self):
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "missing_period.csv"
            pd.DataFrame({
                "DATETIME": ["2025-01-01T00:00:00Z"],
                "ND": [100.0],
            }).to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "SETTLEMENT_PERIOD"):
                create_features.load_clean_data(input_file)

    def test_loader_rejects_invalid_datetime_instead_of_dropping_it(self):
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "invalid_datetime.csv"
            pd.DataFrame({
                "DATETIME": ["invalid", "2025-01-01T00:30:00Z"],
                "SETTLEMENT_PERIOD": [1, 2],
                "ND": [100.0, 101.0],
            }).to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "DATETIME=1"):
                create_features.load_clean_data(input_file)

    def test_loader_rejects_non_finite_demand(self):
        with TemporaryDirectory() as temporary_directory:
            input_file = Path(temporary_directory) / "infinite_demand.csv"
            pd.DataFrame({
                "DATETIME": [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:30:00Z",
                ],
                "SETTLEMENT_PERIOD": [1, 2],
                "ND": [100.0, np.inf],
            }).to_csv(input_file, index=False)

            with self.assertRaisesRegex(ValueError, "Non-finite ND"):
                create_features.load_clean_data(input_file)

    def test_all_demand_features_respect_the_48_period_information_horizon(self):
        target_index = 700
        forecast_origin = target_index - create_features.FORECAST_HORIZON_PERIODS
        original = pd.DataFrame({"ND": np.arange(800, dtype=float)})
        changed_future = original.copy()
        changed_future.loc[forecast_origin + 1:, "ND"] += 1_000_000.0

        original_features = create_features.create_lag_features(original)
        changed_features = create_features.create_lag_features(changed_future)
        demand_features = [
            "lag_48",
            "lag_336",
            "rolling_mean_48",
            "rolling_mean_336",
        ]

        np.testing.assert_allclose(
            original_features.loc[target_index, demand_features],
            changed_features.loc[target_index, demand_features],
        )

    def test_rolling_windows_end_at_the_forecast_origin(self):
        frame = pd.DataFrame({"ND": np.arange(800, dtype=float)})
        result = create_features.create_lag_features(frame)
        first_complete_weekly_window = 383

        self.assertEqual(
            result.loc[first_complete_weekly_window, "lag_48"],
            335.0,
        )
        self.assertEqual(
            result.loc[first_complete_weekly_window, "lag_336"],
            47.0,
        )
        self.assertEqual(
            result.loc[first_complete_weekly_window, "rolling_mean_48"],
            np.mean(np.arange(288.0, 336.0)),
        )
        self.assertEqual(
            result.loc[first_complete_weekly_window, "rolling_mean_336"],
            np.mean(np.arange(0.0, 336.0)),
        )

    def test_cleaning_removes_only_predictable_leading_rows(self):
        frame = self.make_valid_frame()
        expected_first_datetime = frame.loc[383, "DATETIME"]
        frame = create_features.create_time_features(frame)
        frame = create_features.create_bank_holiday_features(frame)
        frame = create_features.create_lag_features(frame)

        result = create_features.clean_feature_data(frame)

        self.assertEqual(len(result), 417)
        self.assertEqual(result.loc[0, "DATETIME"], expected_first_datetime)

    def test_cleaning_rejects_an_internal_missing_feature(self):
        frame = self.make_valid_frame()
        frame = create_features.create_time_features(frame)
        frame = create_features.create_bank_holiday_features(frame)
        frame = create_features.create_lag_features(frame)
        frame.loc[500, "lag_48"] = np.nan

        with self.assertRaisesRegex(ValueError, "expected leading 383 rows"):
            create_features.clean_feature_data(frame)


class BankHolidayFeatureTests(unittest.TestCase):
    def test_gb_regional_holiday_flags(self):
        timestamps = pd.to_datetime(
            [
                "2025-01-02 12:00:00",  # Scotland only
                "2025-04-21 12:00:00",  # England and Wales only
                "2025-12-25 12:00:00",  # Both divisions
                "2025-02-01 12:00:00",  # Neither division
            ]
        ).tz_localize(create_features.LOCAL_TIMEZONE)
        frame = pd.DataFrame({"DATETIME": timestamps})

        result = create_features.create_bank_holiday_features(frame)

        self.assertEqual(
            result["is_bank_holiday_england_wales"].to_list(),
            [0, 1, 1, 0],
        )
        self.assertEqual(
            result["is_bank_holiday_scotland"].to_list(),
            [1, 0, 1, 0],
        )


class DescriptiveStatisticsTests(unittest.TestCase):
    def test_descriptive_input_is_complete_clean_data(self):
        self.assertEqual(
            descriptive_statistics.DATA_PATH.name,
            "clean_demand_data_2021_2025.csv",
        )

    def test_dst_periods_map_to_their_actual_local_clock_time(self):
        spring_periods = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-03-31"] * 46,
            "SETTLEMENT_PERIOD": range(1, 47),
        })
        autumn_periods = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-10-27"] * 50,
            "SETTLEMENT_PERIOD": range(1, 51),
        })
        converted = load_data.create_datetime_column(
            pd.concat([spring_periods, autumn_periods], ignore_index=True)
        )
        converted["ND"] = 100.0

        result = descriptive_statistics.add_calendar_features(converted)

        spring_period_3 = result.loc[
            (result["SETTLEMENT_DATE"] == pd.Timestamp("2024-03-31"))
            & (result["SETTLEMENT_PERIOD"] == 3),
            "local_half_hour",
        ].iloc[0]
        autumn_period_3 = result.loc[
            (result["SETTLEMENT_DATE"] == pd.Timestamp("2024-10-27"))
            & (result["SETTLEMENT_PERIOD"] == 3),
            "local_half_hour",
        ].iloc[0]
        autumn_period_5 = result.loc[
            (result["SETTLEMENT_DATE"] == pd.Timestamp("2024-10-27"))
            & (result["SETTLEMENT_PERIOD"] == 5),
            "local_half_hour",
        ].iloc[0]

        self.assertEqual(spring_period_3, 5)  # 02:00 local
        self.assertEqual(autumn_period_3, 3)  # first 01:00
        self.assertEqual(autumn_period_5, 3)  # repeated 01:00

    def test_full_year_clock_counts_balance_spring_and_autumn_dst(self):
        timestamps = pd.date_range(
            pd.Timestamp("2024-01-01", tz=create_features.LOCAL_TIMEZONE),
            pd.Timestamp("2025-01-01", tz=create_features.LOCAL_TIMEZONE),
            freq="30min",
            inclusive="left",
        )
        frame = pd.DataFrame({
            "DATETIME": timestamps,
            "ND": np.arange(len(timestamps), dtype=float) + 1000.0,
        })
        frame = descriptive_statistics.add_calendar_features(frame)

        result = descriptive_statistics.calculate_grouped_statistics(
            frame,
            "local_half_hour",
        )

        self.assertEqual(len(result), 48)
        self.assertTrue(result["count"].eq(366).all())
        self.assertEqual(int(result["count"].sum()), len(frame))

    def test_complete_coverage_rejects_a_missing_half_hour(self):
        timestamps = pd.date_range(
            pd.Timestamp("2024-01-01", tz=create_features.LOCAL_TIMEZONE),
            pd.Timestamp("2025-01-01", tz=create_features.LOCAL_TIMEZONE),
            freq="30min",
            inclusive="left",
        )
        frame = pd.DataFrame({"DATETIME": timestamps}).drop(index=100)

        with self.assertRaisesRegex(ValueError, "complete half-hourly coverage"):
            descriptive_statistics.validate_complete_coverage(frame, 2024, 2024)

    def test_grouped_statistics_include_robust_summary_measures(self):
        frame = pd.DataFrame({
            "group": [1, 1, 1, 1],
            "ND": [1.0, 2.0, 3.0, 100.0],
        })

        result = descriptive_statistics.calculate_grouped_statistics(
            frame,
            "group",
        ).iloc[0]

        self.assertEqual(result["count"], 4)
        self.assertEqual(result["median"], 2.5)
        self.assertEqual(result["q25"], 1.75)
        self.assertEqual(result["q75"], 27.25)


class DistributionDriftTests(unittest.TestCase):
    def test_every_final_model_input_is_monitored(self):
        self.assertTrue(
            set(feature_config.FINAL_FEATURE_COLUMNS).issubset(
                distribution_drift_analysis.DRIFT_VARIABLES
            )
        )

    def test_psi_is_zero_for_identical_distributions(self):
        values = np.arange(1.0, 101.0)
        psi = distribution_drift_analysis.calculate_psi(values, values)

        self.assertAlmostEqual(psi, 0.0)

    def test_psi_detects_large_shift(self):
        reference = np.arange(1.0, 101.0)
        shifted = np.arange(101.0, 201.0)
        psi = distribution_drift_analysis.calculate_psi(reference, shifted)

        self.assertGreaterEqual(psi, 0.25)
        self.assertEqual(
            distribution_drift_analysis.classify_psi(psi),
            "Large shift",
        )

    def test_training_reference_is_fixed_before_monitoring_years(self):
        training_comparisons = [
            comparison
            for comparison in distribution_drift_analysis.COMPARISONS
            if comparison["Comparison_Type"] == "Training reference"
        ]

        self.assertTrue(training_comparisons)
        for comparison in training_comparisons:
            self.assertEqual(comparison["Reference_Years"], (2021, 2022, 2023))
            self.assertTrue(min(comparison["Current_Years"]) >= 2024)

    def test_common_coverage_removes_partial_calendar_days(self):
        timestamps = list(pd.date_range(
            "2021-01-02 00:00:00",
            periods=48,
            freq="30min",
            tz=train_ml_models.LOCAL_TIMEZONE,
        ))
        timestamps += [pd.Timestamp(
            "2021-01-01 23:30:00",
            tz=train_ml_models.LOCAL_TIMEZONE,
        )]
        timestamps += list(pd.date_range(
            "2022-01-01 00:00:00",
            periods=96,
            freq="30min",
            tz=train_ml_models.LOCAL_TIMEZONE,
        ))
        frame = pd.DataFrame({"DATETIME": timestamps})

        result = (
            distribution_drift_analysis.restrict_to_common_calendar_coverage(
                frame
            )
        )

        self.assertEqual(set(result["DATETIME"].dt.strftime("%m-%d")), {"01-02"})
        self.assertEqual(len(result), 96)


class FixedOriginTests(unittest.TestCase):
    def test_fixed_origin_split_never_trains_on_test_years(self):
        timestamps = pd.to_datetime(
            [f"{year}-06-01 12:00:00" for year in range(2021, 2026)],
            utc=True,
        ).tz_convert(train_ml_models.LOCAL_TIMEZONE)
        frame = pd.DataFrame({"DATETIME": timestamps})

        training, test_frames = fixed_origin_degradation.split_fixed_origin_data(
            frame
        )

        self.assertEqual(set(training["DATETIME"].dt.year), {2021, 2022, 2023})
        self.assertEqual(set(test_frames), {2024, 2025})
        self.assertLess(
            training["DATETIME"].max(),
            min(test_frame["DATETIME"].min() for test_frame in test_frames.values()),
        )

    def test_fixed_origin_model_is_fitted_once(self):
        class CountingRegressor:
            def __init__(self):
                self.fit_calls = 0
                self.prediction = 0.0

            def fit(self, X, y):
                self.fit_calls += 1
                self.prediction = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.prediction)

        columns = train_ml_models.FEATURE_COLUMNS
        training = pd.DataFrame({column: [1.0, 2.0] for column in columns})
        training["ND"] = [100.0, 110.0]
        training["DATETIME"] = pd.to_datetime(
            ["2022-01-01", "2023-01-01"],
            utc=True,
        ).tz_convert(train_ml_models.LOCAL_TIMEZONE)

        test_frames = {}
        for year in [2024, 2025]:
            frame = pd.DataFrame({column: [1.0] for column in columns})
            frame["lag_48"] = [100.0]
            frame["lag_336"] = [100.0]
            frame["ND"] = [100.0]
            frame["DATETIME"] = pd.to_datetime(
                [f"{year}-01-01"],
                utc=True,
            ).tz_convert(train_ml_models.LOCAL_TIMEZONE)
            test_frames[year] = frame

        model = CountingRegressor()
        results = fixed_origin_degradation.evaluate_fixed_origin(
            training,
            test_frames,
            models={"Counting model": model},
            verbose=False,
        )

        self.assertEqual(model.fit_calls, 1)
        self.assertEqual(
            len(results[results["Model"] == "Counting model"]),
            2,
        )

        summary = fixed_origin_degradation.create_degradation_summary(results)
        self.assertNotIn("MAPE_Slope_Per_Year", summary.columns)
        self.assertTrue(
            summary["Evidence_Scope"].str.contains("not a long-term trend").all()
        )


class RobustnessAnalysisTests(unittest.TestCase):
    def test_rolling_origin_labels_fixed_configurations_explicitly(self):
        self.assertEqual(
            robustness_analysis.ROLLING_ARIMA_MODEL_NAME,
            "ARIMA(2,1,0) fixed",
        )
        self.assertEqual(
            set(robustness_analysis.ROLLING_ML_CONFIGURATIONS),
            set(train_ml_models.get_models()),
        )

    def test_rolling_origin_does_not_silently_skip_missing_year(self):
        frame = pd.DataFrame({
            column: [1.0]
            for column in train_ml_models.FEATURE_COLUMNS
        })
        frame["ND"] = [100.0]
        frame["DATETIME"] = pd.to_datetime(
            ["2021-06-01"],
            utc=True,
        ).tz_convert(train_ml_models.LOCAL_TIMEZONE)

        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            robustness_analysis.evaluate_rolling_origin(frame)


class ReproducibilityTests(unittest.TestCase):
    def test_pipeline_scripts_exist_in_dependency_order(self):
        script_names = [path.name for _, path in run_pipeline.PIPELINE_STEPS]

        self.assertEqual(
            script_names,
            [
                "load_data.py",
                "create_features.py",
                "descriptive_statistics.py",
                "baseline_models.py",
                "feature_ablation.py",
                "distribution_drift_analysis.py",
                "train_ml_models.py",
                "training_period_sensitivity.py",
                "feature_importance.py",
                "visualise_results.py",
                "fixed_origin_degradation.py",
                "robustness_analysis.py",
            ],
        )
        self.assertTrue(all(path.exists() for _, path in run_pipeline.PIPELINE_STEPS))

    def test_training_period_sensitivity_keeps_validation_and_test_years_fixed(self):
        keep = training_period_sensitivity.SCENARIOS["Keep 2021"]
        removed = training_period_sensitivity.SCENARIOS["Remove 2021"]

        self.assertEqual(keep["parameter_training_years"], (2021, 2022, 2023))
        self.assertEqual(removed["parameter_training_years"], (2022, 2023))
        self.assertEqual(keep["final_training_years"], (2021, 2022, 2023, 2024))
        self.assertEqual(removed["final_training_years"], (2022, 2023, 2024))
        self.assertEqual(training_period_sensitivity.VALIDATION_YEAR, 2024)
        self.assertEqual(training_period_sensitivity.TEST_YEAR, 2025)

    def test_final_analysis_notebook_is_valid_and_complete(self):
        notebook_path = PROJECT_ROOT / "notebooks" / "final_analysis_report.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 20)

        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        for required_section in [
            "Persistence and ARIMA baselines",
            "Feature-ablation experiment",
            "Distribution-drift analysis",
            "Fixed-origin degradation",
            "Sensitivity to excluding 2021",
            "Limitations and future work",
        ]:
            self.assertIn(required_section, markdown)

        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertTrue(code_cells)
        for index, source in enumerate(code_cells, start=1):
            compile(source, f"final_analysis_report.ipynb:code_cell_{index}", "exec")

        combined_code = "\n".join(code_cells)
        for forbidden_operation in [".to_csv(", ".to_pickle(", "joblib.dump(", ".fit("]:
            self.assertNotIn(forbidden_operation, combined_code)


if __name__ == "__main__":
    unittest.main()
