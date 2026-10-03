"""Shared feature-group definitions for selection and final modelling.

The final feature configuration was selected using 2021-2022 for fitting and
2023 for feature selection. It is deliberately frozen before the separate 2024
hyperparameter-validation stage and the untouched 2025 final test.
"""

from __future__ import annotations


# Candidate inputs are grouped by their forecasting role so the ablation study
# can remove meaningful groups rather than arbitrary individual columns.
LAG_FEATURES = (
    "lag_48",
    "lag_336",
)
ROLLING_FEATURES = (
    "rolling_mean_48",
    "rolling_mean_336",
)
CALENDAR_FEATURES = (
    "settlement_period",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
)
HOLIDAY_FEATURES = (
    "is_bank_holiday_england_wales",
    "is_bank_holiday_scotland",
)

ALL_CANDIDATE_FEATURES = (
    LAG_FEATURES
    + ROLLING_FEATURES
    + CALENDAR_FEATURES
    + HOLIDAY_FEATURES
)

# Freeze the statistically selected 2023 configuration before 2024 tuning and
# the untouched 2025 final evaluation.
FINAL_FEATURE_SET_NAME = "Selected + England/Wales holiday"
FINAL_FEATURE_COLUMNS = (
    LAG_FEATURES
    + CALENDAR_FEATURES
    + (HOLIDAY_FEATURES[0],)
)
