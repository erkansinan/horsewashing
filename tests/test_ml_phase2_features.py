from __future__ import annotations

from datetime import date

from atyaris.ml.features import TJK_FEATURE_COLUMNS, build_leakage_safe_features
from atyaris.ml.provider import SyntheticRacingDataProvider as _FixtureRacingDataProvider


def test_phase2_feature_columns_exist() -> None:
    provider = _FixtureRacingDataProvider()
    data = provider.get_dataset(date(2025, 1, 1), date(2025, 2, 28))
    built = build_leakage_safe_features(data)

    required = {
        "age",
        "handicap_points",
        "career_starts",
        "career_wins",
        "career_places",
        "last_year_starts",
        "last_year_wins",
        "last_year_places",
        "jockey_horse_combo_starts",
        "jockey_horse_combo_wins",
        "history_avg_finish_position",
        "history_avg_field_size",
        "history_avg_weight",
        "history_avg_odds",
        "history_avg_handicap_points",
        "history_avg_race_time_seconds",
        "history_avg_prize",
        "history_avg_s20",
        "workout_count",
        "workout_avg_time_seconds",
        "workout_best_time_seconds",
        "workout_avg_distance",
        "days_since_last_workout",
        "recovery_score",
        "short_rest_flag",
        "long_layoff_flag",
        "race_frequency_3",
        "race_frequency_5",
        "seasonal_race_load",
        "distance_fit",
        "surface_fit",
        "track_fit",
    }

    assert required.issubset(set(built.feature_columns))
    assert built.feature_columns == TJK_FEATURE_COLUMNS


def test_tjk_numeric_features_have_no_missing_values() -> None:
    provider = _FixtureRacingDataProvider()
    data = provider.get_dataset(date(2025, 3, 1), date(2025, 4, 15))
    built = build_leakage_safe_features(data)
    frame = built.frame

    assert frame["recovery_score"].between(0.0, 1.0).all()
    assert not frame[TJK_FEATURE_COLUMNS].isna().any().any()
