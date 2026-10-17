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
        "days_since_last_race",
        "fatigue_score",
        "short_rest_flag",
        "long_layoff_flag",
        "race_frequency_3",
        "race_frequency_5",
        "seasonal_race_load",
        "history_avg_finish_position",
        "history_avg_weight",
        "history_avg_odds",
        "history_avg_handicap_points",
        "history_avg_race_time_seconds",
        "workout_count",
        "workout_avg_distance",
        "workout_avg_speed_index",
        "workout_best_speed_index",
        "days_since_last_workout",
        "class_drop_flag",
        "workout_sudden_improvement",
        "rest_optimal_fit",
    }

    assert required.issubset(set(built.feature_columns))
    assert built.feature_columns == TJK_FEATURE_COLUMNS
    assert "history_avg_s20" not in built.feature_columns
    assert "finish_position" in built.frame.columns
    assert "finish_position" not in built.feature_columns
    assert "workout_avg_time_seconds" not in built.feature_columns
    assert "workout_best_time_seconds" not in built.feature_columns
    assert not {
        "trainer_win_rate",
        "trainer_top3_rate",
        "trainer_top5_rate",
        "trainer_experience_log",
        "trainer_stats_missing",
        "form_avg_5",
        "history_avg_field_size",
        "history_avg_prize",
        "distance_fit",
        "recovery_score",
    }.intersection(built.feature_columns)


def test_tjk_numeric_features_have_no_missing_values() -> None:
    provider = _FixtureRacingDataProvider()
    data = provider.get_dataset(date(2025, 3, 1), date(2025, 4, 15))
    built = build_leakage_safe_features(data)
    frame = built.frame

    assert frame["recovery_score"].between(0.0, 1.0).all()
    assert not frame[TJK_FEATURE_COLUMNS].isna().any().any()
