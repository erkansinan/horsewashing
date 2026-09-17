from __future__ import annotations

from datetime import date

import pandas as pd

from atyaris.ml.features import build_leakage_safe_features, preprocess_dataset
from atyaris.ml.market_blend import _normalized_market_probability
from atyaris.ml.provider import SyntheticRacingDataProvider as _FixtureRacingDataProvider


def _dataset() -> pd.DataFrame:
    provider = _FixtureRacingDataProvider()
    return provider.get_dataset(date(2025, 1, 1), date(2025, 4, 30))


def test_preprocess_preserves_date_ordering() -> None:
    raw = _dataset().sample(frac=1.0, random_state=1).reset_index(drop=True)
    pre = preprocess_dataset(raw)
    assert pre["race_datetime"].is_monotonic_increasing


def test_feature_build_has_no_missing_values_in_feature_columns() -> None:
    built = build_leakage_safe_features(_dataset())
    assert not built.frame[built.feature_columns].isna().any().any()


def test_probability_sums_to_one_per_race_after_normalization() -> None:
    built = build_leakage_safe_features(_dataset())
    race_probs = built.frame.groupby("race_id")["market_probability_norm"].sum().round(6)
    assert (race_probs == 1.0).all()


def test_market_probability_ignores_zero_and_missing_odds_without_warning() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "odds": [0.0, None, 2.0],
        }
    )

    probabilities = _normalized_market_probability(frame)

    assert probabilities.tolist() == [0.0, 0.0, 1.0]


def test_features_do_not_include_known_leakage_columns() -> None:
    built = build_leakage_safe_features(_dataset())
    forbidden = {"finish_position", "is_winner", "latent_true_win_probability"}
    assert forbidden.intersection(set(built.feature_columns)) == set()


def test_features_use_only_previous_races_for_same_day_target() -> None:
    frame = pd.DataFrame(
        [
            {
                "race_id": "R1",
                "date": date(2025, 1, 1),
                "race_datetime": "2025-01-01 12:00:00",
                "horse_id": "H1",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 2,
                "odds": 2.0,
                "market_probability": 0.5,
                "finish_position": 1,
                "is_winner": 1,
                "early_pace": 0.2,
                "surface": "KUM",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
            {
                "race_id": "R2",
                "date": date(2025, 1, 1),
                "race_datetime": "2025-01-01 13:00:00",
                "horse_id": "H1",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 2,
                "odds": 2.0,
                "market_probability": 0.5,
                "finish_position": 2,
                "is_winner": 0,
                "early_pace": 0.2,
                "surface": "KUM",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
        ]
    )

    built = build_leakage_safe_features(frame)
    first, second = built.frame.sort_values("race_datetime").itertuples(index=False)

    assert first.form_avg_3 == 0.45
    assert second.form_avg_3 > first.form_avg_3


def test_no_future_aggregates() -> None:
    frame = pd.DataFrame(
        [
            {
                "race_id": "R1",
                "date": date(2025, 1, 1),
                "race_datetime": "2025-01-01 12:00:00",
                "horse_id": "H1",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 2,
                "odds": 2.0,
                "market_probability": 0.5,
                "finish_position": 1,
                "is_winner": 1,
                "early_pace": 0.2,
                "surface": "KUM",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
            {
                "race_id": "R2",
                "date": date(2025, 1, 2),
                "race_datetime": "2025-01-02 12:00:00",
                "horse_id": "H1",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 2,
                "odds": 2.0,
                "market_probability": 0.5,
                "finish_position": 2,
                "is_winner": 0,
                "early_pace": 0.2,
                "surface": "KUM",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
        ]
    )

    full = build_leakage_safe_features(frame).frame
    prefix = build_leakage_safe_features(frame.iloc[:1]).frame
    aggregate_columns = [
        "form_avg_3",
        "form_avg_5",
        "form_avg_10",
        "form_var_5",
        "last_run_perf",
        "days_since_last_race",
        "distance_fit",
        "surface_fit",
        "track_fit",
        "history_avg_finish_position",
        "history_avg_field_size",
        "history_avg_weight",
        "history_avg_odds",
        "history_avg_handicap_points",
        "history_avg_race_time_seconds",
        "history_avg_prize",
        "history_avg_s20",
        "jockey_horse_combo_wins",
    ]

    pd.testing.assert_series_equal(
        full.loc[0, aggregate_columns],
        prefix.loc[0, aggregate_columns],
        check_names=False,
    )


def test_precomputed_real_features_are_not_replaced_by_no_history_defaults() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "date": [date(2025, 1, 1)] * 2,
            "race_datetime": ["2025-01-01 12:00:00"] * 2,
            "horse_id": ["H1", "H2"],
            "draw": [1, 2],
            "weight": [56.0, 57.0],
            "distance": [1400, 1400],
            "field_size": [2, 2],
            "market_probability": [0.6, 0.4],
            "implied_probability": [0.6, 0.4],
            "form_avg_3": [0.8, 0.3],
            "form_avg_5": [0.7, 0.2],
            "form_avg_10": [0.6, 0.1],
            "form_var_5": [0.02, 0.04],
            "last_run_perf": [0.9, 0.1],
            "trend_3_10": [0.1, 0.1],
            "days_since_last_race": [8.0, 42.0],
            "fatigue_score": [0.2, 0.8],
            "recovery_score": [0.7, 0.5],
            "short_rest_flag": [0.0, 0.0],
            "long_layoff_flag": [0.0, 0.0],
            "race_frequency_3": [0.2, 0.1],
            "race_frequency_5": [0.3, 0.1],
            "seasonal_race_load": [0.05, 0.02],
            "pace_hint": [0.2, 0.0],
            "style_front_prob": [0.5, 0.25],
            "style_presser_prob": [0.25, 0.25],
            "style_stalker_prob": [0.25, 0.25],
            "style_closer_prob": [0.0, 0.25],
            "distance_fit": [0.7, 0.4],
            "surface_fit": [0.8, 0.45],
            "track_fit": [0.6, 0.45],
            "condition_fit": [0.5, 0.45],
            "odds": [1.7, 2.5],
            "track": ["ANKARA", "ANKARA"],
            "surface": ["Kum", "Kum"],
            "track_condition": ["NORMAL", "NORMAL"],
        }
    )

    built = build_leakage_safe_features(frame)

    assert built.frame.loc[0, "days_since_last_race"] == 8.0
    assert built.frame.loc[0, "form_avg_3"] == 0.8
    assert built.frame.loc[1, "surface_fit"] == 0.45
