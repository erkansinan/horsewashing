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
