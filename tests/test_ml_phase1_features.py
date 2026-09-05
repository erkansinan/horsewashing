from __future__ import annotations

from datetime import date

import pandas as pd

from atyaris.ml.features import build_leakage_safe_features, preprocess_dataset
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


def test_features_do_not_include_known_leakage_columns() -> None:
    built = build_leakage_safe_features(_dataset())
    forbidden = {"finish_position", "is_winner", "latent_true_win_probability"}
    assert forbidden.intersection(set(built.feature_columns)) == set()
