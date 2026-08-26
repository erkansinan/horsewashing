from __future__ import annotations

from datetime import date

from atyaris.ml.features import build_leakage_safe_features
from atyaris.ml.provider import SyntheticRacingDataProvider


def test_phase2_feature_columns_exist() -> None:
    provider = SyntheticRacingDataProvider()
    data = provider.get_dataset(date(2025, 1, 1), date(2025, 2, 28))
    built = build_leakage_safe_features(data)

    required = {
        "recovery_score",
        "short_rest_flag",
        "long_layoff_flag",
        "race_frequency_3",
        "race_frequency_5",
        "seasonal_race_load",
        "style_front_prob",
        "style_presser_prob",
        "style_stalker_prob",
        "style_closer_prob",
        "expected_early_pace",
        "pace_pressure",
        "pace_suitability",
        "distance_fit",
        "surface_fit",
        "track_fit",
        "condition_fit",
        "field_strength_index",
        "opponent_strength_mean",
    }

    assert required.issubset(set(built.feature_columns))


def test_phase2_ranges_and_style_simplex() -> None:
    provider = SyntheticRacingDataProvider()
    data = provider.get_dataset(date(2025, 3, 1), date(2025, 4, 15))
    built = build_leakage_safe_features(data)
    frame = built.frame

    assert frame["pace_pressure"].between(0.0, 1.0).all()
    assert frame["recovery_score"].between(0.0, 1.0).all()
    assert frame["pace_suitability"].between(0.0, 1.0).all()

    style_sum = (
        frame["style_front_prob"]
        + frame["style_presser_prob"]
        + frame["style_stalker_prob"]
        + frame["style_closer_prob"]
    )
    assert (style_sum.round(6) == 1.0).all()
