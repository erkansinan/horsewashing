from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from atyaris.ml.features import (
    FIELD_SIZE_DEPENDENT_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS,
    TJK_SELECTED_STAGE1_FEATURE_COLUMNS,
    build_leakage_safe_features,
    preprocess_dataset,
)
from atyaris.ml.market_blend import (
    _normalized_market_probability,
    extract_odds_implied_probability,
)
from atyaris.ml.provider import SyntheticRacingDataProvider as _FixtureRacingDataProvider
from atyaris.ml.pipeline import Phase1Paths, build_features, prepare_prediction_features


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


def test_prediction_feature_preparation_uses_html_program(monkeypatch, tmp_path) -> None:
    target_date = date(2026, 9, 29)
    paths = SimpleNamespace(
        clean_csv=tmp_path / "clean.csv",
        prediction_features_csv=tmp_path / "prediction.csv",
    )
    raw = pd.DataFrame(
        {
            "date": [target_date.isoformat()],
            "race_id": ["Adana-2026-09-29-1"],
            "track": ["ADANA"],
            "horse_id": ["horse-1"],
            "career_starts": [0],
            "history_missing": [1],
        }
    )
    captured = {}
    monkeypatch.setattr(
        "atyaris.ml.pipeline.ingest_real_tjk_data",
        lambda *_args, **kwargs: captured.update(kwargs) or raw,
    )
    monkeypatch.setattr(
        "atyaris.ml.pipeline.build_leakage_safe_features",
        lambda frame, **_kwargs: SimpleNamespace(frame=frame),
    )
    monkeypatch.setattr("atyaris.ml.pipeline._write_csv_atomically", lambda *_args: None)

    prediction = prepare_prediction_features(
        target_date,
        paths,
        hippodrome="Adana",
        race_no=1,
    )

    assert len(prediction) == 1
    assert captured["require_results"] is False
    assert captured["program_from_html"] is True
    assert "program_from_csv" not in captured
    assert captured["hippodrome"] == "Adana"
    assert captured["race_no"] == 1


def test_market_probability_ignores_zero_and_missing_odds_without_warning() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "odds": [0.0, None, 2.0],
        }
    )

    probabilities = _normalized_market_probability(frame)

    assert probabilities.tolist() == [0.0, 0.0, 1.0]


def test_market_probability_prefers_normalized_values_when_odds_are_missing() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "odds": [0.0, None, 2.0],
            "market_probability_norm": [0.5, 0.3, 0.2],
        }
    )

    probabilities = _normalized_market_probability(frame)

    assert probabilities.tolist() == pytest.approx([0.5, 0.3, 0.2])


def test_odds_implied_probability_uses_odds_not_cached_market_values() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "odds": [2.0, 4.0, 4.0],
            "market_probability_norm": [0.2, 0.3, 0.5],
        }
    )

    probabilities = extract_odds_implied_probability(frame)

    assert probabilities.tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_market_probability_uses_fallback_when_all_odds_are_missing() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R1"],
            "odds": [0.0, None, 0.0],
            "market_probability": [1 / 3, 1 / 3, 1 / 3],
        }
    )

    probabilities = _normalized_market_probability(frame)

    assert probabilities.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_features_do_not_include_known_leakage_columns() -> None:
    built = build_leakage_safe_features(_dataset())
    forbidden = {"finish_position", "is_winner", "latent_true_win_probability"}
    assert forbidden.intersection(set(built.feature_columns)) == set()


def test_selected_stage1_features_exclude_field_size_dependent_columns() -> None:
    assert not FIELD_SIZE_DEPENDENT_FEATURE_COLUMNS.intersection(
        TJK_SELECTED_STAGE1_FEATURE_COLUMNS
    )


def test_placer_labels_are_available_but_never_model_features() -> None:
    source = _dataset()
    built = build_leakage_safe_features(source)
    assert {"is_placed", "target_highest_odds_placer"}.issubset(LEAKAGE_COLUMNS)
    assert {"is_placed", "target_highest_odds_placer"}.issubset(built.frame.columns)
    assert not {"is_placed", "target_highest_odds_placer"}.intersection(built.feature_columns)

    labeled = built.frame[built.frame["target_highest_odds_placer"].notna()]
    for _, race in labeled.groupby("race_id"):
        target = str(race["target_highest_odds_placer"].iloc[0])
        source_race = source[source["race_id"] == race["race_id"].iloc[0]]
        eligible = source_race[pd.to_numeric(source_race["finish_position"], errors="coerce") <= 3]
        eligible = eligible[pd.to_numeric(eligible["odds"], errors="coerce") > 1.0]
        assert target in set(eligible["horse_id"].astype(str))
        target_odds = eligible.loc[eligible["horse_id"].astype(str) == target, "odds"].iloc[0]
        assert target_odds == pd.to_numeric(eligible["odds"], errors="coerce").max()


def test_highest_odds_placer_features_use_pre_race_odds_and_prior_history() -> None:
    frame = pd.DataFrame(
        [
            {
                "race_id": "R1",
                "date": "2025-01-01",
                "race_datetime": "2025-01-01 12:00:00",
                "horse_id": "H1",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 5,
                "odds": 10.0,
                "market_probability": 0.1,
                "finish_position": 2,
                "is_winner": 0,
                "early_pace": 0.2,
                "surface": "KUM",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
            {
                "race_id": "R2",
                "date": "2025-01-10",
                "race_datetime": "2025-01-10 12:00:00",
                "horse_id": "H1",
                "draw": 2,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 5,
                "odds": 8.0,
                "market_probability": 0.1,
                "finish_position": 1,
                "is_winner": 1,
                "early_pace": 0.2,
                "surface": "KUM",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
        ]
    )

    built = build_leakage_safe_features(frame)
    first, second = built.frame.sort_values("race_datetime").itertuples(index=False)

    assert set(TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS).issubset(built.frame.columns)
    assert first.history_longshot_starts == 0.0
    assert second.history_longshot_starts == 1.0
    assert second.history_longshot_place_rate > first.history_longshot_place_rate
    assert second.odds_rank_fraction == 0.0
    assert not set(TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS).intersection(LEAKAGE_COLUMNS)


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
    assert first.history_avg_weight == 0.0
    assert second.history_avg_weight == 56.0


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


def test_precomputed_prediction_history_survives_labeled_history_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "race_id": "R1",
                "date": date(2025, 1, 1),
                "race_datetime": "2025-01-01 12:00:00",
                "horse_id": "TJK-123",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 8,
                "odds": 3.0,
                "market_probability": 1 / 3,
                "finish_position": 3,
                "career_starts": 4,
                "career_wins": 1,
                "career_places": 2,
                "history_avg_finish_position": 3.0,
                "history_missing": 0.0,
                "form_avg_3": 0.68,
                "form_avg_5": 0.61,
                "form_avg_10": 0.57,
                "days_since_last_race": 12.0,
                "track_fit": 0.6,
                "surface_fit": 0.7,
                "distance_fit": 0.8,
                "workout_count": 3.0,
                "workout_missing": 0.0,
                "surface": "Kum",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
            {
                "race_id": "R2",
                "date": date(2025, 1, 2),
                "race_datetime": "2025-01-02 12:00:00",
                "horse_id": "TJK-123",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 8,
                "odds": 3.0,
                "market_probability": 1 / 3,
                "finish_position": None,
                "career_starts": 7,
                "career_wins": 2,
                "career_places": 4,
                "history_avg_finish_position": 3.5,
                "history_missing": 0.0,
                "form_avg_3": 0.82,
                "form_avg_5": 0.76,
                "form_avg_10": 0.71,
                "form_var_5": 0.03,
                "last_run_perf": 0.9,
                "trend_3_10": 0.11,
                "days_since_last_race": 5.0,
                "fatigue_score": 0.1,
                "recovery_score": 0.8,
                "short_rest_flag": 1.0,
                "long_layoff_flag": 0.0,
                "race_frequency_3": 0.6,
                "race_frequency_5": 1.0,
                "seasonal_race_load": 0.1,
                "distance_fit": 0.75,
                "surface_fit": 0.8,
                "track_fit": 0.7,
                "workout_count": 4.0,
                "workout_avg_distance": 800.0,
                "workout_avg_speed_index": 1.4,
                "workout_best_speed_index": 1.7,
                "days_since_last_workout": 2.0,
                "career_summary_missing": 0.0,
                "workout_missing": 0.0,
                "age_missing": 0.0,
                "handicap_missing": 0.0,
                "odds_missing": 0.0,
                "jockey_change_upgrade": 0.0,
                "trainer_change_upgrade": 0.0,
                "class_drop_flag": 0.0,
                "workout_sudden_improvement": 0.2,
                "rest_optimal_fit": 0.6,
                "surface": "Kum",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            },
        ]
    )

    built = build_leakage_safe_features(frame)
    prediction = built.frame.loc[built.frame["race_id"] == "R2"].iloc[0]

    assert prediction["career_starts"] == 7.0
    assert prediction["career_wins"] == 2.0
    assert prediction["history_avg_finish_position"] == 3.5
    assert prediction["workout_count"] == 4.0
    assert prediction["form_avg_5"] == 0.76

    training = built.frame.loc[built.frame["race_id"] == "R1"].iloc[0]
    assert training["career_starts"] == 4.0
    assert training["history_avg_finish_position"] == 3.0
    assert training["workout_count"] == 3.0
    assert training["form_avg_5"] == 0.61


def test_field_size_dependent_precomputed_features_can_be_refreshed_selectively() -> None:
    rows = []
    for race_id, day, finish_position in [("R1", 1, 1), ("R2", 2, 2)]:
        rows.append(
            {
                "race_id": race_id,
                "date": date(2025, 1, day),
                "race_datetime": f"2025-01-0{day} 12:00:00",
                "horse_id": "TJK-123",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 10,
                "odds": 2.0,
                "market_probability": 0.5,
                "finish_position": finish_position,
                "is_winner": int(finish_position == 1),
                "career_starts": 7,
                "history_avg_field_size": 99.0,
                "form_avg_3": 0.01,
                "form_avg_5": 0.01,
                "form_avg_10": 0.01,
                "days_since_last_race": 40.0,
                "track_fit": 0.01,
                "surface_fit": 0.01,
                "distance_fit": 0.01,
                "surface": "Kum",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            }
        )

    built = build_leakage_safe_features(
        pd.DataFrame(rows),
        recompute_precomputed_columns=FIELD_SIZE_DEPENDENT_FEATURE_COLUMNS,
    )
    refreshed = built.frame.loc[built.frame["race_id"] == "R2"].iloc[0]

    assert refreshed["last_run_perf"] == 1.0
    assert refreshed["history_avg_field_size"] == 10.0
    assert refreshed["career_starts"] == 7.0


def test_build_features_refreshes_stale_field_size_dependent_precomputed_columns(tmp_path) -> None:
    rows = []
    for race_id, day, finish_position in [("R1", 1, 1), ("R2", 2, 2)]:
        rows.append(
            {
                "race_id": race_id,
                "date": date(2025, 1, day),
                "race_datetime": f"2025-01-0{day} 12:00:00",
                "horse_id": "TJK-123",
                "draw": 1,
                "weight": 56.0,
                "distance": 1400,
                "field_size": 10,
                "odds": 2.0,
                "market_probability": 0.5,
                "finish_position": finish_position,
                "is_winner": int(finish_position == 1),
                "career_starts": 7,
                "history_avg_field_size": 99.0,
                "form_avg_3": 0.01,
                "form_avg_5": 0.01,
                "form_avg_10": 0.01,
                "days_since_last_race": 40.0,
                "track_fit": 0.01,
                "surface_fit": 0.01,
                "distance_fit": 0.01,
                "surface": "Kum",
                "track": "ANKARA",
                "track_condition": "NORMAL",
            }
        )
    paths = Phase1Paths(
        clean_csv=tmp_path / "clean.csv",
        features_csv=tmp_path / "features.csv",
    )
    pd.DataFrame(rows).to_csv(paths.clean_csv, index=False)

    build_features(paths)
    refreshed = pd.read_csv(paths.features_csv).loc[1]

    assert refreshed["last_run_perf"] == 1.0
    assert refreshed["history_avg_field_size"] == 10.0
    assert refreshed["career_starts"] == 7.0
