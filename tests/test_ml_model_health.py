from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atyaris.ml.fundamental_model import fit_conditional_logit
from atyaris.ml.market_blend import BenterTwoStageArtifact
import atyaris.ml.model_health as health


def _splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rows = []
    for race_no in range(8):
        race_date = date(2025, 1, 1) + timedelta(days=race_no)
        for draw in range(3):
            rows.append(
                {
                    "race_id": f"R{race_no}",
                    "date": race_date,
                    "horse_id": f"H{race_no}_{draw}",
                    "signal": float(3 - draw) + race_no * 0.01,
                    "market_probability_norm": [0.6, 0.3, 0.1][draw],
                    "is_winner": int(draw == race_no % 3),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.iloc[:12].copy(), frame.iloc[12:18].copy(), frame.iloc[18:].copy(), ["signal", "market_probability_norm"]


def _artifact(train: pd.DataFrame, features: list[str]) -> BenterTwoStageArtifact:
    model = fit_conditional_logit(train, features, max_iter=40)
    return BenterTwoStageArtifact(stage1_model=model, stage2_model=model, feature_columns=features)


def test_log_loss_uses_race_winner_probability_mass() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["single", "single", "single", "dead_heat", "dead_heat", "dead_heat"],
            "is_winner": [1, 0, 0, 1, 1, 0],
        }
    )

    result = health._log_loss(frame, [0.8, 0.1, 0.1, 0.2, 0.3, 0.5])

    assert result == pytest.approx((-np.log(0.8) - np.log(0.5)) / 2)


def test_health_metrics_name_race_loss_and_row_bce_separately() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "is_winner": [1, 0, 0, 1],
        }
    )
    probabilities = np.array([0.8, 0.2, 0.3, 0.7])

    metrics = health._metrics(frame, probabilities)

    assert metrics["race_based_log_loss"] == pytest.approx((-np.log(0.8) - np.log(0.7)) / 2)
    assert metrics["row_based_bce"] == pytest.approx((-2 * np.log(0.8) - 2 * np.log(0.7)) / 4)


def test_market_health_comparison_reports_same_units_and_normalized_probabilities() -> None:
    train, _, test, features = _splits()
    artifact = _artifact(train, features)

    result = health.test_beats_market_baseline(artifact, train, test, features)

    assert "full_race_based_log_loss" in result.details
    assert "market_only_race_based_log_loss" in result.details
    assert "full_row_based_bce" in result.details
    assert "market_only_row_based_bce" in result.details
    assert result.details["full_max_race_probability_sum_error"] < 1e-12
    assert result.details["market_max_race_probability_sum_error"] < 1e-12


def test_no_leakage_detects_temporal_and_group_contract() -> None:
    train, validation, test, features = _splits()
    result = health.test_no_leakage(train, validation, test, features)
    assert result.name == "test_no_leakage"
    assert result.passed is True


def test_label_shuffle_fails() -> None:
    train, _, test, features = _splits()
    result = health.test_label_shuffle_fails(train, test, features)
    assert result.name == "test_label_shuffle_fails"
    assert "chance_level" in result.details


def test_noise_feature_is_ignored() -> None:
    train, _, test, features = _splits()
    result = health.test_noise_feature_is_ignored(train, test, features)
    assert result.name == "test_noise_feature_is_ignored"
    assert "noise_importance" in result.details


def test_parameter_update_and_prediction_diversity() -> None:
    train, _, test, features = _splits()
    artifact = _artifact(train, features)
    assert health.test_parameter_update(artifact).passed
    assert "probability_variance" in health.test_prediction_diversity(artifact, test).details


def test_absolute_weight_sign_is_not_a_fixed_health_gate() -> None:
    artifact = BenterTwoStageArtifact(
        stage1_model=SimpleNamespace(coef_=np.array([0.1])),
        stage2_model=SimpleNamespace(coef_=np.array([0.1])),
        feature_columns=["weight"],
    )

    result = health.test_feature_signs(artifact)

    assert result.passed


def test_reproducibility() -> None:
    train, _, test, features = _splits()
    result = health.test_reproducibility(train, test, features)
    assert result.passed


def test_conditional_logit_enforces_requested_coefficient_sign() -> None:
    rows = []
    for race_no in range(8):
        for weight in (50.0, 55.0, 60.0):
            rows.append(
                {
                    "race_id": f"R{race_no}",
                    "weight": weight,
                    "is_winner": int(weight == 60.0),
                }
            )

    model = fit_conditional_logit(
        pd.DataFrame(rows),
        ["weight"],
        max_iter=40,
        coefficient_sign_constraints={"weight": -1},
    )

    assert model.coef_[0] < 0.0


def test_dataset_quality_report_deduplicates_fallback_splits() -> None:
    train, validation, test, _ = _splits()
    report = health.dataset_quality_report(train, validation, test)

    assert report["rows"] == 24
    assert report["unique_dates"] == 8
    assert report["unique_races"] == 8
    assert report["production_ready"] is False
