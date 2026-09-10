from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

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


def test_reproducibility() -> None:
    train, _, test, features = _splits()
    result = health.test_reproducibility(train, test, features)
    assert result.passed
