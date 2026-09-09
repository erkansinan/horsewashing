from __future__ import annotations

# Stage-2 market blend inspired by Benter (1994): combine model signal + market signal
# with race-conditional logit so probabilities remain relative within each race.
from dataclasses import dataclass

import numpy as np
import pandas as pd

from atyaris.ml.fundamental_model import (
    ConditionalLogitModel,
    fit_conditional_logit,
    predict_conditional_logit_probability,
)


@dataclass
class BenterTwoStageArtifact:
    stage1_model: ConditionalLogitModel
    stage2_model: ConditionalLogitModel
    feature_columns: list[str]
    logistic_weight_hint: float = 0.0
    form_weight: float = 0.75
    market_weight: float = 0.25

    @property
    def logistic_weight(self) -> float:
        coef = getattr(self.stage2_model, "coef_", None)
        if coef is None:
            return float(self.logistic_weight_hint)
        arr = np.asarray(coef, dtype=float).ravel()
        if arr.size == 0:
            return float(self.logistic_weight_hint)
        return float(arr[0])


def _safe_logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(q / (1.0 - q))


def _normalized_market_probability(frame: pd.DataFrame) -> np.ndarray:
    if "odds" in frame.columns:
        odds = pd.to_numeric(frame["odds"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        implied = np.where(odds > 1.0, 1.0 / odds, 0.0)
    elif "market_probability" in frame.columns:
        implied = pd.to_numeric(frame["market_probability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        implied = np.full(len(frame), 0.1, dtype=float)

    temp = frame[["race_id"]].copy()
    temp["implied"] = implied
    denom = temp.groupby("race_id")["implied"].transform("sum").replace(0.0, 1.0)
    return (temp["implied"] / denom).to_numpy(dtype=float)


def fit_two_stage_benter(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    stage1_penalty: str = "l2",
    stage1_regularization: float = 0.05,
    stage2_penalty: str = "l2",
    stage2_regularization: float = 0.02,
) -> BenterTwoStageArtifact:
    stage1 = fit_conditional_logit(
        train_df,
        feature_columns,
        penalty=stage1_penalty,
        regularization_strength=stage1_regularization,
    )

    blend_df = calibration_df.copy()
    stage1_prob = predict_conditional_logit_probability(stage1, blend_df)
    market_prob = _normalized_market_probability(blend_df)

    blend_df["stage1_logit"] = _safe_logit(stage1_prob)
    blend_df["market_logit"] = _safe_logit(market_prob)
    blend_df["interaction"] = blend_df["stage1_logit"] * blend_df["market_logit"]

    stage2_features = ["stage1_logit", "market_logit", "interaction"]
    stage2 = fit_conditional_logit(
        blend_df,
        stage2_features,
        penalty=stage2_penalty,
        regularization_strength=stage2_regularization,
    )

    return BenterTwoStageArtifact(stage1_model=stage1, stage2_model=stage2, feature_columns=feature_columns)


def predict_two_stage_probability(artifact: BenterTwoStageArtifact, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)

    return blend_form_market_probability(
        artifact,
        frame,
        predict_form_probability(artifact, frame),
    )


def blend_form_market_probability(
    artifact: BenterTwoStageArtifact,
    frame: pd.DataFrame,
    form_probability: np.ndarray,
) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)

    market_prob = _normalized_market_probability(frame)

    form_weight = float(getattr(artifact, "form_weight", 0.75))
    market_weight = float(getattr(artifact, "market_weight", 0.25))
    total_weight = form_weight + market_weight
    if total_weight <= 0.0:
        form_weight, market_weight, total_weight = 0.75, 0.25, 1.0

    return (form_weight * form_probability + market_weight * market_prob) / total_weight


def predict_form_probability(artifact: BenterTwoStageArtifact, frame: pd.DataFrame) -> np.ndarray:
    """Return the form-only conditional probability before market blending."""
    if frame.empty:
        return np.array([], dtype=float)
    return predict_conditional_logit_probability(artifact.stage1_model, frame)


def extract_market_reference_probability(frame: pd.DataFrame) -> np.ndarray:
    return _normalized_market_probability(frame)
