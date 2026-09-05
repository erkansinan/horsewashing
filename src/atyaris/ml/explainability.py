from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atyaris.ml.market_blend import BenterTwoStageArtifact
from atyaris.ml.market_blend import predict_two_stage_probability


def compute_permutation_importance(
    artifact: BenterTwoStageArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
    n_repeats: int = 3,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    if frame.empty:
        return []

    rng = np.random.default_rng(random_state)
    base_prob = np.clip(predict_two_stage_probability(artifact, frame), 1e-9, 1.0)
    y = frame["is_winner"].to_numpy(dtype=float)
    base_loss = float(-np.mean(y * np.log(base_prob)))

    rows = []
    for col in feature_columns:
        losses = []
        for _ in range(n_repeats):
            perm = frame.copy()
            perm[col] = rng.permutation(perm[col].to_numpy())
            p = np.clip(predict_two_stage_probability(artifact, perm), 1e-9, 1.0)
            losses.append(float(-np.mean(y * np.log(p))))
        rows.append({"feature": col, "importance_mean": float(np.mean(losses) - base_loss)})
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return rows


def compute_optional_shap_summary(
    artifact: BenterTwoStageArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
    sample_size: int = 300,
) -> dict[str, Any]:
    if frame.empty:
        return {"available": False, "reason": "bos veri"}

    coefs = artifact.stage1_model.coef_
    rows = [
        {"feature": feature_columns[i], "abs_stage1_coef": float(abs(coefs[i]))}
        for i in range(min(len(feature_columns), len(coefs)))
    ]
    rows.sort(key=lambda r: r["abs_stage1_coef"], reverse=True)
    return {"available": True, "method": "abs_stage1_coefficient", "top_features": rows[:15]}
