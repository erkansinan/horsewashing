from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from atyaris.ml.modeling import EnsembleArtifact


def compute_permutation_importance(
    artifact: EnsembleArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    if frame.empty:
        return []

    x = frame[feature_columns].to_numpy()
    y = frame["is_winner"].to_numpy()

    result = permutation_importance(
        artifact.logistic,
        x,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_log_loss",
    )

    rows = []
    for i, col in enumerate(feature_columns):
        rows.append({"feature": col, "importance_mean": float(result.importances_mean[i])})
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return rows


def compute_optional_shap_summary(
    artifact: EnsembleArtifact,
    frame: pd.DataFrame,
    feature_columns: list[str],
    sample_size: int = 300,
) -> dict[str, Any]:
    try:
        import shap  # type: ignore
    except Exception:
        return {"available": False, "reason": "shap kutuphanesi kurulu degil"}

    if frame.empty:
        return {"available": False, "reason": "bos veri"}

    sample = frame.sample(min(sample_size, len(frame)), random_state=42)
    x = sample[feature_columns]
    explainer = shap.TreeExplainer(artifact.random_forest)
    shap_values = explainer.shap_values(x)

    if isinstance(shap_values, list):
        vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        vals = shap_values

    abs_mean = np.abs(vals).mean(axis=0)
    rows = [{"feature": feature_columns[i], "mean_abs_shap": float(abs_mean[i])} for i in range(len(feature_columns))]
    rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    return {"available": True, "top_features": rows[:15]}
