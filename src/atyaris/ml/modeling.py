from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from atyaris.ml.market_blend import BenterTwoStageArtifact


def _safe_joblib_load(path: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise
        import numpy.core as numpy_core  # type: ignore[attr-defined]

        sys.modules.setdefault("numpy._core", numpy_core)
        return joblib.load(path)


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
    return float(log_loss(y_true, p))


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["is_winner"].to_numpy()
    p = np.clip(frame["calibrated_probability"].to_numpy(), 1e-8, 1.0 - 1e-8)
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def save_phase3_artifact(artifact: BenterTwoStageArtifact, calibrator_payload: dict[str, object], path: str) -> None:
    joblib.dump(
        {
            "stage1_model": artifact.stage1_model,
            "stage2_model": artifact.stage2_model,
            "feature_columns": artifact.feature_columns,
            "logistic_weight": artifact.logistic_weight,
            "form_weight": artifact.form_weight,
            "market_weight": artifact.market_weight,
            "calibrator": calibrator_payload,
        },
        path,
    )


def load_phase3_artifact(path: str) -> tuple[BenterTwoStageArtifact, dict[str, object]]:
    payload = _safe_joblib_load(path)
    # New format
    if "stage1_model" in payload and "stage2_model" in payload:
        artifact = BenterTwoStageArtifact(
            stage1_model=payload["stage1_model"],
            stage2_model=payload["stage2_model"],
            feature_columns=list(payload["feature_columns"]),
            logistic_weight_hint=float(payload.get("logistic_weight", 0.0) or 0.0),
            form_weight=float(payload.get("form_weight", 0.75) or 0.75),
            market_weight=float(payload.get("market_weight", 0.25) or 0.25),
        )
    else:
        # Keep a clear failure mode for obsolete artifacts from the removed ensemble stack.
        raise ValueError("Eski ensemble artefakti desteklenmiyor; modeli tekrar egitin.")

    calibrator_payload = dict(payload.get("calibrator") or {})
    return artifact, calibrator_payload
