from __future__ import annotations

from dataclasses import dataclass
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_joblib_load(path: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise
        import numpy.core as numpy_core  # type: ignore[attr-defined]

        sys.modules.setdefault("numpy._core", numpy_core)
        return joblib.load(path)


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    feature_columns: list[str]


@dataclass
class EnsembleArtifact:
    logistic: Pipeline
    random_forest: RandomForestClassifier
    feature_columns: list[str]
    logistic_weight: float


def train_logistic_baseline(train_df: pd.DataFrame, feature_columns: list[str]) -> ModelArtifact:
    x = train_df[feature_columns].to_numpy()
    y = train_df["is_winner"].to_numpy()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs")),
        ]
    )
    model.fit(x, y)
    return ModelArtifact(pipeline=model, feature_columns=feature_columns)


def _predict_binary_probability(model, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = frame[feature_columns].to_numpy()
    return model.predict_proba(x)[:, 1]


def _safe_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
    return float(log_loss(y_true, p))


def train_phase3_ensemble(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
) -> EnsembleArtifact:
    x_train = train_df[feature_columns].to_numpy()
    y_train = train_df["is_winner"].to_numpy()

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")),
        ]
    )
    logistic.fit(x_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=6,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    logistic_weight = 0.65
    if not valid_df.empty:
        y_valid = valid_df["is_winner"].to_numpy()
        p_log = _predict_binary_probability(logistic, valid_df, feature_columns)
        p_rf = _predict_binary_probability(rf, valid_df, feature_columns)
        best_loss = float("inf")
        best_w = logistic_weight
        for w in np.linspace(0.1, 0.9, 17):
            blended = w * p_log + (1.0 - w) * p_rf
            loss = _safe_log_loss(y_valid, blended)
            if loss < best_loss:
                best_loss = loss
                best_w = float(w)
        logistic_weight = best_w

    return EnsembleArtifact(
        logistic=logistic,
        random_forest=rf,
        feature_columns=feature_columns,
        logistic_weight=logistic_weight,
    )


def predict_ensemble_raw_probability(artifact: EnsembleArtifact, frame: pd.DataFrame) -> np.ndarray:
    p_log = _predict_binary_probability(artifact.logistic, frame, artifact.feature_columns)
    p_rf = _predict_binary_probability(artifact.random_forest, frame, artifact.feature_columns)
    return artifact.logistic_weight * p_log + (1.0 - artifact.logistic_weight) * p_rf


def predict_probabilities(artifact: ModelArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    probs = artifact.pipeline.predict_proba(out[artifact.feature_columns].to_numpy())[:, 1]
    out["raw_probability"] = probs

    # Race-level normalization so probabilities sum to 1 within each race.
    denom = out.groupby("race_id")["raw_probability"].transform("sum").replace(0.0, 1.0)
    out["calibrated_probability"] = out["raw_probability"] / denom
    out["rank"] = out.groupby("race_id")["calibrated_probability"].rank(ascending=False, method="dense")
    return out


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["is_winner"].to_numpy()
    p = np.clip(frame["calibrated_probability"].to_numpy(), 1e-8, 1.0 - 1e-8)
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def save_artifact(artifact: ModelArtifact, path: str) -> None:
    joblib.dump({"pipeline": artifact.pipeline, "feature_columns": artifact.feature_columns}, path)


def load_artifact(path: str) -> ModelArtifact:
    payload = _safe_joblib_load(path)
    return ModelArtifact(pipeline=payload["pipeline"], feature_columns=list(payload["feature_columns"]))


def save_phase3_artifact(artifact: EnsembleArtifact, calibrator_payload: dict[str, object], path: str) -> None:
    joblib.dump(
        {
            "logistic": artifact.logistic,
            "random_forest": artifact.random_forest,
            "feature_columns": artifact.feature_columns,
            "logistic_weight": artifact.logistic_weight,
            "calibrator": calibrator_payload,
        },
        path,
    )


def load_phase3_artifact(path: str) -> tuple[EnsembleArtifact, dict[str, object]]:
    payload = _safe_joblib_load(path)
    artifact = EnsembleArtifact(
        logistic=payload["logistic"],
        random_forest=payload["random_forest"],
        feature_columns=list(payload["feature_columns"]),
        logistic_weight=float(payload["logistic_weight"]),
    )
    calibrator_payload = dict(payload.get("calibrator") or {})
    return artifact, calibrator_payload
