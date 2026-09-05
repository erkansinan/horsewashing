from __future__ import annotations

# Core race-level relative probability model inspired by Benter (1994):
# utility -> softmax within each race (conditional logit / multinomial logit by race set).
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConditionalLogitModel:
    feature_columns: list[str]
    coef_: np.ndarray
    intercept_: float
    mean_: np.ndarray
    scale_: np.ndarray
    penalty: str
    regularization_strength: float


def _race_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for _, idx in frame.groupby("race_id", sort=False).indices.items():
        groups.append(np.asarray(idx, dtype=int))
    return groups


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_ = np.mean(x, axis=0)
    scale_ = np.std(x, axis=0)
    scale_ = np.where(scale_ < 1e-8, 1.0, scale_)
    z = (x - mean_) / scale_
    return z, mean_, scale_


def _standardize_apply(x: np.ndarray, mean_: np.ndarray, scale_: np.ndarray) -> np.ndarray:
    return (x - mean_) / np.where(scale_ < 1e-8, 1.0, scale_)


def _softmax_by_group(utility: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    probs = np.zeros_like(utility, dtype=float)
    for g in groups:
        u = utility[g]
        shift = np.max(u)
        e = np.exp(u - shift)
        denom = np.sum(e)
        if denom <= 0.0:
            probs[g] = 1.0 / max(len(g), 1)
        else:
            probs[g] = e / denom
    return probs


def fit_conditional_logit(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    learning_rate: float = 0.05,
    max_iter: int = 600,
    penalty: str = "l2",
    regularization_strength: float = 0.05,
    random_state: int = 42,
) -> ConditionalLogitModel:
    if frame.empty:
        raise ValueError("fit_conditional_logit icin bos veri verildi")

    x_raw = frame[feature_columns].to_numpy(dtype=float)
    y = frame["is_winner"].to_numpy(dtype=float)
    groups = _race_groups(frame)

    x, mean_, scale_ = _standardize_fit(x_raw)
    rng = np.random.default_rng(random_state)
    coef = rng.normal(loc=0.0, scale=0.03, size=x.shape[1])
    intercept = 0.0

    reg = max(regularization_strength, 1e-8)
    penalty_norm = penalty.lower().strip()
    if penalty_norm not in {"l1", "l2", "none"}:
        penalty_norm = "l2"

    for _ in range(max_iter):
        utility = x @ coef + intercept
        p = _softmax_by_group(utility, groups)
        diff = p - y

        grad_coef = x.T @ diff
        grad_intercept = float(np.sum(diff))

        if penalty_norm == "l2":
            grad_coef = grad_coef + reg * coef
        elif penalty_norm == "l1":
            grad_coef = grad_coef + reg * np.sign(coef)

        coef = coef - learning_rate * grad_coef / max(len(frame), 1)
        intercept = intercept - learning_rate * grad_intercept / max(len(frame), 1)

    return ConditionalLogitModel(
        feature_columns=feature_columns,
        coef_=coef,
        intercept_=float(intercept),
        mean_=mean_,
        scale_=scale_,
        penalty=penalty_norm,
        regularization_strength=reg,
    )


def predict_conditional_logit_probability(model: ConditionalLogitModel, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    x_raw = frame[model.feature_columns].to_numpy(dtype=float)
    x = _standardize_apply(x_raw, model.mean_, model.scale_)
    utility = x @ model.coef_ + model.intercept_
    groups = _race_groups(frame)
    return _softmax_by_group(utility, groups)
