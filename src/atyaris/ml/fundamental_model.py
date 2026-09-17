from __future__ import annotations

# Core race-level relative probability model inspired by Benter (1994):
# utility -> softmax within each race (conditional logit / multinomial logit by race set).
from dataclasses import dataclass, field
from pathlib import Path

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
    loss_history_: list[float] = field(default_factory=list)
    validation_loss_history_: list[float] = field(default_factory=list)
    iterations_: int = 0


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
    if not groups:
        return np.zeros_like(utility, dtype=float)

    group_index = np.empty(len(utility), dtype=np.intp)
    for group_number, indices in enumerate(groups):
        group_index[indices] = group_number

    group_count = len(groups)
    group_max = np.full(group_count, -np.inf, dtype=float)
    np.maximum.at(group_max, group_index, utility)
    exponentials = np.exp(utility - group_max[group_index])
    group_totals = np.zeros(group_count, dtype=float)
    np.add.at(group_totals, group_index, exponentials)
    probabilities = np.divide(
        exponentials,
        group_totals[group_index],
        out=np.zeros_like(exponentials),
        where=group_totals[group_index] > 0.0,
    )

    for group_number, indices in enumerate(groups):
        if group_totals[group_number] <= 0.0:
            probabilities[indices] = 1.0 / max(len(indices), 1)
    return probabilities


def fit_conditional_logit(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    learning_rate: float = 0.05,
    max_iter: int = 600,
    penalty: str = "l2",
    regularization_strength: float = 0.05,
    random_state: int = 42,
    validation_frame: pd.DataFrame | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_interval: int = 50,
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
    start_iteration = 0
    loss_history: list[float] = []
    validation_loss_history: list[float] = []
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and checkpoint.exists():
        saved = np.load(checkpoint, allow_pickle=False)
        if int(saved["feature_count"]) == x.shape[1]:
            coef = saved["coef"].astype(float)
            intercept = float(saved["intercept"])
            start_iteration = int(saved["iteration"]) + 1
            loss_history = saved["loss_history"].astype(float).tolist()
            validation_loss_history = saved["validation_loss_history"].astype(float).tolist()

    reg = max(regularization_strength, 1e-8)
    penalty_norm = penalty.lower().strip()
    if penalty_norm not in {"l1", "l2", "none"}:
        penalty_norm = "l2"

    def _loss(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
        labels = frame["is_winner"].to_numpy(dtype=float)
        winner_indices = np.flatnonzero(labels > 0.5)
        if winner_indices.size == 0:
            return 0.0
        clipped = np.clip(probabilities[winner_indices], 1e-8, 1.0)
        return float(-np.mean(np.log(clipped)))

    validation_x = None
    validation_y = None
    validation_groups: list[np.ndarray] = []
    if validation_frame is not None and not validation_frame.empty:
        validation_x_raw = validation_frame[feature_columns].to_numpy(dtype=float)
        validation_x = _standardize_apply(validation_x_raw, mean_, scale_)
        validation_y = validation_frame["is_winner"].to_numpy(dtype=float)
        validation_groups = _race_groups(validation_frame)

    for iteration in range(start_iteration, max_iter):
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

        loss_history.append(_loss(frame, p))
        if validation_x is not None and validation_y is not None:
            validation_p = _softmax_by_group(validation_x @ coef + intercept, validation_groups)
            validation_loss_history.append(_loss(validation_frame, validation_p))  # type: ignore[arg-type]
        if checkpoint is not None and checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary_checkpoint = checkpoint.with_suffix(".tmp.npz")
            np.savez(
                temporary_checkpoint,
                feature_count=np.array(x.shape[1]),
                iteration=np.array(iteration),
                coef=coef,
                intercept=np.array(intercept),
                loss_history=np.asarray(loss_history),
                validation_loss_history=np.asarray(validation_loss_history),
            )
            temporary_checkpoint.replace(checkpoint)

    return ConditionalLogitModel(
        feature_columns=feature_columns,
        coef_=coef,
        intercept_=float(intercept),
        mean_=mean_,
        scale_=scale_,
        penalty=penalty_norm,
        regularization_strength=reg,
        loss_history_=loss_history,
        validation_loss_history_=validation_loss_history,
        iterations_=max_iter,
    )


def predict_conditional_logit_probability(model: ConditionalLogitModel, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    x_raw = frame[model.feature_columns].to_numpy(dtype=float)
    x = _standardize_apply(x_raw, model.mean_, model.scale_)
    utility = x @ model.coef_ + model.intercept_
    groups = _race_groups(frame)
    return _softmax_by_group(utility, groups)


def predict_conditional_logit_utility(model: ConditionalLogitModel, frame: pd.DataFrame) -> np.ndarray:
    """Return standardized conditional-logit utility before race softmax."""
    if frame.empty:
        return np.array([], dtype=float)
    x_raw = frame[model.feature_columns].to_numpy(dtype=float)
    x = _standardize_apply(x_raw, model.mean_, model.scale_)
    return x @ model.coef_ + model.intercept_
