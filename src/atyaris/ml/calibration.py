from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    method: str
    model: object | None


def fit_calibrator(y_true: np.ndarray, raw_prob: np.ndarray, method: str = "isotonic") -> Calibrator:
    x = np.clip(raw_prob, 1e-8, 1.0 - 1e-8).reshape(-1, 1)
    y = y_true.astype(int)

    if method == "none":
        return Calibrator(method="none", model=None)

    if method == "platt":
        model = LogisticRegression(max_iter=400, solver="lbfgs")
        model.fit(x, y)
        return Calibrator(method="platt", model=model)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x.ravel(), y)
    return Calibrator(method="isotonic", model=iso)


def apply_calibrator(calibrator: Calibrator, raw_prob: np.ndarray) -> np.ndarray:
    p = np.clip(raw_prob, 1e-8, 1.0 - 1e-8)
    if calibrator.method == "none" or calibrator.model is None:
        return p
    if calibrator.method == "platt":
        return calibrator.model.predict_proba(p.reshape(-1, 1))[:, 1]
    return calibrator.model.predict(p)


def to_payload(calibrator: Calibrator) -> dict[str, object]:
    return {"method": calibrator.method, "model": calibrator.model}


def from_payload(payload: dict[str, object]) -> Calibrator:
    return Calibrator(method=str(payload.get("method", "none")), model=payload.get("model"))
