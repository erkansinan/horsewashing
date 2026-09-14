from __future__ import annotations

# Calibration layer using Platt scaling (Platt, 1999) or isotonic regression,
# as commonly recommended in horse-racing probability calibration workflows.

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


ISOTONIC_MIN_SAMPLES = 1000


@dataclass
class Calibrator:
    method: str
    model: object | None


def fit_calibrator(y_true: np.ndarray, raw_prob: np.ndarray, method: str = "isotonic") -> Calibrator:
    x = np.clip(raw_prob, 1e-8, 1.0 - 1e-8).reshape(-1, 1)
    y = y_true.astype(int)

    if method == "none":
        return Calibrator(method="none", model=None)

    if np.unique(y).size < 2:
        return Calibrator(method="none", model=None)

    # Isotonic is highly flexible and creates flat steps on small holdouts.
    # Use the lower-variance sigmoid/Platt mapping until the calibration set
    # is large enough to support a non-parametric fit.
    effective_method = method
    if method == "isotonic" and len(y) < ISOTONIC_MIN_SAMPLES:
        effective_method = "platt"

    if effective_method == "platt":
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


def apply_probability_floor(probabilities: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Keep calibrated probabilities positive before race normalization."""
    return np.maximum(np.asarray(probabilities, dtype=float), floor)


def smooth_race_probabilities(
    probabilities: np.ndarray,
    race_sizes: np.ndarray,
    uniform_weight: float = 0.02,
) -> np.ndarray:
    """Add a small race-size-aware prior after pointwise calibration."""
    p = apply_probability_floor(probabilities)
    sizes = np.maximum(np.asarray(race_sizes, dtype=float), 1.0)
    weight = float(np.clip(uniform_weight, 0.0, 1.0))
    return (1.0 - weight) * p + weight / sizes


def recover_collapsed_calibration(
    calibrated: np.ndarray,
    raw_probability: np.ndarray,
    collapse_tolerance: float = 1e-12,
) -> np.ndarray:
    """Preserve raw ranking when a calibrator maps every row to one value."""
    calibrated_array = np.asarray(calibrated, dtype=float)
    raw_array = np.asarray(raw_probability, dtype=float)
    if calibrated_array.size > 1 and np.ptp(calibrated_array) <= collapse_tolerance:
        return apply_probability_floor(raw_array)
    recovered = calibrated_array.copy()
    collapsed_rows = recovered <= collapse_tolerance
    recovered[collapsed_rows] = raw_array[collapsed_rows]
    return apply_probability_floor(recovered)


def to_payload(calibrator: Calibrator) -> dict[str, object]:
    return {"method": calibrator.method, "model": calibrator.model}


def from_payload(payload: dict[str, object]) -> Calibrator:
    return Calibrator(method=str(payload.get("method", "none")), model=payload.get("model"))
