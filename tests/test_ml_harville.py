from __future__ import annotations

import numpy as np
import pandas as pd

from atyaris.ml.calibration import (
    apply_probability_floor,
    fit_calibrator,
    recover_collapsed_calibration,
    smooth_race_probabilities,
)
from atyaris.ml.harville import add_harville_columns


def test_harville_probabilities_are_valid_for_all_runners() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "calibrated_probability": [0.7, 0.2, 0.09, 0.01],
        }
    )

    result = add_harville_columns(frame)

    assert np.isclose(result["calibrated_probability"].sum(), 1.0)
    assert np.isclose(result["place2_probability"].sum(), 1.0)
    assert np.isclose(result["place3_probability"].sum(), 1.0)
    assert (result["top3_probability"] > 0.0).all()
    assert (result["top3_probability"] <= 1.0).all()


def test_probability_floor_prevents_exact_zero_after_calibration() -> None:
    result = apply_probability_floor(np.array([0.8, 0.0, 0.0]))

    assert (result > 0.0).all()
    assert result[0] == 0.8


def test_race_probability_smoothing_prevents_calibration_collapse() -> None:
    result = smooth_race_probabilities(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([4, 4, 4, 4]),
    )

    assert np.isclose(result.sum(), 0.98 + 0.02)
    assert result[0] < 1.0
    assert (result > 0.0).all()


def test_collapsed_calibration_falls_back_to_raw_ranking() -> None:
    result = recover_collapsed_calibration(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.7, 0.2, 0.1]),
    )

    assert np.argmax(result) == 0
    assert not np.allclose(result, result[0])


def test_partial_calibration_collapse_recovers_zeroed_rows() -> None:
    result = recover_collapsed_calibration(
        np.array([0.98, 0.0, 0.0]),
        np.array([0.7, 0.2, 0.1]),
    )

    assert result[1] > result[2] > 0.0


def test_small_isotonic_calibration_uses_platt_fallback() -> None:
    calibrator = fit_calibrator(
        np.array([1, 0, 0, 1]),
        np.array([0.8, 0.4, 0.2, 0.7]),
        method="isotonic",
    )

    assert calibrator.method == "platt"
