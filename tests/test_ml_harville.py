from __future__ import annotations

import numpy as np
import pandas as pd

from atyaris.ml.calibration import (
    apply_probability_floor,
    fit_calibrator,
    recover_collapsed_calibration,
    smooth_race_probabilities,
)
from atyaris.ml.harville import (
    add_harville_columns,
    fit_harville_gammas,
    mark_highest_odds_placer_predictions,
)


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


def test_place_gamma_preserves_raw_harville_at_one_and_adjusts_longshots() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "calibrated_probability": [0.70, 0.20, 0.08, 0.02],
        }
    )

    raw = add_harville_columns(frame)
    adjusted = add_harville_columns(frame, place2_gamma=0.5, place3_gamma=0.5)

    assert np.allclose(raw["place2_probability"], raw["harville_place2_raw"])
    assert np.allclose(raw["place3_probability"], raw["harville_place3_raw"])
    assert np.isclose(adjusted["place2_probability"].sum(), 1.0)
    assert np.isclose(adjusted["place3_probability"].sum(), 1.0)
    assert adjusted.loc[3, "place2_probability"] > raw.loc[3, "place2_probability"]


def test_place_gammas_are_learned_from_prior_exact_finish_positions() -> None:
    rows = []
    for race_number in range(24):
        for horse_number, (probability, finish) in enumerate(
            zip([0.70, 0.20, 0.08, 0.02], [1, 3, 4, 2]),
            start=1,
        ):
            rows.append(
                {
                    "race_id": f"R{race_number}",
                    "calibrated_probability": probability,
                    "finish_position": finish,
                }
            )

    gammas = fit_harville_gammas(pd.DataFrame(rows))

    assert gammas["place2_training_races"] == 24
    assert gammas["place3_training_races"] == 24
    assert 0.05 <= gammas["place2_gamma"] < 1.0
    assert 0.05 <= gammas["place3_gamma"] < 1.0


def test_highest_odds_selection_is_limited_to_top_three_place_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "horse_id": ["H1", "H2", "H3", "H4"],
            "place_probability": [0.8, 0.7, 0.6, 0.1],
            "odds": [4.0, 8.0, 12.0, 100.0],
        }
    )

    result = mark_highest_odds_placer_predictions(frame)

    assert set(result.loc[result["predicted_top3"], "horse_id"]) == {"H1", "H2", "H3"}
    assert result.loc[result["predicted_highest_odds_placer"], "horse_id"].tolist() == ["H3"]


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


def test_dominant_calibration_plateau_recovers_raw_race_probabilities() -> None:
    calibrated = np.array([0.139, 0.078] + [0.078] * 10)
    raw = np.array([0.112, 0.087, 0.083, 0.084, 0.082, 0.080, 0.094, 0.077, 0.076, 0.079, 0.073, 0.072])

    result = recover_collapsed_calibration(
        calibrated,
        raw,
        group_ids=np.array(["race-1"] * 12),
    )

    assert np.allclose(result, raw)
    assert np.unique(result).size == 12


def test_calibration_plateau_recovery_is_limited_to_its_race() -> None:
    calibrated = np.array([0.4, 0.3, 0.3, 0.3, 0.3, 0.6, 0.25, 0.15])
    raw = np.array([0.5, 0.3, 0.2, 0.15, 0.1, 0.5, 0.3, 0.2])

    result = recover_collapsed_calibration(
        calibrated,
        raw,
        group_ids=np.array(["collapsed"] * 5 + ["normal"] * 3),
    )

    assert np.allclose(result[:5], raw[:5])
    assert np.allclose(result[5:], calibrated[5:])


def test_small_isotonic_calibration_uses_platt_fallback() -> None:
    calibrator = fit_calibrator(
        np.array([1, 0, 0, 1]),
        np.array([0.8, 0.4, 0.2, 0.7]),
        method="isotonic",
    )

    assert calibrator.method == "platt"
