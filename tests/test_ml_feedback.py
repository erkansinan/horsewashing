from __future__ import annotations

import pandas as pd

from atyaris.ml.feedback import attach_results, compare_predictions


def test_compare_predictions_matches_race_suffix_and_draw() -> None:
    predictions = pd.DataFrame(
        [
            {"race_id": "İstanbul-2026-09-06-1", "draw": 4, "calibrated_probability": 0.7},
            {"race_id": "İstanbul-2026-09-06-1", "draw": 2, "calibrated_probability": 0.3},
            {"race_id": "İstanbul-2026-09-06-2", "draw": 7, "calibrated_probability": 0.6},
        ]
    )

    comparison = compare_predictions(predictions, {1: {4: 1, 2: 2}, 2: {7: 3}})

    assert comparison["evaluated_races"] == 2
    assert comparison["evaluated_horses"] == 3
    assert comparison["top1_hit_rate"] == 0.5
    assert comparison["top3_hit_rate"] == 0.5
    assert comparison["rows"]["actual_finish_position"].tolist() == [1, 2, 3]


def test_missing_result_is_not_marked_as_loss_or_win() -> None:
    predictions = pd.DataFrame(
        [{"race_id": "Ankara-2026-09-06-1", "draw": 3, "calibrated_probability": 0.5}]
    )

    attached = attach_results(predictions, {})

    assert pd.isna(attached.loc[0, "actual_finish_position"])
    assert pd.isna(attached.loc[0, "actual_is_winner"])
