from __future__ import annotations

import numpy as np
import pandas as pd

from atyaris.ml.features import TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS
from atyaris.ml.placer_model import (
    fit_highest_odds_placer_model,
    predict_highest_odds_placer_probability,
)


def _placer_training_frame() -> pd.DataFrame:
    rows = []
    for race_number in range(8):
        for runner in range(4):
            odds = float(2 + runner * 4)
            rows.append(
                {
                    "race_id": f"R{race_number}",
                    "horse_id": f"H{runner}",
                    "target_highest_odds_placer": "H3",
                    "odds": odds,
                    "odds_log": np.log(odds),
                    "odds_rank_fraction": runner / 3,
                    "odds_vs_field_median": runner - 1.5,
                    "market_probability_norm": 1.0 / odds,
                    **{
                        column: 0.0
                        for column in TJK_HIGHEST_ODDS_PLACER_FEATURE_COLUMNS
                        if column not in {
                            "odds_log",
                            "odds_rank_fraction",
                            "odds_vs_field_median",
                            "market_probability_norm",
                        }
                    },
                }
            )
    return pd.DataFrame(rows)


def test_direct_model_learns_one_highest_odds_placer_per_race() -> None:
    frame = _placer_training_frame()

    model = fit_highest_odds_placer_model(frame)
    probabilities = predict_highest_odds_placer_probability(model, frame)
    scored = frame.assign(probability=probabilities)

    assert np.allclose(scored.groupby("race_id")["probability"].sum(), 1.0)
    assert (
        scored.loc[scored.groupby("race_id")["probability"].idxmax(), "horse_id"]
        == "H3"
    ).all()

    prediction = frame.iloc[:4].copy()
    prediction.loc[prediction["horse_id"] == "H0", "odds"] = 1.0
    probabilities = predict_highest_odds_placer_probability(model, prediction)

    assert probabilities[0] == 0.0
    assert np.isclose(probabilities.sum(), 1.0)
