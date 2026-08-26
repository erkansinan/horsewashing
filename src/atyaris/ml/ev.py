from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_ev_columns(
    frame: pd.DataFrame,
    min_probability: float,
    min_edge: float,
    min_ev: float,
) -> pd.DataFrame:
    out = frame.copy()

    market_col = "market_probability_norm" if "market_probability_norm" in out.columns else "market_probability"
    out["market_probability_used"] = out[market_col].fillna(0.0)
    out["edge"] = out["calibrated_probability"] - out["market_probability_used"]

    odds = out.get("odds", pd.Series(np.nan, index=out.index)).fillna(0.0)
    out["ev"] = out["calibrated_probability"] * (odds - 1.0) - (1.0 - out["calibrated_probability"])
    out["ev"] = out["ev"].fillna(-1.0)

    bet_mask = (
        (out["calibrated_probability"] >= min_probability)
        & (out["edge"] >= min_edge)
        & (out["ev"] >= min_ev)
    )
    out["bet_decision"] = np.where(bet_mask, "BET", "NO_BET")
    return out
