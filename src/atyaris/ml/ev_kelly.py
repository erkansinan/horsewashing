from __future__ import annotations

# EV + fractional Kelly sizing for value betting.
import numpy as np
import pandas as pd


def add_ev_kelly_columns(
    frame: pd.DataFrame,
    *,
    min_probability: float,
    min_edge: float,
    min_ev: float,
    fractional_kelly: float,
    max_kelly_fraction: float,
) -> pd.DataFrame:
    out = frame.copy()

    market_col = "market_probability_norm" if "market_probability_norm" in out.columns else "market_probability"
    out["market_probability_used"] = pd.to_numeric(out.get(market_col, 0.0), errors="coerce").fillna(0.0)
    out["edge"] = out["calibrated_probability"] - out["market_probability_used"]

    odds = pd.to_numeric(out.get("odds", np.nan), errors="coerce").fillna(0.0)
    out["ev"] = out["calibrated_probability"] * odds - 1.0

    b = (odds - 1.0).to_numpy(dtype=float)
    p = out["calibrated_probability"].to_numpy(dtype=float)
    q = 1.0 - p
    raw_kelly = np.where(b > 1e-12, (b * p - q) / b, 0.0)
    raw_kelly = np.clip(raw_kelly, 0.0, None)

    frac = max(float(fractional_kelly), 0.0)
    cap = max(float(max_kelly_fraction), 0.0)
    out["kelly_fraction"] = np.clip(raw_kelly * frac, 0.0, cap)

    bet_mask = (
        (out["calibrated_probability"] >= min_probability)
        & (out["edge"] >= min_edge)
        & (out["ev"] >= min_ev)
        & (out["kelly_fraction"] > 0.0)
    )
    out["bet_decision"] = np.where(bet_mask, "BET", "NO_BET")
    return out
