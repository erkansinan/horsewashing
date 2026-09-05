from __future__ import annotations

# Harville-style derivation of place probabilities from win probabilities.
# Widely used for exacta/trifecta style rank-order calculations.
import numpy as np
import pandas as pd


def _harville_second(win_probs: np.ndarray) -> np.ndarray:
    n = len(win_probs)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            denom = max(1e-12, 1.0 - win_probs[j])
            s += win_probs[j] * (win_probs[i] / denom)
        out[i] = s
    return out


def _harville_third(win_probs: np.ndarray) -> np.ndarray:
    n = len(win_probs)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                denom_j = max(1e-12, 1.0 - win_probs[j])
                rem = max(1e-12, 1.0 - win_probs[j] - win_probs[k])
                s += win_probs[j] * (win_probs[k] / denom_j) * (win_probs[i] / rem)
        out[i] = s
    return out


def add_harville_columns(frame: pd.DataFrame, win_col: str = "calibrated_probability") -> pd.DataFrame:
    out = frame.copy()
    out["place2_probability"] = 0.0
    out["place3_probability"] = 0.0

    for _, idx in out.groupby("race_id").groups.items():
        race_idx = list(idx)
        if len(race_idx) < 2:
            continue
        p = out.loc[race_idx, win_col].to_numpy(dtype=float)
        p = np.clip(p, 1e-12, 1.0)
        p = p / max(p.sum(), 1e-12)

        p2 = _harville_second(p)
        p3 = _harville_third(p) if len(race_idx) >= 3 else np.zeros_like(p2)

        out.loc[race_idx, "place2_probability"] = p2
        out.loc[race_idx, "place3_probability"] = p3

    out["top3_probability"] = (out[win_col] + out["place2_probability"] + out["place3_probability"]).clip(upper=1.0)
    return out
