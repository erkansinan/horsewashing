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


def _gamma_adjust(probabilities: np.ndarray, gamma: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))
    adjusted = 1.0 / (1.0 + np.exp(-np.clip(gamma * logits, -40.0, 40.0)))
    return adjusted / max(float(adjusted.sum()), 1e-12)


def _fit_position_gamma(races: list[tuple[np.ndarray, int]]) -> float:
    if len(races) < 20:
        return 1.0

    def loss(gamma: float) -> float:
        total = 0.0
        for probabilities, target_index in races:
            adjusted = _gamma_adjust(probabilities, gamma)
            total -= float(np.log(max(adjusted[target_index], 1e-12)))
        return total / len(races)

    left, right = 0.05, 5.0
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = loss(x1), loss(x2)
    for _ in range(64):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = loss(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = loss(x2)
    gamma = (left + right) / 2.0
    candidates = (0.05, 1.0, 5.0, gamma)
    return min(candidates, key=loss)


def fit_harville_gammas(
    frame: pd.DataFrame,
    win_col: str = "calibrated_probability",
    finish_col: str = "finish_position",
    minimum_races: int = 20,
) -> dict[str, float | int]:
    """Learn place-specific logit slopes from prior, labeled races only."""
    if not {"race_id", win_col, finish_col}.issubset(frame.columns):
        return {"place2_gamma": 1.0, "place3_gamma": 1.0, "training_races": 0}

    raw = add_harville_columns(frame, win_col=win_col)
    race_targets: dict[int, list[tuple[np.ndarray, int]]] = {2: [], 3: []}
    for _, indices in raw.groupby("race_id", sort=False).groups.items():
        group = raw.loc[indices]
        finishes = pd.to_numeric(group[finish_col], errors="coerce").to_numpy(dtype=float)
        for position in (2, 3):
            targets = np.flatnonzero(finishes == position)
            if len(targets) != 1:
                continue
            race_targets[position].append(
                (group[f"harville_place{position}_raw"].to_numpy(dtype=float), int(targets[0]))
            )

    result: dict[str, float | int] = {
        "place2_gamma": _fit_position_gamma(race_targets[2]) if len(race_targets[2]) >= minimum_races else 1.0,
        "place3_gamma": _fit_position_gamma(race_targets[3]) if len(race_targets[3]) >= minimum_races else 1.0,
        "place2_training_races": len(race_targets[2]),
        "place3_training_races": len(race_targets[3]),
    }
    return result


def add_harville_columns(
    frame: pd.DataFrame,
    win_col: str = "calibrated_probability",
    place2_gamma: float = 1.0,
    place3_gamma: float = 1.0,
) -> pd.DataFrame:
    out = frame.copy()
    out["harville_place2_raw"] = 0.0
    out["harville_place3_raw"] = 0.0
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

        out.loc[race_idx, "harville_place2_raw"] = p2
        out.loc[race_idx, "harville_place3_raw"] = p3
        out.loc[race_idx, "place2_probability"] = _gamma_adjust(p2, place2_gamma)
        if len(race_idx) >= 3:
            out.loc[race_idx, "place3_probability"] = _gamma_adjust(p3, place3_gamma)

    out["top3_probability"] = (out[win_col] + out["place2_probability"] + out["place3_probability"]).clip(upper=1.0)
    out["place_probability"] = out["top3_probability"]
    return out


def mark_highest_odds_placer_predictions(
    frame: pd.DataFrame,
    place_probability_col: str = "place_probability",
    odds_col: str = "odds",
    top_k: int = 3,
) -> pd.DataFrame:
    """Mark the highest-odds runner among the model's top-k place candidates."""
    out = frame.copy()
    if place_probability_col not in out.columns:
        place_probability_col = "top3_probability" if "top3_probability" in out else "calibrated_probability"
    out["predicted_place_rank"] = np.nan
    out["predicted_top3"] = False
    out["predicted_highest_odds_placer"] = False
    if out.empty or place_probability_col not in out.columns:
        return out

    for _race_id, indices in out.groupby("race_id", sort=False).groups.items():
        probabilities = pd.to_numeric(out.loc[indices, place_probability_col], errors="coerce").fillna(-1.0)
        ranks = probabilities.rank(method="first", ascending=False)
        out.loc[indices, "predicted_place_rank"] = ranks
        candidate_indices = ranks[ranks <= max(1, top_k)].index
        out.loc[candidate_indices, "predicted_top3"] = True
        if odds_col not in out.columns:
            continue
        candidates = out.loc[candidate_indices].copy()
        candidates["_odds"] = pd.to_numeric(candidates[odds_col], errors="coerce")
        candidates = candidates[candidates["_odds"] > 1.0]
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            ["_odds", place_probability_col], ascending=[False, False], kind="stable"
        )
        out.loc[candidates.index[0], "predicted_highest_odds_placer"] = True
    return out
