"""TJK gunluk sonuclarini ML tahminleriyle karsilastirma yardimcilari."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _race_number(race_id: object) -> int | None:
    match = re.search(r"-(\d+)$", str(race_id))
    return int(match.group(1)) if match else None


def attach_results(
    predictions: pd.DataFrame,
    results: dict[int, dict[int, int]],
) -> pd.DataFrame:
    """Tahmin satirlarina TJK bitis derecesini start numarasiyla ekler."""
    output = predictions.copy()
    output["race_no"] = output["race_id"].map(_race_number)
    output["actual_finish_position"] = [
        results.get(race_no, {}).get(int(draw))
        if race_no is not None and pd.notna(draw)
        else None
        for race_no, draw in zip(output["race_no"], output.get("draw", pd.Series(dtype=float)))
    ]
    output["actual_is_winner"] = output["actual_finish_position"].eq(1).astype(int)
    output.loc[output["actual_finish_position"].isna(), "actual_is_winner"] = np.nan
    return output


def compare_predictions(predictions: pd.DataFrame, results: dict[int, dict[int, int]]) -> dict[str, object]:
    """Tahminleri gercek sonuclarla eslestirip ozet metrikleri hesaplar."""
    scored = attach_results(predictions, results)
    scored = scored.dropna(subset=["actual_finish_position"]).copy()
    if scored.empty:
        return {
            "evaluated_races": 0,
            "evaluated_horses": 0,
            "top1_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "brier": None,
            "rows": scored,
        }

    race_groups = scored.groupby("race_no", sort=True)
    top1_hits = []
    top3_hits = []
    brier_values = []
    for _, group in race_groups:
        ranked = group.sort_values("calibrated_probability", ascending=False)
        winner = group["actual_is_winner"].eq(1)
        top1_hits.append(int(bool(winner.loc[ranked.index[:1]].any())))
        top3_hits.append(int(bool(winner.loc[ranked.index[:3]].any())))
        if "calibrated_probability" in group:
            brier_values.extend(
                (group["calibrated_probability"].astype(float) - group["actual_is_winner"].astype(float)) ** 2
            )

    return {
        "evaluated_races": len(top1_hits),
        "evaluated_horses": int(len(scored)),
        "top1_hit_rate": float(np.mean(top1_hits)),
        "top3_hit_rate": float(np.mean(top3_hits)),
        "brier": float(np.mean(brier_values)) if brier_values else None,
        "rows": scored,
    }
