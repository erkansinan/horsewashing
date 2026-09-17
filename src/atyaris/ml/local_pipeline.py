"""Build a labelled local training frame from the raw JSONL collections."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atyaris.ml.features import TJK_STAGE1_FEATURE_COLUMNS


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def build_local_raw_frame(paths) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Materialize labelled rows from local raw collections without network access."""
    programs = _load_jsonl(paths.raw_daily_program_jsonl)
    results = _load_jsonl(paths.raw_race_results_jsonl)
    histories = _load_jsonl(paths.raw_history_jsonl)
    if not programs:
        return pd.read_csv(paths.raw_csv) if paths.raw_csv.exists() else pd.DataFrame()

    result_by_key = {
        (str(row.get("race_id")), str(row.get("horse_id"))): row
        for row in results
    }
    history_by_horse: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in histories:
        horse_id = str(row.get("horse_id") or row.get("target_horse_id") or "")
        if horse_id:
            history_by_horse[horse_id].append(row)

    rows: list[dict[str, Any]] = []
    for program in programs:
        target_date = _date(program.get("race_date") or program.get("date"))
        if target_date is None:
            continue
        horse_id = str(program.get("horse_id") or "")
        result = result_by_key.get((str(program.get("race_id")), horse_id), {})
        finish = result.get("finish_position")
        if finish is None:
            continue
        history = [
            item for item in history_by_horse.get(horse_id, [])
            if (_date(item.get("race_date")) or date.max) < target_date
        ]
        finishes = [float(item["finish_position"]) for item in history if item.get("finish_position") is not None]
        performance = [1.0 - ((value - 1.0) / max(float(item.get("field_size") or 10) - 1.0, 1.0)) for value, item in zip(finishes, history)]
        recent = performance[-10:]
        row: dict[str, Any] = {
            "race_id": str(program.get("race_id")),
            "date": target_date.isoformat(),
            "race_datetime": program.get("race_datetime"),
            "horse_id": horse_id,
            "horse_name": program.get("horse_name", ""),
            "draw": program.get("draw", 0),
            "weight": program.get("weight_kg", 0),
            "distance": program.get("distance_m", 0),
            "field_size": program.get("field_size", 0),
            "age": program.get("age", 0),
            "handicap_points": program.get("handicap_points", 0),
            "odds": program.get("odds", 0),
            "is_winner": int(float(finish) == 1),
            "track": str(program.get("hippodrome", "")).upper(),
            "surface": program.get("surface", ""),
            "track_condition": program.get("track_condition", "NORMAL"),
            "career_starts": len(history),
            "career_wins": sum(1 for item in history if item.get("finish_position") == 1),
            "career_places": sum(1 for item in history if item.get("finish_position") is not None and int(item["finish_position"]) <= 3),
            "form_avg_3": sum(recent[-3:]) / len(recent[-3:]) if recent else 0.45,
            "form_avg_5": sum(recent[-5:]) / len(recent[-5:]) if recent else 0.45,
            "form_avg_10": sum(recent) / len(recent) if recent else 0.45,
            "last_run_perf": recent[-1] if recent else 0.45,
            "history_avg_finish_position": sum(finishes) / len(finishes) if finishes else 0.0,
            "history_avg_field_size": sum(float(item.get("field_size") or 0) for item in history) / len(history) if history else 0.0,
            "history_avg_weight": sum(float(item.get("weight_kg") or 0) for item in history) / len(history) if history else 0.0,
            "history_avg_odds": sum(float(item.get("odds") or 0) for item in history) / len(history) if history else 0.0,
            "history_missing": float(not history),
            "career_summary_missing": float(not history),
            "workout_missing": 1.0,
            "trainer_stats_missing": 1.0,
            "odds_missing": float(not program.get("odds")),
        }
        for column in TJK_STAGE1_FEATURE_COLUMNS:
            row.setdefault(column, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
