"""Build a labelled local training frame from the raw JSONL collections."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
import math
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


def find_missing_local_targets(
    paths,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    """Find locally indexed races whose result or horse history is incomplete."""
    programs = _load_jsonl(paths.raw_daily_program_jsonl)
    results = _load_jsonl(paths.raw_race_results_jsonl)
    histories = _load_jsonl(paths.raw_history_jsonl)
    result_keys = {
        (str(row.get("race_id")), str(row.get("horse_id")))
        for row in results
        if row.get("finish_position") is not None
    }
    history_targets = {
        (str(row.get("target_race_id")), str(row.get("target_horse_id")))
        for row in histories
        if row.get("target_race_id") and row.get("target_horse_id")
    }
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for program in programs:
        target_date = _date(program.get("race_date") or program.get("date"))
        if target_date is None or (start_date and target_date < start_date) or (end_date and target_date > end_date):
            continue
        try:
            race_no = int(program.get("race_no"))
        except (TypeError, ValueError):
            continue
        key = (target_date.isoformat(), str(program.get("hippodrome") or ""), race_no)
        grouped[key].append(program)

    missing: list[dict[str, Any]] = []
    for (date_text, hippodrome, race_no), entries in grouped.items():
        missing_results = [
            entry for entry in entries
            if (str(entry.get("race_id")), str(entry.get("horse_id"))) not in result_keys
        ]
        missing_history = [
            entry for entry in entries
            if (str(entry.get("race_id")), str(entry.get("horse_id"))) not in history_targets
        ]
        if missing_results or missing_history:
            missing.append(
                {
                    "date": date_text,
                    "hippodrome": hippodrome,
                    "race_no": race_no,
                    "missing_results": len(missing_results),
                    "missing_history": len(missing_history),
                }
            )
    return missing


def build_local_raw_frame(paths) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Materialize labelled rows from local raw collections without network access."""
    programs = _load_jsonl(paths.raw_daily_program_jsonl)
    results = _load_jsonl(paths.raw_race_results_jsonl)
    histories = _load_jsonl(paths.raw_history_jsonl)
    workouts = _load_jsonl(getattr(paths, "raw_workouts_jsonl", None)) if getattr(paths, "raw_workouts_jsonl", None) else []
    trainer_statistics = _load_jsonl(getattr(paths, "raw_trainer_statistics_jsonl", None)) if getattr(paths, "raw_trainer_statistics_jsonl", None) else []
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
    workouts_by_horse: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in workouts:
        horse_id = str(row.get("horse_id") or "")
        if horse_id:
            workouts_by_horse[horse_id].append(row)
    trainers_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trainer_statistics:
        trainer_id = str(row.get("trainer_id") or "")
        as_of_date = _date(row.get("as_of_date"))
        if trainer_id and as_of_date is not None:
            trainers_by_id[trainer_id].append(row)

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
        odds = pd.to_numeric(program.get("odds"), errors="coerce")
        market_probability = 1.0 / float(odds) if pd.notna(odds) and float(odds) > 1.0 else 0.1
        history = [
            item for item in history_by_horse.get(horse_id, [])
            if (_date(item.get("race_date")) or date.max) < target_date
        ]
        past_workouts = [
            item for item in workouts_by_horse.get(horse_id, [])
            if (_date(item.get("workout_date")) or date.max) < target_date
        ]
        workout_speeds = [
            float(item.get("distance_m") or 0.0) / max(float(item.get("time_seconds") or 0.0), 1e-9)
            for item in past_workouts
            if float(item.get("time_seconds") or 0.0) > 0.0
        ]
        trainer_id = str(program.get("trainer_id") or "")
        trainer_snapshots = [
            item for item in trainers_by_id.get(trainer_id, [])
            if (_date(item.get("as_of_date")) or date.max) < target_date
        ]
        trainer_stats = max(
            trainer_snapshots,
            key=lambda item: str(item.get("as_of_date", "")),
            default=None,
        )
        trainer_starts = int(float((trainer_stats or {}).get("total_starts") or 0))
        trainer_strength = min(max(trainer_starts, 0), 1000) / (min(max(trainer_starts, 0), 1000) + 30.0)
        trainer_win_rate = (
            (float(trainer_stats.get("first_rate") or 0.0) / 100.0) * trainer_strength
            + 0.10 * (1.0 - trainer_strength)
            if trainer_stats else 0.0
        )
        trainer_top3_rate = (
            sum(float(trainer_stats.get(key) or 0.0) for key in ("first_rate", "second_rate", "third_rate")) / 100.0 * trainer_strength
            + 0.30 * (1.0 - trainer_strength)
            if trainer_stats else 0.0
        )
        trainer_top5_rate = (
            sum(float(trainer_stats.get(key) or 0.0) for key in ("first_rate", "second_rate", "third_rate", "fourth_rate", "fifth_rate")) / 100.0 * trainer_strength
            + 0.50 * (1.0 - trainer_strength)
            if trainer_stats else 0.0
        )
        finishes = [float(item["finish_position"]) for item in history if item.get("finish_position") is not None]
        performance = [1.0 - ((value - 1.0) / max(float(item.get("field_size") or 10) - 1.0, 1.0)) for value, item in zip(finishes, history)]
        recent = performance[-10:]
        row: dict[str, Any] = {
            "race_id": str(program.get("race_id")),
            "date": target_date.isoformat(),
            "race_datetime": program.get("race_datetime") or f"{target_date.isoformat()} 12:00:00",
            "horse_id": horse_id,
            "horse_name": program.get("horse_name", ""),
            "draw": program.get("draw", 0),
            "weight": program.get("weight_kg", 0),
            "distance": program.get("distance_m", 0),
            "field_size": program.get("field_size", 0),
            "age": program.get("age", 0),
            "handicap_points": program.get("handicap_points", 0),
            "odds": odds if pd.notna(odds) else 0.0,
            "market_probability": market_probability,
            # Retain the label so the feature builder can replay history and
            # calculate point-in-time features; it is excluded from features.
            "finish_position": int(float(finish)),
            "is_winner": int(float(finish) == 1),
            "track": str(program.get("hippodrome", "")).upper(),
            "surface": program.get("surface", ""),
            "track_condition": program.get("track_condition", "NORMAL"),
            "pace_hint": 0.0,
            "style_front_prob": 0.25,
            "style_presser_prob": 0.25,
            "style_stalker_prob": 0.25,
            "style_closer_prob": 0.25,
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
            "workout_missing": float(not past_workouts),
            "trainer_starts": trainer_starts,
            "trainer_first_place": int(float((trainer_stats or {}).get("first_place") or 0)),
            "trainer_second_place": int(float((trainer_stats or {}).get("second_place") or 0)),
            "trainer_third_place": int(float((trainer_stats or {}).get("third_place") or 0)),
            "trainer_fourth_place": int(float((trainer_stats or {}).get("fourth_place") or 0)),
            "trainer_fifth_place": int(float((trainer_stats or {}).get("fifth_place") or 0)),
            "trainer_win_rate": trainer_win_rate,
            "trainer_top3_rate": trainer_top3_rate,
            "trainer_top5_rate": trainer_top5_rate,
            "trainer_experience_log": math.log1p(trainer_starts),
            "trainer_stats_missing": float(trainer_stats is None),
            "odds_missing": float(not program.get("odds")),
            "jockey_name": program.get("jockey_name", ""),
            "trainer_name": program.get("trainer_name", ""),
            "race_class": program.get("race_class", ""),
            "workout_latest_speed_index": workout_speeds[-1] if workout_speeds else 0.0,
            "workout_prior_avg_speed_index": sum(workout_speeds[:-1]) / len(workout_speeds[:-1]) if len(workout_speeds) > 1 else 0.0,
        }
        for column in TJK_STAGE1_FEATURE_COLUMNS:
            row.setdefault(column, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
