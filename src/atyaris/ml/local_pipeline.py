"""Build a labelled local training frame from the raw JSONL collections."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from atyaris.ml.features import TJK_STAGE1_FEATURE_COLUMNS, class_level


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


def _number(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else None


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
    trainer_statistics_path = getattr(paths, "raw_trainer_statistics_jsonl", None)
    trainer_statistics = (
        _load_jsonl(trainer_statistics_path)
        if trainer_statistics_path is not None else []
    )
    if not programs:
        return pd.read_csv(paths.raw_csv) if paths.raw_csv.exists() else pd.DataFrame()

    result_by_key = {
        (str(row.get("race_id")), str(row.get("horse_id"))): row
        for row in results
    }

    history_by_target: dict[
        tuple[str, str], dict[tuple[date, str], dict[str, Any]]
    ] = defaultdict(dict)
    legacy_history_by_horse: dict[
        str, dict[tuple[date, str], dict[str, Any]]
    ] = defaultdict(dict)
    if paths.raw_history_jsonl.exists():
        with paths.raw_history_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                race_date = _date(item.get("race_date"))
                if race_date is None:
                    continue
                compact = {
                    "race_id": str(item.get("race_id") or ""),
                    "race_date": race_date,
                    "finish_position": _number(item.get("finish_position")),
                    "weight_kg": _number(item.get("weight_kg")),
                    "odds": _number(item.get("odds")),
                    "handicap_points": _number(item.get("handicap_points")),
                    "race_time_seconds": _number(item.get("race_time_seconds")),
                    "jockey_name": str(item.get("jockey_name") or "").strip().casefold(),
                    "race_class": str(item.get("race_class") or ""),
                }
                target_race_id = str(item.get("target_race_id") or "")
                target_horse_id = str(item.get("target_horse_id") or "")
                key = (race_date, compact["race_id"])
                if target_race_id and target_horse_id:
                    history_by_target[(target_race_id, target_horse_id)][key] = compact
                else:
                    horse_id = str(item.get("horse_id") or "")
                    if horse_id:
                        legacy_history_by_horse[horse_id][key] = compact

    workouts_by_horse: dict[str, dict[tuple[date, float, float], None]] = defaultdict(dict)
    workout_path = getattr(paths, "raw_workouts_jsonl", None)
    if workout_path is not None and workout_path.exists():
        with workout_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                horse_id = str(item.get("horse_id") or "")
                workout_date = _date(item.get("workout_date"))
                distance = _number(item.get("distance_m"))
                seconds = _number(item.get("time_seconds"))
                if horse_id and workout_date is not None and distance is not None:
                    workouts_by_horse[horse_id][
                        (workout_date, distance, seconds or 0.0)
                    ] = None
    trainers_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trainer_statistics:
        trainer_id = str(item.get("trainer_id") or "")
        if trainer_id and _date(item.get("as_of_date")) is not None:
            trainers_by_id[trainer_id].append(item)

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
        race_id = str(program.get("race_id"))
        exact_history = history_by_target.get((race_id, horse_id), {})
        history_records = exact_history.values() if exact_history else legacy_history_by_horse.get(horse_id, {}).values()
        history = sorted(
            (
                item for item in history_records
                if item["race_date"] < target_date
            ),
            key=lambda item: (item["race_date"], item["race_id"]),
        )
        finishes = [
            float(item["finish_position"])
            for item in history
            if item["finish_position"] is not None and item["finish_position"] > 0
        ]
        odds = _number(program.get("odds"))
        market_probability = 1.0 / odds if odds is not None and odds > 1.0 else 0.1
        weight = _number(program.get("weight_kg")) or 0.0
        distance = _number(program.get("distance_m")) or 0.0
        field_size = _number(program.get("field_size")) or 0.0
        age_value = _number(program.get("age"))
        handicap_value = _number(program.get("handicap_points"))
        past_workouts = sorted(
            (
                (workout_date, workout_distance, seconds)
                for workout_date, workout_distance, seconds
                in workouts_by_horse.get(horse_id, {})
                if workout_date < target_date
                and (distance <= 0.0 or abs(workout_distance - distance) <= 200.0)
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        workout_speeds = [
            workout_distance / seconds
            for _, workout_distance, seconds in past_workouts
            if seconds > 0.0 and workout_distance > 0.0
        ]
        workout_distances = [
            workout_distance
            for _, workout_distance, _ in past_workouts
            if workout_distance > 0.0
        ]
        days_since_workout = (
            (target_date - max(item[0] for item in past_workouts)).days
            if past_workouts else 0
        )
        last_year_history = [
            item for item in history
            if 0 <= (target_date - item["race_date"]).days <= 365
        ]
        current_jockey = str(program.get("jockey_name") or "").strip().casefold()
        combo_history = [
            item for item in history
            if current_jockey and item["jockey_name"] == current_jockey
        ]
        history_dates = [item["race_date"] for item in history]
        days_since_last_race = (
            (target_date - history_dates[-1]).days if history_dates else 30
        )
        recent_gaps = [
            (history_dates[index] - history_dates[index - 1]).days
            for index in range(1, len(history_dates))
        ]
        gaps_near_optimal = [gap for gap in recent_gaps[-5:] if 14 <= gap <= 28]
        history_weights = [
            float(item["weight_kg"])
            for item in history
            if item["weight_kg"] is not None and item["weight_kg"] > 0
        ]
        history_odds = [
            float(item["odds"])
            for item in history
            if item["odds"] is not None and item["odds"] > 0
        ]
        history_handicaps = [
            float(item["handicap_points"])
            for item in history
            if item["handicap_points"] is not None
        ]
        history_race_times = [
            float(item["race_time_seconds"])
            for item in history
            if item["race_time_seconds"] is not None and item["race_time_seconds"] > 0
        ]
        longshot_history = [
            item for item in history
            if item["odds"] is not None and item["odds"] >= 5.0
        ]
        longshot_places = sum(
            1 for item in longshot_history
            if item["finish_position"] is not None
            and 0 < item["finish_position"] <= 3
        )
        current_class = str(program.get("group_info") or program.get("race_class") or "")
        previous_class = next(
            (item["race_class"] for item in reversed(history) if item["race_class"]),
            "",
        )
        trainer_id = str(program.get("trainer_id") or "")
        trainer_stats = max(
            (
                item for item in trainers_by_id.get(trainer_id, [])
                if (_date(item.get("as_of_date")) or date.max) < target_date
            ),
            key=lambda item: str(item.get("as_of_date", "")),
            default=None,
        )
        trainer_starts = int(_number((trainer_stats or {}).get("total_starts")) or 0)
        trainer_strength = min(max(trainer_starts, 0), 1000) / (
            min(max(trainer_starts, 0), 1000) + 30.0
        )
        trainer_win_rate = (
            (_number(trainer_stats.get("first_rate")) or 0.0) / 100.0 * trainer_strength
            + 0.10 * (1.0 - trainer_strength)
            if trainer_stats else 0.0
        )
        trainer_top3_rate = (
            sum(_number(trainer_stats.get(key)) or 0.0 for key in (
                "first_rate", "second_rate", "third_rate"
            )) / 100.0 * trainer_strength
            + 0.30 * (1.0 - trainer_strength)
            if trainer_stats else 0.0
        )
        trainer_top5_rate = (
            sum(_number(trainer_stats.get(key)) or 0.0 for key in (
                "first_rate", "second_rate", "third_rate", "fourth_rate", "fifth_rate"
            )) / 100.0 * trainer_strength
            + 0.50 * (1.0 - trainer_strength)
            if trainer_stats else 0.0
        )
        speed_average = sum(workout_speeds) / len(workout_speeds) if workout_speeds else 0.0
        prior_speed_average = (
            sum(workout_speeds[:-1]) / len(workout_speeds[:-1])
            if len(workout_speeds) > 1 else 0.0
        )
        workout_improvement = (
            workout_speeds[-1] - prior_speed_average
            if workout_speeds and prior_speed_average > 0.0 else 0.0
        )
        race_frequency_3 = 3.0 / max(
            (target_date - history_dates[max(0, len(history_dates) - 3)]).days,
            1,
        ) if history_dates else 0.1
        race_frequency_5 = 5.0 / max(
            (target_date - history_dates[max(0, len(history_dates) - 5)]).days,
            1,
        ) if history_dates else (5.0 / 30.0)
        season_starts = sum(
            1 for item in history
            if 0 <= (target_date - item["race_date"]).days <= 120
        )
        fatigue_score = 1.0 / (1.0 + math.exp(-((days_since_last_race - 18.0) / 8.0)))
        rest_optimal_fit = (
            max(0.0, 1.0 - abs(days_since_last_race - median(gaps_near_optimal)) / 21.0)
            if gaps_near_optimal else 0.0
        )
        current_level = class_level(current_class)
        previous_level = class_level(previous_class)
        class_drop = (
            float(previous_level - current_level > 0.0)
            if previous_level is not None and current_level is not None else 0.0
        )
        row: dict[str, Any] = {
            "_tjk_point_in_time_features": 1.0,
            "race_id": race_id,
            "date": target_date.isoformat(),
            "race_datetime": program.get("race_datetime") or f"{target_date.isoformat()} 12:00:00",
            "horse_id": horse_id,
            "horse_name": program.get("horse_name", ""),
            "draw": program.get("draw", 0),
            "weight": weight,
            "distance": distance,
            "field_size": field_size,
            "age": age_value or 0.0,
            "handicap_points": handicap_value or 0.0,
            "odds": odds if pd.notna(odds) else 0.0,
            "market_probability": market_probability,
            "finish_position": int(float(finish)),
            "is_winner": int(float(finish) == 1),
            "is_placed": int(float(finish) <= 3),
            "track": str(program.get("hippodrome", "")).upper(),
            "surface": program.get("surface", ""),
            "track_condition": program.get("track_condition", "NORMAL"),
            "pace_hint": 0.0,
            "style_front_prob": 0.25,
            "style_presser_prob": 0.25,
            "style_stalker_prob": 0.25,
            "style_closer_prob": 0.25,
            "race_class": current_class,
            "jockey_name": program.get("jockey_name", ""),
            "trainer_name": program.get("trainer_name", ""),
            "career_starts": len(history),
            "career_wins": sum(1 for item in history if item["finish_position"] == 1),
            "career_places": sum(
                1 for item in history
                if item["finish_position"] is not None
                and 0 < item["finish_position"] <= 3
            ),
            "last_year_starts": len(last_year_history),
            "last_year_wins": sum(1 for item in last_year_history if item["finish_position"] == 1),
            "last_year_places": sum(
                1 for item in last_year_history
                if item["finish_position"] is not None
                and 0 < item["finish_position"] <= 3
            ),
            "jockey_horse_combo_starts": len(combo_history),
            "jockey_horse_combo_wins": sum(
                1 for item in combo_history if item["finish_position"] == 1
            ),
            "days_since_last_race": days_since_last_race,
            "fatigue_score": fatigue_score,
            "short_rest_flag": float(0 <= days_since_last_race < 8),
            "long_layoff_flag": float(days_since_last_race > 75),
            "race_frequency_3": race_frequency_3,
            "race_frequency_5": race_frequency_5,
            "seasonal_race_load": season_starts / 120.0,
            "history_avg_finish_position": sum(finishes) / len(finishes) if finishes else 0.0,
            "history_avg_weight": sum(history_weights) / len(history_weights) if history_weights else 0.0,
            "history_avg_odds": sum(history_odds) / len(history_odds) if history_odds else 0.0,
            "history_avg_handicap_points": (
                sum(history_handicaps) / len(history_handicaps) if history_handicaps else 0.0
            ),
            "history_avg_race_time_seconds": (
                sum(history_race_times) / len(history_race_times) if history_race_times else 0.0
            ),
            "workout_count": len(past_workouts),
            "workout_avg_distance": (
                sum(workout_distances) / len(workout_distances) if workout_distances else 0.0
            ),
            "workout_avg_speed_index": speed_average,
            "workout_best_speed_index": max(workout_speeds, default=0.0),
            "days_since_last_workout": days_since_workout,
            "history_missing": float(not history),
            "career_summary_missing": float(not finishes),
            "workout_missing": float(not past_workouts),
            "age_missing": float(age_value is None),
            "handicap_missing": float(handicap_value is None),
            "odds_missing": float(odds is None or odds <= 1.0),
            "trainer_starts": trainer_starts,
            "trainer_first_place": int(_number((trainer_stats or {}).get("first_place")) or 0),
            "trainer_second_place": int(_number((trainer_stats or {}).get("second_place")) or 0),
            "trainer_third_place": int(_number((trainer_stats or {}).get("third_place")) or 0),
            "trainer_fourth_place": int(_number((trainer_stats or {}).get("fourth_place")) or 0),
            "trainer_fifth_place": int(_number((trainer_stats or {}).get("fifth_place")) or 0),
            "trainer_win_rate": trainer_win_rate,
            "trainer_top3_rate": trainer_top3_rate,
            "trainer_top5_rate": trainer_top5_rate,
            "trainer_experience_log": math.log1p(trainer_starts),
            "trainer_stats_missing": float(trainer_stats is None),
            "class_drop_flag": class_drop,
            "workout_sudden_improvement": workout_improvement,
            "rest_optimal_fit": rest_optimal_fit,
            "history_longshot_starts": len(longshot_history),
            "history_longshot_place_rate": (
                (longshot_places + 0.9) / (len(longshot_history) + 3.0)
            ),
            "form_avg_3": 0.45,
            "form_avg_5": 0.45,
            "form_avg_10": 0.45,
            "last_run_perf": 0.45,
            "workout_latest_speed_index": workout_speeds[-1] if workout_speeds else 0.0,
            "workout_prior_avg_speed_index": prior_speed_average,
        }
        for column in TJK_STAGE1_FEATURE_COLUMNS:
            row.setdefault(column, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)
