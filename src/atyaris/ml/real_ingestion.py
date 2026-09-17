from __future__ import annotations

from datetime import date, timedelta
from collections.abc import Callable
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.models.entities import HorseStatistics, TrainerStatistics
from atyaris.ml.raw_store import JsonlRawStore
from atyaris.services import build_data_source


logger = logging.getLogger(__name__)


def _track_label(hippodrome: str) -> str:
    value = (hippodrome or "").strip().upper()
    return value.replace("İ", "I")


def _market_probability_from_odds(odds: float | None, default: float) -> float:
    if odds is None or odds <= 1.0:
        return default
    return 1.0 / odds


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "hesaplaniyor"
    rounded = max(0, int(seconds))
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes:
        return f"{minutes} dk {remaining_seconds} sn"
    return f"{remaining_seconds} sn"


def _estimate_remaining(started_at: float, completed: int, total: int) -> str:
    if completed <= 0 or total <= completed:
        return _format_duration(0.0 if total <= completed else None)
    elapsed = monotonic() - started_at
    return _format_duration((elapsed / completed) * (total - completed))


def _append_raw_history_records(path: Path | None, records: list[dict[str, object]]) -> None:
    if path is None or not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[object, ...]] = set()
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                payload = json.loads(line)
                existing_keys.add(
                    (
                        payload.get("target_date"),
                        payload.get("target_race_id"),
                        payload.get("target_horse_id"),
                        payload.get("race_date"),
                        payload.get("finish_position"),
                    )
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ham gecmis JSONL okunamadi; yeni satirlar yine eklenecek: %s", path)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            key = (
                record.get("target_date"),
                record.get("target_race_id"),
                record.get("target_horse_id"),
                record.get("race_date"),
                record.get("finish_position"),
            )
            if key in existing_keys:
                continue
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            existing_keys.add(key)


def ingest_real_tjk_data(
    start_date: date,
    end_date: date,
    paths,
    progress_callback: Callable[[str], None] | None = None,
    require_results: bool = True,
    hippodrome: str | None = None,
    race_no: int | None = None,
) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Build labelled training or unlabelled prediction rows from live TJK pages."""
    source = build_data_source("tjk", __import__("atyaris.config", fromlist=["get_settings"]).get_settings())
    if not isinstance(source, TJKHtmlDataSource):
        raise RuntimeError("ML egitimi icin gercek TJK kaynagi bekleniyor.")

    rows: list[dict[str, object]] = []
    raw_history_records: list[dict[str, object]] = []
    raw_program_records: list[dict[str, object]] = []
    raw_result_records: list[dict[str, object]] = []
    raw_workout_records: list[dict[str, object]] = []
    raw_trainer_records: list[dict[str, object]] = []
    incomplete_workout_records = 0
    raw_history_path = getattr(paths, "raw_history_jsonl", None)
    daily_races: list[tuple[date, object, list[object], dict[int, dict[int, int]]]] = []
    unavailable_days: list[str] = []
    unavailable_result_sets: list[str] = []
    trainer_stats_cache: dict[int, TrainerStatistics | None] = {}
    started_at = monotonic()
    total_days = max((end_date - start_date).days + 1, 1)
    completed_days = 0

    # Collect the complete labelled window before reading horse histories. This
    # keeps result acquisition separate from the historical replay phase.
    current = start_date
    while current <= end_date:
        if progress_callback is not None:
            progress_callback(
                f"Sonuclar aliniyor: {current.isoformat()} "
                f"({completed_days + 1}/{total_days} gun) | "
                f"tahmini kalan: {_estimate_remaining(started_at, completed_days, total_days)}"
            )
        try:
            hippodromes = (
                [hippodrome]
                if hippodrome
                else source.get_available_hippodromes(current)
            )
        except DataSourceError as exc:
            unavailable_days.append(f"{current.isoformat()}: {exc}")
            logger.warning(
                "TJK egitim hatasi | tarih=%s | hipodrom-listesi | hata=%s",
                current.isoformat(),
                exc,
            )
            if progress_callback is not None:
                progress_callback(
                    f"    Uyari: {current.isoformat()} icin hipodrom listesi alinamadi; "
                    f"gun atlanacak ({exc})"
                )
            completed_days += 1
            current += timedelta(days=1)
            continue

        for hippodrome in hippodromes:
            if progress_callback is not None:
                progress_callback(
                    f"Sonuc sayfasi okunuyor: {current.isoformat()} / {hippodrome} | "
                    f"toplanan yarislari: {len(daily_races)} | "
                    f"tahmini kalan: {_estimate_remaining(started_at, completed_days, total_days)}"
                )
            try:
                races = source.get_daily_races(current, hippodrome)
            except DataSourceError as exc:
                unavailable_result_sets.append(f"{current.isoformat()} / {hippodrome}: bulten: {exc}")
                logger.warning(
                    "TJK egitim hatasi | tarih=%s | hipodrom=%s | bulten | hata=%s",
                    current.isoformat(),
                    hippodrome,
                    exc,
                )
                if progress_callback is not None:
                    progress_callback(
                        f"    Uyari: {current.isoformat()} / {hippodrome} bulteni alinamadi; "
                        f"hipodrom atlanacak ({exc})"
                    )
                continue
            if not races:
                continue
            if race_no is not None:
                races = [race for race in races if race.race_no == race_no]
                if not races:
                    continue
            if require_results:
                try:
                    results = source.get_daily_race_results(current, hippodrome)
                except DataSourceError as exc:
                    unavailable_result_sets.append(f"{current.isoformat()} / {hippodrome}: {exc}")
                    logger.warning(
                        "TJK egitim hatasi | tarih=%s | hipodrom=%s | sonuclar | hata=%s",
                        current.isoformat(),
                        hippodrome,
                        exc,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            f"    Uyari: {current.isoformat()} / {hippodrome} sonuclari alinamadi; "
                            f"hipodrom atlanacak ({exc})"
                        )
                    continue
                if not results:
                    unavailable_result_sets.append(f"{current.isoformat()} / {hippodrome}: bos sonuc")
                    logger.warning(
                        "TJK egitim hatasi | tarih=%s | hipodrom=%s | sonuclar bos",
                        current.isoformat(),
                        hippodrome,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            f"    Uyari: {current.isoformat()} / {hippodrome} icin sonuc tablosu bos; "
                            "hipodrom atlanacak"
                        )
                    continue
                missing_races = sorted(
                    {race.race_no for race in races}.difference(results)
                )
                if missing_races:
                    unavailable_result_sets.append(
                        f"{current.isoformat()} / {hippodrome}: eksik kosular {missing_races}"
                    )
                    logger.warning(
                        "TJK egitim uyarisi | tarih=%s | hipodrom=%s | eksik-kosular=%s",
                        current.isoformat(),
                        hippodrome,
                        missing_races,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            f"    Uyari: {current.isoformat()} / {hippodrome} icin "
                            f"eksik kosular atlandi: {missing_races}; tekrar denenecek"
                        )
            else:
                results = {}
            daily_races.append((current, hippodrome, races, results))
        completed_days += 1
        current += timedelta(days=1)

    total_entries = sum(len(race.entries) for _, _, races, _ in daily_races for race in races)
    completed_entries = 0
    replay_started_at = monotonic()

    for current, hippodrome, races, results in daily_races:
            for race in races:
                race_result = results.get(race.race_no, {})
                field_size = len(race.entries)
                if field_size <= 0:
                    continue

                for entry in race.entries:
                    completed_entries += 1
                    if progress_callback is not None:
                        progress_callback(
                            f"Tarihsel istatistik hesaplaniyor: {current.isoformat()} / "
                            f"{hippodrome} / {race.race_no}. kosu / {entry.horse_name} | "
                            f"gunluk at: {completed_entries}/{total_entries} | "
                            f"gunluk uretilen satir: {len(rows)} | "
                            f"tahmini kalan: {_estimate_remaining(replay_started_at, completed_entries, total_entries)}"
                        )
                    finish_position = None if not require_results else entry.actual_finish_position
                    if finish_position is None and require_results:
                        finish_position = race_result.get(entry.number)
                    if require_results and finish_position is None:
                        # Without a real race result, this row cannot be used for
                        # academic training/evaluation without inventing labels.
                        continue

                    raw_program_records.append(
                        {
                            "race_id": str(race.id),
                            "horse_id": str(entry.horse_id),
                            "race_date": current.isoformat(),
                            "race_datetime": race.start_time.isoformat() if race.start_time else None,
                            "hippodrome": str(hippodrome),
                            "race_no": race.race_no,
                            "field_size": len(race.entries),
                            "horse_name": entry.horse_name,
                            "draw": entry.number,
                            "age": entry.age,
                            "jockey_id": getattr(entry.jockey, "source_jockey_id", None),
                            "jockey_name": entry.jockey.name,
                            "trainer_id": entry.trainer.source_trainer_id,
                            "trainer_name": entry.trainer.name,
                            "weight_kg": entry.weight_kg,
                            "handicap_points": entry.handicap_points,
                            "odds": entry.odds,
                            "distance_m": race.distance_m,
                            "surface": race.surface.value if hasattr(race.surface, "value") else str(race.surface),
                        }
                    )
                    raw_result_records.append(
                        {
                            "race_id": str(race.id),
                            "horse_id": str(entry.horse_id),
                            "race_date": current.isoformat(),
                            "hippodrome": str(hippodrome),
                            "race_no": race.race_no,
                            "horse_number": entry.number,
                            "horse_name": entry.horse_name,
                            "finish_position": finish_position,
                        }
                    )

                    try:
                        stats = source.get_horse_statistics(entry, include_workouts=True)
                    except DataSourceError as exc:
                        # TJK may omit an individual horse history while still
                        # exposing the race and its official result. Keep the
                        # labelled row and let feature defaults represent the
                        # missing history instead of aborting the whole run.
                        stats = HorseStatistics(
                            horse_id=str(entry.horse_id),
                            horse_name=str(entry.horse_name),
                        )
                        logger.warning(
                            "TJK egitim uyarisi | tarih=%s | hipodrom=%s | at=%s | gecmis yok | hata=%s",
                            current.isoformat(),
                            hippodrome,
                            entry.horse_name,
                            exc,
                        )
                        if progress_callback is not None:
                            progress_callback(
                                f"    Uyari: {entry.horse_name} gecmisi alinamadi; varsayilan istatistik kullaniliyor ({exc})"
                            )
                    retrieved_at = datetime.now(timezone.utc).isoformat()
                    for performance in stats.past_performances:
                        raw_history_records.append(
                            {
                                "retrieved_at": retrieved_at,
                                "horse_id": str(entry.horse_id),
                                "race_id": f"{entry.horse_id}|{performance.race_date.isoformat()}|{performance.hippodrome}|{race.race_no}",
                                "race_date": performance.race_date.isoformat(),
                                "target_date": current.isoformat(),
                                "target_hippodrome": hippodrome,
                                "target_race_no": race.race_no,
                                "target_race_id": str(race.id),
                                "target_horse_id": str(entry.horse_id),
                                "target_horse_name": entry.horse_name,
                                "source_horse_id": entry.source_horse_id,
                                **performance.model_dump(mode="json"),
                            }
                        )
                    trainer_statistics = None
                    trainer_id = entry.trainer.source_trainer_id
                    if trainer_id is not None:
                        if trainer_id not in trainer_stats_cache:
                            try:
                                trainer_stats_cache[trainer_id] = source.get_trainer_statistics(trainer_id)
                            except DataSourceError as exc:
                                trainer_stats_cache[trainer_id] = None
                                logger.warning(
                                    "TJK antrenor istatistigi alinamadi | antrenor=%s | id=%s | hata=%s",
                                    entry.trainer.name,
                                    trainer_id,
                                    exc,
                                )
                        trainer_statistics = trainer_stats_cache.get(trainer_id)
                    if trainer_statistics is not None:
                        raw_trainer_records.append(
                            {
                                "trainer_id": str(trainer_statistics.trainer_id),
                                "as_of_date": current.isoformat(),
                                "trainer_name": trainer_statistics.trainer_name,
                                "total_starts": trainer_statistics.total_starts,
                                "first_place": trainer_statistics.first_place,
                                "second_place": trainer_statistics.second_place,
                                "third_place": trainer_statistics.third_place,
                                "fourth_place": trainer_statistics.fourth_place,
                                "fifth_place": trainer_statistics.fifth_place,
                                "first_rate": trainer_statistics.first_rate,
                                "second_rate": trainer_statistics.second_rate,
                                "third_rate": trainer_statistics.third_rate,
                                "fourth_rate": trainer_statistics.fourth_rate,
                                "fifth_rate": trainer_statistics.fifth_rate,
                                "retrieved_at": retrieved_at,
                            }
                        )
                    for workout in stats.workout_records:
                        workout_date = (
                            workout.workout_date.isoformat() if workout.workout_date else None
                        )
                        if (
                            workout_date is None
                            or workout.distance_m is None
                            or workout.time_seconds is None
                            or workout.detail is None
                        ):
                            incomplete_workout_records += 1
                            logger.warning(
                                "Eksik anahtar alanli workout kaydi atlandi: horse_id=%s, "
                                "date=%s, distance=%s, time=%s, detail=%s",
                                entry.horse_id,
                                workout_date,
                                workout.distance_m,
                                workout.time_seconds,
                                workout.detail,
                            )
                            continue
                        raw_workout_records.append(
                            {
                                "horse_id": str(entry.horse_id),
                                "workout_date": workout_date,
                                "distance_m": workout.distance_m,
                                "time_seconds": workout.time_seconds,
                                "detail": workout.detail,
                                "hippodrome": workout.hippodrome,
                                "surface": workout.surface,
                                "workout_type": workout.workout_type,
                                "workout_jockey": workout.workout_jockey,
                                "status": workout.status,
                                "ranking_status": workout.ranking_status,
                            }
                        )

                    trainer_starts = trainer_statistics.total_starts if trainer_statistics else 0
                    # Small samples are shrunk toward conservative field priors.
                    trainer_strength = min(max(trainer_starts, 0), 1000) / (min(max(trainer_starts, 0), 1000) + 30.0)
                    trainer_win_rate = (
                        (trainer_statistics.first_rate / 100.0) * trainer_strength
                        + 0.10 * (1.0 - trainer_strength)
                        if trainer_statistics else 0.0
                    )
                    trainer_top3_rate = (
                        ((trainer_statistics.first_rate + trainer_statistics.second_rate + trainer_statistics.third_rate) / 100.0)
                        * trainer_strength
                        + 0.30 * (1.0 - trainer_strength)
                        if trainer_statistics else 0.0
                    )
                    trainer_top5_rate = (
                        ((trainer_statistics.first_rate + trainer_statistics.second_rate + trainer_statistics.third_rate + trainer_statistics.fourth_rate + trainer_statistics.fifth_rate) / 100.0)
                        * trainer_strength
                        + 0.50 * (1.0 - trainer_strength)
                        if trainer_statistics else 0.0
                    )
                    history = [
                        p
                        for p in stats.past_performances
                        if p.race_date < current
                    ]
                    history.sort(key=lambda p: p.race_date)
                    current_jockey = entry.jockey.name.strip().casefold()
                    jockey_horse_combo_wins = sum(
                        1
                        for performance in history
                        if (performance.jockey_name or "").strip().casefold() == current_jockey
                        and performance.finish_position == 1
                    )
                    historical_workouts = [
                        workout
                        for workout in stats.workout_records
                        if workout.workout_date is not None
                        and workout.workout_date < current
                        and workout.distance_m is not None
                        and abs(float(workout.distance_m) - float(race.distance_m)) <= 200.0
                    ]

                    def _mean(values: list[float]) -> float:
                        return float(sum(values) / len(values)) if values else 0.0

                    history_finish = [float(p.finish_position) for p in history if p.finish_position is not None]
                    history_field_sizes = [float(p.field_size) for p in history if p.field_size is not None]
                    history_weights = [float(p.weight_kg) for p in history if p.weight_kg is not None]
                    history_odds = [float(p.odds) for p in history if p.odds is not None]
                    history_handicap = [
                        float(p.handicap_points) for p in history if p.handicap_points is not None
                    ]
                    history_times = [
                        float(p.race_time_seconds) for p in history if p.race_time_seconds is not None
                    ]
                    history_prizes = [
                        float(value)
                        for p in history
                        for value in [pd.to_numeric(p.prize_info, errors="coerce")]
                        if pd.notna(value)
                    ]
                    history_s20 = [
                        float(value)
                        for p in history
                        for value in [pd.to_numeric(p.s20, errors="coerce")]
                        if pd.notna(value)
                    ]
                    workout_times = [
                        float(workout.time_seconds)
                        for workout in historical_workouts
                        if workout.time_seconds is not None
                    ]
                    workout_distances = [
                        float(workout.distance_m)
                        for workout in historical_workouts
                        if workout.distance_m is not None
                    ]
                    workout_speed_indices = [
                        100.0 / (float(workout.time_seconds) / (float(workout.distance_m) / 100.0))
                        for workout in historical_workouts
                        if workout.time_seconds is not None
                        and workout.distance_m is not None
                        and float(workout.time_seconds) > 0.0
                        and float(workout.distance_m) > 0.0
                    ]
                    workout_dates = [
                        workout.workout_date
                        for workout in historical_workouts
                    ]
                    career_wins = sum(1 for p in history if p.finish_position == 1)
                    career_places = sum(
                        1 for p in history if p.finish_position is not None and p.finish_position <= 3
                    )
                    last_year_history = [
                        p for p in history if (current - p.race_date).days <= 365
                    ]

                    perf_hist = [
                        1.0 - ((float(p.finish_position) - 1.0) / max(float(p.field_size or field_size) - 1.0, 1.0))
                        for p in history
                        if p.finish_position is not None
                    ]
                    recent_3 = perf_hist[-3:] if perf_hist else []
                    recent_5 = perf_hist[-5:] if perf_hist else []
                    recent_10 = perf_hist[-10:] if perf_hist else []

                    days_since_last = (
                        float((current - history[-1].race_date).days) if history else 30.0
                    )
                    last_distance = float(history[-1].distance_m) if history else float(race.distance_m)
                    distance_shock = abs(float(race.distance_m) - last_distance) / max(float(race.distance_m), 1.0)
                    last_perf = recent_3[-1] if recent_3 else 0.45
                    short_rest_flag = 1.0 if days_since_last < 8 else 0.0
                    long_layoff_flag = 1.0 if days_since_last > 75 else 0.0
                    seasonal_load = float(
                        sum(1 for p in history if (current - p.race_date).days <= 120)
                    ) / 120.0
                    race_frequency_3 = 3.0 / max(days_since_last, 1.0)
                    race_frequency_5 = 5.0 / max(days_since_last, 1.0)

                    fatigue_load = (
                        1.8 * short_rest_flag
                        + 0.6 * race_frequency_3
                        + 0.4 * race_frequency_5
                        + 0.8 * seasonal_load
                        + 0.5 * distance_shock
                        + 0.5 * (1.0 - last_perf)
                        + 0.4 * long_layoff_flag
                    )
                    recovery_score = float(1.0 / (1.0 + np.exp(fatigue_load - 1.5)))
                    fatigue_score = float(1.0 / (1.0 + np.exp(-(days_since_last - 18.0) / 8.0)))

                    pace_values = [
                        float(p.early_pace_index)
                        for p in history
                        if p.early_pace_index is not None
                    ]
                    pace_hint = float(pace_values[-1]) if pace_values else 0.0
                    style_front_prob = (
                        float(sum(1 for p in pace_values if p >= 0.6)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )
                    style_presser_prob = (
                        float(sum(1 for p in pace_values if 0.2 <= p < 0.6)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )
                    style_stalker_prob = (
                        float(sum(1 for p in pace_values if -0.25 <= p < 0.2)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )
                    style_closer_prob = (
                        float(sum(1 for p in pace_values if p < -0.25)) / len(pace_values)
                        if pace_values
                        else 0.25
                    )

                    surface = race.surface.value if hasattr(race.surface, "value") else str(race.surface)
                    surface_fit = (
                        float(
                            sum(
                                perf
                                for perf, p in zip(perf_hist, history)
                                if (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == surface
                            )
                            / max(
                                1,
                                sum(
                                    1
                                    for p in history
                                    if (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == surface
                                ),
                            )
                        )
                        if history
                        else 0.45
                    )
                    track_fit = (
                        float(
                            sum(
                                perf
                                for perf, p in zip(perf_hist, history)
                                if str(p.hippodrome) == str(race.hippodrome)
                            )
                            / max(1, sum(1 for p in history if str(p.hippodrome) == str(race.hippodrome)))
                        )
                        if history
                        else 0.45
                    )
                    condition_fit = 0.45
                    distance_fit = (
                        float(
                            sum(
                                perf
                                for perf, p in zip(perf_hist, history)
                                if abs(float(p.distance_m) - float(race.distance_m)) <= 200.0
                            )
                            / max(
                                1,
                                sum(
                                    1
                                    for p in history
                                    if abs(float(p.distance_m) - float(race.distance_m)) <= 200.0
                                ),
                            )
                        )
                        if history
                        else 0.45
                    )

                    market_probability = _market_probability_from_odds(entry.odds, 1.0 / max(field_size, 1))
                    implied_probability = market_probability

                    rows.append(
                        {
                            "race_id": str(race.id),
                            "date": current,
                            "race_datetime": race.start_time,
                            "horse_id": str(entry.horse_id),
                            "horse_name": str(entry.horse_name),
                            "draw": int(entry.number),
                            "weight": float(entry.weight_kg),
                            "distance": float(race.distance_m),
                            "field_size": float(field_size),
                            "age": float(entry.age) if entry.age is not None else 0.0,
                            "handicap_points": float(entry.handicap_points) if entry.handicap_points is not None else 0.0,
                            "form_avg_3": float(sum(recent_3) / len(recent_3)) if recent_3 else 0.45,
                            "form_avg_5": float(sum(recent_5) / len(recent_5)) if recent_5 else 0.45,
                            "form_avg_10": float(sum(recent_10) / len(recent_10)) if recent_10 else 0.45,
                            "form_var_5": float(pd.Series(recent_5).var()) if len(recent_5) >= 2 else 0.03,
                            "last_run_perf": float(last_perf),
                            "trend_3_10": (
                                float(sum(recent_3) / len(recent_3)) if recent_3 else 0.45
                            )
                            - (
                                float(sum(recent_10) / len(recent_10)) if recent_10 else 0.45
                            ),
                            "days_since_last_race": days_since_last,
                            "fatigue_score": fatigue_score,
                            "recovery_score": recovery_score,
                            "short_rest_flag": short_rest_flag,
                            "long_layoff_flag": long_layoff_flag,
                            "race_frequency_3": race_frequency_3,
                            "race_frequency_5": race_frequency_5,
                            "seasonal_race_load": seasonal_load,
                            "pace_hint": pace_hint,
                            "style_front_prob": style_front_prob,
                            "style_presser_prob": style_presser_prob,
                            "style_stalker_prob": style_stalker_prob,
                            "style_closer_prob": style_closer_prob,
                            "distance_fit": distance_fit,
                            "surface_fit": surface_fit,
                            "track_fit": track_fit,
                            "career_starts": float(len(history)),
                            "career_wins": float(career_wins),
                            "career_places": float(career_places),
                            "last_year_starts": float(len(last_year_history)),
                            "last_year_wins": float(sum(1 for p in last_year_history if p.finish_position == 1)),
                            "last_year_places": float(
                                sum(
                                    1
                                    for p in last_year_history
                                    if p.finish_position is not None and p.finish_position <= 3
                                )
                            ),
                            "jockey_horse_combo_starts": float(stats.jockey_horse_combo_starts),
                            "jockey_horse_combo_wins": float(jockey_horse_combo_wins),
                            "trainer_starts": float(trainer_starts),
                            "trainer_first_place": float(trainer_statistics.first_place if trainer_statistics else 0),
                            "trainer_second_place": float(trainer_statistics.second_place if trainer_statistics else 0),
                            "trainer_third_place": float(trainer_statistics.third_place if trainer_statistics else 0),
                            "trainer_fourth_place": float(trainer_statistics.fourth_place if trainer_statistics else 0),
                            "trainer_fifth_place": float(trainer_statistics.fifth_place if trainer_statistics else 0),
                            "trainer_win_rate": float(trainer_win_rate),
                            "trainer_top3_rate": float(trainer_top3_rate),
                            "trainer_top5_rate": float(trainer_top5_rate),
                            "trainer_experience_log": float(np.log1p(trainer_starts)),
                            "trainer_stats_missing": float(trainer_statistics is None),
                            "history_avg_finish_position": _mean(history_finish),
                            "history_avg_field_size": _mean(history_field_sizes),
                            "history_avg_weight": _mean(history_weights),
                            "history_avg_odds": _mean(history_odds),
                            "history_avg_handicap_points": _mean(history_handicap),
                            "history_avg_race_time_seconds": _mean(history_times),
                            "history_avg_prize": _mean(history_prizes),
                            "history_avg_s20": _mean(history_s20),
                            "workout_count": float(len(historical_workouts)),
                            "workout_avg_time_seconds": _mean(workout_times),
                            "workout_best_time_seconds": min(workout_times) if workout_times else 0.0,
                            "workout_avg_distance": _mean(workout_distances),
                            "workout_avg_speed_index": _mean(workout_speed_indices),
                            "workout_best_speed_index": max(workout_speed_indices) if workout_speed_indices else 0.0,
                            "days_since_last_workout": float((current - max(workout_dates)).days)
                            if workout_dates else 0.0,
                            "history_missing": float(not history),
                            "career_summary_missing": float(not history),
                            "workout_missing": float(not historical_workouts),
                            "age_missing": float(entry.age is None),
                            "handicap_missing": float(entry.handicap_points is None),
                            "odds_missing": float(entry.odds is None or entry.odds <= 1.0),
                            "market_probability": market_probability,
                            "implied_probability": implied_probability,
                            "odds": float(entry.odds) if entry.odds is not None else 0.0,
                            "is_winner": int(finish_position == 1),
                            "track": _track_label(race.hippodrome),
                            "surface": surface,
                            "track_condition": "NORMAL",
                        }
                    )

    if raw_history_path is not None and raw_history_records:
        JsonlRawStore(raw_history_path, "horse_history").upsert(raw_history_records)
    raw_stores = [
        (getattr(paths, "raw_daily_program_jsonl", None), "daily_program", raw_program_records),
        (getattr(paths, "raw_race_results_jsonl", None), "race_results", raw_result_records),
        (getattr(paths, "raw_workouts_jsonl", None), "workouts", raw_workout_records),
        (getattr(paths, "raw_trainer_statistics_jsonl", None), "trainer_statistics", raw_trainer_records),
    ]
    for raw_path, collection, records in raw_stores:
        if raw_path is not None and records:
            JsonlRawStore(raw_path, collection).upsert(records)
    frame = pd.DataFrame(rows)
    if frame.empty:
        details = []
        if unavailable_days:
            details.append(f"veri alinamayan gun sayisi: {len(unavailable_days)}")
        if unavailable_result_sets:
            details.append(f"sonuc alinamayan hipodrom sayisi: {len(unavailable_result_sets)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        purpose = "ML egitimi" if require_results else "ML tahmini"
        if require_results and (unavailable_days or unavailable_result_sets):
            frame.attrs["retry_targets"] = [*unavailable_days, *unavailable_result_sets]
            frame.attrs["incomplete_workout_records"] = incomplete_workout_records
            return frame
        raise RuntimeError(
            f"Gercek TJK verisiyle {purpose} icin kullanilabilir satir bulunamadi. "
            f"Bulten/sonuc/at gecmisi scraping dogrulanmali{suffix}."
        )

    required_columns = {
        "race_id",
        "date",
        "race_datetime",
        "horse_id",
        "draw",
        "weight",
        "distance",
        "field_size",
        "market_probability",
        "implied_probability",
        "odds",
        "is_winner",
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Gercek TJK veri cercevesinde eksik zorunlu kolonlar var: {missing}")

    if unavailable_days or unavailable_result_sets:
        frame.attrs["retry_targets"] = [*unavailable_days, *unavailable_result_sets]
    frame.attrs["incomplete_workout_records"] = incomplete_workout_records
    return frame
