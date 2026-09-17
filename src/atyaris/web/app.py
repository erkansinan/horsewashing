"""FastAPI tabanli basit HTML arayuzu.

``atyaris web`` komutuyla baslatilir (bkz. ``atyaris/cli/app.py``). Ayni
``atyaris.services`` yardimci fonksiyonlarini ve ``PredictionEngine``'i CLI
ile paylasir; boylece iki arayuz arasinda is mantigi tekrarlanmaz.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
import json
from threading import Event, RLock, Thread, Timer
from pathlib import Path
import re
from types import SimpleNamespace
from time import monotonic
from urllib.parse import urlencode
from uuid import uuid4
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from atyaris.config import get_settings
from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import KNOWN_HIPPODROMES, TJKHtmlDataSource
from atyaris.ml.pipeline import (
    build_features,
    ingest_real_data,
    MIN_TRAINING_DATES,
    optimize_for_date,
    paths_from_settings,
    prepare_prediction_features,
    predict_for_date,
    preprocess_raw,
    run_phase1_backtest,
    train_phase1_model,
)
from atyaris.ml.real_ingestion import ingest_real_tjk_data
from atyaris.ml.raw_store import JsonlRawStore
from atyaris.ml.local_pipeline import build_local_raw_frame
from atyaris.ml.features import TJK_SELECTED_STAGE1_FEATURE_COLUMNS
from atyaris.ml.explainability import compute_optional_shap_summary, compute_permutation_importance
from atyaris.ml.modeling import load_phase3_artifact
from atyaris.ml.phase5 import register_training_run
from atyaris.ml.tracking import load_training_history
from atyaris.prediction.engine import PredictionEngine
from atyaris.services import InvalidSourceError, build_data_source, fetch_races, parse_date
from atyaris.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

_SORTABLE_FIELDS = {
    "strategy": "Stratejik Sira",
    "number": "No",
    "odds": "Ganyan",
    "total": "Toplam",
    "form": "Form",
    "jockey_trainer": "Jokey",
    "distance_surface": "Pist",
    "weight": "Agirlik",
    "rest": "Dinlenme",
    "win_probability": "Kazanma Olasiligi",
    "confidence": "Guven",
    "ev": "EV",
    "kelly": "Kelly",
}

_ML_SORTABLE_FIELDS = {
    "rank": "Rank",
    "horse_id": "At ID",
    "odds": "Ganyan",
    "calibrated_probability": "P(win)",
    "place2_probability": "P(2.)",
    "place3_probability": "P(3.)",
    "top3_probability": "P(Top3)",
    "confidence": "Guven",
    "edge": "Edge",
    "ev": "EV",
    "kelly_fraction": "Kelly",
    "form_strength": "TJK Veri Gucu",
}

_TRACK_DISPLAY_MAP = {
    "ANKARA": "Ankara",
    "ISTANBUL": "İstanbul",
    "IZMIR": "İzmir",
}

_TR_ASCII_MAP = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
_TRAINING_LOCK = RLock()
_TRAINING_JOBS: dict[str, dict[str, object]] = {}
_TRAINING_CANCEL_EVENTS: dict[str, Event] = {}
_TRAINING_PAUSE_REQUESTS: set[str] = set()
_TRAINING_RETRY_TIMERS: dict[str, Timer] = {}
_COLLECTION_JOBS: dict[str, dict[str, object]] = {}
_COLLECTION_CANCEL_EVENTS: dict[str, Event] = {}
_COLLECTION_PAUSE_REQUESTS: set[str] = set()


class _TrainingCancelled(Exception):
    pass


class _TrainingPaused(Exception):
    pass


def _training_jobs_manifest_path() -> Path:
    settings = get_settings()
    return paths_from_settings(settings).raw_csv.with_suffix(".jobs.json")


def _collection_jobs_manifest_path() -> Path:
    settings = get_settings()
    return paths_from_settings(settings).raw_csv.with_suffix(".collection.jobs.json")


def _persist_training_jobs() -> None:
    """Persist resumable jobs without serializing transient thread state."""
    path = _training_jobs_manifest_path()
    with _TRAINING_LOCK:
        jobs = [
            {
                "job_id": job_id,
                "status": str(job.get("status")),
                "message": str(job.get("message", "")),
                "retry_targets": list(job.get("retry_targets", [])),
                "can_accept_partial": bool(job.get("can_accept_partial", False)),
                "start_date": str(job.get("start_date", "")),
                "end_date": str(job.get("end_date", "")),
            }
            for job_id, job in _TRAINING_JOBS.items()
            if job.get("status") in {"running", "paused", "awaiting_decision", "retrying", "completed_with_warnings"}
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jobs, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _persist_collection_jobs() -> None:
    path = _collection_jobs_manifest_path()
    with _TRAINING_LOCK:
        jobs = [
            {"job_id": job_id, **job}
            for job_id, job in _COLLECTION_JOBS.items()
            if job.get("status") in {"running", "paused", "completed", "failed"}
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jobs, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _restore_training_jobs() -> None:
    """Recover interrupted jobs as paused so the user can resume them safely."""
    path = _training_jobs_manifest_path()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Training job manifest could not be read: %s", path)
        return
    if not isinstance(payload, list):
        return
    restored_warning_jobs: list[str] = []
    with _TRAINING_LOCK:
        for item in payload:
            if not isinstance(item, dict):
                continue
            job_id = str(item.get("job_id", "")).strip()
            start_date = str(item.get("start_date", "")).strip()
            end_date = str(item.get("end_date", "")).strip()
            if not job_id or not start_date or not end_date:
                continue
            try:
                date.fromisoformat(start_date)
                date.fromisoformat(end_date)
            except ValueError:
                continue
            _TRAINING_JOBS.setdefault(
                job_id,
                {
                    "status": (
                        "retrying"
                        if item.get("status") == "running" and item.get("retry_targets")
                        else "paused"
                        if item.get("status") == "running"
                        else "retrying"
                        if item.get("status") == "awaiting_decision"
                        else str(item.get("status", "paused"))
                    ),
                    "message": (
                        "Atlanan veri parcalari icin otomatik tekrar denemesi bekleniyor"
                        if item.get("status") == "running" and item.get("retry_targets")
                        else "Sunucu yeniden baslatildi; checkpointten devam edilebilir"
                        if item.get("status") == "running"
                        else "Atlanan veri parcalari icin otomatik tekrar denemesi bekleniyor"
                        if item.get("status") == "awaiting_decision"
                        else str(item.get("message", ""))
                    ),
                    "retry_targets": list(item.get("retry_targets", [])),
                    "can_accept_partial": bool(item.get("can_accept_partial", False)),
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            if item.get("status") in {"running", "awaiting_decision", "retrying", "completed_with_warnings"}:
                restored_warning_jobs.append(job_id)
    for job_id in restored_warning_jobs:
        _schedule_skipped_training_retry(job_id)


def _active_training_job_id() -> str:
    with _TRAINING_LOCK:
        return next(
            (job_id for job_id, job in _TRAINING_JOBS.items() if job.get("status") == "running"),
            "",
        )


def _start_retry_skipped_training_locked(job_id: str) -> None:
    """Start another skipped-data attempt; caller must hold _TRAINING_LOCK."""
    job = _TRAINING_JOBS.get(job_id)
    if job is None or job.get("status") not in {"awaiting_decision", "retrying", "completed_with_warnings"}:
        return
    timer = _TRAINING_RETRY_TIMERS.pop(job_id, None)
    if timer is not None:
        timer.cancel()
    job["status"] = "running"
    job["message"] = "Atlanan veri parcalari tekrar deneniyor"
    _TRAINING_CANCEL_EVENTS[job_id] = Event()
    Thread(
        target=_run_training_job,
        args=(
            job_id,
            date.fromisoformat(str(job["start_date"])),
            date.fromisoformat(str(job["end_date"])),
        ),
        name=f"atyaris-train-{job_id[:8]}",
        daemon=True,
    ).start()
    _persist_training_jobs()


def _schedule_skipped_training_retry(job_id: str) -> None:
    """Retry warning-completed training in 30 seconds unless it was cancelled."""
    def retry() -> None:
        with _TRAINING_LOCK:
            _TRAINING_RETRY_TIMERS.pop(job_id, None)
            _start_retry_skipped_training_locked(job_id)

    with _TRAINING_LOCK:
        existing = _TRAINING_RETRY_TIMERS.pop(job_id, None)
        if existing is not None:
            existing.cancel()
        timer = Timer(30.0, retry)
        timer.daemon = True
        _TRAINING_RETRY_TIMERS[job_id] = timer
        timer.start()


def _restore_collection_jobs() -> None:
    path = _collection_jobs_manifest_path()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Collection job manifest could not be read: %s", path)
        return
    if not isinstance(payload, list):
        return
    restored_running = False
    with _TRAINING_LOCK:
        for item in payload:
            if not isinstance(item, dict):
                continue
            job_id = str(item.get("job_id", "")).strip()
            if not job_id or not item.get("start_date") or not item.get("end_date"):
                continue
            restored = dict(item)
            restored.pop("job_id", None)
            if restored.get("status") == "running":
                restored["status"] = "paused"
                restored["message"] = "Sunucu yeniden baslatildi; checkpointten devam edilebilir"
                restored["pause_requested"] = False
                restored_running = True
            _COLLECTION_JOBS[job_id] = restored
            _COLLECTION_CANCEL_EVENTS[job_id] = Event()
            if restored.get("status") == "paused":
                _COLLECTION_CANCEL_EVENTS[job_id].set()
    if restored_running:
        _persist_collection_jobs()


def _run_collection_job(job_id: str, start_date: date, end_date: date) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    cancel_event = _COLLECTION_CANCEL_EVENTS[job_id]
    total_days = max((end_date - start_date).days + 1, 1)
    collection_checkpoint = paths.raw_csv.with_suffix(".collection.progress.json")
    completed_units: set[str] = set()
    failed_units: set[str] = set()
    if collection_checkpoint.exists():
        try:
            payload = json.loads(collection_checkpoint.read_text(encoding="utf-8"))
            if (
                payload.get("start_date") == start_date.isoformat()
                and payload.get("end_date") == end_date.isoformat()
            ):
                completed_units = {str(value) for value in payload.get("completed_units", [])}
                failed_units = {str(value) for value in payload.get("failed_units", [])}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            completed_units = set()

    def write_checkpoint() -> None:
        payload = {
            "collection_version": 1,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "completed_units": sorted(completed_units),
            "failed_units": sorted(failed_units),
        }
        temporary = collection_checkpoint.with_suffix(collection_checkpoint.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        temporary.replace(collection_checkpoint)

    def append_frame(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        existing = pd.read_csv(paths.raw_csv) if paths.raw_csv.exists() else pd.DataFrame()
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
        keys = [column for column in ["date", "race_id", "horse_id", "draw"] if column in combined.columns]
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="last")
        temporary = paths.raw_csv.with_suffix(paths.raw_csv.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(temporary, index=False)
        temporary.replace(paths.raw_csv)

    source = _bulletin_source_for_ml(settings)
    failed_targets = 0
    skipped_days = 0
    missing_data_records = 0
    completed_days: set[date] = {
        date.fromisoformat(unit.split("|", 1)[0])
        for unit in completed_units
        if "|" in unit
    }
    started_at = monotonic()
    processed_units = len(completed_units) + len(failed_units)
    observed_units_by_day: dict[date, int] = {}
    active_collection_date = end_date

    def check_collection_control() -> None:
        with _TRAINING_LOCK:
            job = _COLLECTION_JOBS.get(job_id)
            paused = job is not None and job.get("status") == "paused"
            pause_requested = job_id in _COLLECTION_PAUSE_REQUESTS
        if paused or pause_requested:
            raise _TrainingPaused()
        if cancel_event.is_set() or job is None or job.get("status") != "running":
            raise _TrainingCancelled()

    def progress(message: str) -> None:
        nonlocal processed_units
        check_collection_control()
        with _TRAINING_LOCK:
            job = _COLLECTION_JOBS.get(job_id)
            if job is not None:
                job["message"] = message
                job["completed_days"] = len(completed_days)
                job["remaining_days"] = max(total_days - len(completed_days), 0)
                elapsed = monotonic() - started_at
                units_per_second = processed_units / elapsed if processed_units and elapsed > 0 else 0.0
                completed_day_units = [
                    count for day, count in observed_units_by_day.items()
                    if day in completed_days and count > 0
                ]
                average_units_per_day = (
                    sum(completed_day_units) / len(completed_day_units)
                    if completed_day_units
                    else max(observed_units_by_day.get(active_collection_date, 1), 1)
                )
                remaining_units = max(
                    (total_days * average_units_per_day) - processed_units,
                    0,
                )
                job["estimated_remaining"] = (
                    f"{max(remaining_units / units_per_second / 60.0, 0.1):.1f} dk"
                    if units_per_second > 0 and remaining_units > 0
                    else f"{max(elapsed / 60.0, 0.1):.1f} dk; ilk tamamlanan yaris bekleniyor"
                )
                job["current_date"] = message[:10] if len(message) >= 10 else ""
                job["current_target"] = message
                _persist_collection_jobs()

    try:
        for current in (end_date - timedelta(days=offset) for offset in range(total_days)):
            active_collection_date = current
            check_collection_control()
            progress(f"Hipodromlar aliniyor: {current.isoformat()}")
            try:
                hippodromes = source.get_available_hippodromes(current)
            except DataSourceError as exc:
                skipped_days += 1
                failed_targets += 1
                progress(f"Gun atlandi: {current.isoformat()} | {exc}")
                continue
            day_units = 0
            for hippodrome in hippodromes:
                check_collection_control()
                progress(f"Hipodrom okunuyor: {current.isoformat()} / {hippodrome}")
                try:
                    races = source.get_daily_races(current, hippodrome)
                except DataSourceError as exc:
                    failed_targets += 1
                    progress(f"Hipodrom atlandi: {current.isoformat()} / {hippodrome} | {exc}")
                    continue
                for race in races:
                    check_collection_control()
                    unit = f"{current.isoformat()}|{hippodrome}|{race.race_no}"
                    day_units += 1
                    observed_units_by_day[current] = day_units
                    if unit in completed_units:
                        continue
                    progress(f"Yaris aliniyor: {unit}")
                    last_error = None
                    succeeded = False
                    for attempt in range(1, 4):
                        check_collection_control()
                        try:
                            race_data = ingest_real_tjk_data(
                                current,
                                current,
                                paths,
                                progress_callback=progress,
                                hippodrome=hippodrome,
                                race_no=race.race_no,
                            )
                            check_collection_control()
                            append_frame(race_data)
                            missing_data_records += int(race_data.attrs.get("incomplete_workout_records", 0))
                            succeeded = True
                            break
                        except (DataSourceError, RuntimeError) as exc:
                            last_error = exc
                            if attempt < 3:
                                progress(f"Tekrar deneme {attempt}/3: {unit}")
                    if not succeeded:
                        failed_targets += 1
                        failed_units.add(unit)
                        progress(f"Yaris basarisiz: {unit} | {last_error}")
                    else:
                        completed_units.add(unit)
                    processed_units += 1
                    write_checkpoint()
            if day_units == 0:
                skipped_days += 1
            else:
                observed_units_by_day[current] = day_units
                completed_days.add(current)
            with _TRAINING_LOCK:
                job = _COLLECTION_JOBS[job_id]
                job["completed_days"] = len(completed_days)
                job["remaining_days"] = max(total_days - len(completed_days), 0)
                job["skipped_days"] = skipped_days
                job["failed_targets"] = failed_targets
                job["missing_data_records"] = missing_data_records
                job["message"] = f"Gun tamamlandi: {current.isoformat()}"
            _persist_collection_jobs()
        with _TRAINING_LOCK:
            _COLLECTION_JOBS[job_id]["status"] = "completed"
            _COLLECTION_JOBS[job_id]["message"] = "Veri toplama tamamlandi"
        _persist_collection_jobs()
    except _TrainingPaused:
        with _TRAINING_LOCK:
            _COLLECTION_PAUSE_REQUESTS.discard(job_id)
            _COLLECTION_JOBS[job_id]["status"] = "paused"
            _COLLECTION_JOBS[job_id]["message"] = "Veri toplama duraklatildi; checkpoint korundu"
        _persist_collection_jobs()
    except _TrainingCancelled:
        with _TRAINING_LOCK:
            _COLLECTION_JOBS[job_id]["status"] = "cancelled"
            _COLLECTION_JOBS[job_id]["message"] = "Veri toplama durduruldu"
        _persist_collection_jobs()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Web collection job failed")
        with _TRAINING_LOCK:
            _COLLECTION_JOBS[job_id]["status"] = "failed"
            _COLLECTION_JOBS[job_id]["message"] = str(exc)
        _persist_collection_jobs()


@dataclass
class MLRaceSummary:
    race_id: str
    race_name: str
    race_time: str
    horse_names: str
    horse_count: int
    top_horse_name: str
    top_probability: float
    value_bet_count: int


def _bulletin_source_for_ml(settings):  # type: ignore[no-untyped-def]
    """ML web flow uses real daily bulletin for date/city/race selection."""
    return build_data_source("tjk", settings)


def _safe_metric(value: float | None, fallback: float) -> float:
    return value if value is not None else fallback


def _safe_float(value, default: float = 0.0) -> float:  # type: ignore[no-untyped-def]
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_placeholder_horse_name(value: object) -> bool:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return True
    if re.fullmatch(r"H\d{4}", text):
        return True
    if re.fullmatch(r"HORSE[_-]?\d+", text):
        return True
    return False


def _position_points(position: int | None, field_size: int | None) -> float:
    if position is None:
        return 30.0
    if position in {1: 100, 2: 80, 3: 65}:
        return {1: 100, 2: 80, 3: 65}[position]
    size = field_size or 10
    return max(5.0, 30.0 * (1 - (position - 3) / max(size - 3, 1)))


def _derive_horse_stats_metrics(stats, race_distance: int | None = None, race_surface: str | None = None, race_track: str | None = None):
    if stats is None:
        return {
            "form_avg_3": "-",
            "form_avg_5": "-",
            "form_avg_10": "-",
            "days_since_last_race": "-",
            "track_fit": "-",
            "surface_fit": "-",
            "distance_fit": "-",
            "pace_pressure": "-",
            "weight": "-",
        }

    performances = sorted(stats.past_performances, key=lambda p: p.race_date, reverse=True)
    recent_3 = performances[:3]
    recent_5 = performances[:5]
    recent_10 = performances[:10]

    def _average_position_score(items):
        if not items:
            return 0.0
        score = sum(_position_points(p.finish_position, p.field_size) for p in items)
        return round(score / len(items), 2)

    form_avg_3 = _average_position_score(recent_3)
    form_avg_5 = _average_position_score(recent_5)
    form_avg_10 = _average_position_score(recent_10)
    days_since_last_race = stats.days_since_last_race if stats.days_since_last_race is not None else "-"

    valid_weights = [float(p.weight_kg) for p in performances if p.weight_kg is not None]
    weight = round(sum(valid_weights) / len(valid_weights), 1) if valid_weights else "-"

    def _matching_score(items):
        if not items:
            return 0.0
        points = [_position_points(p.finish_position, p.field_size) for p in items]
        wins = sum(1 for p in items if p.finish_position == 1)
        return round((sum(points) / len(points)) * 0.7 + (wins / len(items)) * 100.0 * 0.3, 2)

    track_fit = 0.0
    if race_track:
        same_track = [
            p for p in performances
            if str(p.hippodrome).strip().lower() == str(race_track).strip().lower()
        ]
        track_fit = _matching_score(same_track)
    surface_fit = 0.0
    if race_surface:
        same_surface = [
            p for p in performances
            if str(p.surface.value if hasattr(p.surface, "value") else str(p.surface)).strip().lower()
            == str(race_surface).strip().lower()
        ]
        surface_fit = _matching_score(same_surface)
    distance_fit = 0.0
    if race_distance is not None:
        similar_distance = [
            p for p in performances
            if abs(int(p.distance_m) - int(race_distance)) <= 200
        ]
        distance_fit = _matching_score(similar_distance)

    pace_values = [
        float(p.early_pace_index)
        for p in performances[:10]
        if p.early_pace_index is not None
    ]
    workout_times = [
        float(workout.time_seconds)
        for workout in stats.workout_records
        if workout.time_seconds is not None
    ]
    workout_dates = [
        workout.workout_date
        for workout in stats.workout_records
        if workout.workout_date is not None
    ]
    workout_distances = [
        float(workout.distance_m)
        for workout in stats.workout_records
        if workout.distance_m is not None
    ]
    if pace_values:
        front_share = sum(1 for value in pace_values if value >= 0.6) / len(pace_values)
        presser_share = sum(1 for value in pace_values if 0.2 <= value < 0.6) / len(pace_values)
        pace_pressure = max(0.0, min(1.0, front_share + 0.7 * presser_share))
    else:
        workout_pace = []
        matching_workouts = [
            workout
            for workout in stats.workout_records
            if race_distance is not None
            and workout.distance_m is not None
            and abs(float(workout.distance_m) - float(race_distance)) <= 200.0
        ]
        for workout in sorted(
            matching_workouts,
            key=lambda w: (w.workout_date or date.min),
            reverse=True,
        )[:5]:
            if workout.distance_m and workout.time_seconds and workout.distance_m > 0:
                sec_per_100 = workout.time_seconds / max(workout.distance_m / 100.0, 1.0)
                # ivme/tempo sinyali: daha hizli idman daha yuksek pace pressure olarak yorumlanir
                workout_pace.append(max(0.0, min(1.0, 1.0 - ((sec_per_100 - 55.0) / 45.0))))
        pace_pressure = round(sum(workout_pace) / len(workout_pace), 3) if workout_pace else 0.0

    return {
        "form_avg_3": round(form_avg_3, 2),
        "form_avg_5": round(form_avg_5, 2),
        "form_avg_10": round(form_avg_10, 2),
        "days_since_last_race": days_since_last_race,
        "weight": round(weight, 1) if isinstance(weight, float) else weight,
        "track_fit": round(track_fit, 2),
        "surface_fit": round(surface_fit, 2),
        "distance_fit": round(distance_fit, 2),
        "pace_pressure": round(pace_pressure, 3),
        "workout_count": float(len(stats.workout_records)),
        "workout_avg_time_seconds": round(sum(workout_times) / len(workout_times), 2) if workout_times else 0.0,
        "workout_best_time_seconds": min(workout_times) if workout_times else 0.0,
        "workout_avg_distance": round(sum(workout_distances) / len(workout_distances), 2) if workout_distances else 0.0,
        "days_since_last_workout": float((date.today() - max(workout_dates)).days) if workout_dates else 0.0,
    }


def _prediction_sort_key(item, sort_by: str):  # type: ignore[no-untyped-def]
    if sort_by == "strategy":
        win_probability = _safe_metric(getattr(item, "win_probability", None), -1.0)
        confidence_score = _safe_metric(getattr(item, "confidence_score", None), -1.0)
        value_bet = getattr(item, "value_bet", None)
        expected_value = _safe_metric(
            getattr(value_bet, "expected_value", None) if value_bet else None,
            -9999.0,
        )
        kelly_stake = _safe_metric(
            getattr(value_bet, "fractional_kelly_stake", None) if value_bet else None,
            -9999.0,
        )
        score = getattr(item, "score", None)
        total_score = _safe_metric(getattr(score, "total_score", None) if score is not None else None, -1.0)
        return (win_probability, confidence_score, total_score, expected_value, kelly_stake)
    if sort_by == "number":
        return item.entry.number
    if sort_by == "odds":
        return item.entry.odds if item.entry.odds is not None else 9999.0
    if sort_by == "total":
        return item.score.total_score
    if sort_by == "form":
        return item.score.form_score
    if sort_by == "jockey_trainer":
        return item.score.jockey_trainer_score
    if sort_by == "distance_surface":
        return item.score.distance_surface_score
    if sort_by == "weight":
        return item.score.weight_score
    if sort_by == "rest":
        return item.score.rest_score
    if sort_by == "win_probability":
        return _safe_metric(getattr(item, "win_probability", None), -1.0)
    if sort_by == "confidence":
        return _safe_metric(getattr(item, "confidence_score", None), -1.0)
    if sort_by == "ev":
        value_bet = getattr(item, "value_bet", None)
        return _safe_metric(getattr(value_bet, "expected_value", None) if value_bet else None, -9999.0)
    if sort_by == "kelly":
        value_bet = getattr(item, "value_bet", None)
        return _safe_metric(getattr(value_bet, "fractional_kelly_stake", None) if value_bet else None, -9999.0)
    return item.score.total_score


def _ml_sort_value(row, sort_by: str):  # type: ignore[no-untyped-def]
    if sort_by == "rank":
        return int(row.get("rank", 9999))
    if sort_by == "horse_id":
        return str(row.get("horse_id", ""))
    if sort_by == "odds":
        return _safe_float(row.get("odds"), 9999.0)
    if sort_by == "calibrated_probability":
        return _safe_float(row.get("calibrated_probability"), -1.0)
    if sort_by == "place2_probability":
        return _safe_float(row.get("place2_probability"), -1.0)
    if sort_by == "place3_probability":
        return _safe_float(row.get("place3_probability"), -1.0)
    if sort_by == "top3_probability":
        return _safe_float(row.get("top3_probability"), -1.0)
    if sort_by == "confidence":
        return _safe_float(row.get("confidence"), -1.0)
    if sort_by == "edge":
        return _safe_float(row.get("edge"), -9999.0)
    if sort_by == "ev":
        return _safe_float(row.get("ev"), -9999.0)
    if sort_by == "kelly_fraction":
        return _safe_float(row.get("kelly_fraction"), -9999.0)
    if sort_by == "form_strength":
        form_3 = _safe_float(row.get("form_avg_3"), 0.0)
        form_5 = _safe_float(row.get("form_avg_5"), 0.0)
        form_10 = _safe_float(row.get("form_avg_10"), 0.0)
        track_fit = _safe_float(row.get("track_fit"), 0.0)
        surface_fit = _safe_float(row.get("surface_fit"), 0.0)
        distance_fit = _safe_float(row.get("distance_fit"), 0.0)
        career_wins = _safe_float(row.get("career_wins"), 0.0)
        career_starts = max(_safe_float(row.get("career_starts"), 0.0), 1.0)
        workout_count = _safe_float(row.get("workout_count"), 0.0)
        return (
            0.32 * form_5
            + 0.20 * form_10
            + 0.12 * form_3
            + 0.12 * track_fit
            + 0.10 * surface_fit
            + 0.08 * distance_fit
            + 0.04 * (career_wins / career_starts)
            + 0.02 * min(workout_count / 10.0, 1.0)
        )
    return _safe_float(row.get("calibrated_probability"), -1.0)


def _ml_race_summaries(frame: pd.DataFrame) -> list[MLRaceSummary]:
    if frame.empty:
        return []

    rows: list[MLRaceSummary] = []
    grouped = frame.sort_values(["race_id", "rank"]).groupby("race_id", sort=True)
    for race_id, race_df in grouped:
        first = race_df.iloc[0]
        rows.append(
            MLRaceSummary(
                race_id=str(race_id),
                race_name=str(first.get("race_name", race_id)),
                race_time=str(first.get("race_time", "-")),
                horse_names=", ".join(
                    str(name)
                    for name in race_df.get("horse_name", race_df.get("horse_id", "-"))
                    if str(name).strip() and str(name).lower() != "nan"
                ) or "-",
                horse_count=int(len(race_df)),
                top_horse_name=str(first.get("horse_name", first.get("horse_id", "-"))),
                top_probability=float(first.get("calibrated_probability", 0.0)),
                value_bet_count=int((race_df.get("bet_decision", "NO_BET") == "BET").sum()),
            )
        )
    return rows


def _ml_race_summaries_from_races(  # type: ignore[no-untyped-def]
    races: list,
    data_source,
    settings,
) -> list[MLRaceSummary]:
    _ = (data_source, settings)
    rows: list[MLRaceSummary] = []
    for race in sorted(races, key=lambda r: (r.hippodrome, r.race_no)):
        active = [e for e in race.entries if not e.is_scratched]
        # Index sayfasinda yalnizca bulten listesi gosterilir; agir ML tahmini
        # kullanici kosu secince /predict adiminda calistirilir.
        top = "-"
        top_probability = 0.0
        value_bet_count = 0

        rows.append(
            MLRaceSummary(
                race_id=_to_ml_frame_race_id(str(race.id), race.start_time.date()) or str(race.id),
                race_name=f"{race.hippodrome} - {race.race_no}. Kosu",
                race_time=race.start_time.strftime("%H:%M"),
                horse_names=", ".join(
                    str(entry.horse_name)
                    for entry in active
                    if str(entry.horse_name).strip()
                ) or "-",
                horse_count=len(active),
                top_horse_name=top,
                top_probability=top_probability,
                value_bet_count=value_bet_count,
            )
        )
    return rows


def _parse_race_no(race_id: str) -> int | None:
    m = re.search(r"_(\d+)$", race_id)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None

    m = re.search(r"-(\d+)$", race_id)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _to_ml_frame_race_id(race_id: str, target_date: date) -> str | None:
    """Map external race identifiers to synthetic ML frame id format: YYYYMMDD_NN."""
    if re.fullmatch(r"\d{8}_\d{2}", race_id):
        return race_id
    race_no = _parse_race_no(race_id)
    if race_no is None:
        return None
    return f"{target_date.strftime('%Y%m%d')}_{race_no:02d}"


def _display_track_name(track: str) -> str:
    key = _normalize_track_key(track)
    return _TRACK_DISPLAY_MAP.get(key, track)


def _normalize_track_key(value: str) -> str:
    return value.strip().translate(_TR_ASCII_MAP).upper()


def _normalize_name_key(value: object) -> str:
    text = str(value or "").translate(_TR_ASCII_MAP).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _record_number(rec: dict) -> int | None:
    for key in ("number", "draw"):
        raw = rec.get(key)
        try:
            if raw is None or pd.isna(raw):
                continue
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _race_matches_bulletin(records: list[dict], race) -> bool:  # type: ignore[no-untyped-def]
    if not records:
        return False
    expected_numbers = {int(e.number) for e in race.entries}
    if not expected_numbers:
        return False
    observed_numbers = {n for n in (_record_number(r) for r in records) if n is not None}
    if not observed_numbers:
        return False
    return len(records) == len(expected_numbers) and observed_numbers.issubset(expected_numbers)


def _place_probabilities_from_records(records: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    if len(records) < 2:
        return {}, {}
    win_probs = np.asarray([_safe_float(r.get("calibrated_probability"), 0.0) for r in records], dtype=float)
    win_probs = np.clip(win_probs, 1e-12, 1.0)
    win_probs = win_probs / max(win_probs.sum(), 1e-12)

    p2 = np.zeros_like(win_probs)
    p3 = np.zeros_like(win_probs)
    n = len(records)
    for i in range(n):
        s2 = 0.0
        s3 = 0.0
        for j in range(n):
            if i == j:
                continue
            denom_j = max(1e-12, 1.0 - win_probs[j])
            s2 += win_probs[j] * (win_probs[i] / denom_j)
            for k in range(n):
                if k == i or k == j:
                    continue
                denom_k = max(1e-12, 1.0 - win_probs[j] - win_probs[k])
                s3 += win_probs[j] * (win_probs[k] / denom_j) * (win_probs[i] / denom_k)
        p2[i] = s2
        p3[i] = s3

    keys = [str(r.get("horse_id", "")) for r in records]
    return dict(zip(keys, p2.tolist())), dict(zip(keys, p3.tolist()))


def _build_ml_analysis_context(paths, pred: pd.DataFrame, records: list[dict]) -> dict[str, object]:  # type: ignore[no-untyped-def]
    context: dict[str, object] = {
        "data_notes": [
            "Gunluk bulten, kosu sonucu ve at gecmisi TJK canli sayfalarindan cekilir.",
            "Gercek veri eksikse model bu ekranda uydurma deger uretmez.",
            "Harville sira olasiliklari, ayni kosudaki gercek kazanma olasiliklarindan turetilir.",
        ],
        "feature_notes": [],
        "data_summary": [],
        "horse_stats": [],
        "backtest_summary": None,
        "method_notes": [
            "Cekirdek model: Benter iki asamali conditional logit.",
            "Sira olasiliklari: Harville formulu.",
            "Karar katmani: kalibrasyon + EV + fractional Kelly.",
        ],
    }

    try:
        if paths.features_csv.exists():
            features = pd.read_csv(paths.features_csv)
            context["data_summary"] = [
                f"Toplam feature satiri: {len(features)}",
                f"Benzersiz kosu sayisi: {int(features['race_id'].nunique()) if 'race_id' in features.columns else 0}",
                f"Tarih araligi: {features['date'].min() if 'date' in features.columns and not features.empty else '-'} -> {features['date'].max() if 'date' in features.columns and not features.empty else '-'}",
            ]
    except Exception:
        pass

    try:
        artifact, _ = load_phase3_artifact(str(paths.model_path))
        feature_cols = artifact.feature_columns
        sample = pred.head(200).copy()
        if "is_winner" not in sample.columns:
            sample["is_winner"] = 0
        perm = compute_permutation_importance(artifact, sample, feature_cols, n_repeats=2)
        shap = compute_optional_shap_summary(artifact, sample, feature_cols)
        top_features = perm[:8]
        if not top_features and shap.get("available"):
            top_features = shap.get("top_features", [])[:8]
        context["feature_notes"] = [
            f"{item['feature']}: {float(item.get('importance_mean', item.get('abs_stage1_coef', 0.0))):.4f}"
            for item in top_features
        ]
    except Exception:
        pass

    try:
        if records:
            record_frame = pd.DataFrame(records)
            summary_lines = [
                f"Secili kosu at sayisi: {int(len(record_frame))}",
                f"Ortalama P(win): {float(pd.to_numeric(record_frame.get('calibrated_probability', pd.Series(dtype=float)), errors='coerce').mean() or 0.0):.3f}",
                f"EV > 0 aday sayisi: {int((pd.to_numeric(record_frame.get('ev', pd.Series(dtype=float)), errors='coerce') > 0).sum()) if 'ev' in record_frame.columns else 0}",
            ]
            context["data_summary"] = context.get("data_summary", []) + summary_lines
    except Exception:
        pass

    try:
        stat_rows = []
        for rec in records:
            stat_rows.append(
                {
                    "horse": str(rec.get("horse_name") or rec.get("horse_id") or "-"),
                    "number": rec.get("number", "-"),
                    "draw": rec.get("draw", rec.get("number", "-")),
                    "weight": rec.get("weight", "-"),
                    "distance": rec.get("distance", "-"),
                    "field_size": rec.get("field_size", "-"),
                    "age": rec.get("age", "-"),
                    "handicap_points": rec.get("handicap_points", "-"),
                    "form_avg_3": rec.get("form_avg_3", "-"),
                    "form_avg_5": rec.get("form_avg_5", "-"),
                    "form_avg_10": rec.get("form_avg_10", "-"),
                    "days_since_last_race": rec.get("days_since_last_race", "-"),
                    "track_fit": rec.get("track_fit", "-"),
                    "surface_fit": rec.get("surface_fit", "-"),
                    "distance_fit": rec.get("distance_fit", "-"),
                    "career_starts": rec.get("career_starts", "-"),
                    "career_wins": rec.get("career_wins", "-"),
                    "career_places": rec.get("career_places", "-"),
                    "last_year_starts": rec.get("last_year_starts", "-"),
                    "last_year_wins": rec.get("last_year_wins", "-"),
                    "last_year_places": rec.get("last_year_places", "-"),
                    "jockey_horse_combo_wins": rec.get("jockey_horse_combo_wins", "-"),
                    "history_avg_finish_position": rec.get("history_avg_finish_position", "-"),
                    "history_avg_odds": rec.get("history_avg_odds", "-"),
                    "history_avg_race_time_seconds": rec.get("history_avg_race_time_seconds", "-"),
                    "workout_count": rec.get("workout_count", "-"),
                    "workout_avg_time_seconds": rec.get("workout_avg_time_seconds", "-"),
                    "workout_best_time_seconds": rec.get("workout_best_time_seconds", "-"),
                    "days_since_last_workout": rec.get("days_since_last_workout", "-"),
                    "market_probability_norm": rec.get("market_probability_norm", rec.get("market_probability_used", "-")),
                    "edge": rec.get("edge", "-"),
                    "ev": rec.get("ev", "-"),
                    "kelly_fraction": rec.get("kelly_fraction", "-"),
                }
            )
        context["horse_stats"] = stat_rows
    except Exception:
        pass

    return context


def _attach_ml_labels(paths, target_date: date, frame: pd.DataFrame) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    out = frame.copy()
    if out.empty:
        if "race_name" not in out.columns:
            out["race_name"] = pd.Series(dtype="object")
        if "horse_name" not in out.columns:
            out["horse_name"] = pd.Series(dtype="object")
        if "number" not in out.columns:
            out["number"] = pd.Series(dtype="float64")
        return out

    if not paths.clean_csv.exists():
        out["race_name"] = out["race_id"].astype(str)
        out["horse_name"] = out["horse_id"].astype(str)
        out["number"] = pd.to_numeric(out.get("draw", pd.Series(np.nan, index=out.index)), errors="coerce")
        return out

    clean = pd.read_csv(paths.clean_csv)
    clean["date"] = pd.to_datetime(clean["date"]).dt.date
    day_clean = clean[clean["date"] == target_date].copy()
    if day_clean.empty:
        out["race_name"] = out["race_id"].astype(str)
        out["horse_name"] = out["horse_id"].astype(str)
        out["number"] = pd.to_numeric(out.get("draw", pd.Series(np.nan, index=out.index)), errors="coerce")
        return out

    labels = day_clean[["race_id", "horse_id", "horse_name", "track", "draw"]].drop_duplicates()
    out = out.merge(labels, on=["race_id", "horse_id"], how="left", suffixes=("", "_label"))

    race_label_map: dict[str, str] = {}
    for rid in out["race_id"].astype(str).unique().tolist():
        sub = out[out["race_id"].astype(str) == rid]
        track = str(sub["track"].dropna().iloc[0]) if "track" in sub.columns and not sub["track"].dropna().empty else "-"
        track_display = _display_track_name(track)
        race_no = _parse_race_no(rid)
        if race_no is None:
            race_label_map[rid] = f"{track_display} - {rid}"
        else:
            race_label_map[rid] = f"{track_display} - {race_no}. Kosu"

    out["race_name"] = out["race_id"].astype(str).map(race_label_map)
    out["horse_name"] = out["horse_name"].fillna(out["horse_id"]).astype(str)
    out["number"] = pd.to_numeric(out.get("draw", pd.Series(np.nan, index=out.index)), errors="coerce")
    return out


def _overlay_bulletin_horse_names(
    records: list[dict],
    *,
    settings,
    target_date: date,
    city: str,
    requested_race_id: str,
) -> list[dict]:  # type: ignore[no-untyped-def]
    """Replace synthetic horse labels with bulletin names using race+number mapping."""
    if not records or not city:
        return records

    try:
        data_source = _bulletin_source_for_ml(settings)
        races = fetch_races(data_source, target_date, city, None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Bulletin name overlay skipped: %s", exc)
        return records

    requested_no = _parse_race_no(requested_race_id)
    target_race = None
    for race in races:
        race_id = str(race.id)
        if race_id == requested_race_id:
            target_race = race
            break
        mapped = _to_ml_frame_race_id(race_id, target_date)
        if mapped and mapped == requested_race_id:
            target_race = race
            break
        if requested_no is not None and race.race_no == requested_no:
            target_race = race
            break

    if target_race is None:
        return records

    number_to_name = {
        int(entry.number): str(entry.horse_name)
        for entry in target_race.entries
    }
    horse_id_to_name = {
        str(entry.horse_id): str(entry.horse_name)
        for entry in target_race.entries
        if entry.horse_id is not None
    }
    source_horse_id_to_name = {
        str(entry.source_horse_id): str(entry.horse_name)
        for entry in target_race.entries
        if entry.source_horse_id is not None
    }

    for rec in records:
        current_name = rec.get("horse_name")
        if not _is_placeholder_horse_name(current_name):
            continue

        number = rec.get("number")
        try:
            num = int(number) if number is not None and not pd.isna(number) else None
        except (TypeError, ValueError):
            num = None
        if num is not None and num in number_to_name:
            rec["horse_name"] = number_to_name[num]
            continue

        horse_id = str(rec.get("horse_id", "")).strip()
        if horse_id and horse_id in horse_id_to_name:
            rec["horse_name"] = horse_id_to_name[horse_id]
            continue

        source_horse_id = rec.get("source_horse_id")
        if source_horse_id is not None:
            src_key = str(source_horse_id).strip()
            if src_key in source_horse_id_to_name:
                rec["horse_name"] = source_horse_id_to_name[src_key]

    return records


def _ml_has_date(
    paths,
    target_date: date,
    city: str = "",
    race_no: int | None = None,
) -> bool:  # type: ignore[no-untyped-def]
    # Training features may contain results for a day that is already over.
    # They must never make a live prediction look ready for that same day.
    for feature_path in (paths.prediction_features_csv,):
        if not feature_path.exists():
            continue
        try:
            dates = pd.read_csv(
                feature_path,
                usecols=lambda column: column in {"date", "track", "race_id"},
            )
        except Exception:  # noqa: BLE001
            continue
        parsed = pd.to_datetime(dates["date"], errors="coerce").dt.date
        matching = parsed == target_date
        if city and "track" in dates.columns:
            matching &= dates["track"].astype(str).map(_normalize_track_key) == _normalize_track_key(city)
        if race_no is not None and "race_id" in dates.columns:
            matching &= dates["race_id"].astype(str).map(_parse_race_no) == race_no
        if bool(matching.any()):
            return True
    return False


def _ml_model_loadable(paths) -> tuple[bool, str]:  # type: ignore[no-untyped-def]
    if not paths.model_path.exists():
        return False, f"Model dosyasi bulunamadi: {paths.model_path}"
    try:
        artifact, _ = load_phase3_artifact(str(paths.model_path))
        expected_columns = TJK_SELECTED_STAGE1_FEATURE_COLUMNS
        if artifact.feature_columns != expected_columns:
            missing = sorted(set(expected_columns) - set(artifact.feature_columns))
            obsolete = sorted(set(artifact.feature_columns) - set(expected_columns))
            return False, (
                "Model eski TJK feature semasiyla kaydedilmis. "
                f"Eksik: {missing}; artik kullanilmamasi gerekenler: {obsolete}"
            )
    except Exception as exc:  # noqa: BLE001
        return False, f"Model artifact'i yuklenemedi: {exc}"
    return True, ""


def _safe_tjk_hippodromes(data_source: TJKHtmlDataSource, target_date: date) -> tuple[list[str], str | None]:
    try:
        return data_source.get_available_hippodromes(target_date), None
    except DataSourceError as exc:
        logger.warning("TJK hipodrom listesi alinamadi, bilinen listeyle devam edilecek: %s", exc)
        return list(KNOWN_HIPPODROMES), (
            "TJK'ya su an ulasilamiyor (DNS/ag gecici hatasi). "
            "Bilinen hipodrom listesi gosteriliyor; birazdan tekrar deneyin."
        )


def _ensure_ml_ready_for_date(
    settings,
    target_date: date,
    city: str = "",
    race_no: int | None = None,
) -> None:  # type: ignore[no-untyped-def]
    paths = paths_from_settings(settings)
    loadable_result = _ml_model_loadable(paths)
    if isinstance(loadable_result, tuple):
        model_loadable, model_error = loadable_result
    else:
        model_loadable = bool(loadable_result)
        model_error = ""
    if not model_loadable:
        raise RuntimeError(
            "ML modeli hazir degil. Once Model Egitimi bolumunden yeni TJK feature'lariyla "
            f"egitimi tamamlayin. Ayrinti: {model_error}"
        )
    try:
        has_date = _ml_has_date(paths, target_date, city=city, race_no=race_no)
    except TypeError as exc:
        if "unexpected keyword argument 'city'" not in str(exc):
            raise
        has_date = _ml_has_date(paths, target_date)
    if not has_date:
        try:
                prepare_prediction_features(
                    target_date,
                    paths,
                    hippodrome=city or None,
                    race_no=race_no,
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"{target_date.isoformat()} icin labelsiz tahmin feature verisi hazirlanamadi: {exc}"
            ) from exc
        if not _ml_has_date(paths, target_date, city=city, race_no=race_no):
            raise RuntimeError(
                f"{target_date.isoformat()} icin tahmin feature verisi olusmadi. "
                "TJK gunluk programi kontrol edilmeli."
            )



def _predict_ml_for_date_with_recovery(
    settings,
    target_date: date,
    city: str = "",
    race_no: int | None = None,
) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    paths = paths_from_settings(settings)
    if city:
        try:
            _ensure_ml_ready_for_date(settings, target_date, city=city, race_no=race_no)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            _ensure_ml_ready_for_date(settings, target_date)
    else:
        try:
            _ensure_ml_ready_for_date(settings, target_date, race_no=race_no)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            _ensure_ml_ready_for_date(settings, target_date)
    try:
        pred = predict_for_date(
            paths,
            target_date,
            enable_ev=settings.phase3_enable_ev,
            ev_probability_threshold=settings.ev_probability_threshold,
            ev_min_edge=settings.ev_min_edge,
            ev_min_value=settings.ev_min_value,
        )
        return _attach_ml_labels(paths, target_date, pred)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ML prediction failed")
        raise RuntimeError(f"ML tahmini uretilemedi: {exc}") from exc


def _run_training_job(
    job_id: str,
    start_date: date,
    end_date: date,
    allow_partial: bool = False,
) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)

    cancel_event = _TRAINING_CANCEL_EVENTS[job_id]

    def progress(message: str) -> None:
        if cancel_event.is_set():
            with _TRAINING_LOCK:
                if job_id in _TRAINING_PAUSE_REQUESTS:
                    raise _TrainingPaused()
            raise _TrainingCancelled()
        with _TRAINING_LOCK:
            job = _TRAINING_JOBS.get(job_id)
            if job is not None:
                job["message"] = message

    try:
        progress("1/5 Veri cekme basladi")
        data = ingest_real_data(start_date, end_date, paths, progress_callback=progress)
        retry_targets = list(data.attrs.get("retry_targets", []))
        if retry_targets and not allow_partial:
            with _TRAINING_LOCK:
                _TRAINING_JOBS[job_id] = {
                    "status": "retrying",
                    "message": (
                        f"{len(retry_targets)} veri parcasi alinamadi. "
                        "Otomatik olarak 30 saniye sonra tekrar denenecek."
                    ),
                    "retry_targets": retry_targets,
                    "can_accept_partial": not data.empty,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                }
                _persist_training_jobs()
            _schedule_skipped_training_retry(job_id)
            return
        progress("Veri cekme tamamlandi")
        progress(f"1/5 tamamlandi: {len(data)} satir")
        progress("2/5 Preprocess basladi")
        cleaned = preprocess_raw(paths)
        progress(f"2/5 tamamlandi: {len(cleaned)} satir")
        progress("3/5 Feature uretimi basladi")
        built = build_features(paths)
        progress(f"3/5 tamamlandi: {len(built.frame)} satir")
        progress("4/5 Model egitimi basladi")
        holdout_days = settings.phase1_holdout_days
        artifact = train_phase1_model(
            paths,
            holdout_days=holdout_days,
            calibration_days=settings.phase3_calibration_days,
            calibration_method=settings.phase3_calibration_method,
        )
        progress("Model egitimi tamamlandi")
        model_version = register_training_run(
            settings,
            paths,
            holdout_days=holdout_days,
            calibration_method=settings.phase3_calibration_method,
            blend_weight=float(artifact.logistic_weight),
            feature_frame=built.frame,
            feature_columns=artifact.feature_columns,
        )
        progress(f"4/5 tamamlandi: {model_version}")
        progress("5/5 Walk-forward backtest basladi")
        backtest = run_phase1_backtest(
            paths,
            min_train_days=settings.phase1_min_train_days,
            calibration_days=settings.phase3_calibration_days,
            calibration_method=settings.phase3_calibration_method,
            ev_probability_threshold=settings.ev_probability_threshold,
            ev_min_edge=settings.ev_min_edge,
            ev_min_value=settings.ev_min_value,
        )
        progress("Walk-forward backtest tamamlandi")
        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id] = {
                "status": "completed_with_warnings" if retry_targets else "completed",
                "message": (
                    "Egitim tamamlandi; bazi veri parcalari atlandi."
                    if retry_targets else "Egitim tamamlandi"
                ),
                "model_version": model_version,
                "backtest": backtest,
                "retry_targets": retry_targets,
                "can_accept_partial": not data.empty,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            _persist_training_jobs()
            if retry_targets:
                _schedule_skipped_training_retry(job_id)
    except _TrainingPaused:
        with _TRAINING_LOCK:
            _TRAINING_PAUSE_REQUESTS.discard(job_id)
            _TRAINING_JOBS[job_id] = {
                "status": "paused",
                "message": "Egitim duraklatildi; checkpoint korundu",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            _persist_training_jobs()
    except _TrainingCancelled:
        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id] = {
                "status": "cancelled",
                "message": "Egitim kullanici istegiyle durduruldu",
            }
            _persist_training_jobs()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Web training job failed")
        message = str(exc)
        if "bos veri uretti" in message:
            message = (
                f"{message} TJK sonuclarinin erisilebilir oldugu bir tarih araligi secin "
                "veya canli veri kaynagini kontrol edin."
            )
        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id] = {
                "status": "failed",
                "message": message,
            }
        _persist_training_jobs()


def _run_local_training_job(job_id: str, start_date: date, end_date: date) -> None:
    settings = get_settings()
    paths = paths_from_settings(settings)
    try:
        local_frame = build_local_raw_frame(paths)
        if not local_frame.empty:
            local_frame.to_csv(paths.raw_csv, index=False)
        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id]["message"] = "Lokal ham arşiv okunuyor"
        cleaned = preprocess_raw(paths)
        if not cleaned.empty:
            dates = pd.to_datetime(cleaned["date"], errors="coerce").dt.date
            cleaned = cleaned[(dates >= start_date) & (dates <= end_date)].copy()
            cleaned.to_csv(paths.clean_csv, index=False)
        built = build_features(paths)
        artifact = train_phase1_model(
            paths,
            holdout_days=settings.phase1_holdout_days,
            calibration_days=settings.phase3_calibration_days,
            calibration_method=settings.phase3_calibration_method,
        )
        model_version = register_training_run(
            settings,
            paths,
            holdout_days=settings.phase1_holdout_days,
            calibration_method=settings.phase3_calibration_method,
            blend_weight=float(artifact.logistic_weight),
            feature_frame=built.frame,
            feature_columns=artifact.feature_columns,
        )
        backtest = run_phase1_backtest(
            paths,
            min_train_days=settings.phase1_min_train_days,
            calibration_days=settings.phase3_calibration_days,
            calibration_method=settings.phase3_calibration_method,
            ev_probability_threshold=settings.ev_probability_threshold,
            ev_min_edge=settings.ev_min_edge,
            ev_min_value=settings.ev_min_value,
        )
        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id] = {
                "status": "completed",
                "message": "Lokal veriden egitim tamamlandi",
                "model_version": model_version,
                "backtest": backtest,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Local training job failed")
        with _TRAINING_LOCK:
            _TRAINING_JOBS[job_id] = {"status": "failed", "message": f"Lokal egitim: {exc}"}
    _persist_training_jobs()


def _validate_training_date_range(start_date: date, end_date: date) -> None:
    if end_date - start_date < timedelta(days=MIN_TRAINING_DATES - 1):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model egitimi icin en az {MIN_TRAINING_DATES} farkli tarih gerekli. "
                f"En az {MIN_TRAINING_DATES} gunluk bir tarih araligi secin."
            ),
        )


def create_app() -> FastAPI:
    """FastAPI uygulamasini olusturur (uvicorn factory olarak kullanilir)."""
    configure_logging()
    _restore_training_jobs()
    _restore_collection_jobs()
    app = FastAPI(title="Turkiye At Yarisi Tahmin Araci", docs_url="/api/docs")
    templates.env.globals["active_training_job_id"] = _active_training_job_id

    @app.get("/collect", response_class=RedirectResponse)
    def start_collection(
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    ) -> RedirectResponse:
        parsed_start = date.fromisoformat(start_date)
        parsed_end = date.fromisoformat(end_date)
        if parsed_start > parsed_end:
            raise HTTPException(status_code=400, detail="Baslangic tarihi bitis tarihinden sonra olamaz.")
        with _TRAINING_LOCK:
            active = next(
                (job_id for job_id, job in _COLLECTION_JOBS.items() if job.get("status") == "running"),
                None,
            )
            if active is not None:
                job_id = active
            else:
                resumable = next(
                    (
                        job_id for job_id, job in _COLLECTION_JOBS.items()
                        if job.get("status") == "paused"
                        and job.get("start_date") == start_date
                        and job.get("end_date") == end_date
                    ),
                    None,
                )
                if resumable is not None:
                    job_id = resumable
                else:
                    job_id = uuid4().hex
                    _COLLECTION_JOBS[job_id] = {
                        "status": "running",
                        "message": "Veri toplama siraya alindi",
                        "start_date": start_date,
                        "end_date": end_date,
                        "completed_days": 0,
                        "remaining_days": (parsed_end - parsed_start).days + 1,
                        "skipped_days": 0,
                        "failed_targets": 0,
                        "missing_data_records": 0,
                    }
                _COLLECTION_JOBS[job_id]["status"] = "running"
                _COLLECTION_CANCEL_EVENTS[job_id] = Event()
                Thread(
                    target=_run_collection_job,
                    args=(job_id, parsed_start, parsed_end),
                    name=f"atyaris-collect-{job_id[:8]}",
                    daemon=True,
                ).start()
                _persist_collection_jobs()
        return RedirectResponse(url=f"/training?collection_job={job_id}", status_code=303)

    @app.get("/collect/status/{job_id}")
    def collection_status(job_id: str) -> JSONResponse:
        with _TRAINING_LOCK:
            job = _COLLECTION_JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Veri toplama isi bulunamadi.")
            return JSONResponse({"job_id": job_id, **job})

    @app.post("/collect/{action}/{job_id}")
    def collection_control(action: str, job_id: str) -> JSONResponse:
        if action not in {"pause", "resume", "cancel", "remove"}:
            raise HTTPException(status_code=404, detail="Gecersiz veri toplama islemi.")
        with _TRAINING_LOCK:
            job = _COLLECTION_JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Veri toplama isi bulunamadi.")
            if action == "remove":
                if job.get("status") == "running":
                    raise HTTPException(status_code=409, detail="Calisan veri toplama isi kaldirilamaz.")
                _COLLECTION_JOBS.pop(job_id, None)
                _COLLECTION_PAUSE_REQUESTS.discard(job_id)
                _COLLECTION_CANCEL_EVENTS.pop(job_id, None)
                _persist_collection_jobs()
                return JSONResponse({"job_id": job_id, "status": "removed"})
            if action == "pause" and job.get("status") == "running":
                _COLLECTION_PAUSE_REQUESTS.add(job_id)
                _COLLECTION_CANCEL_EVENTS[job_id].set()
                job["message"] = "Veri toplama duraklatiliyor; mevcut TJK istegi tamamlaninca duracak..."
                job["pause_requested"] = True
            elif action == "cancel":
                _COLLECTION_PAUSE_REQUESTS.discard(job_id)
                event = _COLLECTION_CANCEL_EVENTS.get(job_id)
                if event is not None:
                    event.set()
                if job.get("status") != "running":
                    job["status"] = "cancelled"
                    job["message"] = "Veri toplama iptal edildi"
            elif action == "resume" and job.get("status") == "paused":
                job["status"] = "running"
                job["message"] = "Veri toplama checkpointten devam ediyor"
                _COLLECTION_CANCEL_EVENTS[job_id] = Event()
                Thread(
                    target=_run_collection_job,
                    args=(job_id, date.fromisoformat(str(job["start_date"])), date.fromisoformat(str(job["end_date"]))),
                    name=f"atyaris-collect-{job_id[:8]}",
                    daemon=True,
                ).start()
            _persist_collection_jobs()
            return JSONResponse({"job_id": job_id, **job})

    @app.get("/train", response_class=RedirectResponse)
    def start_training(
        start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
        mode: str = Query("online", pattern="^(online|local)$"),
    ) -> RedirectResponse:
        try:
            parsed_start = date.fromisoformat(start_date)
            parsed_end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Tarih YYYY-AA-GG formatinda olmali.") from exc
        if parsed_start > parsed_end:
            raise HTTPException(status_code=400, detail="Baslangic tarihi bitis tarihinden sonra olamaz.")
        _validate_training_date_range(parsed_start, parsed_end)

        with _TRAINING_LOCK:
            active = next(
                (job_id for job_id, job in _TRAINING_JOBS.items() if job.get("status") == "running"),
                None,
            )
            if active is not None:
                job_id = active
            else:
                resumable = next(
                    (
                        job_id
                        for job_id, job in _TRAINING_JOBS.items()
                        if job.get("status") == "paused"
                        and job.get("start_date") == start_date
                        and job.get("end_date") == end_date
                    ),
                    None,
                )
                if resumable is not None:
                    job_id = resumable
                    _TRAINING_JOBS[job_id]["status"] = "running"
                    _TRAINING_JOBS[job_id]["message"] = "Egitim checkpointten devam ediyor"
                    _TRAINING_CANCEL_EVENTS[job_id] = Event()
                    Thread(
                        target=_run_local_training_job if mode == "local" else _run_training_job,
                        args=(job_id, parsed_start, parsed_end),
                        name=f"atyaris-train-{job_id[:8]}",
                        daemon=True,
                    ).start()
                    _persist_training_jobs()
                    return RedirectResponse(url=f"/training?train_job={job_id}", status_code=303)
                job_id = uuid4().hex
                _TRAINING_JOBS[job_id] = {
                    "status": "running",
                    "message": "Egitim siraya alindi",
                    "start_date": start_date,
                    "end_date": end_date,
                }
                _TRAINING_CANCEL_EVENTS[job_id] = Event()
                Thread(
                    target=_run_local_training_job if mode == "local" else _run_training_job,
                    args=(job_id, parsed_start, parsed_end),
                    name=f"atyaris-train-{job_id[:8]}",
                    daemon=True,
                ).start()
                _persist_training_jobs()
        return RedirectResponse(url=f"/training?train_job={job_id}", status_code=303)

    @app.get("/train/status/{job_id}")
    def training_status(job_id: str) -> JSONResponse:
        with _TRAINING_LOCK:
            job = _TRAINING_JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Egitim isi bulunamadi.")
            return JSONResponse(dict(job))

    @app.post("/train/cancel/{job_id}")
    def cancel_training(job_id: str) -> JSONResponse:
        with _TRAINING_LOCK:
            job = _TRAINING_JOBS.get(job_id)
            cancel_event = _TRAINING_CANCEL_EVENTS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Egitim isi bulunamadi.")
            if job.get("status") == "running" and cancel_event is not None:
                cancel_event.set()
                job["message"] = "Egitim durduruluyor..."
            elif job.get("status") in {"paused", "awaiting_decision", "retrying", "completed_with_warnings"}:
                job["status"] = "cancelled"
                job["message"] = "Egitim kullanici istegiyle iptal edildi"
                _TRAINING_PAUSE_REQUESTS.discard(job_id)
                _TRAINING_CANCEL_EVENTS.pop(job_id, None)
                retry_timer = _TRAINING_RETRY_TIMERS.pop(job_id, None)
                if retry_timer is not None:
                    retry_timer.cancel()
                _persist_training_jobs()
            return JSONResponse(dict(job))

    @app.post("/train/pause/{job_id}")
    def pause_training(job_id: str) -> JSONResponse:
        with _TRAINING_LOCK:
            job = _TRAINING_JOBS.get(job_id)
            pause_event = _TRAINING_CANCEL_EVENTS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Egitim isi bulunamadi.")
            if job.get("status") == "running" and pause_event is not None:
                _TRAINING_PAUSE_REQUESTS.add(job_id)
                pause_event.set()
                job["message"] = "Egitim duraklatiliyor..."
                _persist_training_jobs()
            return JSONResponse(dict(job))

    @app.post("/train/resume/{job_id}")
    def resume_training(job_id: str) -> JSONResponse:
        with _TRAINING_LOCK:
            job = _TRAINING_JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Egitim isi bulunamadi.")
            if job.get("status") != "paused":
                return JSONResponse(dict(job))
            job["status"] = "running"
            job["message"] = "Egitim checkpointten devam ediyor"
            _TRAINING_CANCEL_EVENTS[job_id] = Event()
            Thread(
                target=_run_training_job,
                args=(job_id, date.fromisoformat(str(job["start_date"])), date.fromisoformat(str(job["end_date"]))),
                name=f"atyaris-train-{job_id[:8]}",
                daemon=True,
            ).start()
            _persist_training_jobs()
            return JSONResponse(dict(job))

    @app.post("/train/accept-partial/{job_id}")
    def accept_partial_training(job_id: str) -> JSONResponse:
        with _TRAINING_LOCK:
            job = _TRAINING_JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Egitim isi bulunamadi.")
            if job.get("status") != "awaiting_decision":
                return JSONResponse(dict(job))
            job["status"] = "running"
            job["message"] = "Mevcut verilerle egitim devam ediyor"
            _TRAINING_CANCEL_EVENTS[job_id] = Event()
            Thread(
                target=_run_training_job,
                args=(
                    job_id,
                    date.fromisoformat(str(job["start_date"])),
                    date.fromisoformat(str(job["end_date"])),
                    True,
                ),
                name=f"atyaris-train-{job_id[:8]}",
                daemon=True,
            ).start()
            _persist_training_jobs()
            return JSONResponse(dict(job))

    @app.post("/train/retry-skipped/{job_id}")
    def retry_skipped_training(job_id: str) -> JSONResponse:
        with _TRAINING_LOCK:
            job = _TRAINING_JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Egitim isi bulunamadi.")
            if job.get("status") not in {"awaiting_decision", "retrying", "completed_with_warnings"}:
                return JSONResponse(dict(job))
            if job.get("status") == "awaiting_decision":
                job["status"] = "completed_with_warnings"
            _start_retry_skipped_training_locked(job_id)
            return JSONResponse(dict(job))

    @app.get("/api/training/history")
    def training_history_api(
        limit: int = Query(10, ge=1, le=100),
        start_date: str = Query("", pattern=r"^$|^\d{4}-\d{2}-\d{2}$"),
        end_date: str = Query("", pattern=r"^$|^\d{4}-\d{2}-\d{2}$"),
    ) -> JSONResponse:
        if start_date and end_date and start_date > end_date:
            raise HTTPException(status_code=400, detail="Baslangic tarihi bitis tarihinden sonra olamaz.")
        settings = get_settings()
        return JSONResponse(
            load_training_history(
                settings.phase5_tracking_db_path,
                limit=limit,
                start_date=start_date or None,
                end_date=end_date or None,
            )
        )

    @app.get("/training", response_class=HTMLResponse)
    def training_page(
        request: Request,
        start_date: str = Query("", alias="start_date"),
        end_date: str = Query("", alias="end_date"),
        limit: int = Query(10, ge=1, le=100),
        train_job: str = Query(""),
    ) -> HTMLResponse:
        error = None
        history = {"latest_training": None, "runs": [], "count": 0}
        try:
            if start_date:
                date.fromisoformat(start_date)
            if end_date:
                date.fromisoformat(end_date)
            if start_date and end_date and start_date > end_date:
                raise ValueError("Baslangic tarihi bitis tarihinden sonra olamaz.")
            settings = get_settings()
            paths = paths_from_settings(settings)
            raw_maintenance = []
            for path, collection in (
                (paths.raw_daily_program_jsonl, "daily_program"),
                (paths.raw_race_results_jsonl, "race_results"),
                (paths.raw_history_jsonl, "horse_history"),
                (paths.raw_workouts_jsonl, "workouts"),
                (paths.raw_trainer_statistics_jsonl, "trainer_statistics"),
            ):
                try:
                    store = JsonlRawStore(path, collection)
                    recommendation = store.compact_recommendation()
                    raw_maintenance.append({"name": collection, **recommendation.__dict__})
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    raw_maintenance.append({"name": collection, "recommended": False, "reason": f"uyumsuz/eski format: {exc}", "active_records": 0, "file_bytes": path.stat().st_size if path.exists() else 0})
            history = load_training_history(
                settings.phase5_tracking_db_path,
                limit=limit,
                start_date=start_date or None,
                end_date=end_date or None,
            )
        except ValueError as exc:
            error = f"Gecersiz istek: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training history page failure")
            error = f"Egitim gecmisi okunamadi: {exc}"

        return templates.TemplateResponse(
            request,
            "training_history.html",
            {
                "history": history,
                "error": error,
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "train_job": train_job,
                "training_jobs": [
                    {"job_id": job_id, **job}
                    for job_id, job in reversed(list(_TRAINING_JOBS.items()))
                    if job.get("status") != "cancelled"
                ],
                "collection_jobs": [
                    {"job_id": job_id, **job}
                    for job_id, job in reversed(list(_COLLECTION_JOBS.items()))
                    if job.get("status") != "cancelled"
                ],
                "raw_maintenance": raw_maintenance,
            },
        )

    @app.post("/raw/compact")
    def compact_raw_data() -> JSONResponse:
        settings = get_settings()
        paths = paths_from_settings(settings)
        results = []
        for path, collection in (
            (paths.raw_daily_program_jsonl, "daily_program"),
            (paths.raw_race_results_jsonl, "race_results"),
            (paths.raw_history_jsonl, "horse_history"),
            (paths.raw_workouts_jsonl, "workouts"),
            (paths.raw_trainer_statistics_jsonl, "trainer_statistics"),
        ):
            store = JsonlRawStore(path, collection)
            results.append({"name": collection, **store.compact().__dict__})
        return JSONResponse({"status": "completed", "collections": results})

    @app.get("/training-history", response_class=RedirectResponse)
    def training_history_redirect(
        start_date: str = Query(""),
        end_date: str = Query(""),
        limit: int = Query(10, ge=1, le=100),
    ) -> RedirectResponse:
        query = urlencode({"start_date": start_date, "end_date": end_date, "limit": limit})
        return RedirectResponse(url=f"/training?{query}", status_code=307)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        source: str = Query("sample", pattern="^(sample|tjk|ml)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
    ) -> HTMLResponse:
        settings = get_settings()
        error = None
        info = None
        races = []
        ml_races: list[MLRaceSummary] = []
        hippodromes: list[str] = []
        resolved_date = date_str
        try:
            parsed_date = parse_date(date_str or None)
            resolved_date = parsed_date.isoformat()

            if source == "ml":
                data_source = _bulletin_source_for_ml(settings)
                if isinstance(data_source, TJKHtmlDataSource):
                    hippodromes, source_info = _safe_tjk_hippodromes(data_source, parsed_date)
                    if source_info:
                        info = source_info
                else:
                    all_races = fetch_races(data_source, parsed_date, None, None)
                    hippodromes = sorted({r.hippodrome for r in all_races})

                if city:
                    races = fetch_races(data_source, parsed_date, city, None)
                    ml_races = _ml_race_summaries_from_races(races, data_source, settings)
                    if ml_races:
                        quick_info = "Bulten listesi hazir. Tahminler yalnizca kosu secildiginde hesaplanir."
                        info = f"{info} {quick_info}".strip() if info else quick_info
                else:
                    city_prompt = "Lutfen listeden bir hipodrom secin."
                    info = f"{info} {city_prompt}".strip() if info else city_prompt
                    ml_races = []

                if not ml_races:
                    if city:
                        info = "Secili tarih ve hipodrom icin ML kosu bulunamadi."

                return templates.TemplateResponse(
                    request,
                    "index.html",
                    {
                        "races": races,
                        "ml_races": ml_races,
                        "error": error,
                        "info": info,
                        "source": source,
                        "date": resolved_date,
                        "city": city,
                        "hippodromes": hippodromes,
                    },
                )

            data_source = build_data_source(source, settings)
            if isinstance(data_source, TJKHtmlDataSource):
                hippodromes, source_info = _safe_tjk_hippodromes(data_source, parsed_date)
                if source_info:
                    info = source_info
                if city:
                    races = fetch_races(data_source, parsed_date, city, None)
                    if not races:
                        info = (
                            "Secili hipodrom icin kosu listesi alinmadi. "
                            "TJK'nin JS/AJAX yapisi nedeniyle bu tarih icin veri gelmiyor olabilir."
                        )
                else:
                    races = []
                    city_prompt = "Lutfen listeden bir hipodrom secin."
                    info = f"{info} {city_prompt}".strip() if info else city_prompt
            else:
                all_races = fetch_races(data_source, parsed_date, None, None)
                hippodromes = sorted({r.hippodrome for r in all_races})
                races = [r for r in all_races if not city or r.hippodrome == city]
        except (InvalidSourceError, ValueError) as exc:
            error = f"Gecersiz istek: {exc}"
        except DataSourceError as exc:
            error = f"Veri kaynagi hatasi: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Index endpoint failure")
            error = f"Beklenmeyen hata: {exc}"

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "races": races,
                "ml_races": ml_races,
                "error": error,
                "info": info,
                "source": source,
                "date": resolved_date,
                "city": city,
                "hippodromes": hippodromes,
            },
        )

    @app.get("/predict", response_class=HTMLResponse)
    def predict(
        request: Request,
        race_id: str,
        source: str = Query("sample", pattern="^(sample|tjk|ml)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
        sort_by: str = Query(
            "calibrated_probability",
            pattern="^(strategy|number|odds|total|form|jockey_trainer|distance_surface|weight|rest|win_probability|confidence|ev|kelly|rank|horse_id|calibrated_probability|place2_probability|place3_probability|top3_probability|edge|kelly_fraction|form_strength)$",
        ),
        sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    ) -> HTMLResponse:
        settings = get_settings()
        error = None
        prediction = None
        ml_prediction = None
        display_rows = []
        sort_urls: dict[str, str] = {}
        pdf_url = ""
        table_usage_notes = [
            "1) Once Kazanma Olasiligi ve Guven ile guclu adaylari ayiklayin.",
            "2) Sonra EV > 0 olanlari value adayi olarak filtreleyin.",
            "3) Bahis buyuklugunu Kelly (fractional) degerine gore sinirlayin.",
        ]

        def _entry_key(entry) -> str:  # type: ignore[no-untyped-def]
            if entry.source_horse_id is not None:
                return f"source:{entry.source_horse_id}"
            return f"horse:{entry.horse_id}"

        def _build_predict_url(*, sort_field: str, sort_direction: str) -> str:
            return "/predict?" + urlencode(
                {
                    "race_id": race_id,
                    "source": source,
                    "date": date_str,
                    "city": city,
                    "sort_by": sort_field,
                    "sort_order": sort_direction,
                }
            )

        try:
            parsed_date = parse_date(date_str or None)

            if source == "ml":
                paths = paths_from_settings(settings)
                requested_race_no = _parse_race_no(str(race_id))
                pred = _predict_ml_for_date_with_recovery(
                    settings,
                    parsed_date,
                    city=city,
                    race_no=requested_race_no,
                )
                target_track_key = _normalize_track_key(city) if city else ""

                data_source = _bulletin_source_for_ml(settings)
                try:
                    if city:
                        bulletin_races = fetch_races(data_source, parsed_date, city, None)
                    else:
                        bulletin_races = fetch_races(data_source, parsed_date, None, None)
                except Exception:  # noqa: BLE001
                    bulletin_races = []

                selected_race = next((r for r in bulletin_races if str(r.id) == str(race_id)), None)
                if selected_race is None:
                    mapped_for_lookup = _to_ml_frame_race_id(str(race_id), parsed_date)
                    if mapped_for_lookup is not None:
                        selected_race = next(
                            (
                                r
                                for r in bulletin_races
                                if _to_ml_frame_race_id(str(r.id), parsed_date) == mapped_for_lookup
                            ),
                            None,
                        )
                if selected_race is None and requested_race_no is not None and city:
                    selected_race = next((r for r in bulletin_races if int(r.race_no) == requested_race_no), None)

                scoped_pred = pred.copy()
                if target_track_key and "track" in scoped_pred.columns:
                    track_keys = scoped_pred["track"].astype(str).map(_normalize_track_key)
                    scoped_pred = scoped_pred[track_keys == target_track_key].copy()

                race_df = pd.DataFrame()
                if selected_race is not None:
                    selected_race_no = int(selected_race.race_no)
                    scoped_race_no = scoped_pred["race_id"].astype(str).map(_parse_race_no)
                    race_df = scoped_pred[scoped_race_no == selected_race_no].copy()
                elif mapped_race_id := _to_ml_frame_race_id(race_id, parsed_date):
                    race_df = scoped_pred[scoped_pred["race_id"].astype(str) == mapped_race_id].copy()
                elif requested_race_no is not None:
                    scoped_race_no = scoped_pred["race_id"].astype(str).map(_parse_race_no)
                    race_df = scoped_pred[scoped_race_no == requested_race_no].copy()
                else:
                    race_df = scoped_pred[scoped_pred["race_id"].astype(str) == str(race_id)].copy()

                if selected_race is not None and not race_df.empty:
                    candidate_records = race_df.to_dict(orient="records")
                    if not _race_matches_bulletin(candidate_records, selected_race):
                        race_df = pd.DataFrame()

                if not race_df.empty:
                    ml_sort_by = sort_by if sort_by in _ML_SORTABLE_FIELDS else "calibrated_probability"
                    reverse = sort_order == "desc"
                    records = race_df.to_dict(orient="records")
                    for rec in records:
                        horse_name = rec.get("horse_name")
                        if horse_name in (None, "", "nan"):
                            rec["horse_name"] = str(rec.get("horse_id", "-"))

                        if rec.get("number") is None:
                            draw_val = rec.get("draw")
                            try:
                                rec["number"] = int(draw_val) if draw_val is not None and not pd.isna(draw_val) else None
                            except (TypeError, ValueError):
                                rec["number"] = None

                    records = _overlay_bulletin_horse_names(
                        records,
                        settings=settings,
                        target_date=parsed_date,
                        city=city,
                        requested_race_id=race_id,
                    )

                    p2_fallback, p3_fallback = _place_probabilities_from_records(records)
                    for rec in records:
                        hid = str(rec.get("horse_id", ""))
                        if hid in p2_fallback:
                            rec["place2_probability"] = p2_fallback[hid]
                        if hid in p3_fallback:
                            rec["place3_probability"] = p3_fallback[hid]
                        rec["top3_probability"] = min(
                            1.0,
                            _safe_float(rec.get("calibrated_probability"), 0.0)
                            + _safe_float(rec.get("place2_probability"), 0.0)
                            + _safe_float(rec.get("place3_probability"), 0.0),
                        )

                    records.sort(key=lambda r: _ml_sort_value(r, ml_sort_by), reverse=reverse)
                    for idx, rec in enumerate(records, start=1):
                        rec["rank"] = idx

                    horse_stats_rows: list[dict[str, object]] = []
                    for rec in records:
                        horse_stats_rows.append(
                            {
                                "horse": str(rec.get("horse_name") or rec.get("horse_id") or "-"),
                                "horse_id": rec.get("horse_id", "-"),
                                "number": rec.get("number", "-"),
                                "draw": rec.get("draw", rec.get("number", "-")),
                                "weight": rec.get("weight", "-"),
                                "distance": rec.get("distance", "-"),
                                "field_size": rec.get("field_size", "-"),
                                "age": rec.get("age", "-"),
                                "handicap_points": rec.get("handicap_points", "-"),
                                "form_avg_3": rec.get("form_avg_3", "-"),
                                "form_avg_5": rec.get("form_avg_5", "-"),
                                "form_avg_10": rec.get("form_avg_10", "-"),
                                "days_since_last_race": rec.get("days_since_last_race", "-"),
                                "track_fit": rec.get("track_fit", "-"),
                                "surface_fit": rec.get("surface_fit", "-"),
                                "distance_fit": rec.get("distance_fit", "-"),
                                "career_starts": rec.get("career_starts", "-"),
                                "career_wins": rec.get("career_wins", "-"),
                                "career_places": rec.get("career_places", "-"),
                                "last_year_starts": rec.get("last_year_starts", "-"),
                                "last_year_wins": rec.get("last_year_wins", "-"),
                                "last_year_places": rec.get("last_year_places", "-"),
                                "jockey_horse_combo_wins": rec.get("jockey_horse_combo_wins", "-"),
                                "history_avg_finish_position": rec.get("history_avg_finish_position", "-"),
                                "history_avg_odds": rec.get("history_avg_odds", "-"),
                                "history_avg_race_time_seconds": rec.get("history_avg_race_time_seconds", "-"),
                                "workout_count": rec.get("workout_count", "-"),
                                "workout_avg_time_seconds": rec.get("workout_avg_time_seconds", "-"),
                                "workout_best_time_seconds": rec.get("workout_best_time_seconds", "-"),
                                "days_since_last_workout": rec.get("days_since_last_workout", "-"),
                                "market_probability_norm": rec.get("market_probability_norm", rec.get("market_probability_used", "-")),
                                "edge": rec.get("edge", "-"),
                                "ev": rec.get("ev", "-"),
                                "kelly_fraction": rec.get("kelly_fraction", "-"),
                                "avg_race_time_shared_combo": "-",
                                "tjk_stats_url": "",
                                "tjk_workout_url": "",
                                "source_horse_id": rec.get("source_horse_id", ""),
                                "tjk_error": "",
                                "tjk_summary": [],
                                "tjk_history": [],
                            }
                        )

                    shared_combo_label: str | None = None
                    if selected_race is not None:
                        target_date_value = parsed_date
                        target_distance = int(selected_race.distance_m)
                        target_surface = selected_race.surface.value if hasattr(selected_race.surface, "value") else str(selected_race.surface)

                        entry_by_number = {int(entry.number): entry for entry in selected_race.entries}
                        entry_by_horse_id = {str(entry.horse_id): entry for entry in selected_race.entries}
                        entry_by_source_horse_id = {
                            int(entry.source_horse_id): entry
                            for entry in selected_race.entries
                            if entry.source_horse_id is not None
                        }
                        entry_by_name = {
                            _normalize_name_key(entry.horse_name): entry
                            for entry in selected_race.entries
                        }
                        stats_by_entry_key: dict[str, object] = {}
                        horse_history_map: dict[str, list] = {}
                        combo_counter: dict[tuple[int, str], int] = {}
                        for entry in selected_race.entries:
                            try:
                                stats = data_source.get_horse_statistics(entry)
                            except Exception:
                                continue
                            stats_by_entry_key[f"horse:{entry.horse_id}"] = stats
                            if entry.source_horse_id is not None:
                                stats_by_entry_key[f"source:{entry.source_horse_id}"] = stats
                            horse_history_map[str(entry.horse_id)] = stats.past_performances
                            for perf in stats.past_performances:
                                if perf.race_date >= target_date_value:
                                    continue
                                key = (int(perf.distance_m), perf.surface.value if hasattr(perf.surface, "value") else str(perf.surface))
                                combo_counter[key] = combo_counter.get(key, 0) + 1

                        shared_combo = None
                        if combo_counter:
                            shared_combo = max(combo_counter.items(), key=lambda item: item[1])[0]
                            shared_distance, shared_surface = shared_combo
                            shared_combo_label = f"Ortak pist+mesafe kombinasyonu: {shared_distance}m / {shared_surface}"

                        for row in horse_stats_rows:
                            row_number = row.get("number")
                            try:
                                row_number_int = int(row_number) if row_number not in (None, "-") else None
                            except (TypeError, ValueError):
                                row_number_int = None
                            matching_entry = entry_by_number.get(row_number_int) if row_number_int is not None else None
                            if matching_entry is None:
                                row_source_horse_id = row.get("source_horse_id")
                                try:
                                    src_id = int(row_source_horse_id) if row_source_horse_id not in (None, "") else None
                                except (TypeError, ValueError):
                                    src_id = None
                                if src_id is not None:
                                    matching_entry = entry_by_source_horse_id.get(src_id)
                            if matching_entry is None:
                                row_horse_id = str(row.get("horse_id", "")).strip()
                                if row_horse_id:
                                    matching_entry = entry_by_horse_id.get(row_horse_id)
                            if matching_entry is None:
                                row_horse_name_key = _normalize_name_key(row.get("horse"))
                                if row_horse_name_key:
                                    matching_entry = entry_by_name.get(row_horse_name_key)

                            if matching_entry is not None:
                                row["source_horse_id"] = str(matching_entry.source_horse_id or "")
                                if matching_entry.source_horse_id is not None:
                                    row["tjk_stats_url"] = (
                                        "https://www.tjk.org/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri?"
                                        f"QueryParameter_AtId={matching_entry.source_horse_id}&&Era=past"
                                    )
                                    row["tjk_workout_url"] = (
                                        "https://www.tjk.org/TR/YarisSever/Query/Page/IdmanIstatistikleri?"
                                        f"QueryParameter_AtId={matching_entry.source_horse_id}"
                                    )
                                stats = stats_by_entry_key.get(f"source:{matching_entry.source_horse_id}")
                                if stats is None:
                                    stats = stats_by_entry_key.get(f"horse:{matching_entry.horse_id}")
                                if stats is None:
                                    try:
                                        stats = data_source.get_horse_statistics(matching_entry)
                                    except Exception as exc:
                                        row["tjk_error"] = str(exc)
                                        stats = None
                                if stats is not None:
                                    row.update(
                                        _derive_horse_stats_metrics(
                                            stats,
                                            race_distance=int(selected_race.distance_m),
                                            race_surface=(selected_race.surface.value if hasattr(selected_race.surface, "value") else str(selected_race.surface)),
                                            race_track=(selected_race.hippodrome or ""),
                                        )
                                    )
                                    row["weight"] = row.get("weight", "-") if row.get("weight", "-") not in ("-", None) else (
                                        round(sum(float(p.weight_kg) for p in stats.past_performances if p.weight_kg is not None) / max(1, len([p for p in stats.past_performances if p.weight_kg is not None])), 1)
                                        if any(p.weight_kg is not None for p in stats.past_performances)
                                        else "-"
                                    )
                                    row["tjk_summary"] = [
                                        f"Kariyer kosu: {stats.career_starts}",
                                        f"Kariyer galibiyet: {stats.career_wins}",
                                        f"Kariyer plase: {stats.career_places}",
                                        f"Son yil kosu: {stats.last_year_starts}",
                                        f"Son yil galibiyet: {stats.last_year_wins}",
                                        f"Son yil plase: {stats.last_year_places}",
                                        f"Idman kaydi: {len(stats.workout_records)}",
                                    ]
                                    history_rows = []
                                    for perf in stats.past_performances[:10]:
                                        history_rows.append(
                                            {
                                                "race_date": perf.race_date.isoformat(),
                                                "hippodrome": perf.hippodrome,
                                                "distance_m": perf.distance_m,
                                                "surface": perf.surface.value if hasattr(perf.surface, "value") else str(perf.surface),
                                                "finish_position": perf.finish_position,
                                                "race_time_seconds": perf.race_time_seconds,
                                                "weight_kg": perf.weight_kg,
                                                "equipment": perf.equipment,
                                                "jockey_name": perf.jockey_name,
                                                "field_size": perf.field_size,
                                                "odds": perf.odds,
                                                "group_info": perf.group_info,
                                                "race_name": perf.race_name,
                                                "race_class": perf.race_class,
                                                "trainer_name": perf.trainer_name,
                                                "owner_name": perf.owner_name,
                                                "handicap_points": perf.handicap_points,
                                                "prize_info": perf.prize_info,
                                                "s20": perf.s20,
                                                "extra1": None,
                                                "extra2": None,
                                            }
                                        )
                                    row["tjk_history"] = history_rows

                                history = horse_history_map.get(str(matching_entry.horse_id), [])
                                filtered_times = [
                                    float(p.race_time_seconds)
                                    for p in history
                                    if p.race_time_seconds is not None
                                    and p.race_date < target_date_value
                                    and int(p.distance_m) == target_distance
                                    and (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == target_surface
                                ]
                                if not filtered_times and shared_combo is not None:
                                    shared_distance, shared_surface = shared_combo
                                    filtered_times = [
                                        float(p.race_time_seconds)
                                        for p in history
                                        if p.race_time_seconds is not None
                                        and p.race_date < target_date_value
                                        and int(p.distance_m) == int(shared_distance)
                                        and (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == str(shared_surface)
                                    ]
                                if filtered_times:
                                    row["avg_race_time_shared_combo"] = round(sum(filtered_times) / len(filtered_times), 2)

                    horse_name_lookup: dict[str, str] = {}
                    race_name_lookup: dict[str, str] = {}
                    race_horse_number_lookup: dict[tuple[str, str], int] = {}

                    def _upsert_horse_name(horse_key: str, horse_name: str) -> None:
                        if not horse_key:
                            return
                        existing = horse_name_lookup.get(horse_key)
                        if existing is None or (_is_placeholder_horse_name(existing) and not _is_placeholder_horse_name(horse_name)):
                            horse_name_lookup[horse_key] = horse_name

                    for row in pred.to_dict(orient="records"):
                        rid = str(row.get("race_id", ""))
                        hid = str(row.get("horse_id", ""))
                        hname = str(row.get("horse_name", hid))
                        rname = str(row.get("race_name", rid))

                        if rid and rid not in race_name_lookup:
                            race_name_lookup[rid] = rname
                        _upsert_horse_name(hid, hname)

                        number_val = row.get("number", row.get("draw"))
                        try:
                            num = int(number_val) if number_val is not None and not pd.isna(number_val) else None
                        except (TypeError, ValueError):
                            num = None
                        if rid and hid and num is not None:
                            race_horse_number_lookup[(rid, hid)] = num

                    for row in records:
                        rid = str(row.get("race_id", ""))
                        hid = str(row.get("horse_id", ""))
                        hname = str(row.get("horse_name", hid))
                        rname = str(row.get("race_name", rid))
                        if rid and rid not in race_name_lookup:
                            race_name_lookup[rid] = rname
                        _upsert_horse_name(hid, hname)

                    bulletin_by_race_no_and_number: dict[tuple[int, int], str] = {}
                    if city:
                        try:
                            data_source = _bulletin_source_for_ml(settings)
                            bulletin_races = fetch_races(data_source, parsed_date, city, None)
                            for race in bulletin_races:
                                race_no = int(race.race_no)
                                for entry in race.entries:
                                    bulletin_by_race_no_and_number[(race_no, int(entry.number))] = str(entry.horse_name)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("Ticket preview bulletin overlay skipped: %s", exc)

                    opt = optimize_for_date(
                        paths,
                        parsed_date,
                        settings,
                        budget=settings.phase4_default_budget,
                        prediction=pred,
                    )
                    ticket_preview = []
                    selected_race_id = str(race_df["race_id"].iloc[0]) if not race_df.empty else race_id
                    allowed_combo_race_ids = set(scoped_pred["race_id"].astype(str).tolist())
                    if requested_race_no is not None:
                        allowed_combo_race_ids = {
                            rid
                            for rid in allowed_combo_race_ids
                            if (_parse_race_no(rid) is not None and _parse_race_no(rid) >= requested_race_no)
                        }

                    for c in opt.get("columns", []):
                        combo = c.get("combination", {})
                        combo_race_ids = {str(k) for k in combo.keys()}
                        if selected_race_id not in combo:
                            continue
                        if not combo_race_ids.issubset(allowed_combo_race_ids):
                            continue
                        combo_readable: dict[str, str] = {}
                        for k, v in combo.items():
                            rid = str(k)
                            hid = str(v)
                            readable_race = race_name_lookup.get(rid, rid)
                            readable_horse = horse_name_lookup.get(hid, hid)

                            if _is_placeholder_horse_name(readable_horse):
                                race_no = _parse_race_no(rid)
                                horse_no = race_horse_number_lookup.get((rid, hid))
                                if race_no is not None and horse_no is not None:
                                    bulletin_name = bulletin_by_race_no_and_number.get((race_no, horse_no))
                                    if bulletin_name:
                                        readable_horse = bulletin_name

                            combo_readable[readable_race] = readable_horse
                        ticket_preview.append({**c, "combination": combo_readable})

                    ml_prediction = {
                        "race_id": race_id,
                        "race_name": str(records[0].get("race_name", race_id)) if records else race_id,
                        "rows": records,
                        "summary": {
                            "horse_count": int(len(records)),
                            "bet_count": int(sum(1 for r in records if r.get("bet_decision") == "BET")),
                            "top_probability": _safe_float(records[0].get("calibrated_probability"), 0.0) if records else 0.0,
                        },
                        "ticket_preview": ticket_preview,
                        "optimization_summary": opt.get("summary", {}),
                        "analysis": {
                            **_build_ml_analysis_context(paths, pred, records),
                            "horse_stats": horse_stats_rows,
                        },
                        "disclaimer": "Bu tahminler istatistiksel analize dayanir (Benter + Harville + fractional Kelly), kesinlik tasimaz; sorumlu bahis oynayin.",
                    }
                    if shared_combo_label:
                        ml_prediction["analysis"].setdefault("data_summary", []).append(shared_combo_label)

                    for field in _ML_SORTABLE_FIELDS:
                        next_direction = "asc" if field == ml_sort_by and sort_order == "desc" else "desc"
                        if field == ml_sort_by and sort_order == "asc":
                            next_direction = "desc"
                        sort_urls[field] = _build_predict_url(sort_field=field, sort_direction=next_direction)
                else:
                    # If race is from daily bulletin (e.g., TJK ids), generate ML-style table
                    # from PredictionEngine for the selected real race.
                    if selected_race is None:
                        error = "Secilen kosu bulunamadi. Lutfen listeden yeniden secin."
                    else:
                        race = selected_race
                        engine = PredictionEngine(data_source, settings)
                        race_pred = engine.predict(race)

                        records = []
                        for i, hp in enumerate(race_pred.ranked, start=1):
                            ev = hp.value_bet.expected_value if hp.value_bet and hp.value_bet.expected_value is not None else None
                            edge = hp.value_bet.edge if hp.value_bet and hp.value_bet.edge is not None else None
                            decision = "BET" if hp.value_bet and hp.value_bet.is_value_bet else "NO_BET"
                            records.append(
                                {
                                    "rank": i,
                                    "number": hp.entry.number,
                                    "horse_id": hp.entry.horse_id,
                                    "horse_name": hp.entry.horse_name,
                                    "odds": hp.entry.odds,
                                    "calibrated_probability": hp.win_probability if hp.win_probability is not None else 0.0,
                                    "confidence": hp.confidence_score if hp.confidence_score is not None else 0.0,
                                    "edge": edge,
                                    "ev": ev,
                                    "kelly_fraction": hp.value_bet.fractional_kelly_stake if hp.value_bet else 0.0,
                                    "bet_decision": decision,
                                    "race_id": str(race.id),
                                    "race_name": f"{race.hippodrome} - {race.race_no}. Kosu",
                                }
                            )

                        p2_fallback, p3_fallback = _place_probabilities_from_records(records)
                        for rec in records:
                            hid = str(rec.get("horse_id", ""))
                            rec["place2_probability"] = p2_fallback.get(hid, 0.0)
                            rec["place3_probability"] = p3_fallback.get(hid, 0.0)
                            rec["top3_probability"] = min(
                                1.0,
                                _safe_float(rec.get("calibrated_probability"), 0.0)
                                + _safe_float(rec.get("place2_probability"), 0.0)
                                + _safe_float(rec.get("place3_probability"), 0.0),
                            )

                        ml_sort_by = sort_by if sort_by in _ML_SORTABLE_FIELDS else "calibrated_probability"
                        reverse = sort_order == "desc"
                        records.sort(key=lambda r: _ml_sort_value(r, ml_sort_by), reverse=reverse)
                        for idx, rec in enumerate(records, start=1):
                            rec["rank"] = idx

                        ml_prediction = {
                            "race_id": str(race.id),
                            "race_name": f"{race.hippodrome} - {race.race_no}. Kosu",
                            "rows": records,
                            "summary": {
                                "horse_count": len(records),
                                "bet_count": int(sum(1 for r in records if r.get("bet_decision") == "BET")),
                                "top_probability": _safe_float(records[0].get("calibrated_probability"), 0.0) if records else 0.0,
                            },
                            "ticket_preview": [],
                            "optimization_summary": {
                                "status": "INFO",
                                "budget": settings.phase4_default_budget,
                                "spent": 0.0,
                                "column_count": 0,
                            },
                            "analysis": {
                                "data_notes": [
                                    "Bu ekran klasik skor motoru fallback'i ile calisiyor.",
                                    "ML pipeline yerine secili gercek kosu verisi kullanildi.",
                                    "Bu nedenle model katsayilari yerine kosu bazli istatistikler gosteriliyor.",
                                ],
                                "feature_notes": [
                                    f"Top at sayisi: {len(records)}",
                                    f"En yuksek P(win): {_safe_float(records[0].get('calibrated_probability'), 0.0) if records else 0.0:.3f}",
                                    f"BET sinyali: {int(sum(1 for r in records if r.get('bet_decision') == 'BET'))}",
                                ],
                                "data_summary": [
                                    f"Hipodrom: {race.hippodrome}",
                                    f"Kosu no: {race.race_no}",
                                    f"Mesafe: {race.distance_m}m",
                                    f"Pist: {race.surface.value if hasattr(race.surface, 'value') else race.surface}",
                                ],
                                "horse_stats": [],
                                "method_notes": [
                                    "Fallback mod: gercek kosu verisi uzerinden klasik skor + EV/Kelly turetimi.",
                                    "Sentetik veri ile doldurma yapilmaz.",
                                ],
                            },
                            "disclaimer": "Bu tahminler istatistiksel analize dayanir (Benter + Harville + fractional Kelly), kesinlik tasimaz; sorumlu bahis oynayin.",
                        }

                        combo_counter: dict[tuple[int, str], int] = {}
                        for hp in race_pred.ranked:
                            try:
                                stats = data_source.get_horse_statistics(hp.entry)
                            except Exception:
                                continue
                            for perf in stats.past_performances:
                                if perf.race_date >= parsed_date:
                                    continue
                                key = (int(perf.distance_m), perf.surface.value if hasattr(perf.surface, "value") else str(perf.surface))
                                combo_counter[key] = combo_counter.get(key, 0) + 1

                        shared_combo = max(combo_counter.items(), key=lambda item: item[1])[0] if combo_counter else None
                        if shared_combo is not None:
                            shared_distance, shared_surface = shared_combo
                            ml_prediction["analysis"]["data_summary"].append(
                                f"Ortak pist+mesafe kombinasyonu: {shared_distance}m / {shared_surface}"
                            )

                        for r in records:
                            hp = next((x for x in race_pred.ranked if x.entry.horse_id == r.get("horse_id")), None)
                            row = {
                                "horse": r.get("horse_name", r.get("horse_id", "-")),
                                "horse_id": r.get("horse_id", "-"),
                                "number": r.get("number", "-"),
                                "draw": r.get("number", "-"),
                                "weight": getattr(hp.entry, "weight_kg", "-") if hp else "-",
                                "distance": getattr(race, "distance_m", "-"),
                                "field_size": len(records),
                                "form_avg_3": getattr(hp.score, "form_score", "-") if hp else "-",
                                "form_avg_5": "-",
                                "form_avg_10": "-",
                                "days_since_last_race": "-",
                                "track_fit": getattr(hp.score, "distance_surface_score", "-") if hp else "-",
                                "surface_fit": getattr(hp.score, "distance_surface_score", "-") if hp else "-",
                                "distance_fit": getattr(hp.score, "distance_surface_score", "-") if hp else "-",
                                "pace_pressure": "-",
                                "market_probability_norm": r.get("edge", "-"),
                                "edge": r.get("edge", "-"),
                                "ev": r.get("ev", "-"),
                                "kelly_fraction": r.get("kelly_fraction", "-"),
                                "avg_race_time_shared_combo": "-",
                                "source_horse_id": str(getattr(hp.entry, "source_horse_id", "") or "") if hp else "",
                                "tjk_stats_url": "",
                                "tjk_workout_url": "",
                                "tjk_error": "",
                                "tjk_summary": [],
                                "tjk_history": [],
                            }
                            if hp and hp.entry.source_horse_id is not None:
                                row["tjk_stats_url"] = (
                                    "https://www.tjk.org/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri?"
                                    f"QueryParameter_AtId={hp.entry.source_horse_id}&&Era=past"
                                )
                                row["tjk_workout_url"] = (
                                    "https://www.tjk.org/TR/YarisSever/Query/Page/IdmanIstatistikleri?"
                                    f"QueryParameter_AtId={hp.entry.source_horse_id}"
                                )
                            elif hp:
                                row["tjk_error"] = "AtKosuBilgileri icin gerekli AtId (QueryParameter_AtId) bulunamadi."

                            stats = None
                            if hp:
                                try:
                                    stats = data_source.get_horse_statistics(hp.entry)
                                except Exception as exc:
                                    row["tjk_error"] = str(exc)

                            if stats is not None:
                                row.update(
                                    _derive_horse_stats_metrics(
                                        stats,
                                        race_distance=int(race.distance_m),
                                        race_surface=(race.surface.value if hasattr(race.surface, "value") else str(race.surface)),
                                        race_track=(race.hippodrome or ""),
                                    )
                                )
                                row["weight"] = row.get("weight", "-") if row.get("weight", "-") not in ("-", None) else (
                                    round(sum(float(p.weight_kg) for p in stats.past_performances if p.weight_kg is not None) / max(1, len([p for p in stats.past_performances if p.weight_kg is not None])), 1)
                                    if any(p.weight_kg is not None for p in stats.past_performances)
                                    else "-"
                                )
                                row["tjk_summary"] = [
                                    f"Kariyer kosu: {stats.career_starts}",
                                    f"Kariyer galibiyet: {stats.career_wins}",
                                    f"Kariyer plase: {stats.career_places}",
                                    f"Son yil kosu: {stats.last_year_starts}",
                                    f"Son yil galibiyet: {stats.last_year_wins}",
                                    f"Son yil plase: {stats.last_year_places}",
                                    f"Idman kaydi: {len(stats.workout_records)}",
                                ]

                                history_rows = []
                                for perf in stats.past_performances[:10]:
                                    history_rows.append(
                                        {
                                            "race_date": perf.race_date.isoformat(),
                                            "hippodrome": perf.hippodrome,
                                            "distance_m": perf.distance_m,
                                            "surface": perf.surface.value if hasattr(perf.surface, "value") else str(perf.surface),
                                            "finish_position": perf.finish_position,
                                            "race_time_seconds": perf.race_time_seconds,
                                            "weight_kg": perf.weight_kg,
                                            "equipment": perf.equipment,
                                            "jockey_name": perf.jockey_name,
                                            "field_size": perf.field_size,
                                            "odds": perf.odds,
                                            "group_info": perf.group_info,
                                            "race_name": perf.race_name,
                                            "race_class": perf.race_class,
                                            "trainer_name": perf.trainer_name,
                                            "owner_name": perf.owner_name,
                                            "handicap_points": perf.handicap_points,
                                            "prize_info": perf.prize_info,
                                            "s20": perf.s20,
                                            "extra1": None,
                                            "extra2": None,
                                        }
                                    )
                                row["tjk_history"] = history_rows

                                target_distance = int(race.distance_m)
                                target_surface = race.surface.value if hasattr(race.surface, "value") else str(race.surface)
                                filtered_times = [
                                    float(p.race_time_seconds)
                                    for p in stats.past_performances
                                    if p.race_time_seconds is not None
                                    and p.race_date < parsed_date
                                    and int(p.distance_m) == target_distance
                                    and (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == target_surface
                                ]
                                if not filtered_times and shared_combo is not None:
                                    shared_distance, shared_surface = shared_combo
                                    filtered_times = [
                                        float(p.race_time_seconds)
                                        for p in stats.past_performances
                                        if p.race_time_seconds is not None
                                        and p.race_date < parsed_date
                                        and int(p.distance_m) == int(shared_distance)
                                        and (p.surface.value if hasattr(p.surface, "value") else str(p.surface)) == str(shared_surface)
                                    ]
                                if filtered_times:
                                    row["avg_race_time_shared_combo"] = round(sum(filtered_times) / len(filtered_times), 2)

                            ml_prediction["analysis"]["horse_stats"].append(row)

                        for field in _ML_SORTABLE_FIELDS:
                            next_direction = "asc" if field == ml_sort_by and sort_order == "desc" else "desc"
                            if field == ml_sort_by and sort_order == "asc":
                                next_direction = "desc"
                            sort_urls[field] = _build_predict_url(sort_field=field, sort_direction=next_direction)

                return templates.TemplateResponse(
                    request,
                    "predict.html",
                    {
                        "prediction": prediction,
                        "ml_prediction": ml_prediction,
                        "rows": display_rows,
                        "pdf_url": pdf_url,
                        "table_usage_notes": table_usage_notes,
                        "error": error,
                        "source": source,
                        "date": date_str,
                        "city": city,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                        "sort_urls": sort_urls,
                        "sortable_fields": _ML_SORTABLE_FIELDS,
                    },
                )

            data_source = build_data_source(source, settings)
            if source == "tjk" and not city:
                error = "TJK kaynaginda tahmin uretmeden once bir hipodrom secin."
                return templates.TemplateResponse(
                    request,
                    "predict.html",
                    {
                        "prediction": prediction,
                        "ml_prediction": ml_prediction,
                        "rows": display_rows,
                        "table_usage_notes": table_usage_notes,
                        "error": error,
                        "source": source,
                        "date": date_str,
                        "city": city,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                        "sort_urls": sort_urls,
                        "sortable_fields": _SORTABLE_FIELDS,
                    },
                )

            races = fetch_races(data_source, parsed_date, city or None, None)
            race = next((r for r in races if r.id == race_id), None)
            if race is None:
                error = "Yaris bulunamadi (bulten degismis olabilir; lutfen tekrar secin)."
            else:
                engine = PredictionEngine(data_source, settings)
                prediction = engine.predict(race)
                reverse = sort_order == "desc"
                ranked_by_key = {_entry_key(hp.entry): hp for hp in prediction.ranked}
                active_rows = [
                    SimpleNamespace(
                        entry=hp.entry,
                        score=hp.score,
                        reasoning=hp.reasoning,
                        tag=hp.tag,
                        win_probability=hp.win_probability,
                        confidence_score=hp.confidence_score,
                        value_bet=hp.value_bet,
                        feature_snapshot=hp.feature_snapshot,
                        is_scratched=False,
                    )
                    for hp in prediction.ranked
                ]

                unscored_active_rows = [
                    SimpleNamespace(
                        entry=entry,
                        score=None,
                        reasoning=["Bu at aktif durumda; ancak istatistik verisi alinamadigi icin skorlanamadi."],
                        tag="Veri Eksik",
                        win_probability=None,
                        confidence_score=None,
                        value_bet=None,
                        feature_snapshot={},
                        is_scratched=False,
                    )
                    for entry in race.entries
                    if not entry.is_scratched and _entry_key(entry) not in ranked_by_key
                ]

                scratched_rows = [
                    SimpleNamespace(
                        entry=entry,
                        score=None,
                        reasoning=["Bu at kosmaz (scratch) olarak isaretli."],
                        tag="Koşmaz",
                        win_probability=None,
                        confidence_score=None,
                        value_bet=None,
                        feature_snapshot={},
                        is_scratched=True,
                    )
                    for entry in race.entries
                    if entry.is_scratched
                ]

                if sort_by == "number":
                    display_rows = sorted(
                        active_rows + unscored_active_rows + scratched_rows,
                        key=lambda row: _prediction_sort_key(row, sort_by),
                        reverse=reverse,
                    )
                else:
                    display_rows = sorted(
                        active_rows,
                        key=lambda row: _prediction_sort_key(row, sort_by),
                        reverse=reverse,
                    ) + sorted(
                        unscored_active_rows, key=lambda row: row.entry.number
                    ) + sorted(
                        scratched_rows, key=lambda row: row.entry.number
                    )

                pdf_url = "/predict-all-pdf?" + urlencode(
                    {
                        "source": source,
                        "date": date_str,
                        "city": city,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                    }
                )

                for field in _SORTABLE_FIELDS:
                    next_direction = "asc" if field == sort_by and sort_order == "desc" else "desc"
                    if field == sort_by and sort_order == "asc":
                        next_direction = "desc"
                    sort_urls[field] = _build_predict_url(sort_field=field, sort_direction=next_direction)
        except (InvalidSourceError, ValueError) as exc:
            error = f"Gecersiz istek: {exc}"
        except RuntimeError as exc:
            error = (
                f"Tahmin uretilemedi: {exc} "
                "(TJK kaynaginda at istatistikleri su an sinirli/erisilemez olabilir)."
            )
        except DataSourceError as exc:
            error = f"Veri kaynagi hatasi: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Predict endpoint failure")
            error = f"Beklenmeyen hata: {exc}"

        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "prediction": prediction,
                "ml_prediction": ml_prediction,
                "rows": display_rows,
                "pdf_url": pdf_url,
                "table_usage_notes": table_usage_notes,
                "error": error,
                "source": source,
                "date": date_str,
                "city": city,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "sort_urls": sort_urls,
                "sortable_fields": _SORTABLE_FIELDS,
            },
        )

    @app.get("/predict-all-pdf")
    def predict_all_pdf(
        source: str = Query("sample", pattern="^(sample|tjk|ml)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
        sort_by: str = Query(
            "strategy",
            pattern="^(strategy|number|odds|total|form|jockey_trainer|distance_surface|weight|rest|win_probability|confidence|ev|kelly)$",
        ),
        sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    ) -> Response:
        settings = get_settings()
        if source == "ml":
            raise HTTPException(status_code=400, detail="ML modunda PDF ciktisi simdilik desteklenmiyor.")
        if not city:
            raise HTTPException(status_code=400, detail="PDF olusturmak icin once bir hipodrom secin.")

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, landscape
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            import reportlab
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail="PDF olusturma icin 'reportlab' kurulu olmali. Lutfen bagimliliklari guncelleyin.",
            ) from exc

        parsed_date = parse_date(date_str or None)
        data_source = build_data_source(source, settings)
        races = fetch_races(data_source, parsed_date, city or None, None)
        if not races:
            raise HTTPException(
                status_code=400,
                detail="Secili tarih/hipodrom icin yaris bulunamadi; PDF olusturulamadi.",
            )

        engine = PredictionEngine(data_source, settings)
        from io import BytesIO

        def _resolve_pdf_font_paths() -> tuple[Path | None, Path | None]:
            candidates = [
                (
                    Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
                    Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf",
                ),
                (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
                (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
                (
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                ),
                (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
            ]
            for candidate in candidates:
                if isinstance(candidate, tuple):
                    regular, bold = candidate
                    if regular.exists() and bold.exists():
                        return regular, bold
                elif candidate.exists():
                    return candidate, None
            return None, None

        body_font_name = "Helvetica"
        header_font_name = "Helvetica-Bold"
        body_font_path, header_font_path = _resolve_pdf_font_paths()
        fallback_ascii_pdf = body_font_path is None
        if body_font_path is not None:
            body_font_name = "TurkishSans"
            if body_font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(body_font_name, str(body_font_path)))
            if header_font_path is not None:
                header_font_name = "TurkishSansBold"
                if header_font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(header_font_name, str(header_font_path)))
            else:
                header_font_name = body_font_name

        _pdf_ascii_map = str.maketrans(
            {
                "ç": "c",
                "Ç": "C",
                "ğ": "g",
                "Ğ": "G",
                "ı": "i",
                "İ": "I",
                "ö": "o",
                "Ö": "O",
                "ş": "s",
                "Ş": "S",
                "ü": "u",
                "Ü": "U",
            }
        )

        def _pdf_text(value: str) -> str:
            text = value.translate(_pdf_ascii_map) if fallback_ascii_pdf else value
            return escape(text)

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=landscape(A4),
            leftMargin=12 * mm,
            rightMargin=12 * mm,
            topMargin=12 * mm,
            bottomMargin=12 * mm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleTurkish",
            parent=styles["Heading2"],
            fontName=header_font_name,
            fontSize=13,
            leading=16,
        )
        info_style = ParagraphStyle(
            "InfoTurkish",
            parent=styles["Normal"],
            fontName=body_font_name,
            fontSize=9,
            leading=12,
        )
        race_style = ParagraphStyle(
            "RaceTurkish",
            parent=styles["Normal"],
            fontName=header_font_name,
            fontSize=10,
            leading=13,
        )
        cell_style = ParagraphStyle(
            "CellTurkish",
            parent=styles["Normal"],
            fontName=body_font_name,
            fontSize=8,
            leading=10,
        )
        cell_numeric_style = ParagraphStyle(
            "CellNumericTurkish",
            parent=cell_style,
            alignment=2,
        )
        header_cell_style = ParagraphStyle(
            "HeaderCellTurkish",
            parent=cell_style,
            fontName=header_font_name,
            fontSize=8.5,
            leading=10.5,
            alignment=1,
        )

        elements = [
            Paragraph(
                _pdf_text(f"Tahmin Raporu - {city} - {parsed_date.strftime('%d.%m.%Y')}"),
                title_style,
            ),
            Paragraph(_pdf_text(f"Kaynak: {source}"), info_style),
            Spacer(1, 6),
        ]

        generated_race_count = 0
        skipped_races: list[str] = []

        for race in races:
            try:
                prediction = engine.predict(
                    race,
                    include_backtest=False,
                    include_detailed_reasoning=False,
                )
            except (ValueError, RuntimeError, DataSourceError) as exc:
                skipped_races.append(f"{race.race_no}. kosu: {exc}")
                logger.warning(
                    "PDF olusturulurken %s %s. kosu atlandi: %s",
                    race.hippodrome,
                    race.race_no,
                    exc,
                )
                continue

            generated_race_count += 1
            elements.append(
                Paragraph(
                    _pdf_text(
                        f"Koşu {race.race_no} ({race.start_time.strftime('%H:%M')}) - "
                        f"{race.distance_m}m {race.surface.value}"
                    ),
                    race_style,
                )
            )

            reverse = sort_order == "desc"
            ranked = sorted(
                prediction.ranked,
                key=lambda hp: _prediction_sort_key(hp, sort_by),
                reverse=reverse,
            )
            rows = [
                [
                    Paragraph("No", header_cell_style),
                    Paragraph("At", header_cell_style),
                    Paragraph("Ganyan", header_cell_style),
                    Paragraph("K.Olas.", header_cell_style),
                    Paragraph("Guven", header_cell_style),
                    Paragraph("EV", header_cell_style),
                    Paragraph("Kelly", header_cell_style),
                    Paragraph("Toplam", header_cell_style),
                    Paragraph("Etiket", header_cell_style),
                ]
            ]
            for hp in ranked:
                odds_text = f"{hp.entry.odds:.2f}" if hp.entry.odds is not None else "-"
                prob_text = f"{(hp.win_probability or 0.0) * 100:.2f}%" if hp.win_probability is not None else "-"
                conf_text = f"{hp.confidence_score:.2f}" if hp.confidence_score is not None else "-"
                ev_text = (
                    f"{hp.value_bet.expected_value:.3f}"
                    if hp.value_bet and hp.value_bet.expected_value is not None
                    else "-"
                )
                kelly_text = (
                    f"{hp.value_bet.fractional_kelly_stake:.3f}"
                    if hp.value_bet and hp.value_bet.fractional_kelly_stake is not None
                    else "-"
                )
                rows.append(
                    [
                        Paragraph(str(hp.entry.number), cell_numeric_style),
                        Paragraph(_pdf_text(hp.entry.horse_name), cell_style),
                        Paragraph(odds_text, cell_numeric_style),
                        Paragraph(prob_text, cell_numeric_style),
                        Paragraph(conf_text, cell_numeric_style),
                        Paragraph(ev_text, cell_numeric_style),
                        Paragraph(kelly_text, cell_numeric_style),
                        Paragraph(f"{hp.score.total_score:.1f}", cell_numeric_style),
                        Paragraph(_pdf_text(hp.tag), cell_style),
                    ]
                )

            table = LongTable(
                rows,
                colWidths=[10 * mm, 58 * mm, 16 * mm, 18 * mm, 15 * mm, 14 * mm, 14 * mm, 15 * mm, 30 * mm],
                repeatRows=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 8))

        if generated_race_count == 0:
            detail = "Secili tarih/hipodrom icin PDF uretilemedi; kosular tahminlenemedi."
            if skipped_races:
                detail = f"{detail} Ilk hata: {skipped_races[0]}"
            raise HTTPException(status_code=400, detail=detail)

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        file_name = f"tahmin-raporu-{city}-{parsed_date.isoformat()}.pdf".replace(" ", "_")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
        )

    return app
