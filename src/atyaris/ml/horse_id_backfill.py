"""Resumable repair of source horse IDs in previously collected TJK data."""
from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from atyaris.data_sources.tjk_scraper import (
    TJKHtmlDataSource,
    _match_hippodrome,
    _normalize_city,
)
from atyaris.ml.horse_id_mapping import HorseIdMappingStore, resolve_race_horse_ids
from atyaris.ml.pipeline import Phase1Paths
from atyaris.ml.raw_store import JsonlRawStore
from atyaris.models.entities import Jockey, Race, RaceEntry, TrackSurface, Trainer


def _unit_key(record: dict[str, Any]) -> tuple[str, str] | None:
    try:
        return str(record["race_date"]), _normalize_city(str(record["hippodrome"]))
    except (KeyError, TypeError, ValueError):
        return None


def _checkpoint_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _race_objects(records: list[dict[str, Any]]) -> list[Race]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record.get("race_id", "")), _as_int(record.get("race_no"))), []).append(record)

    races = []
    for (race_id, race_no), rows in grouped.items():
        if not rows:
            continue
        first = rows[0]
        try:
            start_time = datetime.fromisoformat(str(first.get("race_datetime") or first["race_date"]))
        except (KeyError, TypeError, ValueError):
            start_time = datetime.combine(date.fromisoformat(str(first["race_date"])), datetime.min.time())
        try:
            surface = TrackSurface(str(first.get("surface", TrackSurface.KUM.value)))
        except ValueError:
            surface = TrackSurface.KUM
        entries = []
        for row in rows:
            source_id = row.get("source_horse_id")
            entries.append(
                RaceEntry(
                    number=_as_int(row.get("draw")),
                    horse_id=str(row.get("horse_id", "")),
                    source_horse_id=_as_int(source_id) if source_id not in (None, "") else None,
                    horse_name=str(row.get("horse_name", "")),
                    age=_as_int(row.get("age")) if row.get("age") not in (None, "") else None,
                    jockey=Jockey(name=str(row.get("jockey_name") or "Bilinmiyor")),
                    trainer=Trainer(
                        name=str(row.get("trainer_name") or "Bilinmiyor"),
                        source_trainer_id=(
                            _as_int(row.get("trainer_id"))
                            if row.get("trainer_id") not in (None, "")
                            else None
                        ),
                    ),
                    weight_kg=float(row.get("weight_kg") or 0.0),
                )
            )
        races.append(
            Race(
                id=race_id,
                hippodrome=str(first.get("hippodrome", "")),
                race_no=race_no,
                start_time=start_time,
                distance_m=_as_int(first.get("distance_m")),
                surface=surface,
                entries=entries,
            )
        )
    return races


def backfill_horse_ids(
    paths: Phase1Paths,
    source: TJKHtmlDataSource,
    progress_callback: Callable[[str], None] | None = None,
    batch_units: int = 10,
    retry_unresolved: bool = False,
) -> dict[str, Any]:
    """Resolve legacy JSONL/CSV IDs; replay uncheckpointed batches safely."""
    program_store = JsonlRawStore(paths.raw_daily_program_jsonl, "daily_program")
    result_store = JsonlRawStore(paths.raw_race_results_jsonl, "race_results")
    history_store = JsonlRawStore(paths.raw_history_jsonl, "horse_history")
    mapping = HorseIdMappingStore(paths.raw_horse_id_mapping_jsonl)
    program_records = program_store.read_latest_records()
    result_records = result_store.read_latest_records()
    history_records = history_store.read_latest_records()
    csv_frame = pd.read_csv(paths.raw_csv, dtype=str, keep_default_na=False) if paths.raw_csv.exists() else pd.DataFrame()

    checkpoint = paths.raw_horse_id_mapping_jsonl.with_suffix(".backfill.progress.json")
    state: dict[str, Any] = {"version": 1, "completed_units": []}
    if checkpoint.exists():
        try:
            previous_state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if previous_state.get("version") == 1:
                state = previous_state
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    completed_units = set(str(item) for item in state.get("completed_units", []))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    program_keys: set[tuple[str, str]] = set()
    for record in program_records:
        key = _unit_key(record)
        if key is not None:
            grouped.setdefault(key, []).append(record)
            program_keys.add((str(record.get("race_id", "")), str(record.get("horse_id", ""))))

    csv_records_by_unit: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not csv_frame.empty:
        for row in csv_frame.to_dict(orient="records"):
            try:
                record_date = str(row.get("date", ""))[:10]
                race_id = str(row.get("race_id", ""))
                race_match = __import__("re").search(r"-(\d+)$", race_id)
                race_no = _as_int(row.get("race_no"), int(race_match.group(1)) if race_match else 0)
                raw_city = str(row.get("hippodrome") or row.get("track") or "")
                hippodrome = _match_hippodrome(raw_city) or raw_city
                horse_name = str(row.get("horse_name", ""))
                horse_id = str(row.get("horse_id", ""))
                if not record_date or not race_id or not race_no or not horse_name:
                    continue
                csv_record = {
                    "race_id": race_id,
                    "horse_id": horse_id,
                    "race_date": record_date,
                    "race_datetime": str(row.get("race_datetime") or record_date),
                    "hippodrome": hippodrome,
                    "race_no": race_no,
                    "horse_name": horse_name,
                    "draw": _as_int(row.get("draw")),
                    "age": _as_int(row.get("age")) if row.get("age") not in (None, "") else None,
                    "weight_kg": float(row.get("weight") or 0.0),
                    "distance_m": _as_int(row.get("distance")),
                    "surface": str(row.get("surface") or TrackSurface.KUM.value),
                    "_backfill_csv_only": True,
                }
            except (TypeError, ValueError):
                continue
            key = _unit_key(csv_record)
            if key is None:
                continue
            identity = (race_id, horse_id)
            if identity in program_keys:
                continue
            grouped.setdefault(key, []).append(csv_record)
            csv_records_by_unit.setdefault(key, []).append(csv_record)

    results_by_unit: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in result_records:
        key = _unit_key(record)
        if key is not None:
            results_by_unit.setdefault(key, []).append(record)
    history_by_unit: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in history_records:
        try:
            key = (
                str(record["target_date"]),
                _normalize_city(str(record["target_hippodrome"])),
            )
        except (KeyError, TypeError, ValueError):
            continue
        history_by_unit.setdefault(key, []).append(record)

    pending_program: dict[tuple[str, str], dict[str, Any]] = {}
    pending_results: dict[tuple[str, str], dict[str, Any]] = {}
    pending_history: dict[tuple[str, str], dict[str, Any]] = {}
    pending_units: list[str] = []
    resolved_count = unresolved_count = processed_units = 0
    unresolved_details: list[dict[str, Any]] = []
    csv_dirty = False

    def persist_batch() -> None:
        nonlocal csv_dirty
        if pending_program or pending_results or pending_history or pending_units:
            if pending_program:
                program_store.upsert(pending_program.values())
            if pending_results:
                result_store.upsert(pending_results.values())
            if pending_history:
                history_store.upsert(pending_history.values())
            if csv_dirty:
                temporary = paths.raw_csv.with_suffix(paths.raw_csv.suffix + ".horse-id.tmp")
                csv_frame.to_csv(temporary, index=False)
                temporary.replace(paths.raw_csv)
                csv_dirty = False
            completed_units.update(pending_units)
            state["completed_units"] = sorted(completed_units)
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _checkpoint_write(checkpoint, state)
            pending_program.clear()
            pending_results.clear()
            pending_history.clear()
            pending_units.clear()

    for (date_text, city_key), records in sorted(grouped.items()):
        unit = f"{date_text}|{city_key}"
        if unit in completed_units and (
            not retry_unresolved or all(record.get("source_horse_id") for record in records)
        ):
            resolved_count += sum(bool(record.get("source_horse_id")) for record in records)
            for record in records:
                if record.get("source_horse_id"):
                    continue
                unresolved_count += 1
                candidate_ids = record.get("id_candidate_ids", [])
                if isinstance(candidate_ids, str):
                    try:
                        candidate_ids = json.loads(candidate_ids)
                    except json.JSONDecodeError:
                        candidate_ids = []
                unresolved_details.append(
                    {
                        "race_date": date_text,
                        "hippodrome": record.get("hippodrome", city_key),
                        "race_no": _as_int(record.get("race_no")),
                        "horse_name": record.get("horse_name", ""),
                        "status": record.get("id_resolution_status", "unresolved"),
                        "candidate_ids": candidate_ids,
                    }
                )
            continue
        target_date = date.fromisoformat(date_text)
        hippodrome = str(records[0].get("hippodrome", city_key))
        races = _race_objects(records)
        event_ids: dict[tuple[str, str, int, str], tuple[int | None, str, list[int]]] = {}
        try:
            report = resolve_race_horse_ids(
                source, races, target_date, hippodrome, mapping, program_records
            )
        except (OSError, ValueError) as exc:
            report = {"resolved": 0, "unresolved": sum(len(r.entries) for r in races), "collision_names": []}
            if progress_callback:
                progress_callback(f"{unit}: ID onarimi ertelendi: {exc}")

        entries_by_event = {
            (target_date.isoformat(), city_key, race.race_no, mapping.name_key(entry.horse_name)): entry
            for race in races
            for entry in race.entries
        }
        resolved_count += int(report.get("resolved", 0))
        unresolved_count += int(report.get("unresolved", 0))
        unresolved_details.extend(report.get("unresolved_entries", []))
        for record in records:
            key = (
                target_date.isoformat(), city_key, _as_int(record.get("race_no")),
                mapping.name_key(str(record.get("horse_name", ""))),
            )
            entry = entries_by_event.get(key)
            if entry is None:
                continue
            updated = dict(record)
            updated.update(
                source_horse_id=entry.source_horse_id,
                id_unresolved=entry.id_unresolved,
                id_resolution_status=entry.id_resolution_status,
                id_candidate_ids=entry.id_candidate_ids,
            )
            if not record.get("_backfill_csv_only"):
                pending_program[(str(updated["race_id"]), str(updated["horse_id"]))] = updated
            event_ids[key] = (
                entry.source_horse_id,
                entry.id_resolution_status,
                list(entry.id_candidate_ids),
            )

        for record in results_by_unit.get((date_text, city_key), []):
            key = (
                str(record.get("race_date", "")),
                _normalize_city(str(record.get("hippodrome", ""))),
                _as_int(record.get("race_no")),
                mapping.name_key(str(record.get("horse_name", ""))),
            )
            if key not in event_ids:
                continue
            source_id, status, candidates = event_ids[key]
            updated = dict(record)
            updated.update(
                source_horse_id=source_id,
                id_unresolved=source_id is None,
                id_resolution_status=status,
                id_candidate_ids=candidates,
            )
            pending_results[(str(updated["race_id"]), str(updated["horse_id"]))] = updated

        for record in history_by_unit.get((date_text, city_key), []):
            try:
                key = (
                    str(record["target_date"]),
                    _normalize_city(str(record["target_hippodrome"])),
                    int(record["target_race_no"]),
                    mapping.name_key(str(record["target_horse_name"])),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if key not in event_ids:
                continue
            source_id, status, candidates = event_ids[key]
            updated = dict(record)
            updated.update(
                source_horse_id=source_id,
                id_unresolved=source_id is None,
                id_resolution_status=status,
                id_candidate_ids=candidates,
            )
            pending_history[(str(updated["horse_id"]), str(updated["race_id"]), str(updated["race_date"]))] = updated

        if not csv_frame.empty:
            for event_key, (source_id, status, candidates) in event_ids.items():
                event_date, _event_city, race_no, name_key = event_key
                matching = (
                    (csv_frame["date"].astype(str).str[:10] == event_date)
                    & (csv_frame["race_id"].astype(str) == str(next(
                        (race.id for race in races if race.race_no == race_no), ""
                    )))
                    & csv_frame["horse_name"].map(mapping.name_key).eq(name_key)
                )
                if not bool(matching.any()):
                    continue
                csv_frame.loc[matching, "source_horse_id"] = str(source_id or "")
                csv_frame.loc[matching, "id_unresolved"] = source_id is None
                csv_frame.loc[matching, "id_resolution_status"] = status
                csv_frame.loc[matching, "id_candidate_ids"] = json.dumps(candidates)
                csv_dirty = True

        pending_units.append(unit)
        processed_units += 1
        if progress_callback:
            progress_callback(
                f"At ID onarimi: {unit} | cozuldu={report.get('resolved', 0)} "
                f"| unresolved={report.get('unresolved', 0)}"
            )
        if len(pending_units) >= max(batch_units, 1):
            persist_batch()

    persist_batch()
    collision_report = mapping.collision_report()
    report = {
        "units_processed": processed_units,
        "units_checkpointed": len(completed_units),
        "resolved_entries": resolved_count,
        "unresolved_entries": unresolved_count,
        "collision_report": collision_report,
        "checkpoint": str(checkpoint),
    }
    report_path = paths.raw_horse_id_mapping_jsonl.with_suffix(".collision-report.json")
    unresolved_report_path = paths.raw_horse_id_mapping_jsonl.with_suffix(
        ".unresolved-report.json"
    )
    _checkpoint_write(
        unresolved_report_path,
        {"unresolved_entries": unresolved_count, "entries": unresolved_details},
    )
    report["unresolved_report"] = str(unresolved_report_path)
    _checkpoint_write(report_path, report)
    return report