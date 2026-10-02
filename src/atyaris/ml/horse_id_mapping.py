"""Persistent, collision-aware TJK horse ID observations."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import random
from typing import Any

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import (
    TJKHtmlDataSource,
    _normalize_city,
    _horse_identity_key,
)
from atyaris.models.entities import Race
from atyaris.ml.raw_store import JsonlRawStore


class HorseIdMappingStore:
    """Keep one record per normalized name and verified TJK ID pair."""

    def __init__(self, path: Path) -> None:
        self._store = JsonlRawStore(path, "horse_id_mapping")
        self._records: dict[tuple[str, int], dict[str, Any]] = {}
        self._by_name: dict[str, set[int]] = {}
        for record in self._store.read_latest_records():
            name_key = str(record.get("horse_name_key", ""))
            resolved_id = record.get("resolved_at_id")
            if not name_key or resolved_id is None:
                continue
            pair = (name_key, int(resolved_id))
            self._records[pair] = record
            self._by_name.setdefault(name_key, set()).add(int(resolved_id))

    @staticmethod
    def name_key(horse_name: str) -> str:
        return _horse_identity_key(horse_name)

    def observe(
        self,
        horse_name: str,
        resolved_at_id: int,
        seen_date: date,
        confidence_source: str = "html_program_link",
    ) -> list[int]:
        collisions = self.observe_many([(horse_name, resolved_at_id, seen_date)])
        return collisions.get(self.name_key(horse_name), [])

    def observe_many(
        self,
        observations: list[tuple[str, int, date]],
    ) -> dict[str, list[int]]:
        incoming: dict[tuple[str, int], dict[str, Any]] = {}
        incoming_names: dict[str, set[int]] = {}
        for horse_name, resolved_at_id, seen_date in observations:
            name_key = self.name_key(horse_name)
            resolved_id = int(resolved_at_id)
            pair = (name_key, resolved_id)
            previous = incoming.get(pair, self._records.get(pair))
            seen_text = seen_date.isoformat()
            incoming[pair] = {
                "horse_name": horse_name,
                "horse_name_key": name_key,
                "resolved_at_id": resolved_id,
                "first_seen_date": min(
                    seen_text,
                    str(previous.get("first_seen_date", seen_text)) if previous else seen_text,
                ),
                "last_seen_date": max(
                    seen_text,
                    str(previous.get("last_seen_date", seen_text)) if previous else seen_text,
                ),
                "confidence_source": "html_program_link",
                "collision_detected": bool(previous and previous.get("collision_detected")),
                "collision_ids": list(previous.get("collision_ids", [])) if previous else [],
            }
            incoming_names.setdefault(name_key, set()).add(resolved_id)

        collision_report: dict[str, list[int]] = {}
        for name_key, new_ids in incoming_names.items():
            all_ids = sorted(self._by_name.get(name_key, set()) | new_ids)
            if len(all_ids) < 2:
                continue
            collision_report[name_key] = all_ids
            for resolved_id in all_ids:
                pair = (name_key, resolved_id)
                record = dict(incoming[pair] if pair in incoming else self._records[pair])
                record["collision_detected"] = True
                record["collision_ids"] = all_ids
                incoming[pair] = record

        if incoming:
            self._store.upsert(incoming.values())
            for pair, record in incoming.items():
                self._records[pair] = record
                self._by_name.setdefault(pair[0], set()).add(pair[1])
        return collision_report

    def resolve_for_date(self, horse_name: str, target_date: date) -> tuple[int | None, str, list[int]]:
        name_key = self.name_key(horse_name)
        ids = self._by_name.get(name_key, set())
        eligible = {
            resolved_id
            for candidate_key, resolved_id in self._records
            if candidate_key == name_key
            and self._records[(candidate_key, resolved_id)]["first_seen_date"]
            <= target_date.isoformat()
            <= self._records[(candidate_key, resolved_id)]["last_seen_date"]
        }
        if len(eligible) == 1:
            resolved_id = next(iter(eligible))
            return resolved_id, "cached", [resolved_id]
        if len(eligible) > 1 or (ids and not eligible and len(ids) > 1):
            return None, "ambiguous", sorted(eligible or ids)
        if len(ids) == 1:
            resolved_id = next(iter(ids))
            record = self._records[(name_key, resolved_id)]
            if target_date.isoformat() >= str(record["first_seen_date"]):
                return resolved_id, "cached", [resolved_id]
        return None, "unresolved", sorted(ids)

    def collision_report(self) -> dict[str, Any]:
        collisions = []
        for name_key, resolved_ids in sorted(self._by_name.items()):
            if len(resolved_ids) < 2:
                continue
            records = [self._records[(name_key, resolved_id)] for resolved_id in sorted(resolved_ids)]
            collisions.append(
                {
                    "horse_name_key": name_key,
                    "horse_names": sorted({str(record["horse_name"]) for record in records}),
                    "resolved_at_ids": sorted(resolved_ids),
                    "first_seen_date": min(str(record["first_seen_date"]) for record in records),
                    "last_seen_date": max(str(record["last_seen_date"]) for record in records),
                }
            )
        return {
            "names_with_multiple_ids": len(collisions),
            "conflicting_name_id_pairs": sum(len(item["resolved_at_ids"]) for item in collisions),
            "collisions": collisions,
        }

    def records(self) -> list[dict[str, Any]]:
        return list(self._records.values())


def validate_random_mappings(
    source: TJKHtmlDataSource,
    mapping: HorseIdMappingStore,
    sample_size: int = 20,
    seed: int = 20260924,
) -> dict[str, Any]:
    records = mapping.records()
    sample = random.Random(seed).sample(records, min(max(sample_size, 0), len(records)))
    checks = []
    for record in sample:
        horse_id = int(record["resolved_at_id"])
        horse_name = str(record["horse_name"])
        try:
            status, observed_names = source.validate_horse_id_name(horse_id, horse_name)
        except DataSourceError as exc:
            status, observed_names = "error", [str(exc)]
        checks.append(
            {
                "horse_name": horse_name,
                "resolved_at_id": horse_id,
                "expected_name_key": str(record["horse_name_key"]),
                "status": status,
                "observed_names": observed_names,
            }
        )
    return {
        "sample_size_requested": sample_size,
        "sample_size_checked": len(checks),
        "seed": seed,
        "matched": sum(item["status"] == "matched" for item in checks),
        "mismatched": sum(item["status"] == "mismatch" for item in checks),
        "unverifiable": sum(item["status"] == "unverifiable" for item in checks),
        "errors": sum(item["status"] == "error" for item in checks),
        "checks": checks,
    }


def resolve_race_horse_ids(
    source: TJKHtmlDataSource,
    races: list[Race],
    target_date: date,
    hippodrome: str,
    mapping: HorseIdMappingStore,
    cached_program_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a race's entries from exact event cache, HTML, mapping, then Atlar search."""
    event_prefix = (target_date.isoformat(), _normalize_city(hippodrome))
    exact_cache: dict[tuple[str, int, str], set[int]] = {}
    for record in cached_program_records or []:
        if record.get("source_horse_id") is None:
            continue
        try:
            record_date = str(record["race_date"])
            record_city = _normalize_city(str(record["hippodrome"]))
            race_no = int(record["race_no"])
            name_key = mapping.name_key(str(record["horse_name"]))
            if (record_date, record_city) == event_prefix:
                exact_cache.setdefault((record_date, race_no, name_key), set()).add(
                    int(record["source_horse_id"])
                )
        except (KeyError, TypeError, ValueError):
            continue

    by_name: dict[tuple[int, str], set[int]] = {}
    unresolved_entries = []
    for race in races:
        for entry in race.entries:
            key = (race.race_no, mapping.name_key(entry.horse_name))
            if entry.source_horse_id is not None:
                ids = {int(entry.source_horse_id)}
                entry.id_resolution_status = "html_program_link"
            else:
                ids = exact_cache.get((target_date.isoformat(), *key), set())
                if not ids:
                    resolved_id, status, candidate_ids = mapping.resolve_for_date(
                        entry.horse_name, target_date
                    )
                    if resolved_id is not None:
                        ids = {resolved_id}
                        entry.id_resolution_status = status
                    elif status == "ambiguous":
                        entry.id_resolution_status = status
                        entry.id_candidate_ids = candidate_ids
                if len(ids) == 1:
                    entry.source_horse_id = next(iter(ids))
                    event_key = (target_date.isoformat(), *key)
                    entry.id_resolution_status = (
                        "event_cache" if event_key in exact_cache else entry.id_resolution_status
                    )
                elif len(ids) > 1:
                    entry.id_resolution_status = "ambiguous"
                    entry.id_candidate_ids = sorted(ids)
            if entry.source_horse_id is not None:
                by_name.setdefault(key, set()).add(int(entry.source_horse_id))
            else:
                unresolved_entries.append((race, entry, key))

    html_program_error = None
    if unresolved_entries:
        try:
            html_races = source.get_daily_races_html(target_date, hippodrome)
            TJKHtmlDataSource.merge_source_ids(races, html_races)
            for race in races:
                for entry in race.entries:
                    if entry.source_horse_id is not None:
                        key = (race.race_no, mapping.name_key(entry.horse_name))
                        by_name.setdefault(key, set()).add(int(entry.source_horse_id))
                        entry.id_resolution_status = "html_program_link"
        except DataSourceError as exc:
            html_program_error = str(exc)

    still_unresolved = [
        (race, entry, key)
        for race, entry, key in unresolved_entries
        if entry.source_horse_id is None
    ]
    html_result_error = None
    if still_unresolved:
        try:
            for race_no, horse_name, resolved_id in source.get_daily_result_horse_ids_html(
                target_date, hippodrome
            ):
                key = (race_no, mapping.name_key(horse_name))
                by_name.setdefault(key, set()).add(resolved_id)
        except DataSourceError as exc:
            html_result_error = str(exc)

    search_unavailable = False
    for race, entry, key in still_unresolved:
        observed_ids = by_name.get(key, set())
        if len(observed_ids) == 1:
            entry.source_horse_id = next(iter(observed_ids))
            entry.id_resolution_status = "html_result_link"
        elif len(observed_ids) > 1:
            entry.id_resolution_status = "ambiguous"
            entry.id_candidate_ids = sorted(observed_ids)
        elif search_unavailable:
            entry.id_resolution_status = "unresolved"
        else:
            try:
                found_id, search_status, candidate_ids = source.find_horse_id_from_search(
                    entry.horse_name, entry.age
                )
            except DataSourceError:
                search_unavailable = True
                found_id, search_status, candidate_ids = None, "unresolved", []
            if found_id is not None:
                entry.source_horse_id = found_id
                entry.id_resolution_status = "atlar_search"
            else:
                entry.id_resolution_status = search_status
                entry.id_candidate_ids = candidate_ids

    collision_names: set[str] = set()
    resolved = 0
    unresolved = []
    mapping_observations: list[tuple[str, int, date]] = []
    for race in races:
        for entry in race.entries:
            entry.id_unresolved = entry.source_horse_id is None
            if entry.source_horse_id is not None:
                resolved += 1
                if entry.id_resolution_status in {
                    "html_program_link",
                    "html_result_link",
                    "resolved_html_name_collision",
                }:
                    mapping_observations.append(
                        (entry.horse_name, int(entry.source_horse_id), target_date)
                    )
            else:
                unresolved.append(
                    {
                        "race_date": target_date.isoformat(),
                        "hippodrome": hippodrome,
                        "race_no": race.race_no,
                        "horse_name": entry.horse_name,
                        "candidate_ids": entry.id_candidate_ids,
                        "status": entry.id_resolution_status,
                    }
                )

    collision_report = mapping.observe_many(mapping_observations)
    collision_names.update(collision_report)
    for race in races:
        for entry in race.entries:
            if mapping.name_key(entry.horse_name) in collision_names and entry.source_horse_id is not None:
                entry.id_resolution_status = "resolved_html_name_collision"

    return {
        "resolved": resolved,
        "unresolved": len(unresolved),
        "unresolved_entries": unresolved,
        "collision_names": sorted(collision_names),
        "html_program_error": html_program_error,
        "html_result_error": html_result_error,
    }