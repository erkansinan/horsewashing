"""Append-friendly storage for the five TJK raw-data collections."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable


RAW_STORE_THRESHOLDS = {
    "warning_records": 100_000,
    "indexed_records": 250_000,
    "warning_bytes": 256 * 1024 * 1024,
    "indexed_bytes": 512 * 1024 * 1024,
    "compact_ratio": 0.20,
}

RAW_COLLECTIONS: dict[str, tuple[str, ...]] = {
    "daily_program": ("race_id", "horse_id"),
    "race_results": ("race_id", "horse_id"),
    "horse_history": ("horse_id", "race_id", "race_date"),
    "workouts": ("horse_id", "workout_date", "distance_m", "time_seconds", "detail"),
    "trainer_statistics": ("trainer_id", "as_of_date"),
}


@dataclass(frozen=True)
class RawStoreStats:
    added: int = 0
    updated: int = 0
    duplicates: int = 0
    records: int = 0
    bytes: int = 0


@dataclass(frozen=True)
class CompactRecommendation:
    recommended: bool
    reason: str
    active_records: int
    superseded_records: int
    file_bytes: int


def _atomic_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)


def _record_key(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    values = [record.get(field) for field in fields]
    if any(value is None for value in values):
        raise ValueError(f"Raw kayit benzersiz anahtar alanlari eksik: {fields}")
    return "|".join(str(value) for value in values)


def _record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonlRawStore:
    """JSONL store with safe small-file upsert and resumable index migration."""

    def __init__(self, path: Path, collection: str, index_dir: Path | None = None) -> None:
        if collection not in RAW_COLLECTIONS:
            raise ValueError(f"Bilinmeyen ham veri koleksiyonu: {collection}")
        self.path = path
        self.collection = collection
        self.key_fields = RAW_COLLECTIONS[collection]
        self.index_dir = index_dir or path.parent / "indexes"
        self.index_path = self.index_dir / f"{path.stem}.sqlite3"
        self.migration_state_path = self.index_dir / f"{path.stem}.migration.json"
        self.migration_temp_path = self.index_dir / f"{path.stem}.sqlite3.tmp"

    def _read_latest(self) -> dict[str, tuple[dict[str, Any], str]]:
        latest: dict[str, tuple[dict[str, Any], str]] = {}
        if not self.path.exists():
            return latest
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                latest[_record_key(record, self.key_fields)] = (record, _record_hash(record))
        return latest

    def upsert(self, records: Iterable[dict[str, Any]]) -> RawStoreStats:
        incoming: dict[str, dict[str, Any]] = {}
        for record in records:
            incoming[_record_key(record, self.key_fields)] = dict(record)
        if not incoming:
            return RawStoreStats(records=self.count_records(), bytes=self.path.stat().st_size if self.path.exists() else 0)
        if self.index_path.exists():
            return self._upsert_indexed(incoming)
        latest = self._read_latest()
        added = updated = duplicates = 0
        for key, record in incoming.items():
            digest = _record_hash(record)
            previous = latest.get(key)
            if previous is None:
                added += 1
            elif previous[1] == digest:
                duplicates += 1
                continue
            else:
                updated += 1
            latest[key] = (record, digest)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record, _digest in latest.values():
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        _atomic_replace(temporary, self.path)
        return RawStoreStats(
            added=added,
            updated=updated,
            duplicates=duplicates,
            records=len(latest),
            bytes=self.path.stat().st_size,
        )

    def count_records(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def compact_recommendation(self) -> CompactRecommendation:
        physical_records = self.count_records()
        active_records = len(self._read_latest())
        file_bytes = self.path.stat().st_size if self.path.exists() else 0
        superseded = max(physical_records - active_records, 0)
        recommended = (
            superseded > max(active_records, 1) * RAW_STORE_THRESHOLDS["compact_ratio"]
            or file_bytes >= RAW_STORE_THRESHOLDS["indexed_bytes"]
        )
        reason = "eski surumler aktif kayitlarin %20'sini asti" if superseded > max(active_records, 1) * RAW_STORE_THRESHOLDS["compact_ratio"] else "dosya 512 MB esigini asti" if file_bytes >= RAW_STORE_THRESHOLDS["indexed_bytes"] else "compact gerekmiyor"
        return CompactRecommendation(recommended, reason, active_records, superseded, file_bytes)

    def compact(self) -> RawStoreStats:
        latest = self._read_latest()
        temporary = self.path.with_suffix(self.path.suffix + ".compact.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record, _digest in latest.values():
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        _atomic_replace(temporary, self.path)
        if self.index_path.exists():
            self.index_path.unlink(missing_ok=True)
            self.migration_state_path.unlink(missing_ok=True)
            self.migrate_to_index()
        return RawStoreStats(records=len(latest), bytes=self.path.stat().st_size)

    def migrate_to_index(self, batch_size: int = 10_000) -> bool:
        """Build the index in a temp DB; return True only when migration is complete."""
        if self.index_path.exists():
            return True
        self.index_dir.mkdir(parents=True, exist_ok=True)
        state = {"status": "migration_in_progress", "line": 0}
        if self.migration_state_path.exists():
            state = json.loads(self.migration_state_path.read_text(encoding="utf-8"))
        if not self.migration_temp_path.exists():
            with sqlite3.connect(self.migration_temp_path) as connection:
                connection.execute("CREATE TABLE records (record_key TEXT PRIMARY KEY, record_hash TEXT NOT NULL, line_number INTEGER NOT NULL, payload TEXT NOT NULL)")
                connection.commit()
        start_line = int(state.get("line", 0))
        processed = start_line
        batch: list[tuple[str, str, int, str]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number < start_line or not line.strip():
                    continue
                record = json.loads(line)
                batch.append((_record_key(record, self.key_fields), _record_hash(record), line_number, json.dumps(record, ensure_ascii=True, sort_keys=True)))
                processed = line_number + 1
                if len(batch) >= batch_size:
                    self._write_migration_batch(batch)
                    batch.clear()
                    self._write_migration_state(processed)
        if batch:
            self._write_migration_batch(batch)
        self._write_migration_state(processed)
        migration_target = self.index_path.with_suffix(".sqlite3.new")
        shutil.copyfile(self.migration_temp_path, migration_target)
        _atomic_replace(migration_target, self.index_path)
        # Windows may keep SQLite's temporary handle open briefly. The temp DB
        # is harmless after the active index is promoted and can be cleaned up
        # by the next maintenance pass.
        self._write_migration_state(processed, status="completed")
        return True

    def _write_migration_batch(self, batch: list[tuple[str, str, int, str]]) -> None:
        connection = sqlite3.connect(self.migration_temp_path)
        try:
            connection.executemany("INSERT OR REPLACE INTO records(record_key, record_hash, line_number, payload) VALUES (?, ?, ?, ?)", batch)
            connection.commit()
        finally:
            connection.close()

    def _write_migration_state(self, line: int, status: str = "migration_in_progress") -> None:
        temporary = self.migration_state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"status": status, "line": line}, ensure_ascii=True), encoding="utf-8")
        _atomic_replace(temporary, self.migration_state_path)

    def _upsert_indexed(self, incoming: dict[str, dict[str, Any]]) -> RawStoreStats:
        added = updated = duplicates = 0
        with sqlite3.connect(self.index_path) as connection:
            pending: list[tuple[str, str, int, str]] = []
            next_line = self.count_records()
            for key, record in incoming.items():
                digest = _record_hash(record)
                previous = connection.execute(
                    "SELECT record_hash FROM records WHERE record_key = ?", (key,)
                ).fetchone()
                if previous is not None and previous[0] == digest:
                    duplicates += 1
                    continue
                if previous is None:
                    added += 1
                else:
                    updated += 1
                payload = json.dumps(record, ensure_ascii=True, sort_keys=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload + "\n")
                pending.append((key, digest, next_line, payload))
                next_line += 1
            connection.executemany(
                "INSERT OR REPLACE INTO records(record_key, record_hash, line_number, payload) VALUES (?, ?, ?, ?)",
                pending,
            )
            connection.commit()
        return RawStoreStats(added, updated, duplicates, self.count_records(), self.path.stat().st_size)

    def _rewrite_with_stats(self, incoming: dict[str, dict[str, Any]], latest: dict[str, tuple[dict[str, Any], str]]) -> RawStoreStats:
        added = updated = duplicates = 0
        for key, record in incoming.items():
            digest = _record_hash(record)
            previous = latest.get(key)
            if previous is None:
                added += 1
            elif previous[1] == digest:
                duplicates += 1
                continue
            else:
                updated += 1
            latest[key] = (record, digest)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record, _digest in latest.values():
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        _atomic_replace(temporary, self.path)
        return RawStoreStats(added, updated, duplicates, len(latest), self.path.stat().st_size)
