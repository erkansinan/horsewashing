from __future__ import annotations

import json

import atyaris.ml.raw_store as raw_store
from atyaris.ml.raw_store import JsonlRawStore


def test_no_duplicate_raw_records(tmp_path) -> None:
    store = JsonlRawStore(tmp_path / "history.jsonl", "horse_history")
    record = {
        "horse_id": "horse-1",
        "race_id": "race-1",
        "race_date": "2026-09-01",
        "jockey_name": "Jokey",
    }

    first = store.upsert([record])
    second = store.upsert([record])

    assert first.added == 1
    assert second.duplicates == 1
    assert store.count_records() == 1


def test_index_migration_can_resume_from_checkpoint(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    store = JsonlRawStore(path, "race_results")
    records = [
        {"race_id": f"race-{index}", "horse_id": f"horse-{index}", "finish_position": index}
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    assert store.migrate_to_index(batch_size=1)
    assert store.index_path.exists()
    assert store.migration_state_path.exists()
    migration_state = json.loads(store.migration_state_path.read_text(encoding="utf-8"))
    assert migration_state["status"] == "completed"
    assert migration_state["line_count"] == 3


def test_upsert_recovers_from_empty_index_file(tmp_path) -> None:
    store = JsonlRawStore(tmp_path / "program.jsonl", "daily_program")
    first = {"race_id": "race-1", "horse_id": "horse-1", "value": 1}
    store.upsert([first])
    store.index_path.parent.mkdir(parents=True, exist_ok=True)
    store.index_path.write_bytes(b"")

    result = store.upsert(
        [
            first,
            {"race_id": "race-2", "horse_id": "horse-2", "value": 2},
        ]
    )

    assert result.records == 2
    assert store.index_path.exists() is False
    assert len(list(store.index_dir.glob("program.sqlite3.invalid-*"))) == 1
    assert len(store.read_latest_records()) == 2


def test_upsert_recovers_from_index_without_records_table(tmp_path) -> None:
    store = JsonlRawStore(tmp_path / "program.jsonl", "daily_program")
    first = {"race_id": "race-1", "horse_id": "horse-1", "value": 1}
    store.upsert([first])
    store.index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = raw_store.sqlite3.connect(store.index_path)
    with connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    connection.close()

    result = store.upsert(
        [
            first,
            {"race_id": "race-2", "horse_id": "horse-2", "value": 2},
        ]
    )

    assert result.records == 2
    assert store.index_path.exists() is False
    assert len(list(store.index_dir.glob("program.sqlite3.invalid-*"))) == 1
    assert len(store.read_latest_records()) == 2


def test_index_migration_rebuilds_empty_index_with_stale_checkpoint(tmp_path) -> None:
    path = tmp_path / "program.jsonl"
    records = [
        {"race_id": f"race-{index}", "horse_id": f"horse-{index}"}
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    store = JsonlRawStore(path, "daily_program")
    store.index_dir.mkdir()
    store.index_path.write_bytes(b"")
    store.migration_state_path.write_text(
        json.dumps({"status": "completed", "line": 2, "line_count": 2}),
        encoding="utf-8",
    )

    assert store.migrate_to_index()
    assert store.has_valid_index
    with raw_store.sqlite3.connect(store.index_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 2


def test_indexed_upsert_does_not_rewrite_existing_records(tmp_path) -> None:
    store = JsonlRawStore(tmp_path / "program.jsonl", "daily_program")
    store.upsert([{"race_id": "race-1", "horse_id": "horse-1", "value": 1}])
    store.migrate_to_index()
    before = store.path.read_text(encoding="utf-8")

    result = store.upsert([{"race_id": "race-1", "horse_id": "horse-1", "value": 2}])

    assert result.updated == 1
    assert store.path.read_text(encoding="utf-8").startswith(before)


def test_indexed_compact_recommendation_uses_index_counts(tmp_path, monkeypatch) -> None:
    path = tmp_path / "history.jsonl"
    first = {"horse_id": "horse-1", "race_id": "race-1", "race_date": "2026-09-01"}
    updated = {**first, "jockey_name": "Jokey"}
    second = {"horse_id": "horse-2", "race_id": "race-2", "race_date": "2026-09-02"}
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in (first, updated, second)),
        encoding="utf-8",
    )
    store = JsonlRawStore(path, "horse_history")
    store.migrate_to_index()

    def fail_if_scanned():
        raise AssertionError("Indexed recommendation must not scan the JSONL file")

    monkeypatch.setattr(store, "count_records", fail_if_scanned)
    monkeypatch.setattr(store, "_read_latest", fail_if_scanned)

    recommendation = store.compact_recommendation()

    assert recommendation.recommended is True
    assert recommendation.active_records == 2
    assert recommendation.superseded_records == 1
    assert recommendation.file_bytes == path.stat().st_size


def test_indexed_upsert_uses_persisted_line_count(tmp_path, monkeypatch) -> None:
    store = JsonlRawStore(tmp_path / "program.jsonl", "daily_program")
    store.upsert([{"race_id": "race-1", "horse_id": "horse-1", "value": 1}])
    store.migrate_to_index()

    def fail_if_scanned():
        raise AssertionError("Indexed upsert must use the persisted line count")

    monkeypatch.setattr(store, "count_records", fail_if_scanned)

    result = store.upsert([{"race_id": "race-1", "horse_id": "horse-1", "value": 2}])

    assert result.records == 2
    assert json.loads(store.migration_state_path.read_text(encoding="utf-8"))["line_count"] == 2


def test_indexed_line_count_bootstraps_legacy_state_once(tmp_path, monkeypatch) -> None:
    path = tmp_path / "history.jsonl"
    store = JsonlRawStore(path, "horse_history")
    records = [
        {"horse_id": f"horse-{index}", "race_id": f"race-{index}", "race_date": "2026-09-01"}
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    store.migrate_to_index()
    state = json.loads(store.migration_state_path.read_text(encoding="utf-8"))
    state.pop("line_count")
    store.migration_state_path.write_text(json.dumps(state), encoding="utf-8")
    calls = 0
    original_count_records = store.count_records

    def count_records_once():
        nonlocal calls
        calls += 1
        return original_count_records()

    monkeypatch.setattr(store, "count_records", count_records_once)

    assert store._indexed_line_count() == 2
    assert store._indexed_line_count() == 2
    assert calls == 1


def test_atomic_replace_retries_windows_file_lock(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.jsonl"
    source.write_text("new\n", encoding="utf-8")
    target.write_text("old\n", encoding="utf-8")
    original_replace = raw_store.Path.replace
    attempts = 0

    def replace_with_transient_lock(path, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = PermissionError("file is in use")
            error.winerror = 32
            raise error
        return original_replace(path, destination)

    monkeypatch.setattr(raw_store.Path, "replace", replace_with_transient_lock)
    monkeypatch.setattr(raw_store.time, "sleep", lambda _seconds: None)

    raw_store._atomic_replace(source, target)

    assert attempts == 2
    assert target.read_text(encoding="utf-8") == "new\n"


def test_upsert_skips_truncated_jsonl_record(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    valid = {
        "horse_id": "horse-1",
        "race_id": "race-1",
        "race_date": "2026-09-01",
    }
    path.write_text(json.dumps(valid) + "\nnot-json\n", encoding="utf-8")

    result = JsonlRawStore(path, "horse_history").upsert(
        [{**valid, "jockey_name": "Jokey"}]
    )

    assert result.updated == 1
    assert path.read_text(encoding="utf-8").count("not-json") == 0
    assert JsonlRawStore(path, "horse_history").count_records() == 1


def test_repair_reports_and_removes_invalid_and_duplicate_records(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    valid = {"horse_id": "horse-1", "race_id": "race-1", "race_date": "2026-09-01"}
    path.write_text(
        json.dumps(valid) + "\n"
        + json.dumps(valid) + "\n"
        + '{"horse_id": "incomplete"}\n'
        + "truncated\n",
        encoding="utf-8",
    )

    result = JsonlRawStore(path, "horse_history").repair()

    assert result.scanned == 4
    assert result.valid_records == 1
    assert result.duplicates_removed == 1
    assert result.missing_key_records == 1
    assert result.malformed_records == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1