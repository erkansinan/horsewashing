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
    assert store.migration_state_path.read_text(encoding="utf-8").find("completed") >= 0


def test_indexed_upsert_does_not_rewrite_existing_records(tmp_path) -> None:
    store = JsonlRawStore(tmp_path / "program.jsonl", "daily_program")
    store.upsert([{"race_id": "race-1", "horse_id": "horse-1", "value": 1}])
    store.migrate_to_index()
    before = store.path.read_text(encoding="utf-8")

    result = store.upsert([{"race_id": "race-1", "horse_id": "horse-1", "value": 2}])

    assert result.updated == 1
    assert store.path.read_text(encoding="utf-8").startswith(before)


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