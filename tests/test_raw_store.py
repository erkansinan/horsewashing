from __future__ import annotations

import json

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