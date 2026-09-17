from __future__ import annotations

import json

from atyaris.ml.real_ingestion import _append_raw_history_records


def test_raw_history_jsonl_preserves_full_records_and_deduplicates(tmp_path) -> None:
    path = tmp_path / "past_performances.jsonl"
    record = {
        "retrieved_at": "2026-09-17T00:00:00+00:00",
        "target_date": "2025-09-17",
        "target_race_id": "2025-09-17-Istanbul-1",
        "target_horse_id": "horse-1",
        "race_date": "2025-08-20",
        "jockey_name": "A.JOKEY",
        "finish_position": 2,
        "distance_m": 1400,
    }

    _append_raw_history_records(path, [record, record])
    _append_raw_history_records(path, [record])

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [record]
    assert rows[0]["jockey_name"] == "A.JOKEY"
    assert rows[0]["distance_m"] == 1400