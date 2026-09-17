from __future__ import annotations

import json
from types import SimpleNamespace

from atyaris.ml.local_pipeline import build_local_raw_frame


def test_local_pipeline_builds_labelled_rows_without_network(tmp_path) -> None:
    program = tmp_path / "program.jsonl"
    results = tmp_path / "results.jsonl"
    history = tmp_path / "history.jsonl"
    program.write_text(json.dumps({
        "race_id": "race-1", "horse_id": "horse-1", "race_date": "2026-09-10",
        "horse_name": "Horse", "draw": 1, "weight_kg": 58, "distance_m": 1400,
        "field_size": 2, "surface": "Cim", "hippodrome": "ISTANBUL",
    }) + "\n", encoding="utf-8")
    results.write_text(json.dumps({
        "race_id": "race-1", "horse_id": "horse-1", "finish_position": 1,
    }) + "\n", encoding="utf-8")
    history.write_text(json.dumps({
        "horse_id": "horse-1", "race_id": "old-race", "race_date": "2026-09-01",
        "finish_position": 2, "field_size": 5,
    }) + "\n", encoding="utf-8")

    paths = SimpleNamespace(
        raw_daily_program_jsonl=program,
        raw_race_results_jsonl=results,
        raw_history_jsonl=history,
        raw_csv=tmp_path / "legacy.csv",
    )

    frame = build_local_raw_frame(paths)

    assert len(frame) == 1
    assert frame.iloc[0]["is_winner"] == 1
    assert frame.iloc[0]["career_starts"] == 1