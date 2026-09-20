from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from atyaris.ml.features import build_leakage_safe_features
from atyaris.ml.local_pipeline import build_local_raw_frame, find_missing_local_targets


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
    assert frame.iloc[0]["market_probability"] == 0.1
    built = build_leakage_safe_features(frame)
    assert "expected_early_pace" in built.frame.columns


def test_local_pipeline_finds_missing_result_or_history_targets(tmp_path) -> None:
    program = tmp_path / "program.jsonl"
    results = tmp_path / "results.jsonl"
    history = tmp_path / "history.jsonl"
    program.write_text(
        "\n".join(
            json.dumps({
                "race_id": race_id,
                "horse_id": horse_id,
                "race_date": "2026-09-10",
                "hippodrome": "Istanbul",
                "race_no": race_no,
            })
            for race_id, horse_id, race_no in (
                ("race-complete", "horse-1", 1),
                ("race-missing", "horse-2", 2),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    results.write_text(json.dumps({
        "race_id": "race-complete", "horse_id": "horse-1", "finish_position": 1,
    }) + "\n", encoding="utf-8")
    history.write_text(json.dumps({
        "target_race_id": "race-complete", "target_horse_id": "horse-1",
    }) + "\n", encoding="utf-8")
    paths = SimpleNamespace(
        raw_daily_program_jsonl=program,
        raw_race_results_jsonl=results,
        raw_history_jsonl=history,
    )

    missing = find_missing_local_targets(paths, date(2026, 9, 10), date(2026, 9, 10))

    assert missing == [{
        "date": "2026-09-10",
        "hippodrome": "Istanbul",
        "race_no": 2,
        "missing_results": 1,
        "missing_history": 1,
    }]