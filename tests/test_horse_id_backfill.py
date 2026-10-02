from __future__ import annotations

from datetime import date, datetime
import json

import pandas as pd

from atyaris.ml.horse_id_backfill import backfill_horse_ids
from atyaris.ml.pipeline import Phase1Paths
from atyaris.ml.raw_store import JsonlRawStore
from atyaris.models.entities import Jockey, Race, RaceEntry, TrackSurface, Trainer


def test_backfill_updates_existing_jsonl_csv_and_checkpoints(tmp_path) -> None:
    target_date = date(2025, 1, 1)
    paths = Phase1Paths(
        raw_csv=tmp_path / "races.csv",
        raw_history_jsonl=tmp_path / "history.jsonl",
        raw_daily_program_jsonl=tmp_path / "program.jsonl",
        raw_race_results_jsonl=tmp_path / "results.jsonl",
        raw_horse_id_mapping_jsonl=tmp_path / "horse_ids.jsonl",
    )
    program_record = {
        "race_id": "Ankara-2025-01-01-1",
        "horse_id": "TEKRAR-1",
        "race_date": target_date.isoformat(),
        "race_datetime": "2025-01-01T12:00:00",
        "hippodrome": "Ankara",
        "race_no": 1,
        "horse_name": "TEKRAR",
        "draw": 1,
        "age": 3,
        "weight_kg": 57.0,
        "surface": "Kum",
    }
    JsonlRawStore(paths.raw_daily_program_jsonl, "daily_program").upsert([program_record])
    JsonlRawStore(paths.raw_race_results_jsonl, "race_results").upsert(
        [{
            "race_id": program_record["race_id"],
            "horse_id": program_record["horse_id"],
            "race_date": target_date.isoformat(),
            "hippodrome": "Ankara",
            "race_no": 1,
            "horse_name": "TEKRAR",
            "finish_position": 1,
        }]
    )
    JsonlRawStore(paths.raw_history_jsonl, "horse_history").upsert(
        [{
            "horse_id": program_record["horse_id"],
            "race_id": "TEKRAR-1|2024-12-01|Ankara|1",
            "race_date": "2024-12-01",
            "target_date": target_date.isoformat(),
            "target_hippodrome": "Ankara",
            "target_race_no": 1,
            "target_horse_name": "TEKRAR",
            "source_horse_id": None,
        }]
    )
    pd.DataFrame(
        [{
            "date": target_date.isoformat(),
            "race_id": program_record["race_id"],
            "horse_id": program_record["horse_id"],
            "horse_name": "TEKRAR",
            "draw": 1,
        }]
    ).to_csv(paths.raw_csv, index=False)

    html_race = Race(
        id=program_record["race_id"],
        hippodrome="Ankara",
        race_no=1,
        start_time=datetime.fromisoformat(program_record["race_datetime"]),
        distance_m=1200,
        surface=TrackSurface.KUM,
        entries=[
            RaceEntry(
                number=8,
                horse_id="TEKRAR-8",
                source_horse_id=98765,
                horse_name="TEKRAR",
                jockey=Jockey(name="Jokey"),
                trainer=Trainer(name="Trainer"),
                weight_kg=57.0,
            )
        ],
    )

    class Source:
        def get_daily_races_html(self, *_args):
            return [html_race]

        def get_daily_result_horse_ids_html(self, *_args):
            raise AssertionError("program HTML already resolved the ID")

        def find_horse_id_from_search(self, *_args):
            raise AssertionError("program HTML already resolved the ID")

    report = backfill_horse_ids(paths, Source())  # type: ignore[arg-type]

    program = JsonlRawStore(paths.raw_daily_program_jsonl, "daily_program").read_latest_records()[0]
    result = JsonlRawStore(paths.raw_race_results_jsonl, "race_results").read_latest_records()[0]
    history = JsonlRawStore(paths.raw_history_jsonl, "horse_history").read_latest_records()[0]
    csv_row = pd.read_csv(paths.raw_csv).iloc[0]
    checkpoint = json.loads(
        paths.raw_horse_id_mapping_jsonl.with_suffix(".backfill.progress.json").read_text(encoding="utf-8")
    )

    assert program["source_horse_id"] == 98765
    assert result["source_horse_id"] == 98765
    assert history["source_horse_id"] == 98765
    assert csv_row["source_horse_id"] == 98765
    assert not bool(csv_row["id_unresolved"])
    assert checkpoint["completed_units"] == ["2025-01-01|ankara"]
    assert report["resolved_entries"] == 1


def test_backfill_discovers_csv_only_legacy_rows(tmp_path) -> None:
    target_date = date(2025, 1, 1)
    paths = Phase1Paths(
        raw_csv=tmp_path / "races.csv",
        raw_history_jsonl=tmp_path / "history.jsonl",
        raw_daily_program_jsonl=tmp_path / "program.jsonl",
        raw_race_results_jsonl=tmp_path / "results.jsonl",
        raw_horse_id_mapping_jsonl=tmp_path / "horse_ids.jsonl",
    )
    paths.raw_daily_program_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "date": target_date.isoformat(),
            "race_id": "Ankara-2025-01-01-1",
            "horse_id": "ESKI AT-1",
            "horse_name": "ESKI AT",
            "draw": 1,
            "track": "ANKARA",
            "age": 4,
            "weight": 56.0,
            "distance": 1400,
            "surface": "Kum",
        }]
    ).to_csv(paths.raw_csv, index=False)
    html_race = Race(
        id="Ankara-2025-01-01-1",
        hippodrome="Ankara",
        race_no=1,
        start_time=datetime(2025, 1, 1),
        distance_m=1400,
        surface=TrackSurface.KUM,
        entries=[
            RaceEntry(
                number=9,
                horse_id="ESKI AT-9",
                source_horse_id=54321,
                horse_name="ESKI AT",
                jockey=Jockey(name="Jokey"),
                trainer=Trainer(name="Trainer"),
                weight_kg=56.0,
            )
        ],
    )

    class Source:
        def get_daily_races_html(self, *_args):
            return [html_race]

    report = backfill_horse_ids(paths, Source())  # type: ignore[arg-type]
    updated = pd.read_csv(paths.raw_csv).iloc[0]

    assert updated["source_horse_id"] == 54321
    assert report["resolved_entries"] == 1
    assert JsonlRawStore(paths.raw_daily_program_jsonl, "daily_program").count_records() == 0


def test_backfill_reports_unresolved_and_skips_completed_unit_by_default(tmp_path) -> None:
    target_date = date(2025, 1, 1)
    paths = Phase1Paths(
        raw_csv=tmp_path / "races.csv",
        raw_history_jsonl=tmp_path / "history.jsonl",
        raw_daily_program_jsonl=tmp_path / "program.jsonl",
        raw_race_results_jsonl=tmp_path / "results.jsonl",
        raw_horse_id_mapping_jsonl=tmp_path / "horse_ids.jsonl",
    )
    JsonlRawStore(paths.raw_daily_program_jsonl, "daily_program").upsert(
        [{
            "race_id": "Ankara-2025-01-01-1",
            "horse_id": "BULUNAMAYAN-1",
            "race_date": target_date.isoformat(),
            "race_datetime": "2025-01-01T12:00:00",
            "hippodrome": "Ankara",
            "race_no": 1,
            "horse_name": "BULUNAMAYAN",
            "draw": 1,
            "age": 3,
            "weight_kg": 57.0,
            "surface": "Kum",
        }]
    )
    html_race = Race(
        id="Ankara-2025-01-01-1",
        hippodrome="Ankara",
        race_no=1,
        start_time=datetime(2025, 1, 1, 12, 0),
        distance_m=1200,
        surface=TrackSurface.KUM,
        entries=[
            RaceEntry(
                number=1,
                horse_id="BULUNAMAYAN-1",
                horse_name="BULUNAMAYAN",
                jockey=Jockey(name="Jokey"),
                trainer=Trainer(name="Trainer"),
                weight_kg=57.0,
            )
        ],
    )

    class Source:
        attempts = 0

        def get_daily_races_html(self, *_args):
            self.attempts += 1
            return [html_race]

        def get_daily_result_horse_ids_html(self, *_args):
            self.attempts += 1
            return []

        def find_horse_id_from_search(self, *_args):
            self.attempts += 1
            return None, "unresolved", []

    source = Source()
    first_report = backfill_horse_ids(paths, source)  # type: ignore[arg-type]
    unresolved_path = paths.raw_horse_id_mapping_jsonl.with_suffix(
        ".unresolved-report.json"
    )
    unresolved_report = json.loads(unresolved_path.read_text(encoding="utf-8"))
    attempts_after_first_run = source.attempts
    second_report = backfill_horse_ids(paths, source)  # type: ignore[arg-type]

    assert first_report["unresolved_entries"] == 1
    assert unresolved_report["entries"] == [
        {
            "race_date": "2025-01-01",
            "hippodrome": "Ankara",
            "race_no": 1,
            "horse_name": "BULUNAMAYAN",
            "status": "unresolved",
            "candidate_ids": [],
        }
    ]
    assert second_report["unresolved_entries"] == 1
    assert source.attempts == attempts_after_first_run