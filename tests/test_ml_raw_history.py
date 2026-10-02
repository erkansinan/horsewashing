from __future__ import annotations

import json
from datetime import date

import pytest
import httpx
from types import SimpleNamespace

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.models.entities import PastPerformance, TrackSurface
from atyaris.ml.pipeline import Phase1Paths
from atyaris.ml.real_ingestion import (
    _append_raw_history_records,
    _field_size_adjusted_history,
    _load_recent_trainer_statistics,
    ingest_real_tjk_data,
)
from atyaris.ml.local_pipeline import build_local_raw_frame


def test_field_size_adjusted_history_ignores_unknown_and_impossible_field_sizes() -> None:
    history = [
        PastPerformance(
            race_date=date(2026, 8, 1),
            hippodrome="Ankara",
            distance_m=1200,
            surface=TrackSurface.KUM,
            finish_position=3,
            field_size=None,
        ),
        PastPerformance(
            race_date=date(2026, 8, 2),
            hippodrome="Ankara",
            distance_m=1200,
            surface=TrackSurface.KUM,
            finish_position=9,
            field_size=6,
        ),
        PastPerformance(
            race_date=date(2026, 8, 3),
            hippodrome="Ankara",
            distance_m=1200,
            surface=TrackSurface.KUM,
            finish_position=3,
            field_size=12,
        ),
    ]

    assert _field_size_adjusted_history(history) == [pytest.approx(9 / 11)]


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


def test_recent_trainer_cache_uses_latest_non_future_record_within_age(tmp_path) -> None:
    path = tmp_path / "trainer_statistics.jsonl"
    records = [
        {"trainer_id": "100", "as_of_date": "2026-09-19", "trainer_name": "Fresh", "total_starts": 10},
        {"trainer_id": "100", "as_of_date": "2026-09-20", "trainer_name": "Fresh", "total_starts": 20},
        {"trainer_id": "200", "as_of_date": "2026-09-18", "trainer_name": "Stale", "total_starts": 30},
        {"trainer_id": "300", "as_of_date": "2026-09-27", "trainer_name": "Future", "total_starts": 40},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    cached = _load_recent_trainer_statistics(path, date(2026, 9, 26), max_age_days=7)

    assert set(cached) == {100}
    assert cached[100][0].total_starts == 20
    assert cached[100][1] == date(2026, 9, 20)


def test_prediction_ingestion_scopes_csv_resolution_and_html_fetch_to_requested_race(tmp_path, monkeypatch) -> None:
    target_date = date(2026, 9, 25)
    races = SampleDataSource().get_daily_races(target_date, "Ankara")
    races[0].entries[0].trainer.source_trainer_id = 9001
    races[0].entries[1].trainer.source_trainer_id = 9002
    source = TJKHtmlDataSource()
    resolved_race_numbers = []
    html_requests = []
    prediction_request_policies = []
    trainer_requests = []
    source_factory_calls = []

    monkeypatch.setattr(source, "get_daily_races_csv", lambda *_args: races)

    def get_daily_races_html(_target_date, _city, race_no=None):
        html_requests.append(race_no)
        return races

    monkeypatch.setattr(source, "get_daily_races_html", get_daily_races_html)

    def missing_statistics(*_args, **_kwargs):
        if _kwargs:
            prediction_request_policies.append(_kwargs)
        raise DataSourceError("history unavailable")

    monkeypatch.setattr(source, "get_horse_statistics", missing_statistics)

    def trainer_statistics(*_args, **kwargs):
        trainer_requests.append((_args[0], kwargs))
        return None

    monkeypatch.setattr(source, "get_trainer_statistics", trainer_statistics)
    monkeypatch.setattr(
        "atyaris.ml.real_ingestion.build_training_data_source",
        lambda *_args: source_factory_calls.append("training") or source,
    )
    monkeypatch.setattr(
        "atyaris.ml.real_ingestion.build_prediction_data_source",
        lambda *_args: source_factory_calls.append("prediction") or source,
    )

    def record_resolved_races(_source, selected_races, *_args):
        resolved_race_numbers.extend(race.race_no for race in selected_races)
        return {"resolved": 0, "unresolved": 0, "collision_names": []}

    monkeypatch.setattr("atyaris.ml.real_ingestion.resolve_race_horse_ids", record_resolved_races)
    paths = Phase1Paths(
        raw_csv=tmp_path / "raw.csv",
        raw_history_jsonl=tmp_path / "history.jsonl",
        raw_daily_program_jsonl=tmp_path / "program.jsonl",
        raw_race_results_jsonl=tmp_path / "results.jsonl",
        raw_horse_id_mapping_jsonl=tmp_path / "horse_ids.jsonl",
        raw_workouts_jsonl=tmp_path / "workouts.jsonl",
        raw_trainer_statistics_jsonl=tmp_path / "trainers.jsonl",
        clean_csv=tmp_path / "clean.csv",
        features_csv=tmp_path / "features.csv",
        prediction_features_csv=tmp_path / "prediction_features.csv",
        model_path=tmp_path / "model.joblib",
    )

    frame = ingest_real_tjk_data(
        target_date,
        target_date,
        paths,
        require_results=False,
        hippodrome="Ankara",
        race_no=1,
        program_from_csv=True,
    )

    assert not frame.empty
    assert set(frame["history_lookup_status"]).issubset({"fetch_failed", "unresolved_id"})
    assert resolved_race_numbers == [1]
    assert source_factory_calls == ["prediction"]
    assert prediction_request_policies
    assert all(
        policy == {"include_workouts": True, "request_timeout": 5.0, "max_retries": 0}
        for policy in prediction_request_policies
    )

    paths.raw_trainer_statistics_jsonl.write_text(
        json.dumps(
            {
                "trainer_id": "9001",
                "as_of_date": target_date.isoformat(),
                "trainer_name": "Cached Trainer",
                "total_starts": 50,
                "first_place": 10,
                "first_rate": 20.0,
            }
        ) + "\n",
        encoding="utf-8",
    )
    resolved_race_numbers.clear()
    prediction_request_policies.clear()
    trainer_requests.clear()
    html_frame = ingest_real_tjk_data(
        target_date,
        target_date,
        paths,
        require_results=False,
        hippodrome="Ankara",
        race_no=1,
        program_from_html=True,
    )

    assert not html_frame.empty
    assert html_requests == [1]
    assert resolved_race_numbers == []
    assert prediction_request_policies
    assert all(
        policy == {"include_workouts": True, "request_timeout": 5.0, "max_retries": 0}
        for policy in prediction_request_policies
    )
    assert all(
        policy == {"request_timeout": 5.0, "max_retries": 0}
        for _trainer_id, policy in trainer_requests
    )
    requested_trainer_ids = {trainer_id for trainer_id, _policy in trainer_requests}
    assert 9001 not in requested_trainer_ids
    assert 9002 in requested_trainer_ids


def test_local_feature_pipeline_does_not_call_network(tmp_path, monkeypatch) -> None:
    paths = SimpleNamespace(
        raw_daily_program_jsonl=tmp_path / "program.jsonl",
        raw_race_results_jsonl=tmp_path / "results.jsonl",
        raw_history_jsonl=tmp_path / "history.jsonl",
        raw_workouts_jsonl=tmp_path / "workouts.jsonl",
        raw_trainer_statistics_jsonl=tmp_path / "trainers.jsonl",
        raw_csv=tmp_path / "raw.csv",
    )
    program = {
        "race_id": "race-1", "race_date": "2026-09-10", "horse_id": "horse-1",
        "trainer_id": "trainer-1", "hippodrome": "Ankara", "race_no": 1,
        "field_size": 8, "odds": 3.0, "draw": 2, "weight_kg": 57,
    }
    result = {"race_id": "race-1", "horse_id": "horse-1", "finish_position": 1}
    trainer_rows = [
        {"trainer_id": "trainer-1", "as_of_date": "2026-09-09", "total_starts": 20,
         "first_place": 4, "first_rate": 20, "second_rate": 10, "third_rate": 5,
         "fourth_rate": 5, "fifth_rate": 5},
        {"trainer_id": "trainer-1", "as_of_date": "2026-09-10", "total_starts": 100,
         "first_place": 50, "first_rate": 50},
        {"trainer_id": "trainer-1", "as_of_date": "2026-09-11", "total_starts": 200,
         "first_place": 100, "first_rate": 50},
    ]
    paths.raw_daily_program_jsonl.write_text(json.dumps(program) + "\n", encoding="utf-8")
    paths.raw_race_results_jsonl.write_text(json.dumps(result) + "\n", encoding="utf-8")
    paths.raw_trainer_statistics_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in trainer_rows), encoding="utf-8"
    )

    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("Local feature generation attempted an HTTP request")

    monkeypatch.setattr(httpx.Client, "request", forbidden_network)

    frame = build_local_raw_frame(paths)

    assert len(frame) == 1
    assert frame.iloc[0]["trainer_starts"] == 20
    assert frame.iloc[0]["trainer_stats_missing"] == 0