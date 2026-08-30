"""Web arayuzu (FastAPI) icin testler. Yalnizca ``sample`` veri kaynagini
kullanir; canli TJK sitesine bagimli degildir.
"""
from __future__ import annotations

from datetime import date
from datetime import datetime

from fastapi.testclient import TestClient
import atyaris.web.app as web_app_module
import pandas as pd

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.models.entities import Jockey, Race, RaceEntry, Trainer, TrackSurface
from atyaris.web.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_index_lists_sample_races() -> None:
    client = _client()
    response = client.get("/", params={"source": "sample"})
    assert response.status_code == 200
    assert "Gunun Yarislari" in response.text
    assert "Tahmin Uret" in response.text


def test_index_with_invalid_date_shows_error() -> None:
    client = _client()
    response = client.get("/", params={"source": "sample", "date": "gecersiz-tarih"})
    assert response.status_code == 200
    assert "Gecersiz istek" in response.text


def test_predict_renders_ranked_horses_and_disclaimer() -> None:
    sample_source = SampleDataSource()
    race = sample_source.get_daily_races(date.today())[0]

    client = _client()
    response = client.get("/predict", params={"race_id": race.id, "source": "sample"})

    assert response.status_code == 200
    assert race.hippodrome in response.text
    assert "istatistiksel analize dayanir" in response.text


def test_predict_with_unknown_race_id_shows_error() -> None:
    client = _client()
    response = client.get("/predict", params={"race_id": "bilinmeyen-id", "source": "sample"})
    assert response.status_code == 200
    assert "Yaris bulunamadi" in response.text


def test_predict_does_not_mark_active_entries_as_scratched_when_stats_missing(monkeypatch) -> None:
    class PartialStatsSource(SampleDataSource):
        def get_horse_statistics(self, entry):  # type: ignore[override,no-untyped-def]
            if entry.number == 1:
                raise DataSourceError("istatistik gecici olarak alinmadi")
            return super().get_horse_statistics(entry)

    source = PartialStatsSource()
    race = source.get_daily_races(date.today())[0]

    monkeypatch.setattr(web_app_module, "build_data_source", lambda source_name, settings: source)

    client = _client()
    response = client.get("/predict", params={"race_id": race.id, "source": "sample"})

    assert response.status_code == 200
    assert "Veri Eksik" in response.text
    assert "Bu at kosmaz (scratch) olarak isaretli." not in response.text


def test_predict_all_pdf_returns_400_when_all_races_unpredictable(monkeypatch) -> None:
    class UnpredictableSource(SampleDataSource):
        def get_daily_races(self, target_date: date, city: str | None = None) -> list[Race]:  # type: ignore[override]
            race = Race(
                id=f"{city or 'X'}-{target_date.isoformat()}-1",
                hippodrome=city or "Ankara",
                race_no=1,
                start_time=datetime.combine(target_date, datetime.min.time()),
                distance_m=1200,
                surface=TrackSurface.KUM,
                entries=[
                    RaceEntry(
                        number=1,
                        horse_id="h1",
                        horse_name="AT 1",
                        jockey=Jockey(name="J1"),
                        trainer=Trainer(name="T1"),
                        weight_kg=55.0,
                        is_scratched=True,
                    )
                ],
            )
            return [race]

        def get_horse_statistics(self, entry):  # type: ignore[override,no-untyped-def]
            raise DataSourceError("istatistik yok")

    source = UnpredictableSource()
    monkeypatch.setattr(web_app_module, "build_data_source", lambda source_name, settings: source)

    client = _client()
    response = client.get("/predict-all-pdf", params={"race_id": "x", "source": "sample", "city": "Ankara"})

    assert response.status_code == 400
    assert "PDF uretilemedi" in response.text


def test_index_ml_mode_lists_ml_races(monkeypatch) -> None:
    fake_pred = pd.DataFrame(
        [
            {
                "race_id": "20260830_01",
                "horse_id": "H0001",
                "rank": 1,
                "calibrated_probability": 0.34,
                "bet_decision": "BET",
                "track": "ANKARA",
            },
            {
                "race_id": "20260830_01",
                "horse_id": "H0002",
                "rank": 2,
                "calibrated_probability": 0.23,
                "bet_decision": "NO_BET",
                "track": "ANKARA",
            },
        ]
    )
    monkeypatch.setattr(web_app_module, "_ensure_ml_ready_for_date", lambda settings, target_date: None)
    monkeypatch.setattr(web_app_module, "predict_for_date", lambda *args, **kwargs: fake_pred)

    client = _client()
    response = client.get("/", params={"source": "ml", "date": date.today().isoformat()})

    assert response.status_code == 200
    assert "ML Yarislari" in response.text
    assert "ML Tahmin Gor" in response.text


def test_predict_ml_mode_renders_ml_table_and_disclaimer(monkeypatch) -> None:
    fake_pred = pd.DataFrame(
        [
            {
                "race_id": "20260830_01",
                "horse_id": "H0001",
                "rank": 1,
                "odds": 2.5,
                "calibrated_probability": 0.34,
                "confidence": 0.88,
                "edge": 0.12,
                "ev": 0.10,
                "bet_decision": "BET",
                "track": "ANKARA",
            },
            {
                "race_id": "20260830_01",
                "horse_id": "H0002",
                "rank": 2,
                "odds": 4.1,
                "calibrated_probability": 0.23,
                "confidence": 0.71,
                "edge": -0.01,
                "ev": -0.02,
                "bet_decision": "NO_BET",
                "track": "ANKARA",
            },
        ]
    )
    fake_opt = {
        "summary": {"status": "OK", "budget": 500.0, "spent": 12.0, "column_count": 1},
        "columns": [
            {
                "column_id": "C1",
                "strategy": "balanced",
                "combination": {"20260830_01": "H0001"},
                "probability": 0.12,
                "ev": 0.34,
                "monte_carlo_hit_rate": 0.08,
            }
        ],
    }

    monkeypatch.setattr(web_app_module, "_ensure_ml_ready_for_date", lambda settings, target_date: None)
    monkeypatch.setattr(web_app_module, "predict_for_date", lambda *args, **kwargs: fake_pred)
    monkeypatch.setattr(web_app_module, "optimize_for_date", lambda *args, **kwargs: fake_opt)

    client = _client()
    response = client.get(
        "/predict",
        params={"race_id": "20260830_01", "source": "ml", "date": date.today().isoformat()},
    )

    assert response.status_code == 200
    assert "ML Tahmin - Race 20260830_01" in response.text
    assert "istatistiksel analize dayanir" in response.text
