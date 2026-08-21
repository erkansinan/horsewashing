"""Web arayuzu (FastAPI) icin testler. Yalnizca ``sample`` veri kaynagini
kullanir; canli TJK sitesine bagimli degildir.
"""
from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient
import atyaris.web.app as web_app_module

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.sample_source import SampleDataSource
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
