"""Web arayuzu (FastAPI) icin testler. Yalnizca ``sample`` veri kaynagini
kullanir; canli TJK sitesine bagimli degildir.
"""
from __future__ import annotations

from datetime import date
from datetime import datetime

from fastapi.testclient import TestClient
import atyaris.web.app as web_app_module

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
