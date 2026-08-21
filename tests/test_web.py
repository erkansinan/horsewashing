"""Web arayuzu (FastAPI) icin testler. Yalnizca ``sample`` veri kaynagini
kullanir; canli TJK sitesine bagimli degildir.
"""
from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

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
