"""Web arayuzu (FastAPI) icin testler. Yalnizca ``sample`` veri kaynagini
kullanir; canli TJK sitesine bagimli degildir.
"""
from __future__ import annotations

from datetime import date
from datetime import datetime

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
import atyaris.web.app as web_app_module
import pandas as pd

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
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


def test_index_tjk_dns_failure_falls_back_to_known_hippodromes(monkeypatch) -> None:
    class FailingTjkSource(TJKHtmlDataSource):
        def get_available_hippodromes(self, target_date: date) -> list[str]:  # type: ignore[override]
            raise DataSourceError("Hipodrom listesi cekilirken hata: [Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(web_app_module, "build_data_source", lambda source_name, settings: FailingTjkSource())

    client = _client()
    response = client.get("/", params={"source": "tjk", "date": date.today().isoformat()})

    assert response.status_code == 200
    assert "Veri kaynagi hatasi" not in response.text
    assert "su an ulasilamiyor" in response.text
    assert "Ankara" in response.text


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
    source = SampleDataSource()
    monkeypatch.setattr(web_app_module, "_bulletin_source_for_ml", lambda settings: source)

    client = _client()
    response = client.get(
        "/",
        params={"source": "ml", "date": date.today().isoformat(), "city": "Ankara"},
    )

    assert response.status_code == 200
    assert "ML Yarislari" in response.text
    assert "ML Tahmin Gor" in response.text
    assert "Ankara" in response.text


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
    assert "ML Tahmin -" in response.text
    assert "istatistiksel analize dayanir" in response.text


def test_predict_ml_mode_falls_back_when_city_track_mismatch(monkeypatch) -> None:
    target_date = date.today()
    fake_pred = pd.DataFrame(
        [
            {
                "race_id": f"{target_date.strftime('%Y%m%d')}_01",
                "horse_id": "H0001",
                "horse_name": "HORSE_0001",
                "rank": 1,
                "odds": 2.5,
                "calibrated_probability": 0.34,
                "confidence": 0.88,
                "edge": 0.12,
                "ev": 0.10,
                "bet_decision": "BET",
                "track": "ISTANBUL",
            }
        ]
    )

    source = SampleDataSource()
    ankara_race = source.get_daily_races(target_date, "Ankara")[0]

    monkeypatch.setattr(web_app_module, "_bulletin_source_for_ml", lambda settings: source)
    monkeypatch.setattr(web_app_module, "_ensure_ml_ready_for_date", lambda settings, d: None)
    monkeypatch.setattr(web_app_module, "predict_for_date", lambda *args, **kwargs: fake_pred)

    client = _client()
    response = client.get(
        "/predict",
        params={
            "race_id": ankara_race.id,
            "source": "ml",
            "date": target_date.isoformat(),
            "city": "Ankara",
        },
    )

    assert response.status_code == 200
    assert "ML Tahmin - Ankara - 1. Kosu" in response.text
    assert "HORSE_" not in response.text


def test_predict_ml_ticket_preview_keeps_same_track_and_future_legs(monkeypatch) -> None:
    target_date = date.today()
    source = SampleDataSource()
    ankara_race = source.get_daily_races(target_date, "Ankara")[0]
    ankara_numbers = [int(entry.number) for entry in ankara_race.entries]

    rid1 = f"{target_date.strftime('%Y%m%d')}_01"
    rid2 = f"{target_date.strftime('%Y%m%d')}_02"
    rid3 = f"{target_date.strftime('%Y%m%d')}_03"
    rid4 = f"{target_date.strftime('%Y%m%d')}_04"

    race1_rows = []
    for idx, num in enumerate(ankara_numbers, start=1):
        race1_rows.append(
            {
                "race_id": rid1,
                "horse_id": f"H1{idx:03d}",
                "horse_name": f"AT A{idx}",
                "rank": idx,
                "odds": 2.1 + idx * 0.2,
                "calibrated_probability": max(0.30 - idx * 0.01, 0.05),
                "confidence": 0.80,
                "edge": 0.05,
                "ev": 0.06,
                "bet_decision": "BET",
                "track": "ANKARA",
                "draw": num,
            }
        )

    fake_pred = pd.DataFrame(
        race1_rows
        + [
            {
                "race_id": rid2,
                "horse_id": "H2001",
                "horse_name": "AT A2",
                "rank": 1,
                "odds": 3.1,
                "calibrated_probability": 0.27,
                "confidence": 0.76,
                "edge": 0.03,
                "ev": 0.04,
                "bet_decision": "BET",
                "track": "ANKARA",
                "draw": 2,
            },
            {
                "race_id": rid3,
                "horse_id": "H3001",
                "horse_name": "AT A3",
                "rank": 1,
                "odds": 4.2,
                "calibrated_probability": 0.20,
                "confidence": 0.70,
                "edge": 0.01,
                "ev": 0.01,
                "bet_decision": "NO_BET",
                "track": "ANKARA",
                "draw": 3,
            },
            {
                "race_id": rid4,
                "horse_id": "H4001",
                "horse_name": "AT I1",
                "rank": 1,
                "odds": 2.8,
                "calibrated_probability": 0.29,
                "confidence": 0.78,
                "edge": 0.04,
                "ev": 0.05,
                "bet_decision": "BET",
                "track": "ISTANBUL",
                "draw": 1,
            },
        ]
    )
    fake_opt = {
        "summary": {"status": "OK", "budget": 500.0, "spent": 24.0, "column_count": 2},
        "columns": [
            {
                "column_id": "C-good",
                "strategy": "balanced",
                "combination": {rid1: race1_rows[0]["horse_id"], rid2: "H2001"},
                "probability": 0.09,
                "ev": 0.22,
                "monte_carlo_hit_rate": 0.07,
            },
            {
                "column_id": "C-bad",
                "strategy": "aggressive",
                "combination": {rid1: race1_rows[0]["horse_id"], rid4: "H4001"},
                "probability": 0.08,
                "ev": 0.21,
                "monte_carlo_hit_rate": 0.06,
            },
        ],
    }

    monkeypatch.setattr(web_app_module, "_bulletin_source_for_ml", lambda settings: source)
    monkeypatch.setattr(web_app_module, "_ensure_ml_ready_for_date", lambda settings, d: None)
    monkeypatch.setattr(web_app_module, "predict_for_date", lambda *args, **kwargs: fake_pred)
    monkeypatch.setattr(web_app_module, "optimize_for_date", lambda *args, **kwargs: fake_opt)

    client = _client()
    response = client.get(
        "/predict",
        params={
            "race_id": ankara_race.id,
            "source": "ml",
            "date": target_date.isoformat(),
            "city": "Ankara",
        },
    )

    assert response.status_code == 200
    assert "C-good" in response.text
    assert "C-bad" not in response.text


def test_predict_ml_rejects_mismatched_race_rows_and_uses_bulletin_fallback(monkeypatch) -> None:
    target_date = date.today()
    source = SampleDataSource()
    ankara_race = source.get_daily_races(target_date, "Ankara")[0]
    expected_count = len(ankara_race.entries)
    expected_numbers = {entry.number for entry in ankara_race.entries}
    max_expected_number = max(expected_numbers)

    rid = f"{target_date.strftime('%Y%m%d')}_01"
    fake_pred = pd.DataFrame(
        [
            {
                "race_id": rid,
                "horse_id": f"H{i:04d}",
                "horse_name": f"HORSE_{i:04d}",
                "rank": i,
                "odds": 2.0 + i,
                "calibrated_probability": 0.2,
                "confidence": 0.7,
                "edge": 0.01,
                "ev": 0.01,
                "bet_decision": "NO_BET",
                "track": "ANKARA",
                "draw": i,
            }
            for i in range(1, expected_count + 3)
        ]
    )

    monkeypatch.setattr(web_app_module, "_bulletin_source_for_ml", lambda settings: source)
    monkeypatch.setattr(web_app_module, "_ensure_ml_ready_for_date", lambda settings, d: None)
    monkeypatch.setattr(web_app_module, "predict_for_date", lambda *args, **kwargs: fake_pred)

    client = _client()
    response = client.get(
        "/predict",
        params={
            "race_id": ankara_race.id,
            "source": "ml",
            "date": target_date.isoformat(),
            "city": "Ankara",
        },
    )

    assert response.status_code == 200
    assert "HORSE_" not in response.text
    assert f"ML Tahmin - Ankara - {ankara_race.race_no}. Kosu" in response.text

    soup = BeautifulSoup(response.text, "lxml")
    ml_table = None
    for table in soup.find_all("table"):
        header_cells = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if "Rank" in header_cells and "No" in header_cells and "At" in header_cells:
            ml_table = table
            break

    assert ml_table is not None
    body_rows = ml_table.find("tbody").find_all("tr") if ml_table.find("tbody") is not None else []
    shown_numbers = []
    for tr in body_rows:
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        no_text = tds[1].get_text(" ", strip=True)
        if no_text.isdigit():
            shown_numbers.append(int(no_text))

    assert shown_numbers
    assert all(num <= max_expected_number for num in shown_numbers)


def test_predict_ml_fallback_renders_nonzero_place_probabilities(monkeypatch) -> None:
    target_date = date.today()
    source = SampleDataSource()
    ankara_race = source.get_daily_races(target_date, "Ankara")[0]

    fake_pred = pd.DataFrame(
        [
            {
                "race_id": f"{target_date.strftime('%Y%m%d')}_01",
                "horse_id": "H0001",
                "horse_name": "HORSE_0001",
                "rank": 1,
                "odds": 2.5,
                "calibrated_probability": 0.34,
                "confidence": 0.88,
                "edge": 0.12,
                "ev": 0.10,
                "bet_decision": "BET",
                "track": "ISTANBUL",
            }
        ]
    )

    monkeypatch.setattr(web_app_module, "_bulletin_source_for_ml", lambda settings: source)
    monkeypatch.setattr(web_app_module, "_ensure_ml_ready_for_date", lambda settings, d: None)
    monkeypatch.setattr(web_app_module, "predict_for_date", lambda *args, **kwargs: fake_pred)

    client = _client()
    response = client.get(
        "/predict",
        params={
            "race_id": ankara_race.id,
            "source": "ml",
            "date": target_date.isoformat(),
            "city": "Ankara",
        },
    )

    assert response.status_code == 200
    assert f"ML Tahmin - Ankara - {ankara_race.race_no}. Kosu" in response.text
    assert "P(2.)" in response.text
    assert "P(3.)" in response.text
    assert "0.00%" not in response.text
