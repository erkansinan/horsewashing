"""Web arayuzu (FastAPI) icin testler. Yalnizca ``sample`` veri kaynagini
kullanir; canli TJK sitesine bagimli degildir.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
import atyaris.web.app as web_app_module
import pandas as pd

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.models.entities import (
    HorseStatistics,
    Jockey,
    PastPerformance,
    Race,
    RaceEntry,
    Trainer,
    TrackSurface,
    WorkoutRecord,
)
from atyaris.ml.tracking import register_model_run
from atyaris.config import Settings
from atyaris.web.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_index_lists_sample_races() -> None:
    client = _client()
    response = client.get("/", params={"source": "sample"})
    assert response.status_code == 200
    assert "Gunun Yarislari" in response.text
    assert "Tahmin Uret" in response.text


def test_training_history_api_and_html_show_model_changes(monkeypatch, tmp_path) -> None:
    tracking_db = tmp_path / "training-history.sqlite3"
    health_path = tmp_path / "new.health.json"
    health_path.write_text(
        '{"passed": true, "metrics": {"train": {"log_loss": 0.2, "top1": 0.5}, '
        '"validation": {"log_loss": 0.3, "top1": 0.4}, "test": {"log_loss": 0.4, "top1": 0.3}}, '
        '"checks": [{"name": "test_beats_baseline", "details": {"full_log_loss": 0.4, '
        '"market_only_log_loss": 0.5, "full_top1": 0.3, "market_only_top1": 0.2, "favorite_top1": 0.1}}], '
        '"calibration_curve": [{"bin_low": 0.0, "bin_high": 0.1, "predicted": 0.05, "observed": 0.04}], '
        '"feature_importance": [{"feature": "speed", "importance": 0.8, "coefficient": 0.8}]}',
        encoding="utf-8",
    )
    register_model_run(
        str(tracking_db),
        model_version="model_old",
        artifact_path="old.joblib",
        train_start_date="2025-01-01",
        train_end_date="2025-03-31",
        holdout_days=20,
        calibration_method="isotonic",
        blend_weight=0.5,
        metrics={"feature_count": 2, "feature_columns": ["speed", "form"]},
    )
    register_model_run(
        str(tracking_db),
        model_version="model_new",
        artifact_path=str(tmp_path / "new.joblib"),
        train_start_date="2025-02-01",
        train_end_date="2025-04-30",
        holdout_days=30,
        calibration_method="platt",
        blend_weight=0.7,
        metrics={"feature_count": 2, "feature_columns": ["speed", "pace"]},
    )
    settings = Settings(phase5_tracking_db_path=str(tracking_db))
    monkeypatch.setattr(web_app_module, "get_settings", lambda: settings)

    client = _client()
    api_response = client.get("/api/training/history", params={"start_date": "2025-04-01"})
    assert api_response.status_code == 200
    payload = api_response.json()
    assert payload["count"] == 1
    assert payload["latest_training"]["model_version"] == "model_new"
    assert payload["latest_training"]["changes_from_previous"]["added_features"] == ["pace"]
    assert payload["latest_training"]["changes_from_previous"]["removed_features"] == ["form"]
    assert payload["latest_training"]["changes_from_previous"]["calibration_changed"] is True
    assert payload["latest_training"]["model_health"]["metrics"]["test"]["top1"] == 0.3

    html_response = client.get("/training-history")
    assert html_response.status_code == 200
    assert html_response.url.path == "/training"
    assert "Son eğitim tarihi" in html_response.text
    assert "model_new" in html_response.text
    assert "pace" in html_response.text
    assert "Model sağlık raporu" in html_response.text
    assert "Baseline karşılaştırması" in html_response.text
    assert "Kalibrasyon eğrisi" in html_response.text
    assert "speed" in html_response.text


def test_training_page_contains_start_form_and_active_jobs(monkeypatch) -> None:
    web_app_module._TRAINING_JOBS["active-training"] = {
        "status": "running",
        "message": "Gunluk veri checkpoint",
        "start_date": "2026-09-09",
        "end_date": "2026-09-10",
    }
    try:
        response = _client().get("/training")
        assert response.status_code == 200
        assert 'action="/train"' in response.text
        assert "Yeni eğitim başlat" in response.text
        assert "Gunluk veri checkpoint" in response.text
        assert "active-training" in response.text
        assert "ML Eğitim" in response.text
    finally:
        web_app_module._TRAINING_JOBS.pop("active-training", None)


def test_training_page_exposes_pause_and_resume_controls() -> None:
    web_app_module._TRAINING_JOBS["running-training"] = {
        "status": "running",
        "message": "Egitim",
        "start_date": "2026-09-09",
        "end_date": "2026-09-10",
    }
    web_app_module._TRAINING_JOBS["paused-training"] = {
        "status": "paused",
        "message": "Egitim duraklatildi; checkpoint korundu",
        "start_date": "2026-09-09",
        "end_date": "2026-09-10",
    }
    try:
        response = _client().get("/training")
        assert response.status_code == 200
        assert 'data-job-id="running-training"' in response.text
        assert 'data-job-id="paused-training"' in response.text
        assert 'class="pause-training"' in response.text
        assert 'class="resume-training"' in response.text
        assert "Eğitimi duraklat" in response.text
        assert "Kaldığı yerden devam et" in response.text
    finally:
        web_app_module._TRAINING_JOBS.pop("running-training", None)
        web_app_module._TRAINING_JOBS.pop("paused-training", None)


def test_pause_and_resume_training_job(monkeypatch) -> None:
    job_id = "pause-route-job"
    web_app_module._TRAINING_JOBS[job_id] = {
        "status": "running",
        "message": "Egitim",
        "start_date": "2026-09-09",
        "end_date": "2026-09-10",
    }
    web_app_module._TRAINING_CANCEL_EVENTS[job_id] = __import__("threading").Event()
    try:
        client = _client()
        pause_response = client.post(f"/train/pause/{job_id}")
        assert pause_response.status_code == 200
        assert pause_response.json()["status"] == "running"
        assert pause_response.json()["message"] == "Egitim duraklatiliyor..."
        assert job_id in web_app_module._TRAINING_PAUSE_REQUESTS
    finally:
        web_app_module._TRAINING_JOBS.pop(job_id, None)
        web_app_module._TRAINING_CANCEL_EVENTS.pop(job_id, None)
        web_app_module._TRAINING_PAUSE_REQUESTS.discard(job_id)


def test_interrupted_training_job_is_restored_as_paused(monkeypatch, tmp_path) -> None:
    settings = Settings(phase1_raw_csv_path=str(tmp_path / "tjk_real_races.csv"))
    monkeypatch.setattr(web_app_module, "get_settings", lambda: settings)
    job_id = "restartable-training"
    web_app_module._TRAINING_JOBS[job_id] = {
        "status": "running",
        "message": "Gunluk veri checkpoint",
        "start_date": "2026-09-01",
        "end_date": "2026-09-10",
    }
    try:
        web_app_module._persist_training_jobs()
        web_app_module._TRAINING_JOBS.clear()

        _client()

        restored = web_app_module._TRAINING_JOBS[job_id]
        assert restored["status"] == "paused"
        assert restored["start_date"] == "2026-09-01"
        assert restored["end_date"] == "2026-09-10"
        assert "checkpointten devam" in str(restored["message"])
    finally:
        web_app_module._TRAINING_JOBS.pop(job_id, None)


def test_training_status_page_contains_interruption_confirmation(monkeypatch) -> None:
    web_app_module._TRAINING_JOBS["active-job"] = {"status": "running", "message": "Egitim"}
    try:
        response = _client().get("/", params={"source": "sample"})
        assert response.status_code == 200
        assert "devam eden eğitim durdurulacaktır" in response.text
        assert "/train/cancel/" in response.text
    finally:
        web_app_module._TRAINING_JOBS.pop("active-job", None)
        web_app_module._TRAINING_CANCEL_EVENTS.pop("active-job", None)


def test_ml_prediction_does_not_start_hidden_refresh_for_missing_date(monkeypatch) -> None:
    settings = Settings()
    monkeypatch.setattr(web_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(web_app_module, "_ml_model_loadable", lambda paths: True)
    monkeypatch.setattr(web_app_module, "_ml_has_date", lambda paths, target_date: False)
    monkeypatch.setattr(
        web_app_module,
        "prepare_prediction_features",
        lambda target_date, paths: (_ for _ in ()).throw(RuntimeError("test TJK hatasi")),
    )

    response = _client().get(
        "/predict",
        params={"race_id": "Ankara-1", "source": "ml", "date": "2026-09-10", "city": "Ankara"},
    )

    assert response.status_code == 200
    assert "2026-09-10 icin labelsiz tahmin feature verisi hazirlanamadi" in response.text


def test_ml_prediction_on_next_day_after_0909_0910_training_is_rejected(monkeypatch) -> None:
    settings = Settings()
    monkeypatch.setattr(web_app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(web_app_module, "_ml_model_loadable", lambda paths: True)
    monkeypatch.setattr(web_app_module, "_ml_has_date", lambda paths, target_date: target_date.isoformat() == "2026-09-10")
    monkeypatch.setattr(
        web_app_module,
        "prepare_prediction_features",
        lambda target_date, paths: (_ for _ in ()).throw(RuntimeError("test TJK hatasi")),
    )

    response = _client().get(
        "/predict",
        params={"race_id": "Ankara-1", "source": "ml", "date": "2026-09-11", "city": "Ankara"},
    )

    assert response.status_code == 200
    assert "2026-09-11 icin labelsiz tahmin feature verisi hazirlanamadi" in response.text


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


def test_derive_horse_stats_metrics_uses_tjk_and_workout_data() -> None:
    stats = HorseStatistics(
        horse_id="H1",
        horse_name="Test Horse",
        past_performances=[
            PastPerformance(
                race_date=date.today() - timedelta(days=12),
                hippodrome="Ankara",
                distance_m=1400,
                surface=TrackSurface.KUM,
                finish_position=1,
                field_size=8,
                race_time_seconds=82.0,
                early_pace_index=0.75,
                weight_kg=56.5,
            ),
            PastPerformance(
                race_date=date.today() - timedelta(days=23),
                hippodrome="Ankara",
                distance_m=1400,
                surface=TrackSurface.KUM,
                finish_position=2,
                field_size=8,
                race_time_seconds=84.0,
                early_pace_index=0.65,
                weight_kg=56.0,
            ),
            PastPerformance(
                race_date=date.today() - timedelta(days=70),
                hippodrome="Istanbul (Veliefendi)",
                distance_m=1600,
                surface=TrackSurface.CIM,
                finish_position=5,
                field_size=9,
                race_time_seconds=88.0,
                early_pace_index=0.45,
                weight_kg=57.0,
            ),
        ],
        workout_records=[
            WorkoutRecord(
                workout_date=date.today() - timedelta(days=9),
                hippodrome="Ankara",
                surface="Kum",
                distance_m=1200,
                time_seconds=72.0,
            )
        ],
    )

    metrics = web_app_module._derive_horse_stats_metrics(
        stats,
        race_distance=1400,
        race_surface="Kum",
        race_track="Ankara",
    )

    assert metrics["form_avg_5"] > 0
    assert metrics["form_avg_10"] > 0
    assert metrics["days_since_last_race"] == 12
    assert metrics["track_fit"] > 0
    assert metrics["surface_fit"] > 0
    assert metrics["distance_fit"] > 0
    assert 0.0 <= metrics["pace_pressure"] <= 1.0


def test_form_strength_sort_prioritizes_form_metrics_over_ev() -> None:
    rows = [
        {
            "horse_id": "DEJAME",
            "form_avg_3": 25.0,
            "form_avg_5": 17.0,
            "form_avg_10": 22.9,
            "track_fit": 24.72,
            "surface_fit": 33.33,
            "distance_fit": 18.29,
            "pace_pressure": 1.0,
            "ev": 0.94,
        },
        {
            "horse_id": "KALI_STRATA",
            "form_avg_3": 60.0,
            "form_avg_5": 59.0,
            "form_avg_10": 56.9,
            "track_fit": 41.54,
            "surface_fit": 48.83,
            "distance_fit": 56.79,
            "pace_pressure": 1.0,
            "ev": -0.635,
        },
        {
            "horse_id": "SCHATZ",
            "form_avg_3": 49.05,
            "form_avg_5": 58.43,
            "form_avg_10": 41.21,
            "track_fit": 48.12,
            "surface_fit": 33.73,
            "distance_fit": 27.45,
            "pace_pressure": 1.0,
            "ev": 0.166,
        },
    ]

    ordered = sorted(rows, key=lambda r: web_app_module._ml_sort_value(r, "form_strength"), reverse=True)

    assert ordered[0]["horse_id"] == "KALI_STRATA"
    assert ordered[1]["horse_id"] == "SCHATZ"
    assert ordered[2]["horse_id"] == "DEJAME"


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
