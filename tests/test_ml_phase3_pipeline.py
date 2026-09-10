from __future__ import annotations

from datetime import date
from datetime import timedelta

import pandas as pd
import pytest

from atyaris.ml.pipeline import (
    Phase1Paths,
    build_features,
    ingest_real_data,
    predict_for_date,
    prepare_prediction_features,
    preprocess_raw,
    run_phase1_backtest,
    train_phase1_model,
)


def _setup_phase3_artifacts(tmp_path, monkeypatch) -> Phase1Paths:
    from atyaris.ml.provider import SyntheticRacingDataProvider as _FixtureRacingDataProvider

    paths = Phase1Paths(
        raw_csv=tmp_path / "raw.csv",
        clean_csv=tmp_path / "clean.csv",
        features_csv=tmp_path / "features.csv",
        model_path=tmp_path / "phase3.joblib",
    )
    provider = _FixtureRacingDataProvider()
    monkeypatch.setattr(
        "atyaris.ml.pipeline.ingest_real_tjk_data",
        lambda start_date, end_date, paths_arg: provider.get_dataset(start_date, end_date),
    )
    ingest_real_data(date(2025, 1, 1), date(2025, 4, 30), paths)
    preprocess_raw(paths)
    build_features(paths)
    train_phase1_model(paths, holdout_days=20, calibration_days=14, calibration_method="isotonic")
    return paths


def test_phase3_predict_has_calibrated_and_ev_outputs(tmp_path, monkeypatch) -> None:
    paths = _setup_phase3_artifacts(tmp_path, monkeypatch)
    pred = predict_for_date(
        paths,
        target_date=date(2025, 4, 30),
        enable_ev=True,
        ev_probability_threshold=0.15,
        ev_min_edge=0.0,
        ev_min_value=-0.5,
    )

    assert not pred.empty
    assert {
        "raw_probability",
        "calibrated_probability",
        "place2_probability",
        "place3_probability",
        "top3_probability",
        "edge",
        "ev",
        "kelly_fraction",
        "bet_decision",
        "confidence",
    }.issubset(pred.columns)

    race_sums = pred.groupby("race_id")["calibrated_probability"].sum().round(6)
    assert (race_sums == 1.0).all()


def test_phase3_backtest_reports_calibration_and_roi(tmp_path, monkeypatch) -> None:
    paths = _setup_phase3_artifacts(tmp_path, monkeypatch)
    out = run_phase1_backtest(
        paths,
        min_train_days=45,
        calibration_days=14,
        calibration_method="platt",
        ev_probability_threshold=0.15,
        ev_min_edge=0.0,
        ev_min_value=-0.5,
    )

    assert out["evaluated_days"] > 0
    assert "ece" in out
    assert "roi" in out
    assert "total_bets" in out


def test_phase3_predict_returns_empty_for_unknown_date(tmp_path, monkeypatch) -> None:
    paths = _setup_phase3_artifacts(tmp_path, monkeypatch)

    frame = pd.read_csv(paths.features_csv)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    unknown_date = max(frame["date"]) + timedelta(days=400)

    pred = predict_for_date(
        paths,
        unknown_date,
        enable_ev=True,
        ev_probability_threshold=0.18,
        ev_min_edge=0.03,
        ev_min_value=0.02,
    )

    assert pred.empty
    assert "raw_probability" in pred.columns
    assert "calibrated_probability" in pred.columns
    assert "rank" in pred.columns
    assert "confidence" in pred.columns


def test_phase3_training_fails_before_model_fit_when_features_are_empty(tmp_path) -> None:
    paths = Phase1Paths(
        raw_csv=tmp_path / "raw.csv",
        clean_csv=tmp_path / "clean.csv",
        features_csv=tmp_path / "features.csv",
        model_path=tmp_path / "phase3.joblib",
    )
    pd.DataFrame(columns=["date"]).to_csv(paths.features_csv, index=False)

    with pytest.raises(ValueError, match="Feature verisi bos veri uretti"):
        train_phase1_model(paths)


def test_phase3_training_handles_single_day_window(tmp_path, monkeypatch) -> None:
    paths = _setup_phase3_artifacts(tmp_path, monkeypatch)
    frame = pd.read_csv(paths.features_csv)
    frame["date"] = "2025-04-30"
    frame.to_csv(paths.features_csv, index=False)

    artifact = train_phase1_model(
        paths,
        holdout_days=30,
        calibration_days=21,
        calibration_method="none",
    )

    assert artifact.feature_columns
    assert paths.model_path.exists()


def test_prediction_features_are_separate_from_labelled_training_features(tmp_path, monkeypatch) -> None:
    paths = _setup_phase3_artifacts(tmp_path, monkeypatch)
    provider = __import__("atyaris.ml.provider", fromlist=["SyntheticRacingDataProvider"]).SyntheticRacingDataProvider()
    target_date = date(2025, 5, 1)

    monkeypatch.setattr(
        "atyaris.ml.pipeline.ingest_real_tjk_data",
        lambda start_date, end_date, paths_arg, progress_callback=None, require_results=True: provider.get_dataset(
            start_date, end_date
        ).drop(columns=["finish_position", "is_winner"], errors="ignore"),
    )

    prediction = prepare_prediction_features(target_date, paths)

    assert not prediction.empty
    assert set(pd.to_datetime(prediction["date"]).dt.date) == {target_date}
    assert paths.prediction_features_csv.exists()
    training_dates = set(pd.to_datetime(pd.read_csv(paths.features_csv)["date"]).dt.date)
    assert target_date not in training_dates
    assert "is_winner" in prediction.columns
    assert prediction["is_winner"].eq(0).all()


def test_training_ingestion_resumes_from_completed_day_checkpoint(tmp_path, monkeypatch) -> None:
    from atyaris.ml.provider import SyntheticRacingDataProvider

    paths = Phase1Paths(
        raw_csv=tmp_path / "raw.csv",
        clean_csv=tmp_path / "clean.csv",
        features_csv=tmp_path / "features.csv",
        model_path=tmp_path / "model.joblib",
    )
    provider = SyntheticRacingDataProvider()
    calls: list[date] = []
    failed_once = {date(2025, 1, 2)}

    def ingest_day(start_date, end_date, paths_arg, progress_callback=None):
        calls.append(start_date)
        if start_date in failed_once:
            failed_once.remove(start_date)
            raise RuntimeError("gecici TJK hatasi")
        return provider.get_dataset(start_date, end_date)

    monkeypatch.setattr("atyaris.ml.pipeline.ingest_real_tjk_data", ingest_day)

    with pytest.raises(RuntimeError, match="gecici TJK hatasi"):
        ingest_real_data(date(2025, 1, 1), date(2025, 1, 3), paths)

    resumed = ingest_real_data(date(2025, 1, 1), date(2025, 1, 3), paths)

    assert calls == [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 2), date(2025, 1, 3)]
    assert set(pd.to_datetime(resumed["date"]).dt.date) == {
        date(2025, 1, 1),
        date(2025, 1, 2),
        date(2025, 1, 3),
    }
