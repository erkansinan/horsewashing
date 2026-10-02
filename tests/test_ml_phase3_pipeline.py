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
    read_csv_cached,
    run_phase1_backtest,
    train_phase1_model,
)
from atyaris.ml.ev_kelly import add_ev_kelly_columns
from atyaris.ml.modeling import load_phase3_artifact


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


def test_read_csv_cached_reuses_unchanged_file_without_sharing_mutations(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("value\n1\n", encoding="utf-8")
    original_read_csv = pd.read_csv
    reads = 0

    def counting_read_csv(path, *args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counting_read_csv)

    first = read_csv_cached(csv_path)
    first.loc[0, "value"] = 99
    second = read_csv_cached(csv_path)

    assert reads == 1
    assert second.loc[0, "value"] == 1


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
    _, calibration_payload = load_phase3_artifact(str(paths.model_path))
    assert "place2_gamma" in calibration_payload
    assert "place3_gamma" in calibration_payload
    assert 0.05 <= calibration_payload["place2_gamma"] <= 5.0
    assert 0.05 <= calibration_payload["place3_gamma"] <= 5.0

    race_sums = pred.groupby("race_id")["calibrated_probability"].sum().round(6)
    assert (race_sums == 1.0).all()


def test_ev_is_unavailable_and_no_bet_without_valid_odds() -> None:
    frame = pd.DataFrame(
        {
            "calibrated_probability": [0.7, 0.2],
            "market_probability": [0.5, 0.5],
            "odds": [0.0, float("nan")],
        }
    )

    result = add_ev_kelly_columns(
        frame,
        min_probability=0.18,
        min_edge=0.03,
        min_ev=0.02,
        fractional_kelly=0.35,
        max_kelly_fraction=0.25,
    )

    assert result["ev"].isna().all()
    assert result["kelly_fraction"].eq(0.0).all()
    assert result["bet_decision"].eq("NO_BET").all()


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


def test_phase3_training_rejects_single_day_window(tmp_path, monkeypatch) -> None:
    paths = _setup_phase3_artifacts(tmp_path, monkeypatch)
    frame = pd.read_csv(paths.features_csv)
    frame["date"] = "2025-04-30"
    frame.to_csv(paths.features_csv, index=False)

    with pytest.raises(ValueError, match="en az 6 farkli tarih"):
        train_phase1_model(
            paths,
            holdout_days=30,
            calibration_days=21,
            calibration_method="none",
        )


def test_prediction_features_are_separate_from_labelled_training_features(tmp_path, monkeypatch) -> None:
    paths = _setup_phase3_artifacts(tmp_path, monkeypatch)
    provider = __import__("atyaris.ml.provider", fromlist=["SyntheticRacingDataProvider"]).SyntheticRacingDataProvider()
    target_date = date(2025, 5, 1)
    ingestion_calls = []

    def fake_ingest(start_date, end_date, paths_arg, progress_callback=None, require_results=True, **kwargs):
        ingestion_calls.append(kwargs)
        return provider.get_dataset(
            start_date, end_date
        ).drop(columns=["finish_position", "is_winner"], errors="ignore")

    monkeypatch.setattr("atyaris.ml.pipeline.ingest_real_tjk_data", fake_ingest)

    prediction = prepare_prediction_features(target_date, paths)

    assert not prediction.empty
    assert ingestion_calls[0]["program_from_html"] is True
    assert "program_from_csv" not in ingestion_calls[0]
    assert set(pd.to_datetime(prediction["date"]).dt.date) == {target_date}
    assert paths.prediction_features_csv.exists()
    training_dates = set(pd.to_datetime(pd.read_csv(paths.features_csv)["date"]).dt.date)
    assert target_date not in training_dates
    assert "is_winner" in prediction.columns
    assert prediction["is_winner"].eq(0).all()

    other_city = prediction.iloc[[0]].copy()
    other_city["race_id"] = f"{target_date:%Y%m%d}_99"
    other_city["track"] = "IZMIR"
    earlier_date = prediction.iloc[[0]].copy()
    earlier_date["date"] = date(2025, 4, 30)
    paths.prediction_features_csv.write_text(
        pd.concat([other_city, earlier_date], ignore_index=True).to_csv(index=False),
        encoding="utf-8",
    )

    prepare_prediction_features(target_date, paths)

    snapshot = pd.read_csv(paths.prediction_features_csv)
    assert ((snapshot["date"] == target_date.isoformat()) & (snapshot["track"] == "IZMIR")).any()
    assert (pd.to_datetime(snapshot["date"]).dt.date == date(2025, 4, 30)).any()


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
