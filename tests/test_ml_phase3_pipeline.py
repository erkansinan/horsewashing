from __future__ import annotations

from datetime import date

from atyaris.ml.pipeline import (
    Phase1Paths,
    build_features,
    ingest_synthetic,
    predict_for_date,
    preprocess_raw,
    run_phase1_backtest,
    train_phase1_model,
)


def _setup_phase3_artifacts(tmp_path) -> Phase1Paths:
    paths = Phase1Paths(
        raw_csv=tmp_path / "raw.csv",
        clean_csv=tmp_path / "clean.csv",
        features_csv=tmp_path / "features.csv",
        model_path=tmp_path / "phase3.joblib",
    )
    ingest_synthetic(date(2025, 1, 1), date(2025, 4, 30), paths)
    preprocess_raw(paths)
    build_features(paths)
    train_phase1_model(paths, holdout_days=20, calibration_days=14, calibration_method="isotonic")
    return paths


def test_phase3_predict_has_calibrated_and_ev_outputs(tmp_path) -> None:
    paths = _setup_phase3_artifacts(tmp_path)
    pred = predict_for_date(
        paths,
        target_date=date(2025, 4, 30),
        enable_ev=True,
        ev_probability_threshold=0.15,
        ev_min_edge=0.0,
        ev_min_value=-0.5,
    )

    assert not pred.empty
    assert {"raw_probability", "calibrated_probability", "edge", "ev", "bet_decision", "confidence", "uncertainty"}.issubset(pred.columns)

    race_sums = pred.groupby("race_id")["calibrated_probability"].sum().round(6)
    assert (race_sums == 1.0).all()


def test_phase3_backtest_reports_calibration_and_roi(tmp_path) -> None:
    paths = _setup_phase3_artifacts(tmp_path)
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
