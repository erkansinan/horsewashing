from __future__ import annotations

from datetime import date

from atyaris.config import Settings
from atyaris.ml.optimizer import optimize_ticket_portfolio
from atyaris.ml.pipeline import (
    Phase1Paths,
    build_features,
    ingest_real_data,
    optimize_for_date,
    preprocess_raw,
    train_phase1_model,
)


def _prepare_prediction_frame(tmp_path, monkeypatch):
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


def test_phase4_budget_constraint_and_cost(tmp_path, monkeypatch) -> None:
    paths = _prepare_prediction_frame(tmp_path, monkeypatch)
    settings = Settings(
        phase4_default_budget=5.0,
        phase4_unit_cost=1.0,
        phase4_simulation_count=3000,
        phase4_beam_width=40,
        phase4_top_per_leg=3,
        phase4_no_bet_ev_threshold=-1.0,
    )
    out = optimize_for_date(paths, date(2025, 4, 30), settings, budget=5.0)

    assert out["summary"]["spent"] <= 5.0
    assert out["summary"]["column_count"] <= 5
    for col in out["columns"]:
        assert abs(col["cost"] - 1.0) < 1e-9


def test_phase4_monte_carlo_distribution_and_no_bet() -> None:
    data = {
        "race_id": [
            "r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4", "r5", "r5", "r6", "r6",
        ],
        "horse_id": [
            "a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2", "f1", "f2",
        ],
        "calibrated_probability": [0.7, 0.3, 0.6, 0.4, 0.65, 0.35, 0.55, 0.45, 0.75, 0.25, 0.6, 0.4],
        "market_probability_used": [0.7, 0.3, 0.6, 0.4, 0.65, 0.35, 0.55, 0.45, 0.75, 0.25, 0.6, 0.4],
        "confidence": [0.8] * 12,
        "ev": [0.1] * 12,
        "edge": [0.01] * 12,
        "odds": [2.0] * 12,
    }
    import pandas as pd

    pred = pd.DataFrame(data)

    out_ok = optimize_ticket_portfolio(
        pred,
        budget=3.0,
        unit_cost=1.0,
        beam_width=20,
        top_per_leg=2,
        simulation_count=4000,
        no_bet_ev_threshold=-1.0,
        min_confidence=0.1,
    )
    assert out_ok["summary"]["column_count"] <= 3
    if out_ok["columns"]:
        for c in out_ok["columns"]:
            assert 0.0 <= c["monte_carlo_hit_rate"] <= 1.0

    out_no_bet = optimize_ticket_portfolio(
        pred,
        budget=3.0,
        unit_cost=1.0,
        beam_width=20,
        top_per_leg=2,
        simulation_count=2000,
        no_bet_ev_threshold=999.0,
        min_confidence=0.9,
    )
    assert out_no_bet["summary"]["status"] == "NO_BET"
