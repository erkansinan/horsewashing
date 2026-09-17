from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from atyaris.config import Settings
from atyaris.ml.phase5 import register_training_run, run_phase5_report
from atyaris.ml.pipeline import Phase1Paths, build_features, ingest_real_data, preprocess_raw, train_phase1_model
from atyaris.ml.tracking import load_recent_runs


def test_phase5_report_generation_and_tracking(tmp_path, monkeypatch) -> None:
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
    artifact = train_phase1_model(paths, holdout_days=20, calibration_days=14, calibration_method="isotonic")

    tracking_db = tmp_path / "exp.sqlite3"
    report_dir = tmp_path / "reports"
    settings = Settings(
        phase5_tracking_db_path=str(tracking_db),
        phase5_report_dir=str(report_dir),
        phase5_top_feature_count=8,
        phase5_auto_promote_candidate=True,
        phase1_min_train_days=40,
        phase4_simulation_count=1500,
        phase4_beam_width=30,
        phase4_top_per_leg=3,
        phase4_default_budget=5.0,
    )

    feat = pd.read_csv(paths.features_csv)
    version = register_training_run(
        settings,
        paths,
        holdout_days=20,
        calibration_method="isotonic",
        blend_weight=float(artifact.logistic_weight),
        feature_frame=feat,
    )
    assert version.startswith("model_v")

    result = run_phase5_report(settings, paths, target_date=date(2025, 4, 30), budget=5.0)

    html_path = Path(result["report_paths"]["html"])
    json_path = Path(result["report_paths"]["json"])
    assert html_path.exists()
    assert json_path.exists()

    recent = load_recent_runs(str(tracking_db), limit=5)
    assert len(recent["models"]) >= 1
    assert len(recent["backtests"]) >= 1
    assert len(recent["tickets"]) >= 1

    fixed_test = result["model_health"]["fixed_test_comparison"]
    assert fixed_test["status"] == "ok"
    assert fixed_test["bootstrap_samples"] == 1000
    assert len(fixed_test["top4_difference_bootstrap_ci"]) == 2
    assert "model_top4" in fixed_test
    assert "market_top4" in fixed_test
    assert "model_log_loss" in fixed_test
    assert "market_log_loss" in fixed_test
    report_payload = json_path.read_text(encoding="utf-8")
    assert '"fixed_test_comparison"' in report_payload

    fixed_test = result["model_health"]["fixed_test_comparison"]
    assert fixed_test["status"] == "ok"
    assert fixed_test["bootstrap_samples"] == 1000
    assert len(fixed_test["top4_difference_bootstrap_ci"]) == 2
    assert "model_top4" in fixed_test
    assert "market_top4" in fixed_test
    assert "model_log_loss" in fixed_test
    assert "market_log_loss" in fixed_test
