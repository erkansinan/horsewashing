from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atyaris.config import Settings
from atyaris.ml.explainability import compute_optional_shap_summary, compute_permutation_importance
from atyaris.ml.modeling import load_phase3_artifact
from atyaris.ml.pipeline import (
    Phase1Paths,
    optimize_for_date,
    predict_for_date,
    run_phase1_backtest,
)
from atyaris.ml.reporting import generate_phase5_report
from atyaris.ml.tracking import (
    generate_model_version,
    init_tracking_db,
    latest_model_version,
    load_recent_runs,
    log_backtest_run,
    log_ticket_run,
    mark_model_production,
    register_model_run,
)


def register_training_run(
    settings: Settings,
    paths: Phase1Paths,
    *,
    holdout_days: int,
    calibration_method: str,
    blend_weight: float,
    feature_frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> str:
    db_path = settings.phase5_tracking_db_path
    init_tracking_db(db_path)

    model_version = generate_model_version()
    train_start = str(min(feature_frame["date"])) if not feature_frame.empty else None
    train_end = str(max(feature_frame["date"])) if not feature_frame.empty else None

    tracked_features = feature_columns or [
        column for column in feature_frame.columns
        if column not in {"date", "race_id", "horse_id", "is_winner"}
    ]
    register_model_run(
        db_path,
        model_version=model_version,
        artifact_path=str(paths.model_path),
        train_start_date=train_start,
        train_end_date=train_end,
        holdout_days=holdout_days,
        calibration_method=calibration_method,
        blend_weight=blend_weight,
        metrics={
            "note": "phase5 training registration",
            "training_rows": int(len(feature_frame)),
            "feature_count": len(tracked_features),
            "feature_columns": tracked_features,
        },
        status="candidate",
    )

    if settings.phase5_auto_promote_candidate:
        mark_model_production(db_path, model_version)
    return model_version


def run_phase5_report(
    settings: Settings,
    paths: Phase1Paths,
    *,
    target_date: date,
    budget: float | None,
) -> dict[str, Any]:
    db_path = settings.phase5_tracking_db_path
    init_tracking_db(db_path)

    backtest = run_phase1_backtest(
        paths,
        min_train_days=settings.phase1_min_train_days,
        calibration_days=settings.phase3_calibration_days,
        calibration_method=settings.phase3_calibration_method,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )

    pred = predict_for_date(
        paths,
        target_date,
        enable_ev=True,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )

    opt = optimize_for_date(paths, target_date, settings, budget=budget)

    model_version = latest_model_version(db_path) or "model_v_unknown"
    log_backtest_run(db_path, model_version, backtest)
    log_ticket_run(db_path, model_version, target_date.isoformat(), float(budget or settings.phase4_default_budget), opt.get("summary", {}))

    artifact, _ = load_phase3_artifact(str(paths.model_path))
    feature_cols = artifact.feature_columns
    pred_for_explain = pred.copy()
    if "is_winner" not in pred_for_explain.columns:
        pred_for_explain["is_winner"] = 0

    perm = compute_permutation_importance(artifact, pred_for_explain, feature_cols)
    shap_summary = compute_optional_shap_summary(artifact, pred_for_explain, feature_cols)

    explainability = {
        "permutation_top": perm[: settings.phase5_top_feature_count],
        "shap": shap_summary,
    }

    health_path = paths.model_path.with_suffix(".health.json")
    try:
        health_report = json.loads(health_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        health_report = {"passed": False, "checks": [], "reason": "model health report unavailable"}

    tracking_snapshot = load_recent_runs(db_path, limit=10)
    report_paths = generate_phase5_report(
        settings.phase5_report_dir,
        model_version=model_version,
        target_date=target_date.isoformat(),
        backtest_metrics=backtest,
        prediction_frame=pred,
        optimization_result=opt,
        explainability=explainability,
        tracking_snapshot=tracking_snapshot,
        health_report=health_report,
    )

    return {
        "model_version": model_version,
        "backtest": backtest,
        "optimization_summary": opt.get("summary", {}),
        "report_paths": report_paths,
        "explainability": explainability,
        "model_health": health_report,
    }
