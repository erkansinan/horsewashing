from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from atyaris.config import Settings
from atyaris.ml.backtest import WalkForwardResult, walk_forward_backtest
from atyaris.ml.calibration import apply_calibrator, fit_calibrator, from_payload, to_payload
from atyaris.ml.ev import add_market_ev_columns
from atyaris.ml.features import FeatureBuildResult, assert_no_leakage_columns, build_leakage_safe_features, preprocess_dataset
from atyaris.ml.modeling import (
    EnsembleArtifact,
    load_phase3_artifact,
    predict_ensemble_raw_probability,
    save_phase3_artifact,
    train_phase3_ensemble,
)
from atyaris.ml.optimizer import optimize_ticket_portfolio
from atyaris.ml.provider import SyntheticProviderConfig, SyntheticRacingDataProvider


@dataclass
class Phase1Paths:
    raw_csv: Path = Path("data/raw/synthetic_races.csv")
    clean_csv: Path = Path("data/processed/clean_races.csv")
    features_csv: Path = Path("data/processed/features_phase1.csv")
    model_path: Path = Path("models/phase1_logreg.joblib")


def paths_from_settings(settings: Settings) -> Phase1Paths:
    return Phase1Paths(
        raw_csv=Path(settings.phase1_raw_csv_path),
        clean_csv=Path(settings.phase1_clean_csv_path),
        features_csv=Path(settings.phase1_features_csv_path),
        model_path=Path(settings.phase1_model_path),
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ingest_synthetic(start_date: date, end_date: date, paths: Phase1Paths) -> pd.DataFrame:
    provider = SyntheticRacingDataProvider(SyntheticProviderConfig())
    data = provider.get_dataset(start_date, end_date)
    _ensure_parent(paths.raw_csv)
    data.to_csv(paths.raw_csv, index=False)
    return data


def preprocess_raw(paths: Phase1Paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.raw_csv)
    cleaned = preprocess_dataset(frame)
    _ensure_parent(paths.clean_csv)
    cleaned.to_csv(paths.clean_csv, index=False)
    return cleaned


def build_features(paths: Phase1Paths) -> FeatureBuildResult:
    frame = pd.read_csv(paths.clean_csv)
    built = build_leakage_safe_features(frame)
    assert_no_leakage_columns(built.feature_columns)
    _ensure_parent(paths.features_csv)
    built.frame.to_csv(paths.features_csv, index=False)
    return built


def train_phase1_model(
    paths: Phase1Paths,
    holdout_days: int = 30,
    calibration_days: int = 21,
    calibration_method: str = "isotonic",
) -> EnsembleArtifact:
    frame = pd.read_csv(paths.features_csv)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    feature_columns = [
        c
        for c in frame.columns
        if c
        not in {
            "race_id",
            "date",
            "race_datetime",
            "horse_id",
            "is_winner",
            "odds",
            "raw_probability",
            "calibrated_probability",
            "rank",
            "bet_decision",
            "edge",
            "ev",
        }
    ]

    split_date = max(frame["date"]) - timedelta(days=holdout_days)
    calibration_split = split_date - timedelta(days=calibration_days)

    train_df = frame[frame["date"] < calibration_split].copy()
    calibration_df = frame[(frame["date"] >= calibration_split) & (frame["date"] < split_date)].copy()

    if train_df.empty:
        train_df = frame[frame["date"] < split_date].copy()
    if calibration_df.empty:
        calibration_df = frame[frame["date"] >= split_date].copy()

    artifact = train_phase3_ensemble(train_df, calibration_df, feature_columns)

    raw_cal = predict_ensemble_raw_probability(artifact, calibration_df)
    calibrator = fit_calibrator(
        y_true=calibration_df["is_winner"].to_numpy(),
        raw_prob=raw_cal,
        method=calibration_method,
    )

    _ensure_parent(paths.model_path)
    save_phase3_artifact(artifact, to_payload(calibrator), str(paths.model_path))
    return artifact


def predict_for_date(
    paths: Phase1Paths,
    target_date: date,
    enable_ev: bool = True,
    ev_probability_threshold: float = 0.18,
    ev_min_edge: float = 0.03,
    ev_min_value: float = 0.02,
) -> pd.DataFrame:
    frame = pd.read_csv(paths.features_csv)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    day_df = frame[frame["date"] == target_date].copy()
    artifact, calibrator_payload = load_phase3_artifact(str(paths.model_path))
    calibrator = from_payload(calibrator_payload)

    out = day_df.copy()
    out["raw_probability"] = predict_ensemble_raw_probability(artifact, out)
    out["calibrated_probability"] = apply_calibrator(calibrator, out["raw_probability"].to_numpy())

    # Race-level normalization.
    denom = out.groupby("race_id")["calibrated_probability"].transform("sum").replace(0.0, 1.0)
    out["calibrated_probability"] = out["calibrated_probability"] / denom
    out["rank"] = out.groupby("race_id")["calibrated_probability"].rank(ascending=False, method="dense")

    # Confidence from model disagreement proxy.
    p_log = artifact.logistic.predict_proba(out[artifact.feature_columns].to_numpy())[:, 1]
    p_rf = artifact.random_forest.predict_proba(out[artifact.feature_columns].to_numpy())[:, 1]
    out["uncertainty"] = np.abs(p_log - p_rf)
    out["confidence"] = (1.0 - out["uncertainty"]).clip(lower=0.0, upper=1.0)

    if enable_ev:
        out = add_market_ev_columns(out, ev_probability_threshold, ev_min_edge, ev_min_value)
    return out


def run_phase1_backtest(
    paths: Phase1Paths,
    min_train_days: int = 90,
    calibration_days: int = 21,
    calibration_method: str = "isotonic",
    ev_probability_threshold: float = 0.18,
    ev_min_edge: float = 0.03,
    ev_min_value: float = 0.02,
) -> dict[str, object]:
    frame = pd.read_csv(paths.clean_csv)
    result: WalkForwardResult = walk_forward_backtest(
        frame,
        min_train_days=min_train_days,
        calibration_days=calibration_days,
        calibration_method=calibration_method,
        ev_probability_threshold=ev_probability_threshold,
        ev_min_edge=ev_min_edge,
        ev_min_value=ev_min_value,
    )
    payload = asdict(result)
    payload["fold_max_train_date"] = [d.isoformat() for d in result.fold_max_train_date]
    payload["fold_test_date"] = [d.isoformat() for d in result.fold_test_date]
    return payload


def optimize_for_date(paths: Phase1Paths, target_date: date, settings: Settings, budget: float | None = None) -> dict[str, object]:
    pred = predict_for_date(
        paths,
        target_date,
        enable_ev=True,
        ev_probability_threshold=settings.ev_probability_threshold,
        ev_min_edge=settings.ev_min_edge,
        ev_min_value=settings.ev_min_value,
    )
    return optimize_ticket_portfolio(
        pred,
        budget=(budget if budget is not None else settings.phase4_default_budget),
        unit_cost=settings.phase4_unit_cost,
        beam_width=settings.phase4_beam_width,
        top_per_leg=settings.phase4_top_per_leg,
        simulation_count=settings.phase4_simulation_count,
        base_payout=settings.phase4_base_payout,
        no_bet_ev_threshold=settings.phase4_no_bet_ev_threshold,
        min_confidence=settings.phase4_min_confidence,
        concentration_penalty=settings.phase4_concentration_penalty,
    )
