from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from atyaris.config import Settings
from atyaris.ml.backtest import WalkForwardResult, walk_forward_backtest
from atyaris.ml.calibration import apply_calibrator, fit_calibrator, from_payload, to_payload
from atyaris.ml.ev_kelly import add_ev_kelly_columns
from atyaris.ml.features import FeatureBuildResult, assert_no_leakage_columns, build_leakage_safe_features, preprocess_dataset
from atyaris.ml.harville import add_harville_columns
from atyaris.ml.market_blend import (
    BenterTwoStageArtifact,
    blend_form_market_probability,
    extract_market_reference_probability,
    fit_two_stage_benter,
    predict_form_probability,
)
from atyaris.ml.modeling import load_phase3_artifact, save_phase3_artifact
from atyaris.ml.optimizer import optimize_ticket_portfolio
from atyaris.ml.real_ingestion import ingest_real_tjk_data


@dataclass
class Phase1Paths:
    raw_csv: Path = Path("data/raw/tjk_real_races.csv")
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


def ingest_real_data(
    start_date: date,
    end_date: date,
    paths: Phase1Paths,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    data = ingest_real_tjk_data(start_date, end_date, paths, progress_callback=progress_callback)
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


def _benter_feature_columns(frame: pd.DataFrame) -> list[str]:
    drop_cols = {
        "race_id",
        "date",
        "race_datetime",
        "horse_id",
        "is_winner",
        "odds",
        "raw_probability",
        "calibrated_probability",
        "place2_probability",
        "place3_probability",
        "top3_probability",
        "rank",
        "bet_decision",
        "edge",
        "ev",
        "kelly_fraction",
        "confidence",
        # Market information is blended explicitly at 25% after the form model.
        "market_probability",
        "implied_probability",
        "market_probability_norm",
    }
    return [c for c in frame.columns if c not in drop_cols]


def train_phase1_model(
    paths: Phase1Paths,
    holdout_days: int = 30,
    calibration_days: int = 21,
    calibration_method: str = "isotonic",
) -> BenterTwoStageArtifact:
    frame = pd.read_csv(paths.features_csv)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    feature_columns = _benter_feature_columns(frame)

    split_date = max(frame["date"]) - timedelta(days=holdout_days)
    calibration_split = split_date - timedelta(days=calibration_days)

    train_df = frame[frame["date"] < calibration_split].copy()
    calibration_df = frame[(frame["date"] >= calibration_split) & (frame["date"] < split_date)].copy()

    if train_df.empty:
        train_df = frame[frame["date"] < split_date].copy()
    if calibration_df.empty:
        calibration_df = frame[frame["date"] >= split_date].copy()

    artifact = fit_two_stage_benter(
        train_df,
        calibration_df,
        feature_columns,
        stage1_penalty="l2",
        stage1_regularization=0.05,
        stage2_penalty="l2",
        stage2_regularization=0.02,
    )

    raw_cal = predict_form_probability(artifact, calibration_df)
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

    # No rows for the requested date: return an empty frame with expected output columns
    # instead of passing zero samples into model/scaler steps.
    if day_df.empty:
        out = day_df.copy()
        for col in [
            "raw_probability",
            "calibrated_probability",
            "place2_probability",
            "place3_probability",
            "top3_probability",
            "rank",
            "confidence",
        ]:
            if col not in out.columns:
                out[col] = pd.Series(dtype="float64")
        if enable_ev:
            for col in ["market_probability", "implied_probability", "edge", "ev", "kelly_fraction", "bet_decision"]:
                if col not in out.columns:
                    if col == "bet_decision":
                        out[col] = pd.Series(dtype="object")
                    else:
                        out[col] = pd.Series(dtype="float64")
        return out

    artifact, calibrator_payload = load_phase3_artifact(str(paths.model_path))
    calibrator = from_payload(calibrator_payload)

    out = day_df.copy()
    form_probability = apply_calibrator(
        calibrator,
        predict_form_probability(artifact, out),
    )
    market_probability = extract_market_reference_probability(out)
    form_weight = float(getattr(artifact, "form_weight", 0.75))
    market_weight = float(getattr(artifact, "market_weight", 0.25))
    total_weight = form_weight + market_weight
    if total_weight <= 0.0:
        form_weight, market_weight, total_weight = 0.75, 0.25, 1.0
    out["form_probability"] = form_probability
    out["raw_probability"] = blend_form_market_probability(
        artifact,
        out,
        form_probability,
    )
    out["calibrated_probability"] = out["raw_probability"]

    # Race-level normalization.
    denom = out.groupby("race_id")["calibrated_probability"].transform("sum").replace(0.0, 1.0)
    out["calibrated_probability"] = out["calibrated_probability"] / denom
    out["market_probability_used"] = market_probability
    out["rank"] = out.groupby("race_id")["calibrated_probability"].rank(ascending=False, method="dense")
    out = add_harville_columns(out, win_col="calibrated_probability")

    if enable_ev:
        out = add_ev_kelly_columns(
            out,
            min_probability=ev_probability_threshold,
            min_edge=ev_min_edge,
            min_ev=ev_min_value,
            fractional_kelly=0.35,
            max_kelly_fraction=0.25,
        )
    else:
        out["edge"] = out["calibrated_probability"] - out["market_probability_used"]
        out["ev"] = out["calibrated_probability"] * out["odds"] - 1.0
        out["kelly_fraction"] = 0.0
        out["bet_decision"] = "NO_BET"

    # Confidence proxy: larger model-market divergence indicates stronger model conviction.
    out["confidence"] = (0.5 + np.abs(out["edge"]).clip(upper=0.5)).astype(float)
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
