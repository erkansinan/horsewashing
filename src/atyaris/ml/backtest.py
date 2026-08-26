from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from atyaris.ml.calibration import apply_calibrator, fit_calibrator
from atyaris.ml.ev import add_market_ev_columns
from atyaris.ml.features import FeatureBuildResult, build_leakage_safe_features
from atyaris.ml.modeling import evaluate_predictions, predict_ensemble_raw_probability, train_phase3_ensemble


@dataclass
class WalkForwardResult:
    evaluated_days: int
    evaluated_races: int
    top1_hit_rate: float
    top2_hit_rate: float
    top3_hit_rate: float
    log_loss: float
    brier: float
    ece: float
    roi: float
    total_bets: int
    fold_max_train_date: list[date]
    fold_test_date: list[date]
    fold_log_loss: list[float]
    fold_brier: list[float]
    fold_roi: list[float]


def _topk_hit_rate(pred: pd.DataFrame, k: int) -> float:
    race_hits = []
    for _, grp in pred.groupby("race_id"):
        topk = grp.nsmallest(k, "rank")
        race_hits.append(int(topk["is_winner"].max() == 1))
    return float(sum(race_hits) / len(race_hits)) if race_hits else 0.0


def _ece(frame: pd.DataFrame, bins: int = 10) -> float:
    if frame.empty:
        return 0.0
    p = frame["calibrated_probability"].to_numpy()
    y = frame["is_winner"].to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(frame)
    acc = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        win = float(np.mean(y[mask]))
        acc += abs(conf - win) * (np.sum(mask) / total)
    return float(acc)


def walk_forward_backtest(
    dataset: pd.DataFrame,
    min_train_days: int = 90,
    test_step_days: int = 1,
    calibration_days: int = 21,
    calibration_method: str = "isotonic",
    ev_probability_threshold: float = 0.18,
    ev_min_edge: float = 0.03,
    ev_min_value: float = 0.02,
) -> WalkForwardResult:
    features: FeatureBuildResult = build_leakage_safe_features(dataset)
    feat = features.frame
    unique_days = sorted(feat["date"].unique())

    fold_predictions: list[pd.DataFrame] = []
    train_dates: list[date] = []
    test_dates: list[date] = []
    fold_log_losses: list[float] = []
    fold_briers: list[float] = []
    fold_rois: list[float] = []

    i = min_train_days
    while i < len(unique_days):
        test_day = unique_days[i]
        train_days = unique_days[:i]

        train_df = feat[feat["date"].isin(train_days)].copy()
        test_df = feat[feat["date"] == test_day].copy()
        if train_df.empty or test_df.empty:
            i += test_step_days
            continue

        calibration_start_ix = max(0, i - calibration_days)
        calibration_start_day = unique_days[calibration_start_ix]
        calibration_df = train_df[train_df["date"] >= calibration_start_day].copy()
        core_train_df = train_df[train_df["date"] < calibration_start_day].copy()
        if core_train_df.empty:
            core_train_df = train_df.copy()
        if calibration_df.empty:
            calibration_df = train_df.tail(min(500, len(train_df))).copy()

        artifact = train_phase3_ensemble(core_train_df, calibration_df, features.feature_columns)
        raw_cal = predict_ensemble_raw_probability(artifact, calibration_df)
        calibrator = fit_calibrator(
            y_true=calibration_df["is_winner"].to_numpy(),
            raw_prob=raw_cal,
            method=calibration_method,
        )

        pred = test_df.copy()
        pred["raw_probability"] = predict_ensemble_raw_probability(artifact, pred)
        pred["calibrated_probability"] = apply_calibrator(calibrator, pred["raw_probability"].to_numpy())
        denom = pred.groupby("race_id")["calibrated_probability"].transform("sum").replace(0.0, 1.0)
        pred["calibrated_probability"] = pred["calibrated_probability"] / denom
        pred["rank"] = pred.groupby("race_id")["calibrated_probability"].rank(ascending=False, method="dense")
        pred = add_market_ev_columns(pred, ev_probability_threshold, ev_min_edge, ev_min_value)
        fold_predictions.append(pred)

        fold_metric = evaluate_predictions(pred)
        fold_log_losses.append(float(fold_metric["log_loss"]))
        fold_briers.append(float(fold_metric["brier"]))

        fold_bet_df = pred[pred["bet_decision"] == "BET"].copy()
        if not fold_bet_df.empty:
            fold_bet_df["stake_return"] = np.where(
                fold_bet_df["is_winner"] == 1,
                fold_bet_df["odds"] - 1.0,
                -1.0,
            )
            fold_rois.append(float(fold_bet_df["stake_return"].mean()))
        else:
            fold_rois.append(0.0)

        train_dates.append(max(train_days))
        test_dates.append(test_day)
        i += test_step_days

    if not fold_predictions:
        return WalkForwardResult(
            evaluated_days=0,
            evaluated_races=0,
            top1_hit_rate=0.0,
            top2_hit_rate=0.0,
            top3_hit_rate=0.0,
            log_loss=0.0,
            brier=0.0,
            ece=0.0,
            roi=0.0,
            total_bets=0,
            fold_max_train_date=[],
            fold_test_date=[],
            fold_log_loss=[],
            fold_brier=[],
            fold_roi=[],
        )

    all_pred = pd.concat(fold_predictions, axis=0, ignore_index=True)
    metrics = evaluate_predictions(all_pred)

    bet_df = all_pred[all_pred["bet_decision"] == "BET"].copy()
    if not bet_df.empty:
        bet_df["stake_return"] = np.where(
            bet_df["is_winner"] == 1,
            bet_df["odds"] - 1.0,
            -1.0,
        )
        roi = float(bet_df["stake_return"].mean())
        total_bets = int(len(bet_df))
    else:
        roi = 0.0
        total_bets = 0

    return WalkForwardResult(
        evaluated_days=len(test_dates),
        evaluated_races=int(all_pred["race_id"].nunique()),
        top1_hit_rate=_topk_hit_rate(all_pred, 1),
        top2_hit_rate=_topk_hit_rate(all_pred, 2),
        top3_hit_rate=_topk_hit_rate(all_pred, 3),
        log_loss=metrics["log_loss"],
        brier=metrics["brier"],
        ece=_ece(all_pred, bins=10),
        roi=roi,
        total_bets=total_bets,
        fold_max_train_date=train_dates,
        fold_test_date=test_dates,
        fold_log_loss=fold_log_losses,
        fold_brier=fold_briers,
        fold_roi=fold_rois,
    )
