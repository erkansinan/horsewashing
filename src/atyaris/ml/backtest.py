from __future__ import annotations

# Validation protocol: time-ordered walk-forward with ROI and calibration checks,
# plus bootstrap confidence interval for model-vs-market edge.

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from atyaris.ml.calibration import (
    apply_calibrator,
    fit_calibrator,
    recover_collapsed_calibration,
    smooth_race_probabilities,
)
from atyaris.ml.ev_kelly import add_ev_kelly_columns
from atyaris.ml.features import FeatureBuildResult, build_leakage_safe_features
from atyaris.ml.harville import add_harville_columns
from atyaris.ml.market_blend import (
    extract_market_reference_probability,
    fit_two_stage_benter,
    predict_two_stage_probability,
)
from atyaris.ml.modeling import evaluate_predictions


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
    correct_bet_ratio: float
    market_roi: float
    edge_roi_diff: float
    edge_roi_ci_low: float
    edge_roi_ci_high: float
    calibration_curve: list[dict[str, float]]
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


def _calibration_curve(frame: pd.DataFrame, bins: int = 10) -> list[dict[str, float]]:
    if frame.empty:
        return []
    p = frame["calibrated_probability"].to_numpy()
    y = frame["is_winner"].to_numpy()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "predicted": float(np.mean(p[mask])),
                "observed": float(np.mean(y[mask])),
                "count": float(np.sum(mask)),
            }
        )
    return rows


def _market_favorite_roi(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    rows = []
    for _, grp in frame.groupby("race_id"):
        market_ix = grp["market_probability_used"].astype(float).idxmax()
        pick = grp.loc[market_ix]
        odds = float(pick.get("odds", 0.0) or 0.0)
        if odds <= 1.0:
            rows.append(0.0)
            continue
        rows.append(float(odds - 1.0) if int(pick.get("is_winner", 0)) == 1 else -1.0)
    return float(np.mean(rows)) if rows else 0.0


def _bootstrap_ci(values: np.ndarray, n_boot: int = 1200, alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = float(np.mean(sample))
    low = float(np.quantile(boots, alpha / 2.0))
    high = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return (low, high)


def walk_forward_backtest(
    dataset: pd.DataFrame,
    min_train_days: int = 90,
    test_step_days: int = 1,
    retrain_interval_days: int = 14,
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
    cached_artifact = None
    cached_calibrator = None
    last_fit_day = None
    retrain_interval_days = max(int(retrain_interval_days), 1)

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

        should_retrain = (
            cached_artifact is None
            or last_fit_day is None
            or (pd.Timestamp(test_day) - pd.Timestamp(last_fit_day)).days
            >= retrain_interval_days
        )
        if should_retrain:
            cached_artifact = fit_two_stage_benter(
                core_train_df,
                calibration_df,
                features.feature_columns,
            )
            calibration_form_probability = predict_two_stage_probability(
                cached_artifact,
                calibration_df,
            )
            cached_calibrator = fit_calibrator(
                y_true=calibration_df["is_winner"].to_numpy(),
                raw_prob=calibration_form_probability,
                method=calibration_method,
            )
            last_fit_day = test_day

        artifact = cached_artifact
        calibrator = cached_calibrator
        if artifact is None or calibrator is None:
            raise RuntimeError("Walk-forward modeli yeniden egitilemedi.")

        pred = test_df.copy()
        pred["raw_probability"] = predict_two_stage_probability(artifact, pred)
        raw_probability = pred["raw_probability"].to_numpy()
        calibrated_probability = recover_collapsed_calibration(
            apply_calibrator(calibrator, raw_probability),
            raw_probability,
        )
        pred["calibrated_probability"] = smooth_race_probabilities(
            calibrated_probability,
            pred.groupby("race_id")["race_id"].transform("size").to_numpy(),
        )
        denom = pred.groupby("race_id")["calibrated_probability"].transform("sum").replace(0.0, 1.0)
        pred["calibrated_probability"] = pred["calibrated_probability"] / denom
        pred["market_probability_used"] = extract_market_reference_probability(pred)
        pred["rank"] = pred.groupby("race_id")["calibrated_probability"].rank(ascending=False, method="dense")
        pred = add_harville_columns(pred, win_col="calibrated_probability")
        pred = add_ev_kelly_columns(
            pred,
            min_probability=ev_probability_threshold,
            min_edge=ev_min_edge,
            min_ev=ev_min_value,
            fractional_kelly=0.35,
            max_kelly_fraction=0.25,
        )
        pred["confidence"] = (0.5 + np.abs(pred["edge"]).clip(upper=0.5)).astype(float)
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
            correct_bet_ratio=0.0,
            market_roi=0.0,
            edge_roi_diff=0.0,
            edge_roi_ci_low=0.0,
            edge_roi_ci_high=0.0,
            calibration_curve=[],
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

    model_race_returns: list[float] = []
    market_race_returns: list[float] = []
    for _, grp in all_pred.groupby("race_id"):
        model_ix = grp["calibrated_probability"].astype(float).idxmax()
        model_pick = grp.loc[model_ix]
        m_odds = float(model_pick.get("odds", 0.0) or 0.0)
        model_race_returns.append(float(m_odds - 1.0) if int(model_pick.get("is_winner", 0)) == 1 and m_odds > 1.0 else -1.0)

        market_ix = grp["market_probability_used"].astype(float).idxmax()
        market_pick = grp.loc[market_ix]
        mk_odds = float(market_pick.get("odds", 0.0) or 0.0)
        market_race_returns.append(float(mk_odds - 1.0) if int(market_pick.get("is_winner", 0)) == 1 and mk_odds > 1.0 else -1.0)

    market_roi = float(np.mean(market_race_returns)) if market_race_returns else 0.0
    diff = np.array(model_race_returns, dtype=float) - np.array(market_race_returns, dtype=float)
    ci_low, ci_high = _bootstrap_ci(diff)

    correct_bet_ratio = float(bet_df["is_winner"].mean()) if not bet_df.empty else 0.0

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
        correct_bet_ratio=correct_bet_ratio,
        market_roi=market_roi,
        edge_roi_diff=float(np.mean(diff)) if diff.size else 0.0,
        edge_roi_ci_low=ci_low,
        edge_roi_ci_high=ci_high,
        calibration_curve=_calibration_curve(all_pred, bins=10),
        fold_max_train_date=train_dates,
        fold_test_date=test_dates,
        fold_log_loss=fold_log_losses,
        fold_brier=fold_briers,
        fold_roi=fold_rois,
    )
