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
from atyaris.ml.features import (
    FeatureBuildResult,
    TJK_SELECTED_STAGE1_FEATURE_COLUMNS,
    build_leakage_safe_features,
)
from atyaris.ml.harville import add_harville_columns
from atyaris.ml.market_blend import (
    extract_market_reference_probability,
    fit_two_stage_benter,
    predict_form_probability,
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
    top4_hit_rate: float
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
    odds_slice_metrics: dict[str, dict[str, float | int]]
    upset_metrics: dict[str, float | int]
    market_dependence: dict[str, float]
    alpha_comparison: list[dict[str, float | int | list[float]]]
    paired_alpha_comparison: dict[str, float | int | list[float] | str]
    bonferroni_paired_alpha_comparison: dict[str, float | int | list[float] | str]
    nested_alpha_validation: dict[str, object]
    feature_market_correlations: dict[str, float]
    race_top4_hits: dict[str, float]
    race_longshot_top4_hits: dict[str, float]
    race_log_loss: dict[str, float]


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

def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    values = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2 or values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return 0.0
    return float(values.iloc[:, 0].corr(values.iloc[:, 1]))


def _feature_market_correlations(frame: pd.DataFrame) -> dict[str, float]:
    names = (
        "jockey_change_upgrade",
        "trainer_change_upgrade",
        "class_drop_flag",
        "workout_sudden_improvement",
        "rest_optimal_fit",
    )
    return {
        name: _safe_correlation(frame[name], frame["market_probability_used"])
        for name in names
        if name in frame.columns
    }

def _odds_slice_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    slice_names = ("favorite", "middle", "longshot")
    for name in slice_names:
        subset = frame[frame["odds_slice"] == name]
        races = subset["race_id"].unique()
        race_hits = []
        for race_id in races:
            group = frame[frame["race_id"] == race_id]
            winner = group[group["is_winner"] == 1]
            if not winner.empty and str(winner.iloc[0]["odds_slice"]) == name:
                race_hits.append(int(group.nsmallest(4, "rank")["is_winner"].max() == 1))
        rows[name] = {
            "races": int(len(race_hits)),
            "horses": int(len(subset)),
            "top4_hit_rate": float(np.mean(race_hits)) if race_hits else 0.0,
            "log_loss": float(-np.mean(
                subset["is_winner"] * np.log(np.clip(subset["calibrated_probability"], 1e-12, 1.0))
                + (1 - subset["is_winner"]) * np.log(np.clip(1 - subset["calibrated_probability"], 1e-12, 1.0))
            )) if not subset.empty else 0.0,
        }
    return rows

def _upset_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    upset_races = []
    for _, group in frame.groupby("race_id"):
        favorite = group.nsmallest(1, "odds")
        if favorite.empty or int(favorite.iloc[0]["is_winner"]) != 1:
            upset_races.append(group)
    upset = pd.concat(upset_races, ignore_index=True) if upset_races else frame.iloc[0:0]
    hits = [int(group.nsmallest(4, "rank")["is_winner"].max() == 1) for group in upset_races]
    return {
        "races": int(len(upset_races)),
        "top4_hit_rate": float(np.mean(hits)) if hits else 0.0,
        "winner_captured": int(sum(hits)),
        "log_loss": float(-np.mean(
            upset["is_winner"] * np.log(np.clip(upset["calibrated_probability"], 1e-12, 1.0))
            + (1 - upset["is_winner"]) * np.log(np.clip(1 - upset["calibrated_probability"], 1e-12, 1.0))
        )) if not upset.empty else 0.0,
    }


def _alpha_comparison(frame: pd.DataFrame) -> list[dict[str, float | int | list[float]]]:
    rows: list[dict[str, float | int | list[float]]] = []
    for alpha in (0.0, 0.2, 0.4, 0.6):
        probabilities = alpha * frame["stage1_probability"] + (1.0 - alpha) * frame["market_probability_used"]
        work = frame.assign(alpha_probability=probabilities)
        top4_hits: list[float] = []
        longshot_hits: list[float] = []
        upset_hits: list[float] = []
        market_top4_hits: list[float] = []
        longshot_races = 0
        upset_races = 0
        for _, group in work.groupby("race_id"):
            top4_hits.append(float(group.nlargest(4, "alpha_probability")["is_winner"].max()))
            market_top4_hits.append(float(group.nlargest(4, "market_probability_used")["is_winner"].max()))
            winner = group[group["is_winner"] == 1]
            winner_is_longshot = not winner.empty and str(winner.iloc[0]["odds_slice"]) == "longshot"
            if winner_is_longshot:
                longshot_races += 1
                longshot_hits.append(float(group.nlargest(4, "alpha_probability")["is_winner"].max()))
            favorite = group.nsmallest(1, "odds")
            is_upset = favorite.empty or int(favorite.iloc[0]["is_winner"]) != 1
            if is_upset:
                upset_races += 1
                upset_hits.append(float(group.nlargest(4, "alpha_probability")["is_winner"].max()))
        probability_values = np.clip(probabilities.to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        labels = frame["is_winner"].to_numpy(dtype=float)
        log_loss = float(-np.mean(labels * np.log(probability_values) + (1.0 - labels) * np.log(1.0 - probability_values)))
        ci_low, ci_high = _bootstrap_ci(
            np.asarray(top4_hits, dtype=float) - np.asarray(market_top4_hits, dtype=float)
        )
        rows.append({
            "alpha_stage1": alpha,
            "market_weight": 1.0 - alpha,
            "top4_hit_rate": float(np.mean(top4_hits)) if top4_hits else 0.0,
            "longshot_races": longshot_races,
            "longshot_top4_hit_rate": float(np.mean(longshot_hits)) if longshot_hits else 0.0,
            "longshot_top4_bootstrap_ci": list(_bootstrap_ci(np.asarray(longshot_hits, dtype=float))),
            "upset_races": upset_races,
            "upset_top4_hit_rate": float(np.mean(upset_hits)) if upset_hits else 0.0,
            "log_loss": log_loss,
            "top4_vs_market_bootstrap_ci": [ci_low, ci_high],
        })
    return rows


def _paired_alpha_longshot_comparison(
    frame: pd.DataFrame,
    alpha_low: float = 0.2,
    alpha_high: float = 0.6,
    confidence_alpha: float = 0.05,
) -> dict[str, float | int | list[float] | str]:
    alpha_hits: dict[float, list[float]] = {alpha_low: [], alpha_high: []}
    for _, group in frame.groupby("race_id"):
        winner = group[group["is_winner"] == 1]
        if winner.empty or str(winner.iloc[0]["odds_slice"]) != "longshot":
            continue
        for alpha in alpha_hits:
            probabilities = alpha * group["stage1_probability"] + (1.0 - alpha) * group["market_probability_used"]
            alpha_hits[alpha].append(float(group.loc[probabilities.nlargest(4).index, "is_winner"].max()))

    alpha02 = np.asarray(alpha_hits[alpha_low], dtype=float)
    alpha06 = np.asarray(alpha_hits[alpha_high], dtype=float)
    differences = alpha06 - alpha02
    ci_low, ci_high = _bootstrap_ci(differences, alpha=confidence_alpha)
    difference = float(np.mean(differences)) if differences.size else 0.0
    return {
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
        "races": int(differences.size),
        "alpha_low_longshot_top4_hit_rate": float(np.mean(alpha02)) if alpha02.size else 0.0,
        "alpha_high_longshot_top4_hit_rate": float(np.mean(alpha06)) if alpha06.size else 0.0,
        "paired_difference_high_minus_low": difference,
        "paired_difference_bootstrap_ci": [ci_low, ci_high],
        "confidence_alpha": confidence_alpha,
        "conclusion": "evidence_for_higher_alpha" if ci_low > 0.0 else "not_established",
    }


def _nested_alpha_validation(frame: pd.DataFrame) -> dict[str, object]:
    dates = sorted(pd.to_datetime(frame["date"]).dt.date.unique())
    split = max(1, int(len(dates) * 0.7))
    selection_dates = set(dates[:split])
    validation_dates = set(dates[split:])
    selection = frame[pd.to_datetime(frame["date"]).dt.date.isin(selection_dates)]
    validation = frame[pd.to_datetime(frame["date"]).dt.date.isin(validation_dates)]
    selection_rows = _alpha_comparison(selection)
    selected = min(
        selection_rows,
        key=lambda row: (-float(row["longshot_top4_hit_rate"]), float(row["log_loss"])),
    ) if selection_rows else {"alpha_stage1": 0.2}
    selected_alpha = float(selected["alpha_stage1"])
    validation_test = _paired_alpha_longshot_comparison(validation, alpha_low=0.2, alpha_high=selected_alpha)
    return {
        "selection_dates": [d.isoformat() for d in dates[:split]],
        "validation_dates": [d.isoformat() for d in dates[split:]],
        "selection_races": int(selection["race_id"].nunique()),
        "validation_races": int(validation["race_id"].nunique()),
        "selected_alpha": selected_alpha,
        "selection_results": selection_rows,
        "validation_test_vs_alpha_02": validation_test,
    }


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
    feature_columns: list[str] | None = None,
    evaluation_dates: set[date] | None = None,
) -> WalkForwardResult:
    features: FeatureBuildResult = build_leakage_safe_features(dataset)
    feat = features.frame
    selected_feature_columns = feature_columns or TJK_SELECTED_STAGE1_FEATURE_COLUMNS
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
        if evaluation_dates is not None and test_day not in evaluation_dates:
            i += test_step_days
            continue
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
                selected_feature_columns,
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
        pred["stage1_probability"] = predict_form_probability(artifact, pred)
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
        pred["market_odds_rank"] = pred.groupby("race_id")["odds"].rank(method="first", ascending=True)
        field_sizes = pred.groupby("race_id")["race_id"].transform("size")
        pred["odds_slice"] = np.select(
            [pred["market_odds_rank"] <= 2, pred["market_odds_rank"] >= np.maximum(3, np.ceil(field_sizes * 0.75))],
            ["favorite", "longshot"],
            default="middle",
        )
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
            top4_hit_rate=0.0,
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
            odds_slice_metrics={},
            upset_metrics={},
            market_dependence={},
            alpha_comparison=[],
            paired_alpha_comparison={},
            bonferroni_paired_alpha_comparison={},
            nested_alpha_validation={},
            feature_market_correlations={},
            race_top4_hits={},
            race_longshot_top4_hits={},
            race_log_loss={},
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
        top4_hit_rate=_topk_hit_rate(all_pred, 4),
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
        odds_slice_metrics=_odds_slice_metrics(all_pred),
        upset_metrics=_upset_metrics(all_pred),
        market_dependence={
            "stage1_market_probability_correlation": _safe_correlation(all_pred["stage1_probability"], all_pred["market_probability_used"]),
            "final_market_probability_correlation": _safe_correlation(all_pred["calibrated_probability"], all_pred["market_probability_used"]),
        },
        alpha_comparison=_alpha_comparison(all_pred),
        paired_alpha_comparison=_paired_alpha_longshot_comparison(all_pred),
        bonferroni_paired_alpha_comparison=_paired_alpha_longshot_comparison(
            all_pred,
            confidence_alpha=0.05 / 4.0,
        ),
        nested_alpha_validation=_nested_alpha_validation(all_pred),
        feature_market_correlations=_feature_market_correlations(all_pred),
        race_top4_hits={
            str(race_id): float(group.nsmallest(4, "rank")["is_winner"].max() == 1)
            for race_id, group in all_pred.groupby("race_id")
        },
        race_longshot_top4_hits={
            str(race_id): float(group.nsmallest(4, "rank")["is_winner"].max() == 1)
            for race_id, group in all_pred.groupby("race_id")
            if not group[group["is_winner"] == 1].empty
            and str(group.loc[group["is_winner"] == 1, "odds_slice"].iloc[0]) == "longshot"
        },
        race_log_loss={
            str(race_id): float(-np.mean(
                group["is_winner"] * np.log(np.clip(group["calibrated_probability"], 1e-12, 1.0))
                + (1 - group["is_winner"]) * np.log(np.clip(1 - group["calibrated_probability"], 1e-12, 1.0))
            ))
            for race_id, group in all_pred.groupby("race_id")
        },
    )


NEW_LONGSHOT_FEATURES = (
    "jockey_change_upgrade",
    "trainer_change_upgrade",
    "class_drop_flag",
    "workout_sudden_improvement",
    "rest_optimal_fit",
)


def _paired_feature_comparison(
    baseline: WalkForwardResult,
    candidate: WalkForwardResult,
    confidence_alpha: float = 0.05,
) -> dict[str, object]:
    common = sorted(set(baseline.race_longshot_top4_hits) & set(candidate.race_longshot_top4_hits))
    top4_delta = np.asarray(
        [candidate.race_longshot_top4_hits[race_id] - baseline.race_longshot_top4_hits[race_id] for race_id in common],
        dtype=float,
    )
    loss_delta = np.asarray(
        [baseline.race_log_loss[race_id] - candidate.race_log_loss[race_id] for race_id in common],
        dtype=float,
    )
    top4_ci = _bootstrap_ci(top4_delta, alpha=confidence_alpha)
    loss_ci = _bootstrap_ci(loss_delta, alpha=confidence_alpha)
    established = bool(top4_ci[0] > 0.0 and loss_ci[0] > 0.0)
    return {
        "races": len(common),
        "baseline_top4": baseline.top4_hit_rate,
        "candidate_top4": candidate.top4_hit_rate,
        "baseline_longshot_top4": baseline.odds_slice_metrics.get("longshot", {}).get("top4_hit_rate", 0.0),
        "candidate_longshot_top4": candidate.odds_slice_metrics.get("longshot", {}).get("top4_hit_rate", 0.0),
        "baseline_log_loss": baseline.log_loss,
        "candidate_log_loss": candidate.log_loss,
        "paired_top4_delta": float(np.mean(top4_delta)) if top4_delta.size else 0.0,
        "paired_top4_ci": list(top4_ci),
        "paired_log_loss_improvement": float(np.mean(loss_delta)) if loss_delta.size else 0.0,
        "paired_log_loss_ci": list(loss_ci),
        "confidence_alpha": confidence_alpha,
        "conclusion": "evidence_for_feature_model" if established else "not_established",
    }


def compare_longshot_features(
    dataset: pd.DataFrame,
    *,
    min_train_days: int = 14,
    calibration_days: int = 7,
    calibration_method: str = "isotonic",
) -> dict[str, object]:
    """Compare the old/new feature sets with nested selection and LOO diagnostics."""
    baseline_columns = [column for column in TJK_SELECTED_STAGE1_FEATURE_COLUMNS if column not in NEW_LONGSHOT_FEATURES]
    enhanced_columns = list(TJK_SELECTED_STAGE1_FEATURE_COLUMNS)
    common = {
        "min_train_days": min_train_days,
        "calibration_days": calibration_days,
        "calibration_method": calibration_method,
    }
    baseline = walk_forward_backtest(dataset, feature_columns=baseline_columns, **common)
    enhanced = walk_forward_backtest(dataset, feature_columns=enhanced_columns, **common)
    loo = {
        name: walk_forward_backtest(
            dataset,
            feature_columns=[column for column in enhanced_columns if column != name],
            **common,
        )
        for name in NEW_LONGSHOT_FEATURES
    }

    dates = sorted(pd.to_datetime(dataset["date"]).dt.date.unique())
    split = max(1, int(len(dates) * 0.7))
    selection_dates = set(dates[:split])
    validation_dates = set(dates[split:])
    selection_base = walk_forward_backtest(
        dataset, feature_columns=baseline_columns, evaluation_dates=selection_dates, **common
    )
    selection_new = walk_forward_backtest(
        dataset, feature_columns=enhanced_columns, evaluation_dates=selection_dates, **common
    )
    selected = "enhanced" if selection_new.odds_slice_metrics.get("longshot", {}).get("top4_hit_rate", 0.0) > selection_base.odds_slice_metrics.get("longshot", {}).get("top4_hit_rate", 0.0) else "baseline"
    validation_base = walk_forward_backtest(
        dataset, feature_columns=baseline_columns, evaluation_dates=validation_dates, **common
    )
    validation_new = walk_forward_backtest(
        dataset, feature_columns=enhanced_columns, evaluation_dates=validation_dates, **common
    )
    nested = _paired_feature_comparison(validation_base, validation_new, confidence_alpha=0.05 / 2.0)
    nested["selection_dates"] = [value.isoformat() for value in sorted(selection_dates)]
    nested["validation_dates"] = [value.isoformat() for value in sorted(validation_dates)]
    nested["selected_on_selection"] = selected

    return {
        "conclusion": nested["conclusion"],
        "baseline_feature_columns": baseline_columns,
        "enhanced_feature_columns": enhanced_columns,
        "full_walk_forward": {
            "baseline": baseline,
            "enhanced": enhanced,
            "feature_market_correlations": enhanced.feature_market_correlations,
        },
        "nested_holdout_bonferroni": nested,
        "leave_one_out": {
            name: _paired_feature_comparison(result, enhanced, confidence_alpha=0.05 / 5.0)
            for name, result in loo.items()
        },
    }
