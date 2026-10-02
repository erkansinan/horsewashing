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
from atyaris.ml.harville import add_harville_columns, fit_harville_gammas
from atyaris.ml.harville import mark_highest_odds_placer_predictions
from atyaris.ml.market_blend import (
    extract_market_reference_probability,
    extract_odds_implied_probability,
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
    place_log_loss: float
    place_brier: float
    place_ece: float
    harville_place_calibration: dict[str, object]
    place_calibration_curve: list[dict[str, float]]
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
    model_vs_odds_ranking: dict[str, object]
    market_probability_vs_odds_ranking: dict[str, object]
    odds_probability_vs_odds_ranking: dict[str, object]
    test_model_diverges_from_market_ranking: dict[str, object]
    production_readiness: dict[str, object]
    alpha_comparison: list[dict[str, object]]
    paired_alpha_comparison: dict[str, float | int | list[float] | str]
    bonferroni_paired_alpha_comparison: dict[str, float | int | list[float] | str]
    nested_alpha_validation: dict[str, object]
    feature_market_correlations: dict[str, float]
    race_top4_hits: dict[str, float]
    race_longshot_top4_hits: dict[str, float]
    race_log_loss: dict[str, float]
    highest_odds_placer_metrics: dict[str, float | int]
    race_highest_odds_placer_exact_match: dict[str, float]
    race_market_highest_odds_placer_exact_match: dict[str, float]
    race_highest_odds_placer_dates: dict[str, str]


def _ranking_order_metrics(
    frame: pd.DataFrame,
    probability_col: str,
    top_k: int = 4,
) -> dict[str, object]:
    exact_matches = 0
    same_set_different_order = 0
    different_sets = 0
    eligible_races = 0
    excluded_races = 0
    if not {"race_id", "horse_id", "odds", probability_col}.issubset(frame.columns):
        return {
            "eligible_races": 0,
            "excluded_races_missing_or_invalid_odds_or_probability": 0,
            "exact_order_matches": 0,
            "exact_order_match_rate": 0.0,
            "exact_order_match_percent": 0.0,
            "same_top4_set_different_order_races": 0,
            "same_top4_set_different_order_rate": 0.0,
            "same_top4_set_different_order_percent": 0.0,
            "different_top4_set_races": 0,
            "different_top4_set_rate": 0.0,
            "different_top4_set_percent": 0.0,
            "top_k": top_k,
        }

    for _, group in frame.groupby("race_id", sort=False):
        odds = pd.to_numeric(group["odds"], errors="coerce").to_numpy(dtype=float)
        probability = pd.to_numeric(group[probability_col], errors="coerce").to_numpy(dtype=float)
        if (
            len(group) < top_k
            or not np.isfinite(odds).all()
            or not np.all(odds > 1.0)
            or not np.isfinite(probability).all()
        ):
            excluded_races += 1
            continue
        horse_ids = group["horse_id"].astype(str).to_numpy()
        model_order = horse_ids[np.lexsort((horse_ids, -probability))[:top_k]].tolist()
        odds_order = horse_ids[np.lexsort((horse_ids, odds))[:top_k]].tolist()
        eligible_races += 1
        if model_order == odds_order:
            exact_matches += 1
        elif set(model_order) == set(odds_order):
            same_set_different_order += 1
        else:
            different_sets += 1

    denominator = max(eligible_races, 1)
    return {
        "eligible_races": eligible_races,
        "excluded_races_missing_or_invalid_odds_or_probability": excluded_races,
        "exact_order_matches": exact_matches,
        "exact_order_match_rate": exact_matches / denominator if eligible_races else 0.0,
        "exact_order_match_percent": 100.0 * exact_matches / denominator if eligible_races else 0.0,
        "same_top4_set_different_order_races": same_set_different_order,
        "same_top4_set_different_order_rate": same_set_different_order / denominator if eligible_races else 0.0,
        "same_top4_set_different_order_percent": (
            100.0 * same_set_different_order / denominator if eligible_races else 0.0
        ),
        "different_top4_set_races": different_sets,
        "different_top4_set_rate": different_sets / denominator if eligible_races else 0.0,
        "different_top4_set_percent": 100.0 * different_sets / denominator if eligible_races else 0.0,
        "top_k": top_k,
    }


def _market_ranking_divergence_test(
    ranking_metrics: dict[str, object],
    threshold: float = 0.90,
) -> dict[str, object]:
    exact_rate = float(ranking_metrics.get("exact_order_match_rate", 0.0))
    eligible_races = int(ranking_metrics.get("eligible_races", 0))
    failed = exact_rate >= threshold
    return {
        "test_name": "test_model_diverges_from_market_ranking",
        "threshold": threshold,
        "exact_order_match_rate": exact_rate,
        "status": "not_run" if eligible_races == 0 else "fail" if failed else "pass",
        "warning": (
            "insufficient_walk_forward_races"
            if eligible_races == 0
            else "model_equals_odds_sort" if failed else None
        ),
        "production_allowed": False,
        "production_block_reason": "model_equals_odds_sort" if failed else "holdout_evidence_required",
    }


def _topk_hit_rate(pred: pd.DataFrame, k: int) -> float:
    race_hits = []
    for _, grp in pred.groupby("race_id"):
        topk = grp.nsmallest(k, "rank")
        race_hits.append(int(topk["is_winner"].max() == 1))
    return float(sum(race_hits) / len(race_hits)) if race_hits else 0.0


def _ece(
    frame: pd.DataFrame,
    bins: int = 10,
    probability_col: str = "calibrated_probability",
    label_col: str = "is_winner",
) -> float:
    if frame.empty or probability_col not in frame or label_col not in frame:
        return 0.0
    valid = frame[probability_col].notna() & frame[label_col].notna()
    p = pd.to_numeric(frame.loc[valid, probability_col], errors="coerce").to_numpy()
    y = pd.to_numeric(frame.loc[valid, label_col], errors="coerce").to_numpy()
    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    if p.size == 0:
        return 0.0
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


def _calibration_curve(
    frame: pd.DataFrame,
    bins: int = 10,
    probability_col: str = "calibrated_probability",
    label_col: str = "is_winner",
) -> list[dict[str, float]]:
    if frame.empty or probability_col not in frame or label_col not in frame:
        return []
    valid = frame[probability_col].notna() & frame[label_col].notna()
    p = pd.to_numeric(frame.loc[valid, probability_col], errors="coerce").to_numpy()
    y = pd.to_numeric(frame.loc[valid, label_col], errors="coerce").to_numpy()
    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    if p.size == 0:
        return []
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


def _place_calibration_metrics(frame: pd.DataFrame) -> dict[str, object]:
    valid = frame["is_placed"].notna() & frame["place_probability"].notna()
    probabilities = np.clip(
        pd.to_numeric(frame.loc[valid, "place_probability"], errors="coerce").to_numpy(dtype=float),
        1e-8,
        1.0 - 1e-8,
    )
    labels = pd.to_numeric(frame.loc[valid, "is_placed"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(probabilities) & np.isfinite(labels)
    probabilities, labels = probabilities[finite], labels[finite]
    if probabilities.size == 0:
        return {
            "place_rows": 0,
            "place_log_loss": 0.0,
            "place_brier": 0.0,
            "place_ece": 0.0,
            "place_calibration_curve": [],
        }
    place_frame = pd.DataFrame(
        {"place_probability": probabilities, "is_placed": labels.astype(int)}
    )
    log_loss = float(-np.mean(
        labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)
    ))
    return {
        "place_rows": int(probabilities.size),
        "place_log_loss": log_loss,
        "place_brier": float(np.mean((probabilities - labels) ** 2)),
        "place_ece": _ece(
            place_frame,
            probability_col="place_probability",
            label_col="is_placed",
        ),
        "place_calibration_curve": _calibration_curve(
            place_frame,
            probability_col="place_probability",
            label_col="is_placed",
        ),
    }


def _harville_place_calibration_comparison(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or "finish_position" not in frame:
        return {"fold_gammas": [], "positions": {}}

    result: dict[str, object] = {
        "fold_gammas": [],
        "positions": {},
    }
    if {"place2_gamma", "place3_gamma"}.issubset(frame.columns):
        fold_gamma_frame = frame.groupby("fold_test_date", sort=False).first()
        result["fold_gammas"] = [
            {
                "test_date": str(test_date),
                "place2_gamma": float(row["place2_gamma"]),
                "place3_gamma": float(row["place3_gamma"]),
            }
            for test_date, row in fold_gamma_frame.iterrows()
        ]

    targets = {
        "second": (
            "harville_place2_raw",
            "place2_probability",
            pd.to_numeric(frame["finish_position"], errors="coerce").eq(2),
        ),
        "third": (
            "harville_place3_raw",
            "place3_probability",
            pd.to_numeric(frame["finish_position"], errors="coerce").eq(3),
        ),
        "top3": (
            "harville_top3_raw",
            "harville_top3_gamma",
            pd.to_numeric(frame["finish_position"], errors="coerce").le(3),
        ),
    }

    def metrics(rows: pd.DataFrame, raw_col: str, adjusted_col: str, label: pd.Series) -> dict[str, object]:
        values: dict[str, object] = {"rows": 0, "raw": {}, "gamma_adjusted": {}}
        if rows.empty or raw_col not in rows or adjusted_col not in rows:
            return values
        labels = label.reindex(rows.index).astype(float).to_numpy()
        output: dict[str, dict[str, float]] = {}
        for name, column in (("raw", raw_col), ("gamma_adjusted", adjusted_col)):
            probability = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(labels) & np.isfinite(probability)
            p = np.clip(probability[valid], 1e-8, 1.0 - 1e-8)
            y = labels[valid]
            output[name] = {
                "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) if p.size else 0.0,
                "brier": float(np.mean((p - y) ** 2)) if p.size else 0.0,
            }
        values["rows"] = int(np.isfinite(labels).sum())
        values.update(output)
        return values

    positions: dict[str, object] = {}
    for name, (raw_col, adjusted_col, target) in targets.items():
        slices: dict[str, object] = {}
        for slice_name, rows in (
            ("all", frame),
            ("longshot", frame[frame.get("odds_slice", pd.Series(index=frame.index, dtype=object)) == "longshot"]),
        ):
            slices[slice_name] = metrics(rows, raw_col, adjusted_col, target)
        positions[name] = slices
    result["positions"] = positions
    gammas = frame[["place2_gamma", "place3_gamma"]].drop_duplicates() if {"place2_gamma", "place3_gamma"}.issubset(frame.columns) else pd.DataFrame()
    result["gamma_summary"] = {
        "place2_mean": float(gammas["place2_gamma"].mean()) if not gammas.empty else 1.0,
        "place3_mean": float(gammas["place3_gamma"].mean()) if not gammas.empty else 1.0,
        "place2_min": float(gammas["place2_gamma"].min()) if not gammas.empty else 1.0,
        "place2_max": float(gammas["place2_gamma"].max()) if not gammas.empty else 1.0,
        "place3_min": float(gammas["place3_gamma"].min()) if not gammas.empty else 1.0,
        "place3_max": float(gammas["place3_gamma"].max()) if not gammas.empty else 1.0,
    }
    return result


def _highest_odds_placer_metrics(
    frame: pd.DataFrame,
) -> tuple[dict[str, float | int], dict[str, float], dict[str, float]]:
    exact_matches: dict[str, float] = {}
    market_exact_matches: dict[str, float] = {}
    model_placed: list[float] = []
    market_placed: list[float] = []
    first_stage_capture: list[float] = []
    conditional_second_stage: list[float] = []
    market_exact: list[float] = []
    target_longshot_exact: list[float] = []
    target_longshot_count = 0

    for race_id, group in frame.groupby("race_id", sort=False):
        targets = group["target_highest_odds_placer"].dropna()
        if targets.empty:
            continue
        target = str(targets.iloc[0])
        target_row = group[group["horse_id"].astype(str) == target]
        if target_row.empty:
            continue

        model_pick = group[group["predicted_highest_odds_placer"]].head(1)
        if model_pick.empty:
            exact = 0.0
            model_placed.append(0.0)
        else:
            exact = float(str(model_pick.iloc[0]["horse_id"]) == target)
            placed = pd.to_numeric(model_pick["is_placed"], errors="coerce").iloc[0]
            model_placed.append(float(placed == 1.0))
        exact_matches[str(race_id)] = exact
        model_top3_ids = set(group.loc[group["predicted_top3"], "horse_id"].astype(str))
        captured = target in model_top3_ids
        first_stage_capture.append(float(captured))
        if captured:
            conditional_second_stage.append(exact)

        market_pick = group[group["market_predicted_highest_odds_placer"]].head(1)
        if not market_pick.empty:
            market_is_exact = float(str(market_pick.iloc[0]["horse_id"]) == target)
            market_exact.append(market_is_exact)
            market_exact_matches[str(race_id)] = market_is_exact
            market_pick_placed = pd.to_numeric(market_pick["is_placed"], errors="coerce").iloc[0]
            market_placed.append(float(market_pick_placed == 1.0))
        else:
            market_exact.append(0.0)
            market_exact_matches[str(race_id)] = 0.0
            market_placed.append(0.0)

        if "odds_slice" in target_row.columns and str(target_row.iloc[0]["odds_slice"]) == "longshot":
            target_longshot_count += 1
            target_longshot_exact.append(exact)

    race_count = len(exact_matches)
    exact_rate = float(np.mean(list(exact_matches.values()))) if race_count else 0.0
    market_exact_rate = float(np.mean(market_exact)) if market_exact else 0.0
    conditional_rate = (
        float(np.mean(conditional_second_stage)) if conditional_second_stage else 0.0
    )
    return (
        {
            "eligible_races": race_count,
            "model_exact_match_rate": exact_rate,
            "model_marked_placer_rate": float(np.mean(model_placed)) if model_placed else 0.0,
            "model_first_stage_target_recall": (
                float(np.mean(first_stage_capture)) if first_stage_capture else 0.0
            ),
            "model_second_stage_exact_match_given_target_captured": conditional_rate,
            "model_second_stage_error_rate_given_target_captured": 1.0 - conditional_rate,
            "market_baseline_exact_match_rate": market_exact_rate,
            "market_baseline_marked_placer_rate": (
                float(np.mean(market_placed)) if market_placed else 0.0
            ),
            "model_vs_market_exact_match_delta": exact_rate - market_exact_rate,
            "target_longshot_races": target_longshot_count,
            "target_longshot_exact_match_rate": (
                float(np.mean(target_longshot_exact)) if target_longshot_exact else 0.0
            ),
        },
        exact_matches,
        market_exact_matches,
    )


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


ALPHA_VALUES = (0.0, 0.2, 0.4, 0.6, 1.0)


def _race_top4_hit_map(frame: pd.DataFrame, probability_col: str) -> dict[str, float]:
    hits: dict[str, float] = {}
    for race_id, group in frame.groupby("race_id", sort=False):
        odds = pd.to_numeric(group["odds"], errors="coerce")
        probability = pd.to_numeric(group[probability_col], errors="coerce").to_numpy(dtype=float)
        winners = pd.to_numeric(group["is_winner"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if (
            len(group) < 4
            or not odds.gt(1.0).all()
            or not odds.notna().all()
            or not np.isfinite(probability).all()
            or not np.isfinite(probability).all()
        ):
            continue
        horse_ids = group["horse_id"].astype(str).to_numpy()
        top_indices = np.lexsort((horse_ids, -probability))[:4]
        hits[str(race_id)] = float(winners[top_indices].max() > 0.0)
    return hits


def _topk_winner_hit(group: pd.DataFrame, probabilities: np.ndarray, top_k: int = 4) -> float:
    horse_ids = group["horse_id"].astype(str).to_numpy()
    winners = pd.to_numeric(group["is_winner"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    top_indices = np.lexsort((horse_ids, -probabilities))[:top_k]
    return float(winners[top_indices].max() > 0.0)


def _paired_top4_vs_odds(
    frame: pd.DataFrame,
    probability_col: str,
    *,
    confidence_alpha: float,
    minimum_dates: int = 30,
) -> dict[str, object]:
    model_hits = _race_top4_hit_map(frame, probability_col)
    odds_hits: dict[str, float] = {}
    race_dates: dict[str, str] = {}
    for race_id, group in frame.groupby("race_id", sort=False):
        odds = pd.to_numeric(group["odds"], errors="coerce")
        probability = pd.to_numeric(group[probability_col], errors="coerce")
        if (
            len(group) < 4
            or not odds.gt(1.0).all()
            or not odds.notna().all()
            or not probability.notna().all()
            or not np.isfinite(probability).all()
            or str(race_id) not in model_hits
        ):
            continue
        eligible = group.copy()
        eligible["_odds"] = pd.to_numeric(eligible["odds"], errors="coerce")
        eligible["_horse_id"] = eligible["horse_id"].astype(str)
        ordered = eligible.sort_values(["_odds", "_horse_id"], ascending=[True, True], kind="stable")
        odds_hits[str(race_id)] = float(pd.to_numeric(ordered.head(4)["is_winner"], errors="coerce").max() == 1)
        race_dates[str(race_id)] = pd.to_datetime(group["date"].iloc[0]).date().isoformat()

    common = sorted(set(model_hits) & set(odds_hits))
    differences = np.asarray([model_hits[race_id] - odds_hits[race_id] for race_id in common], dtype=float)
    clusters = np.asarray([race_dates[race_id] for race_id in common], dtype=str)
    ci_low, ci_high = _cluster_bootstrap_ci(differences, clusters, alpha=confidence_alpha)
    ranking = _ranking_order_metrics(frame, probability_col)
    cluster_count = int(np.unique(clusters).size)
    established = cluster_count >= minimum_dates and ci_low > 0.0
    return {
        "eligible_races": int(differences.size),
        "eligible_dates": cluster_count,
        "model_top4_winner_rate": float(np.mean([model_hits[race_id] for race_id in common])) if common else 0.0,
        "odds_sort_top4_winner_rate": float(np.mean([odds_hits[race_id] for race_id in common])) if common else 0.0,
        "paired_top4_winner_rate_delta": float(np.mean(differences)) if differences.size else 0.0,
        "paired_top4_winner_rate_bonferroni_ci": [ci_low, ci_high],
        "bonferroni_alpha": confidence_alpha,
        "minimum_validation_dates": minimum_dates,
        "ranking_agreement": ranking,
        "conclusion": "evidence_for" if established else "model_equals_odds_sort",
        "performance_evidence": "established" if established else "not_established",
    }


def _alpha_market_column(frame: pd.DataFrame) -> str:
    return "odds_market_probability" if "odds_market_probability" in frame.columns else "market_probability_used"


def _alpha_comparison(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    market_column = _alpha_market_column(frame)
    for alpha in ALPHA_VALUES:
        probabilities = alpha * frame["stage1_probability"] + (1.0 - alpha) * frame[market_column]
        work = frame.assign(alpha_probability=probabilities)
        ranking = _ranking_order_metrics(work, "alpha_probability")
        model_hits = _race_top4_hit_map(work, "alpha_probability")
        odds_hits: dict[str, float] = {}
        for race_id, group in work.groupby("race_id", sort=False):
            odds = pd.to_numeric(group["odds"], errors="coerce")
            probabilities_for_race = pd.to_numeric(group["alpha_probability"], errors="coerce")
            if (
                len(group) < 4
                or not odds.gt(1.0).all()
                or not odds.notna().all()
                or not probabilities_for_race.notna().all()
                or not np.isfinite(probabilities_for_race).all()
                or str(race_id) not in model_hits
            ):
                continue
            horse_ids = group["horse_id"].astype(str).to_numpy()
            winners = pd.to_numeric(group["is_winner"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            odds_top_indices = np.lexsort((horse_ids, odds.to_numpy(dtype=float)))[:4]
            odds_hits[str(race_id)] = float(winners[odds_top_indices].max() > 0.0)
        common_races = sorted(set(model_hits) & set(odds_hits))
        model_top4_rate = (
            float(np.mean([model_hits[race_id] for race_id in common_races]))
            if common_races
            else 0.0
        )
        odds_top4_rate = (
            float(np.mean([odds_hits[race_id] for race_id in common_races]))
            if common_races
            else 0.0
        )
        longshot_hits: list[float] = []
        upset_hits: list[float] = []
        longshot_races = 0
        upset_races = 0
        for _, group in work.groupby("race_id"):
            if str(group["race_id"].iloc[0]) not in common_races:
                continue
            winner = group[group["is_winner"] == 1]
            winner_is_longshot = not winner.empty and str(winner.iloc[0]["odds_slice"]) == "longshot"
            if winner_is_longshot:
                longshot_races += 1
                longshot_hits.append(
                    _topk_winner_hit(group, group["alpha_probability"].to_numpy(dtype=float))
                )
            odds = pd.to_numeric(group["odds"], errors="coerce").to_numpy(dtype=float)
            winners = pd.to_numeric(group["is_winner"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            favorite_index = int(np.nanargmin(odds)) if np.isfinite(odds).any() else -1
            is_upset = favorite_index < 0 or winners[favorite_index] != 1
            if is_upset:
                upset_races += 1
                upset_hits.append(
                    _topk_winner_hit(group, group["alpha_probability"].to_numpy(dtype=float))
                )
        probability_values = np.clip(probabilities.to_numpy(dtype=float), 1e-12, 1.0 - 1e-12)
        labels = frame["is_winner"].to_numpy(dtype=float)
        log_loss = float(-np.mean(labels * np.log(probability_values) + (1.0 - labels) * np.log(1.0 - probability_values)))
        rows.append({
            "alpha_stage1": alpha,
            "market_weight": 1.0 - alpha,
            "top4_hit_rate": model_top4_rate,
            "odds_sort_top4_hit_rate": odds_top4_rate,
            "top4_hit_delta_vs_odds_sort": model_top4_rate - odds_top4_rate,
            "eligible_races_vs_odds_sort": len(common_races),
            "longshot_races": longshot_races,
            "longshot_top4_hit_rate": float(np.mean(longshot_hits)) if longshot_hits else 0.0,
            "longshot_top4_bootstrap_ci": list(_bootstrap_ci(np.asarray(longshot_hits, dtype=float))),
            "upset_races": upset_races,
            "upset_top4_hit_rate": float(np.mean(upset_hits)) if upset_hits else 0.0,
            "log_loss": log_loss,
            "top4_vs_odds_sort_evaluation": "descriptive_only",
            "ranking_exact_order_match_rate": ranking["exact_order_match_rate"],
            "ranking_exact_order_match_percent": ranking["exact_order_match_percent"],
            "ranking_same_top4_set_different_order_rate": ranking[
                "same_top4_set_different_order_rate"
            ],
            "ranking_same_top4_set_different_order_percent": ranking[
                "same_top4_set_different_order_percent"
            ],
            "ranking_different_top4_set_rate": ranking["different_top4_set_rate"],
            "ranking_different_top4_set_percent": ranking["different_top4_set_percent"],
            "ranking_comparison": ranking,
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
            probabilities = alpha * group["stage1_probability"] + (1.0 - alpha) * group[
                _alpha_market_column(group)
            ]
            alpha_hits[alpha].append(
                _topk_winner_hit(group, probabilities.to_numpy(dtype=float))
            )

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
        key=lambda row: (-float(row["top4_hit_rate"]), float(row["log_loss"])),
    ) if selection_rows else {"alpha_stage1": 0.2}
    selected_alpha = float(selected["alpha_stage1"])
    validation_rows = _alpha_comparison(validation)
    selected_validation = next(
        (row for row in validation_rows if float(row["alpha_stage1"]) == selected_alpha),
        {},
    )
    market_column = _alpha_market_column(validation)
    selected_alpha_frame = validation.assign(
        _selected_alpha_probability=(
            selected_alpha * validation["stage1_probability"]
            + (1.0 - selected_alpha) * validation[market_column]
        )
    )
    adjusted_alpha = 0.05 / (len(ALPHA_VALUES) + 2)
    selected_alpha_holdout = _paired_top4_vs_odds(
        selected_alpha_frame,
        "_selected_alpha_probability",
        confidence_alpha=adjusted_alpha,
    )
    final_model_validation = _paired_top4_vs_odds(
        validation,
        "calibrated_probability",
        confidence_alpha=adjusted_alpha,
    )
    market_probability_validation = _paired_top4_vs_odds(
        validation,
        "market_probability_used",
        confidence_alpha=adjusted_alpha,
    )
    validation_test = _paired_alpha_longshot_comparison(
        validation,
        alpha_low=0.2,
        alpha_high=selected_alpha,
        confidence_alpha=0.05 / len(ALPHA_VALUES),
    )
    return {
        "selection_dates": [d.isoformat() for d in dates[:split]],
        "validation_dates": [d.isoformat() for d in dates[split:]],
        "selection_races": int(selection["race_id"].nunique()),
        "validation_races": int(validation["race_id"].nunique()),
        "selected_alpha": selected_alpha,
        "selection_results": selection_rows,
        "validation_alpha_results": validation_rows,
        "selected_alpha_validation_result": selected_validation,
        "selected_alpha_vs_odds_sort": selected_alpha_holdout,
        "final_model_vs_odds_sort": final_model_validation,
        "market_probability_vs_odds_sort": market_probability_validation,
        "bonferroni_comparison_family_size": len(ALPHA_VALUES) + 2,
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


def _cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1200,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if values.size == 0 or clusters.size != values.size:
        return (0.0, 0.0)
    unique_clusters = np.unique(clusters)
    if unique_clusters.size == 0:
        return (0.0, 0.0)
    grouped_values = {
        cluster: values[clusters == cluster]
        for cluster in unique_clusters
    }
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        sample = np.concatenate([grouped_values[cluster] for cluster in sampled_clusters])
        boots[index] = float(np.mean(sample))
    return (
        float(np.quantile(boots, alpha / 2.0)),
        float(np.quantile(boots, 1.0 - alpha / 2.0)),
    )


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
    stage1_coefficient_sign_constraints: dict[str, int] | None = None,
) -> WalkForwardResult:
    features: FeatureBuildResult = build_leakage_safe_features(dataset)
    feat = features.frame
    selected_feature_columns = feature_columns or TJK_SELECTED_STAGE1_FEATURE_COLUMNS
    if "weight_deviation" in selected_feature_columns:
        history_weight = pd.to_numeric(feat["history_avg_weight"], errors="coerce").fillna(0.0)
        current_weight = pd.to_numeric(feat["weight"], errors="coerce").fillna(0.0)
        feat["weight_deviation"] = np.where(
            history_weight > 0.0,
            current_weight - history_weight,
            0.0,
        )
    unique_days = sorted(feat["date"].unique())

    fold_predictions: list[pd.DataFrame] = []
    train_dates: list[date] = []
    test_dates: list[date] = []
    fold_log_losses: list[float] = []
    fold_briers: list[float] = []
    fold_rois: list[float] = []
    cached_artifact = None
    cached_calibrator = None
    cached_place_calibrator = None
    cached_place_gammas = {"place2_gamma": 1.0, "place3_gamma": 1.0}
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
                stage1_coefficient_sign_constraints=stage1_coefficient_sign_constraints,
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
            calibrated_calibration_win = recover_collapsed_calibration(
                apply_calibrator(cached_calibrator, calibration_form_probability),
                calibration_form_probability,
                group_ids=calibration_df["race_id"].to_numpy(),
            )
            calibrated_calibration_win = smooth_race_probabilities(
                calibrated_calibration_win,
                calibration_df.groupby("race_id")["race_id"].transform("size").to_numpy(),
            )
            calibration_normalizer = calibration_df.assign(
                _calibrated_win=calibrated_calibration_win
            ).groupby("race_id")["_calibrated_win"].transform("sum").to_numpy()
            calibrated_calibration_win /= np.maximum(calibration_normalizer, 1e-12)
            place_calibration_frame = calibration_df.copy()
            place_calibration_frame["calibrated_probability"] = calibrated_calibration_win
            cached_place_gammas = fit_harville_gammas(place_calibration_frame)
            place_calibration_frame = add_harville_columns(
                place_calibration_frame,
                place2_gamma=float(cached_place_gammas["place2_gamma"]),
                place3_gamma=float(cached_place_gammas["place3_gamma"]),
            )
            placed_labels = pd.to_numeric(calibration_df["is_placed"], errors="coerce")
            place_label_mask = placed_labels.notna().to_numpy()
            cached_place_calibrator = fit_calibrator(
                y_true=placed_labels.to_numpy(dtype=float)[place_label_mask].astype(int),
                raw_prob=place_calibration_frame["place_probability"].to_numpy(dtype=float)[
                    place_label_mask
                ],
                method=calibration_method,
            )
            last_fit_day = test_day

        artifact = cached_artifact
        calibrator = cached_calibrator
        place_calibrator = cached_place_calibrator
        if artifact is None or calibrator is None or place_calibrator is None:
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
        pred["odds_market_probability"] = extract_odds_implied_probability(pred)
        pred["market_probability_used"] = extract_market_reference_probability(pred)
        pred["market_odds_rank"] = pred.groupby("race_id")["odds"].rank(method="first", ascending=True)
        field_sizes = pred.groupby("race_id")["race_id"].transform("size")
        pred["odds_slice"] = np.select(
            [pred["market_odds_rank"] <= 2, pred["market_odds_rank"] >= np.maximum(3, np.ceil(field_sizes * 0.75))],
            ["favorite", "longshot"],
            default="middle",
        )
        pred["rank"] = pred.groupby("race_id")["calibrated_probability"].rank(ascending=False, method="dense")
        pred = add_harville_columns(
            pred,
            win_col="calibrated_probability",
            place2_gamma=float(cached_place_gammas["place2_gamma"]),
            place3_gamma=float(cached_place_gammas["place3_gamma"]),
        )
        pred["harville_top3_raw"] = (
            pred["calibrated_probability"]
            + pred["harville_place2_raw"]
            + pred["harville_place3_raw"]
        ).clip(upper=1.0)
        pred["harville_top3_gamma"] = (
            pred["calibrated_probability"]
            + pred["place2_probability"]
            + pred["place3_probability"]
        ).clip(upper=1.0)
        pred["place2_gamma"] = float(cached_place_gammas["place2_gamma"])
        pred["place3_gamma"] = float(cached_place_gammas["place3_gamma"])
        pred["fold_test_date"] = str(test_day)
        raw_place_probability = pred["place_probability"].to_numpy(dtype=float)
        pred["place_probability"] = recover_collapsed_calibration(
            apply_calibrator(place_calibrator, raw_place_probability),
            raw_place_probability,
            group_ids=pred["race_id"].to_numpy(),
        )
        pred = mark_highest_odds_placer_predictions(pred)
        market_predictions = mark_highest_odds_placer_predictions(
            pred,
            place_probability_col="market_probability_used",
        )
        pred["market_predicted_top3"] = market_predictions["predicted_top3"]
        pred["market_predicted_highest_odds_placer"] = market_predictions[
            "predicted_highest_odds_placer"
        ]
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
            model_vs_odds_ranking={},
            market_probability_vs_odds_ranking={},
            odds_probability_vs_odds_ranking={},
            test_model_diverges_from_market_ranking={
                "test_name": "test_model_diverges_from_market_ranking",
                "threshold": 0.90,
                "exact_order_match_rate": 0.0,
                "status": "not_run",
                "warning": "insufficient_walk_forward_races",
                "production_allowed": False,
            },
            production_readiness={
                "status": "not_run",
                "recommendation": "DO_NOT_PROMOTE",
                "production_allowed": False,
            },
            alpha_comparison=[],
            paired_alpha_comparison={},
            bonferroni_paired_alpha_comparison={},
            nested_alpha_validation={},
            feature_market_correlations={},
            race_top4_hits={},
            race_longshot_top4_hits={},
            race_log_loss={},
            place_log_loss=0.0,
            place_brier=0.0,
            place_ece=0.0,
            harville_place_calibration={"fold_gammas": [], "positions": {}},
            place_calibration_curve=[],
            highest_odds_placer_metrics={
                "eligible_races": 0,
                "model_exact_match_rate": 0.0,
                "model_marked_placer_rate": 0.0,
                "model_first_stage_target_recall": 0.0,
                "model_second_stage_exact_match_given_target_captured": 0.0,
                "model_second_stage_error_rate_given_target_captured": 0.0,
                "market_baseline_exact_match_rate": 0.0,
                "market_baseline_marked_placer_rate": 0.0,
                "model_vs_market_exact_match_delta": 0.0,
                "target_longshot_races": 0,
                "target_longshot_exact_match_rate": 0.0,
            },
            race_highest_odds_placer_exact_match={},
            race_market_highest_odds_placer_exact_match={},
            race_highest_odds_placer_dates={},
        )

    all_pred = pd.concat(fold_predictions, axis=0, ignore_index=True)
    metrics = evaluate_predictions(all_pred)
    place_metrics = _place_calibration_metrics(all_pred)
    placer_metrics, race_placer_exact, race_market_placer_exact = _highest_odds_placer_metrics(all_pred)
    race_placer_dates = {
        str(race_id): pd.to_datetime(group["date"].iloc[0]).date().isoformat()
        for race_id, group in all_pred.groupby("race_id", sort=False)
        if str(race_id) in race_placer_exact
    }

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
    model_vs_odds_ranking = _ranking_order_metrics(all_pred, "calibrated_probability")
    market_probability_vs_odds_ranking = _ranking_order_metrics(
        all_pred,
        "market_probability_used",
    )
    odds_probability_vs_odds_ranking = _ranking_order_metrics(
        all_pred,
        "odds_market_probability",
    )
    divergence_test = _market_ranking_divergence_test(model_vs_odds_ranking)
    nested_alpha_validation = _nested_alpha_validation(all_pred)
    nested_model_comparison = nested_alpha_validation["final_model_vs_odds_sort"]
    model_beats_odds = nested_model_comparison["conclusion"] == "evidence_for"
    divergence_test["production_allowed"] = divergence_test["status"] == "pass"
    divergence_test["production_block_reason"] = (
        None if divergence_test["status"] == "pass" else "model_equals_odds_sort"
    )
    if divergence_test["status"] == "fail":
        production_readiness = {
            "status": "model_equals_odds_sort",
            "recommendation": "DO_NOT_PROMOTE",
            "production_allowed": False,
            "reason": "The model ranking matches odds ranking in at least 90% of races.",
        }
    elif not model_beats_odds:
        production_readiness = {
            "status": "model_equals_odds_sort",
            "recommendation": "DO_NOT_PROMOTE",
            "production_allowed": False,
            "reason": "The nested holdout did not establish improvement over odds-sorted Top-4.",
        }
    else:
        production_readiness = {
            "status": "review_required",
            "recommendation": "REVIEW_BEFORE_PROMOTION",
            "production_allowed": False,
            "reason": "The odds baseline was exceeded; remaining production gates still apply.",
        }

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
        place_log_loss=place_metrics["place_log_loss"],
        place_brier=place_metrics["place_brier"],
        place_ece=place_metrics["place_ece"],
        harville_place_calibration=_harville_place_calibration_comparison(all_pred),
        place_calibration_curve=place_metrics["place_calibration_curve"],
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
            "stage1_odds_implied_probability_correlation": _safe_correlation(
                all_pred["stage1_probability"], all_pred["odds_market_probability"]
            ),
            "final_odds_implied_probability_correlation": _safe_correlation(
                all_pred["calibrated_probability"], all_pred["odds_market_probability"]
            ),
        },
        model_vs_odds_ranking=model_vs_odds_ranking,
        market_probability_vs_odds_ranking=market_probability_vs_odds_ranking,
        odds_probability_vs_odds_ranking=odds_probability_vs_odds_ranking,
        test_model_diverges_from_market_ranking=divergence_test,
        production_readiness=production_readiness,
        alpha_comparison=_alpha_comparison(all_pred),
        paired_alpha_comparison=_paired_alpha_longshot_comparison(all_pred),
        bonferroni_paired_alpha_comparison=_paired_alpha_longshot_comparison(
            all_pred,
            confidence_alpha=0.05 / 4.0,
        ),
        nested_alpha_validation=nested_alpha_validation,
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
        highest_odds_placer_metrics=placer_metrics,
        race_highest_odds_placer_exact_match=race_placer_exact,
        race_market_highest_odds_placer_exact_match=race_market_placer_exact,
        race_highest_odds_placer_dates=race_placer_dates,
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


def _paired_highest_odds_placer_comparison(
    baseline: WalkForwardResult,
    candidate: WalkForwardResult,
    confidence_alpha: float = 0.05 / 2.0,
) -> dict[str, object]:
    common = sorted(
        set(baseline.race_market_highest_odds_placer_exact_match)
        & set(candidate.race_highest_odds_placer_exact_match)
        & set(candidate.race_highest_odds_placer_dates)
    )
    differences = np.asarray(
        [
            candidate.race_highest_odds_placer_exact_match[race_id]
            - baseline.race_market_highest_odds_placer_exact_match[race_id]
            for race_id in common
        ],
        dtype=float,
    )
    date_clusters = np.asarray(
        [candidate.race_highest_odds_placer_dates[race_id] for race_id in common],
        dtype=str,
    )
    ci_low, ci_high = _cluster_bootstrap_ci(
        differences,
        date_clusters,
        alpha=confidence_alpha,
    )
    delta = float(np.mean(differences)) if differences.size else 0.0
    minimum_validation_dates = 30
    return {
        "eligible_validation_races": int(differences.size),
        "eligible_validation_dates": int(np.unique(date_clusters).size),
        "market_baseline_exact_match_rate": baseline.highest_odds_placer_metrics.get(
            "market_baseline_exact_match_rate", 0.0
        ),
        "model_exact_match_rate": candidate.highest_odds_placer_metrics.get(
            "model_exact_match_rate", 0.0
        ),
        "paired_exact_match_delta": delta,
        "paired_exact_match_bonferroni_ci": [ci_low, ci_high],
        "bonferroni_alpha": confidence_alpha,
        "minimum_validation_dates": minimum_validation_dates,
        "conclusion": (
            "evidence_for"
            if np.unique(date_clusters).size >= minimum_validation_dates and ci_low > 0.0
            else "not_established"
        ),
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
    loo_features = list(enhanced_columns)
    loo = {
        name: walk_forward_backtest(
            dataset,
            feature_columns=[column for column in loo_features if column != name],
            **common,
        )
        for name in loo_features
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

    selected_placer_model = (
        "enhanced"
        if float(selection_new.highest_odds_placer_metrics.get("model_exact_match_rate", 0.0))
        > float(selection_base.highest_odds_placer_metrics.get("model_exact_match_rate", 0.0))
        else "baseline"
    )
    selected_placer_validation = (
        validation_new if selected_placer_model == "enhanced" else validation_base
    )
    nested_placer = _paired_highest_odds_placer_comparison(
        validation_base,
        selected_placer_validation,
        confidence_alpha=0.05 / 2.0,
    )
    nested_placer.update(
        {
            "selection_dates": [value.isoformat() for value in sorted(selection_dates)],
            "validation_dates": [value.isoformat() for value in sorted(validation_dates)],
            "selected_on_selection": selected_placer_model,
            "selection_baseline_exact_match_rate": selection_base.highest_odds_placer_metrics.get(
                "model_exact_match_rate", 0.0
            ),
            "selection_enhanced_exact_match_rate": selection_new.highest_odds_placer_metrics.get(
                "model_exact_match_rate", 0.0
            ),
            "validation_candidate_metrics": selected_placer_validation.highest_odds_placer_metrics,
            "validation_baseline_metrics": validation_base.highest_odds_placer_metrics,
            "selection_secondary_placer_rate": selected_placer_validation.highest_odds_placer_metrics.get(
                "model_marked_placer_rate", 0.0
            ),
        }
    )

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
        "nested_highest_odds_placer": nested_placer,
        "leave_one_out": {
            name: _paired_feature_comparison(
                result,
                enhanced,
                confidence_alpha=0.05 / max(1, len(loo)),
            )
            for name, result in loo.items()
        },
    }
