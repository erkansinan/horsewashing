from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


LEAKAGE_COLUMNS = {
    "finish_position",
    "is_winner",
    "is_placed",
    "target_highest_odds_placer",
    "latent_true_win_probability",
}

# These are numeric values available from TJK race declarations, horse history
# and workout pages. Text-only identifiers (horse, jockey, trainer, equipment,
# class and location names) remain metadata rather than arbitrary numeric codes.
TJK_STAGE1_FEATURE_COLUMNS = [
    "draw", "weight", "distance", "field_size", "age", "handicap_points",
    "career_starts", "career_wins", "career_places",
    "last_year_starts", "last_year_wins", "last_year_places",
    "jockey_horse_combo_starts", "jockey_horse_combo_wins",
    "trainer_win_rate", "trainer_top3_rate", "trainer_top5_rate",
    "trainer_experience_log", "trainer_stats_missing",
    "form_avg_3", "form_avg_5", "form_avg_10", "form_var_5", "last_run_perf",
    "trend_3_10", "days_since_last_race", "fatigue_score", "recovery_score",
    "short_rest_flag", "long_layoff_flag", "race_frequency_3", "race_frequency_5",
    "seasonal_race_load", "distance_fit", "surface_fit", "track_fit",
    "history_avg_finish_position", "history_avg_field_size", "history_avg_weight",
    "history_avg_odds", "history_avg_handicap_points", "history_avg_race_time_seconds",
    "history_avg_prize",
    "workout_count",
    "workout_avg_distance", "workout_avg_speed_index", "workout_best_speed_index",
    "days_since_last_workout",
    "history_missing", "career_summary_missing", "workout_missing",
    "age_missing", "handicap_missing", "odds_missing",
    "jockey_change_upgrade", "trainer_change_upgrade", "class_drop_flag",
    "workout_sudden_improvement", "rest_optimal_fit",
]

TJK_SELECTED_STAGE1_FEATURE_COLUMNS = [
    "handicap_points",
    "draw",
    "weight",
    "class_drop_flag",
    "workout_sudden_improvement",
    "rest_optimal_fit",
]

FIELD_SIZE_DEPENDENT_FEATURE_COLUMNS = {
    "field_size",
    "form_avg_3",
    "form_avg_5",
    "form_avg_10",
    "form_var_5",
    "last_run_perf",
    "trend_3_10",
    "fatigue_score",
    "recovery_score",
    "distance_fit",
    "surface_fit",
    "track_fit",
    "history_avg_field_size",
    "history_avg_finish_position",
    "jockey_change_upgrade",
    "trainer_change_upgrade",
}

MISSINGNESS_INDICATOR_COLUMNS = {
    "odds_missing",
    "trainer_stats_missing",
    "history_missing",
    "workout_missing",
    "career_summary_missing",
    "age_missing",
    "handicap_missing",
}

# Backward-compatible public name; these are stage-1, non-market features.
TJK_FEATURE_COLUMNS = TJK_STAGE1_FEATURE_COLUMNS

STAGE2_MARKET_COLUMNS = ["odds", "market_probability_norm", "implied_probability"]


@dataclass
class FeatureBuildResult:
    frame: pd.DataFrame
    feature_columns: list[str]


def _derive_placer_labels(frame: pd.DataFrame) -> pd.DataFrame:
    finish = pd.to_numeric(
        frame.get("finish_position", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    odds = pd.to_numeric(frame.get("odds", pd.Series(np.nan, index=frame.index)), errors="coerce")
    labels = pd.DataFrame(index=frame.index)
    labels["is_placed"] = np.where(finish.notna(), finish.le(3).astype(float), np.nan)
    labels["target_highest_odds_placer"] = None

    if "race_id" not in frame.columns or "horse_id" not in frame.columns:
        return labels
    eligible = finish.le(3) & finish.notna() & odds.gt(1.0) & odds.notna()
    for race_id, group in frame.loc[eligible].groupby("race_id", sort=False):
        race_odds = odds.loc[group.index]
        target_index = race_odds.idxmax()
        horse_id = frame.loc[target_index, "horse_id"]
        if pd.notna(horse_id):
            race_indices = frame.index[frame["race_id"] == race_id]
            labels.loc[race_indices, "target_highest_odds_placer"] = str(horse_id)
    return labels


def preprocess_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["race_datetime"] = pd.to_datetime(out["race_datetime"])
    out = out.sort_values(["race_datetime", "race_id", "draw"], ascending=[True, True, True]).reset_index(drop=True)
    out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out["market_probability"] = pd.to_numeric(out["market_probability"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["distance"] = pd.to_numeric(out["distance"], errors="coerce")
    return out


def _field_size_adjusted_score(finish_position: float, field_size: float) -> float:
    # 1.0 is best (winner), 0.0 is worst in field.
    if field_size <= 1:
        return 0.5
    return float(1.0 - ((finish_position - 1.0) / (field_size - 1.0)))


def _rolling_mean(values: Iterable[float], n: int, default: float) -> float:
    vals = list(values)
    if not vals:
        return default
    return float(np.mean(vals[-n:]))


def _rolling_var(values: Iterable[float], n: int, default: float) -> float:
    vals = list(values)
    if len(vals) < 2:
        return default
    return float(np.var(vals[-n:]))


def _smoothed_mean(values: list[float], prior: float = 0.45, prior_strength: float = 3.0) -> float:
    if not values:
        return prior
    return float((sum(values) + prior * prior_strength) / (len(values) + prior_strength))


def _class_level(value: object) -> float | None:
    text = str(value or "").upper()
    if not text or text == "NAN":
        return None
    for token in text.replace("/", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    order = ("G1", "G2", "G3", "A", "B", "C", "ŞARTLI", "HANDİKAP", "MAIDEN")
    for index, label in enumerate(order):
        if label in text:
            return float(len(order) - index)
    return None


def _change_upgrade(history: list[dict[str, object]], current: object, key: str) -> float:
    current_name = str(current or "").strip()
    if not current_name or current_name.lower() == "nan" or not history:
        return 0.0
    previous_name = str(history[-1].get(key) or "").strip()
    if not previous_name or previous_name == current_name:
        return 0.0
    current_perf = [float(item["perf"]) for item in history if str(item.get(key) or "") == current_name]
    previous_perf = [float(item["perf"]) for item in history if str(item.get(key) or "") == previous_name]
    if not current_perf or not previous_perf:
        return 0.0
    return float(np.clip(np.mean(current_perf) - np.mean(previous_perf), -1.0, 1.0))


def build_leakage_safe_features(
    frame: pd.DataFrame,
    as_of_date: date | None = None,
    recompute_precomputed_columns: set[str] | None = None,
) -> FeatureBuildResult:
    """Builds features using only events strictly before each row race_datetime.

    This function never uses finish_result columns from the current race row while
    constructing its own features.
    """
    df = preprocess_dataset(frame)
    recompute_precomputed_columns = recompute_precomputed_columns or set()
    if as_of_date is not None:
        df = df[df["date"] <= as_of_date].copy()
    placer_labels = _derive_placer_labels(df)

    if "workout_avg_speed_index" not in df.columns:
        avg_time = pd.to_numeric(
            df.get("workout_avg_time_seconds", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        avg_distance = pd.to_numeric(
            df.get("workout_avg_distance", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        df["workout_avg_speed_index"] = np.divide(
            avg_distance,
            avg_time,
            out=np.zeros(len(df), dtype=float),
            where=avg_time.to_numpy(dtype=float) > 0.0,
        )
    if "workout_best_speed_index" not in df.columns:
        best_time = pd.to_numeric(
            df.get("workout_best_time_seconds", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        avg_distance = pd.to_numeric(
            df.get("workout_avg_distance", pd.Series(0.0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        df["workout_best_speed_index"] = np.divide(
            avg_distance,
            best_time,
            out=np.zeros(len(df), dtype=float),
            where=best_time.to_numpy(dtype=float) > 0.0,
        )

    # Real TJK ingestion computes point-in-time features using the resolved
    # source horse ID. Preserve those values for both labelled and prediction
    # rows; rebuilding by race-local horse_id can lose the horse's history.
    precomputed_columns = {
        "days_since_last_race",
        "form_avg_3",
        "form_avg_5",
        "form_avg_10",
        "track_fit",
        "surface_fit",
        "distance_fit",
    }
    if (
        "finish_position" not in df.columns
        and precomputed_columns.issubset(df.columns)
        and not recompute_precomputed_columns
    ):
        feat_df = df.copy()
        grp_sum = feat_df.groupby("race_id")["market_probability"].transform("sum").replace(0.0, 1.0)
        feat_df["market_probability_norm"] = feat_df["market_probability"] / grp_sum
        race_mean_form = feat_df.groupby("race_id")["form_avg_5"].transform("mean")
        race_sum_form = feat_df.groupby("race_id")["form_avg_5"].transform("sum")
        race_count = feat_df.groupby("race_id")["horse_id"].transform("count").replace(0, 1)
        feat_df["field_strength_index"] = race_mean_form
        feat_df["opponent_strength_mean"] = (
            (race_sum_form - feat_df["form_avg_5"]) / (race_count - 1).clip(lower=1)
        ).fillna(race_mean_form)
        feat_df["expected_early_pace"] = feat_df.groupby("race_id")["pace_hint"].transform("mean")
        feat_df["pace_pressure"] = feat_df.groupby("race_id")["style_front_prob"].transform("mean") + (
            0.7 * feat_df.groupby("race_id")["style_presser_prob"].transform("mean")
        )
        feat_df["pace_pressure"] = feat_df["pace_pressure"].clip(lower=0.0, upper=1.0)
        feat_df["pace_suitability"] = (
            feat_df["style_closer_prob"] * feat_df["pace_pressure"]
            + feat_df["style_front_prob"] * (1.0 - feat_df["pace_pressure"])
        )
        feature_columns = TJK_STAGE1_FEATURE_COLUMNS.copy()
        for column in feature_columns:
            if column not in feat_df:
                feat_df[column] = 0.0
        feat_df[feature_columns] = feat_df[feature_columns].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        feat_df["is_placed"] = placer_labels["is_placed"].to_numpy()
        feat_df["target_highest_odds_placer"] = placer_labels[
            "target_highest_odds_placer"
        ].to_numpy()
        return FeatureBuildResult(frame=feat_df, feature_columns=feature_columns)

    histories: dict[str, list[dict[str, object]]] = defaultdict(list)
    feature_rows: list[dict[str, float | int | str | date | pd.Timestamp]] = []

    for row in df.itertuples(index=False):
        horse_id = str(row.horse_id)
        h = histories[horse_id]

        perf_hist = [float(x["perf"]) for x in h]
        race_days = [float(x["day_ordinal"]) for x in h]
        pace_hist = [float(x.get("early_pace", 0.0)) for x in h]
        dist_hist = [float(x.get("distance", 0.0)) for x in h]
        load_120d = [x for x in h if (float(row.race_datetime.toordinal()) - float(x["day_ordinal"])) <= 120.0]

        if race_days:
            days_since_last = float(row.race_datetime.toordinal() - race_days[-1])
        else:
            days_since_last = 30.0

        form_avg_3 = _rolling_mean(perf_hist, 3, default=0.45)
        form_avg_5 = _rolling_mean(perf_hist, 5, default=0.45)
        form_avg_10 = _rolling_mean(perf_hist, 10, default=0.45)
        form_var_5 = _rolling_var(perf_hist, 5, default=0.03)
        last_run_perf = perf_hist[-1] if perf_hist else 0.45
        trend_3_10 = form_avg_3 - form_avg_10

        # Phase 2: fatigue/recovery signals
        recent_3_days = days_since_last if len(race_days) < 3 else float(row.race_datetime.toordinal() - race_days[-3])
        recent_5_days = days_since_last if len(race_days) < 5 else float(row.race_datetime.toordinal() - race_days[-5])
        race_frequency_3 = 3.0 / max(recent_3_days, 1.0)
        race_frequency_5 = 5.0 / max(recent_5_days, 1.0)
        seasonal_race_load = float(len(load_120d)) / 120.0
        prev_distance = dist_hist[-1] if dist_hist else float(row.distance)
        distance_shock = abs(float(row.distance) - prev_distance) / max(float(row.distance), 1.0)
        intensity_proxy = 1.0 - last_run_perf
        short_rest_flag = 1.0 if days_since_last < 8 else 0.0
        long_layoff_flag = 1.0 if days_since_last > 75 else 0.0
        jockey_change_upgrade = _change_upgrade(h, getattr(row, "jockey_name", ""), "jockey_name")
        trainer_change_upgrade = _change_upgrade(h, getattr(row, "trainer_name", ""), "trainer_name")
        current_class = _class_level(getattr(row, "race_class", ""))
        previous_classes = [_class_level(item.get("race_class")) for item in h]
        previous_class = next((value for value in reversed(previous_classes) if value is not None), None)
        class_drop_flag = float(max(previous_class - current_class, 0.0)) if current_class is not None and previous_class is not None else 0.0
        latest_workout = float(getattr(row, "workout_latest_speed_index", 0.0) or 0.0)
        prior_workout = float(getattr(row, "workout_prior_avg_speed_index", 0.0) or 0.0)
        workout_sudden_improvement = float(latest_workout - prior_workout) if latest_workout and prior_workout else 0.0
        prior_gaps = [race_days[index] - race_days[index - 1] for index in range(1, len(race_days))]
        rest_optimal_fit = float(np.exp(-abs(days_since_last - np.median(prior_gaps)) / max(float(np.median(prior_gaps)), 7.0))) if prior_gaps else 0.0

        fatigue_load = (
            1.8 * short_rest_flag
            + 0.6 * race_frequency_3
            + 0.4 * race_frequency_5
            + 0.8 * seasonal_race_load
            + 0.5 * distance_shock
            + 0.5 * intensity_proxy
            + 0.4 * long_layoff_flag
        )
        recovery_score = float(1.0 / (1.0 + np.exp(fatigue_load - 1.5)))

        fatigue_score = 1.0 / (1.0 + np.exp(-(days_since_last - 18.0) / 8.0))
        pace_hint = float(getattr(row, "early_pace", 0.0))

        # Phase 2: running style priors derived from historical early pace.
        front_style = sum(1 for p in pace_hist if p >= 0.6)
        presser_style = sum(1 for p in pace_hist if 0.2 <= p < 0.6)
        stalker_style = sum(1 for p in pace_hist if -0.25 <= p < 0.2)
        closer_style = sum(1 for p in pace_hist if p < -0.25)
        if not pace_hist:
            style_front_prob = 0.25
            style_presser_prob = 0.25
            style_stalker_prob = 0.25
            style_closer_prob = 0.25
        else:
            hist_n = len(pace_hist)
            style_front_prob = front_style / hist_n
            style_presser_prob = presser_style / hist_n
            style_stalker_prob = stalker_style / hist_n
            style_closer_prob = closer_style / hist_n

        # Phase 2: context fit with Bayesian shrinkage.
        same_surface_perf = [float(x["perf"]) for x in h if str(x.get("surface", "")) == str(row.surface)]
        same_track_perf = [float(x["perf"]) for x in h if str(x.get("track", "")) == str(row.track)]
        same_condition_perf = [float(x["perf"]) for x in h if str(x.get("condition", "")) == str(row.track_condition)]
        similar_distance_perf = [
            float(x["perf"])
            for x in h
            if abs(float(x.get("distance", row.distance)) - float(row.distance)) <= 200.0
        ]
        history_field_sizes = [float(x["field_size"]) for x in h]
        history_weights = [float(x["weight"]) for x in h if float(x.get("weight", 0.0)) > 0.0]

        surface_fit = _smoothed_mean(same_surface_perf)
        track_fit = _smoothed_mean(same_track_perf)
        condition_fit = _smoothed_mean(same_condition_perf)
        distance_fit = _smoothed_mean(similar_distance_perf)

        mprob = float(row.market_probability) if pd.notnull(row.market_probability) else 0.1
        implied_prob = 1.0 / max(1.01, float(row.odds)) if pd.notnull(row.odds) else mprob

        feature_rows.append(
            {
                "race_id": row.race_id,
                "date": row.date,
                "race_datetime": row.race_datetime,
                "horse_id": horse_id,
                "draw": int(row.draw),
                "weight": float(row.weight) if pd.notnull(row.weight) else 56.0,
                "distance": float(row.distance) if pd.notnull(row.distance) else 1400.0,
                "field_size": float(row.field_size) if pd.notnull(row.field_size) else 10.0,
                "age": float(getattr(row, "age", 0.0) or 0.0),
                "handicap_points": float(getattr(row, "handicap_points", 0.0) or 0.0),
                "history_missing": float(getattr(row, "history_missing", 0.0) or 0.0),
                "career_summary_missing": float(getattr(row, "career_summary_missing", 0.0) or 0.0),
                "workout_missing": float(getattr(row, "workout_missing", 0.0) or 0.0),
                "age_missing": float(getattr(row, "age_missing", 0.0) or 0.0),
                "handicap_missing": float(getattr(row, "handicap_missing", 0.0) or 0.0),
                "odds_missing": float(getattr(row, "odds_missing", 0.0) or 0.0),
                "form_avg_3": form_avg_3,
                "form_avg_5": form_avg_5,
                "form_avg_10": form_avg_10,
                "form_var_5": form_var_5,
                "last_run_perf": last_run_perf,
                "trend_3_10": trend_3_10,
                "days_since_last_race": days_since_last,
                "fatigue_score": float(fatigue_score),
                "recovery_score": recovery_score,
                "short_rest_flag": short_rest_flag,
                "long_layoff_flag": long_layoff_flag,
                "race_frequency_3": race_frequency_3,
                "race_frequency_5": race_frequency_5,
                "seasonal_race_load": seasonal_race_load,
                "pace_hint": pace_hint,
                "style_front_prob": style_front_prob,
                "style_presser_prob": style_presser_prob,
                "style_stalker_prob": style_stalker_prob,
                "style_closer_prob": style_closer_prob,
                "distance_fit": distance_fit,
                "surface_fit": surface_fit,
                "track_fit": track_fit,
                "history_avg_field_size": float(np.mean(history_field_sizes)) if history_field_sizes else 0.0,
                "history_avg_weight": float(np.mean(history_weights)) if history_weights else 0.0,
                "market_probability": mprob,
                "implied_probability": implied_prob,
                "odds": float(row.odds) if pd.notnull(row.odds) else 0.0,
                "finish_position": (
                    float(row.finish_position)
                    if pd.notnull(getattr(row, "finish_position", np.nan))
                    else np.nan
                ),
                "is_winner": int(getattr(row, "is_winner", 0)) if pd.notnull(getattr(row, "is_winner", 0)) else 0,
                "jockey_change_upgrade": jockey_change_upgrade,
                "trainer_change_upgrade": trainer_change_upgrade,
                "class_drop_flag": class_drop_flag,
                "workout_sudden_improvement": workout_sudden_improvement,
                "rest_optimal_fit": rest_optimal_fit,
            }
        )

        finish_position = float(getattr(row, "finish_position", np.nan))
        field_size = float(getattr(row, "field_size", np.nan))
        if (
            pd.notnull(finish_position)
            and pd.notnull(field_size)
            and field_size > 1.0
            and finish_position <= field_size
        ):
            perf = _field_size_adjusted_score(finish_position, field_size)
            histories[horse_id].append(
                {
                    "perf": perf,
                    "day_ordinal": float(row.race_datetime.toordinal()),
                    "field_size": field_size,
                    "weight": float(getattr(row, "weight", 0.0) or 0.0),
                    "distance": float(getattr(row, "distance", 1400.0)),
                    "surface": str(getattr(row, "surface", "")),
                    "track": str(getattr(row, "track", "")),
                    "condition": str(getattr(row, "track_condition", "")),
                    "early_pace": float(getattr(row, "early_pace", 0.0)),
                    "jockey_name": str(getattr(row, "jockey_name", "") or ""),
                    "trainer_name": str(getattr(row, "trainer_name", "") or ""),
                    "race_class": str(getattr(row, "race_class", "") or ""),
                }
            )

    feat_df = pd.DataFrame(feature_rows)

    precomputed_rows = pd.Series(False, index=df.index)
    if precomputed_columns.issubset(df.columns):
        precomputed_rows = df[list(precomputed_columns)].notna().all(axis=1)
    if precomputed_rows.any():
        for column in TJK_STAGE1_FEATURE_COLUMNS:
            if column not in df.columns or column in recompute_precomputed_columns:
                continue
            values = pd.to_numeric(df.loc[precomputed_rows, column], errors="coerce")
            available = values.notna()
            if available.any():
                feat_df.loc[values.index[available], column] = values.loc[available].to_numpy()

    # Normalize market probabilities within each race to account for overround.
    grp_sum = feat_df.groupby("race_id")["market_probability"].transform("sum").replace(0.0, 1.0)
    feat_df["market_probability_norm"] = feat_df["market_probability"] / grp_sum

    # Phase 2: opponent strength and race pace context (race-level, then projected to horse rows).
    race_mean_form = feat_df.groupby("race_id")["form_avg_5"].transform("mean")
    race_sum_form = feat_df.groupby("race_id")["form_avg_5"].transform("sum")
    race_count = feat_df.groupby("race_id")["horse_id"].transform("count").replace(0, 1)

    feat_df["field_strength_index"] = race_mean_form
    feat_df["opponent_strength_mean"] = ((race_sum_form - feat_df["form_avg_5"]) / (race_count - 1).clip(lower=1)).fillna(race_mean_form)

    feat_df["expected_early_pace"] = feat_df.groupby("race_id")["pace_hint"].transform("mean")
    feat_df["pace_pressure"] = feat_df.groupby("race_id")["style_front_prob"].transform("mean") + (
        0.7 * feat_df.groupby("race_id")["style_presser_prob"].transform("mean")
    )
    feat_df["pace_pressure"] = feat_df["pace_pressure"].clip(lower=0.0, upper=1.0)
    feat_df["pace_suitability"] = (
        feat_df["style_closer_prob"] * feat_df["pace_pressure"]
        + feat_df["style_front_prob"] * (1.0 - feat_df["pace_pressure"])
    )

    feature_columns = TJK_STAGE1_FEATURE_COLUMNS.copy()
    for column in feature_columns:
        if column not in feat_df:
            feat_df[column] = 0.0

    feat_df[feature_columns] = feat_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feat_df["is_placed"] = placer_labels["is_placed"].to_numpy()
    feat_df["target_highest_odds_placer"] = placer_labels[
        "target_highest_odds_placer"
    ].to_numpy()
    return FeatureBuildResult(frame=feat_df, feature_columns=feature_columns)


def assert_no_leakage_columns(feature_columns: list[str]) -> None:
    leaked = LEAKAGE_COLUMNS.intersection(set(feature_columns))
    if leaked:
        raise ValueError(f"Leakage columns found in features: {sorted(leaked)}")
