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
    "history_avg_prize", "history_avg_s20",
    "workout_count",
    "workout_avg_distance", "workout_avg_speed_index", "workout_best_speed_index",
    "days_since_last_workout",
    "history_missing", "career_summary_missing", "workout_missing",
    "age_missing", "handicap_missing", "odds_missing",
]

TJK_SELECTED_STAGE1_FEATURE_COLUMNS = [
    "trainer_win_rate",
    "form_avg_5",
    "last_run_perf",
    "distance_fit",
    "surface_fit",
    "handicap_points",
    "jockey_horse_combo_wins",
    "history_avg_finish_position",
    "draw",
    "weight",
]

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


def build_leakage_safe_features(frame: pd.DataFrame, as_of_date: date | None = None) -> FeatureBuildResult:
    """Builds features using only events strictly before each row race_datetime.

    This function never uses finish_result columns from the current race row while
    constructing its own features.
    """
    df = preprocess_dataset(frame)
    if as_of_date is not None:
        df = df[df["date"] <= as_of_date].copy()

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

    # Real TJK ingestion already computes historical features while the race
    # history is available. Those rows intentionally do not contain the raw
    # finish_position column, so rebuilding history here would replace valid
    # values with the no-history defaults.
    precomputed_columns = {
        "days_since_last_race",
        "form_avg_3",
        "form_avg_5",
        "form_avg_10",
        "track_fit",
        "surface_fit",
        "distance_fit",
    }
    if "finish_position" not in df.columns and precomputed_columns.issubset(df.columns):
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
                "market_probability": mprob,
                "implied_probability": implied_prob,
                "odds": float(row.odds) if pd.notnull(row.odds) else 0.0,
                "is_winner": int(getattr(row, "is_winner", 0)) if pd.notnull(getattr(row, "is_winner", 0)) else 0,
            }
        )

        finish_position = float(getattr(row, "finish_position", np.nan))
        field_size = float(getattr(row, "field_size", np.nan))
        if pd.notnull(finish_position) and pd.notnull(field_size):
            perf = _field_size_adjusted_score(finish_position, field_size)
            histories[horse_id].append(
                {
                    "perf": perf,
                    "day_ordinal": float(row.race_datetime.toordinal()),
                    "distance": float(getattr(row, "distance", 1400.0)),
                    "surface": str(getattr(row, "surface", "")),
                    "track": str(getattr(row, "track", "")),
                    "condition": str(getattr(row, "track_condition", "")),
                    "early_pace": float(getattr(row, "early_pace", 0.0)),
                }
            )

    feat_df = pd.DataFrame(feature_rows)

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
    return FeatureBuildResult(frame=feat_df, feature_columns=feature_columns)


def assert_no_leakage_columns(feature_columns: list[str]) -> None:
    leaked = LEAKAGE_COLUMNS.intersection(set(feature_columns))
    if leaked:
        raise ValueError(f"Leakage columns found in features: {sorted(leaked)}")
