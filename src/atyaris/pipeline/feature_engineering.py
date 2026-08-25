"""Ozellik muhendisligi: zaman serisi, dinlenme etkisi, tempo uyumu ve goreli guc ozellikleri."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from statistics import mean, pstdev

from atyaris.models.entities import HorseStatistics, Race


@dataclass
class EntryFeatures:
    horse_id: str
    values: dict[str, float]


def _finish_quality(position: int, field_size: int) -> float:
    return max(0.0, min(1.0, 1.0 - ((position - 1) / max(field_size - 1, 1))))


def _linear_slope(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    x_vals = list(range(len(series)))
    x_mean = mean(x_vals)
    y_mean = mean(series)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, series))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _rolling_stats(series: list[float], window: int) -> tuple[float, float]:
    if not series:
        return 0.5, 0.0
    cut = series[:window]
    if len(cut) == 1:
        return cut[0], 0.0
    return mean(cut), pstdev(cut)


def _weighted_moving_average(series: list[float], window: int) -> float:
    if not series:
        return 0.5
    cut = series[:window]
    weights = [window - i for i in range(len(cut))]
    return sum(v * w for v, w in zip(cut, weights)) / sum(weights)


def _infer_optimal_rest_days(stats: HorseStatistics, race_date: date) -> float:
    perfs = sorted(stats.past_performances, key=lambda p: p.race_date, reverse=True)
    if len(perfs) < 2:
        return 24.0

    rest_gaps: list[tuple[int, float]] = []
    for idx in range(len(perfs) - 1):
        current = perfs[idx]
        previous = perfs[idx + 1]
        gap = max(1, (current.race_date - previous.race_date).days)
        quality = _finish_quality(current.finish_position or 6, current.field_size or 8)
        rest_gaps.append((gap, quality))

    if not rest_gaps:
        return 24.0
    top = sorted(rest_gaps, key=lambda item: item[1], reverse=True)[: max(2, len(rest_gaps) // 3)]
    return mean([gap for gap, _ in top])


def _tempo_profile(stats: HorseStatistics) -> tuple[float, float, float, float]:
    perfs = stats.recent_form(8)
    if not perfs:
        return 0.5, 0.5, 0.5, 0.5

    early = [p.early_pace_index if p.early_pace_index is not None else 0.5 for p in perfs]
    mid = [p.mid_pace_index if p.mid_pace_index is not None else 0.5 for p in perfs]
    late = [p.late_pace_index if p.late_pace_index is not None else 0.5 for p in perfs]

    # Tempo tipi: onde gidenler yuksek early, takipciler yuksek late degeri alir.
    style = (mean(early) - mean(late)) * 0.5 + 0.5
    return mean(early), mean(mid), mean(late), max(0.0, min(1.0, style))


def build_entry_features(race: Race, stats_by_horse_id: dict[str, HorseStatistics]) -> list[EntryFeatures]:
    """Ham veriden turetilmis model girdilerini uretir."""
    race_date = race.start_time.date()

    precomputed_strength: dict[str, float] = {}
    precomputed_tempo_style: dict[str, float] = {}
    for entry in race.active_entries:
        stats = stats_by_horse_id[entry.horse_id]
        recent = stats.recent_form(8)
        quality = [
            _finish_quality(p.finish_position or 6, p.field_size or 8)
            for p in recent
        ]
        precomputed_strength[entry.horse_id] = _weighted_moving_average(quality, 8)
        _, _, _, style = _tempo_profile(stats)
        precomputed_tempo_style[entry.horse_id] = style

    field_strength_mean = mean(precomputed_strength.values()) if precomputed_strength else 0.5
    projected_early_pace = mean(precomputed_tempo_style.values()) if precomputed_tempo_style else 0.5

    features: list[EntryFeatures] = []
    for entry in race.active_entries:
        stats = stats_by_horse_id[entry.horse_id]
        recent = stats.recent_form(10)
        quality_series = [
            _finish_quality(p.finish_position or 6, p.field_size or 8)
            for p in recent
        ]

        optimal_rest = _infer_optimal_rest_days(stats, race_date)
        days_since_last = float(stats.days_since_last_race or optimal_rest)
        rest_delta = abs(days_since_last - optimal_rest)
        rest_fit = math.exp(-(rest_delta / 16.0))

        slope = _linear_slope(list(reversed(quality_series)))
        rolling_mean_3, rolling_std_3 = _rolling_stats(quality_series, 3)
        rolling_mean_5, rolling_std_5 = _rolling_stats(quality_series, 5)
        wma_5 = _weighted_moving_average(quality_series, 5)

        similar = [
            p
            for p in stats.past_performances
            if p.surface == race.surface and abs(p.distance_m - race.distance_m) <= 200
        ]
        similar_quality = mean(
            [_finish_quality(p.finish_position or 6, p.field_size or 8) for p in similar]
        ) if similar else 0.5

        early, mid, late, style = _tempo_profile(stats)
        tempo_fit = 1.0 - abs(style - projected_early_pace)

        weight_samples = [p.weight_kg for p in stats.past_performances if p.weight_kg is not None]
        avg_weight = mean(weight_samples) if weight_samples else entry.weight_kg
        weight_adv = max(-5.0, min(5.0, avg_weight - entry.weight_kg)) / 5.0

        odds = entry.odds if entry.odds and entry.odds > 0 else None
        implied_prob = (1.0 / odds) if odds else 0.0

        relative_strength = precomputed_strength[entry.horse_id] - field_strength_mean

        values = {
            "days_since_last_race": days_since_last,
            "optimal_rest_days": optimal_rest,
            "rest_fit": rest_fit,
            "rest_performance_interaction": rest_fit * rolling_mean_5,
            "form_trend_slope": slope,
            "similar_conditions_score": similar_quality,
            "wma_form": wma_5,
            "lag_form_1": quality_series[0] if len(quality_series) >= 1 else 0.5,
            "lag_form_2": quality_series[1] if len(quality_series) >= 2 else 0.5,
            "rolling_mean_3": rolling_mean_3,
            "rolling_std_3": rolling_std_3,
            "rolling_mean_5": rolling_mean_5,
            "rolling_std_5": rolling_std_5,
            "tempo_early": early,
            "tempo_mid": mid,
            "tempo_late": late,
            "tempo_style": style,
            "tempo_fit_score": max(0.0, min(1.0, tempo_fit)),
            "relative_strength": relative_strength,
            "weight_advantage": weight_adv,
            "jockey_win_rate": (entry.jockey.win_rate or 12.0) / 100.0,
            "trainer_win_rate": (entry.trainer.win_rate or 10.0) / 100.0,
            "market_implied_probability": implied_prob,
        }
        features.append(EntryFeatures(horse_id=entry.horse_id, values=values))

    return features
