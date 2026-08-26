"""Walk-forward backtest ve degerlendirme metrikleri."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from statistics import mean, pstdev
from typing import Any

from atyaris.config import Settings
from atyaris.data_sources.base import RaceDataSource
from atyaris.models.entities import BacktestMetrics, Race
from atyaris.pipeline.data_layer import prepare_race_data
from atyaris.pipeline.feature_engineering import build_entry_features
from atyaris.pipeline.modeling import EnsembleRanker


@dataclass
class BacktestRow:
    win_probability: float
    is_winner: int
    stake_return: float


@dataclass
class BacktestCase:
    features: list[Any]
    entries: list[Any]
    winner_horse_id: str


def _actual_winner_horse_id(race: Race) -> str | None:
    finished = [e for e in race.active_entries if e.actual_finish_position]
    if not finished:
        return None
    winner = min(finished, key=lambda e: e.actual_finish_position or 99)
    return winner.horse_id


def _inject_external_results(
    race: Race,
    result_map: dict[int, dict[int, int]] | None,
) -> None:
    """Dis kaynaktan gelen yaris sonucuyla entry.actual_finish_position alanini doldurur."""
    if not result_map:
        return
    race_results = result_map.get(race.race_no)
    if not race_results:
        return
    for entry in race.active_entries:
        if entry.number in race_results:
            entry.actual_finish_position = race_results[entry.number]


def _log_loss(rows: list[BacktestRow]) -> float:
    eps = 1e-9
    losses = []
    for row in rows:
        p = max(eps, min(1.0 - eps, row.win_probability))
        y = row.is_winner
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return mean(losses) if losses else 0.0


def _calibration_error(rows: list[BacktestRow], bins: int = 10) -> float:
    if not rows:
        return 0.0
    bucketed: list[list[BacktestRow]] = [[] for _ in range(bins)]
    for row in rows:
        idx = min(bins - 1, int(row.win_probability * bins))
        bucketed[idx].append(row)

    error = 0.0
    total = len(rows)
    for bucket in bucketed:
        if not bucket:
            continue
        avg_p = mean([r.win_probability for r in bucket])
        avg_y = mean([float(r.is_winner) for r in bucket])
        error += abs(avg_p - avg_y) * (len(bucket) / total)
    return error


def _sharpe_like(returns: list[float]) -> float:
    if not returns:
        return 0.0
    avg = mean(returns)
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    if vol == 0:
        return 0.0
    return avg / vol


def _evaluate_cases(
    cases: list[BacktestCase],
    settings: Settings,
    calibration_temperature: float,
) -> list[BacktestRow]:
    ranker = EnsembleRanker(
        boosting_weight=settings.ensemble_boosting_weight,
        ranking_weight=settings.ensemble_ranking_weight,
        calibration_temperature=calibration_temperature,
    )
    rows: list[BacktestRow] = []
    for case in cases:
        ranked = ranker.rank(case.features)
        by_horse = {r.horse_id: r for r in ranked}
        for entry in case.entries:
            pred = by_horse.get(entry.horse_id)
            if pred is None:
                continue
            is_winner = 1 if entry.horse_id == case.winner_horse_id else 0
            stake_return = 0.0
            if pred.calibrated_probability >= settings.ev_probability_threshold and entry.odds and entry.odds > 1.0:
                stake_return = (entry.odds - 1.0) if is_winner else -1.0
            rows.append(
                BacktestRow(
                    win_probability=pred.calibrated_probability,
                    is_winner=is_winner,
                    stake_return=stake_return,
                )
            )
    return rows


def build_walk_forward_backtest(
    data_source: RaceDataSource,
    reference_date: date,
    settings: Settings,
    lookback_days: int = 8,
) -> BacktestMetrics:
    """Zaman bazli geriye yuruyen mini backtest (walk-forward) uygular."""
    rows: list[BacktestRow] = []
    evaluated_races = 0
    cases: list[BacktestCase] = []
    result_cache: dict[tuple[date, str], dict[int, dict[int, int]]] = {}

    for shift in range(lookback_days, 0, -1):
        target_date = reference_date - timedelta(days=shift)
        races = data_source.get_daily_races(target_date)
        for race in races:
            if _actual_winner_horse_id(race) is None and hasattr(data_source, "get_daily_race_results"):
                cache_key = (target_date, race.hippodrome)
                if cache_key not in result_cache:
                    try:
                        result_cache[cache_key] = data_source.get_daily_race_results(target_date, race.hippodrome)  # type: ignore[attr-defined]
                    except Exception:
                        result_cache[cache_key] = {}
                _inject_external_results(race, result_cache.get(cache_key))

            winner_id = _actual_winner_horse_id(race)
            if not winner_id:
                continue
            prepared = prepare_race_data(
                data_source,
                race,
                as_of_date=(race.start_time.date() if settings.backtest_leakage_safe_mode else None),
                exclude_most_recent_races=(
                    settings.backtest_exclude_recent_races if settings.backtest_leakage_safe_mode else 0
                ),
            )
            features = build_entry_features(race, prepared.stats_by_horse_id)
            evaluated_races += 1
            cases.append(BacktestCase(features=features, entries=race.active_entries, winner_horse_id=winner_id))

    if cases:
        calibration_candidates = [settings.calibration_temperature]
        if settings.backtest_optimize_calibration:
            calibration_candidates = [0.65, 0.8, 0.95, 1.1, 1.25]

        best_temperature = settings.calibration_temperature
        best_rows: list[BacktestRow] = []
        best_log_loss = float("inf")
        for temp in calibration_candidates:
            candidate_rows = _evaluate_cases(cases, settings, temp)
            if not candidate_rows:
                continue
            candidate_loss = _log_loss(candidate_rows)
            if candidate_loss < best_log_loss:
                best_log_loss = candidate_loss
                best_rows = candidate_rows
                best_temperature = temp

        rows = best_rows
    else:
        rows = []

    if not rows:
        return BacktestMetrics(
            windows=0,
            evaluated_races=0,
            notes=[
                "Walk-forward backtest icin yeterli etiketli gecmis yaris bulunamadi.",
                "TJK canli kaynaginda resmi sonuc etiketi olmadiginda bu alan notr kalabilir.",
            ],
        )

    returns = [row.stake_return for row in rows if row.stake_return != 0.0]
    total_staked = len(returns)
    roi = (sum(returns) / total_staked) if total_staked else 0.0

    return BacktestMetrics(
        windows=lookback_days,
        evaluated_races=evaluated_races,
        log_loss=round(_log_loss(rows), 4),
        calibration_error=round(_calibration_error(rows), 4),
        roi=round(roi, 4),
        sharpe_like=round(_sharpe_like(returns), 4) if returns else 0.0,
        notes=[
            "Walk-forward pencereleri gelecege bakmayan sekilde kuruldu (veri sizintisi engeli).",
            (
                f"Leakage-safe mod aktif: as_of kesiti + en yeni {settings.backtest_exclude_recent_races} kosu dislama uygulandi."
                if settings.backtest_leakage_safe_mode
                else "Leakage-safe mod kapali: tum erisilebilir gecmis kayitlar kullanildi."
            ),
            (
                f"Kalibrasyon optimizasyonu aktif: en iyi sicaklik {best_temperature:.2f} secildi (log-loss minimizasyonu)."
                if settings.backtest_optimize_calibration and cases
                else "Kalibrasyon optimizasyonu kapali veya veri yetersiz."
            ),
            "ROI yalnizca esik ustu sinyallerde 1 birim stake ile simule edildi.",
        ],
    )
