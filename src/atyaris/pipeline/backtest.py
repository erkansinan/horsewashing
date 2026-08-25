"""Walk-forward backtest ve degerlendirme metrikleri."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
from statistics import mean, pstdev

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


def _actual_winner_horse_id(race: Race) -> str | None:
    finished = [e for e in race.active_entries if e.actual_finish_position]
    if not finished:
        return None
    winner = min(finished, key=lambda e: e.actual_finish_position or 99)
    return winner.horse_id


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


def build_walk_forward_backtest(
    data_source: RaceDataSource,
    reference_date: date,
    settings: Settings,
    lookback_days: int = 8,
) -> BacktestMetrics:
    """Zaman bazli geriye yuruyen mini backtest (walk-forward) uygular."""
    ranker = EnsembleRanker(
        boosting_weight=settings.ensemble_boosting_weight,
        ranking_weight=settings.ensemble_ranking_weight,
        calibration_temperature=settings.calibration_temperature,
    )

    rows: list[BacktestRow] = []
    evaluated_races = 0

    for shift in range(lookback_days, 0, -1):
        target_date = reference_date - timedelta(days=shift)
        races = data_source.get_daily_races(target_date)
        for race in races:
            winner_id = _actual_winner_horse_id(race)
            if not winner_id:
                continue
            prepared = prepare_race_data(data_source, race)
            features = build_entry_features(race, prepared.stats_by_horse_id)
            ranked = ranker.rank(features)
            by_horse = {r.horse_id: r for r in ranked}
            evaluated_races += 1

            for entry in race.active_entries:
                pred = by_horse.get(entry.horse_id)
                if pred is None:
                    continue
                is_winner = 1 if entry.horse_id == winner_id else 0
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
            "ROI yalnizca esik ustu sinyallerde 1 birim stake ile simule edildi.",
        ],
    )
