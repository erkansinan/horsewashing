"""Agirlikli puanlama fonksiyonlari icin testler."""
from __future__ import annotations

from datetime import date, timedelta

from atyaris.config import Settings
from atyaris.models.entities import HorseStatistics, Jockey, PastPerformance, RaceEntry, Trainer, TrackSurface
from atyaris.prediction.scoring import compute_score, score_form, score_rest, score_weight


def _perf(
    days_ago: int,
    position: int,
    weight: float = 56.0,
    surface: TrackSurface = TrackSurface.KUM,
    distance: int = 1400,
) -> PastPerformance:
    return PastPerformance(
        race_date=date.today() - timedelta(days=days_ago),
        hippodrome="Ankara", distance_m=distance, surface=surface,
        finish_position=position, field_size=8, weight_kg=weight,
    )


def test_score_form_higher_for_better_recent_finishes() -> None:
    good = HorseStatistics(horse_id="A", horse_name="A", past_performances=[_perf(10, 1), _perf(40, 2)])
    bad = HorseStatistics(horse_id="B", horse_name="B", past_performances=[_perf(10, 8), _perf(40, 7)])
    assert score_form(good, window=5) > score_form(bad, window=5)


def test_score_form_neutral_without_history() -> None:
    empty = HorseStatistics(horse_id="C", horse_name="C")
    assert score_form(empty, window=5) == 40.0


def test_score_weight_rewards_lighter_than_average() -> None:
    stats = HorseStatistics(
        horse_id="A", horse_name="A",
        past_performances=[_perf(10, 3, weight=58.0), _perf(40, 4, weight=58.0)],
    )
    lighter_score = score_weight(stats, current_weight=55.5)
    heavier_score = score_weight(stats, current_weight=60.5)
    assert lighter_score > 50 > heavier_score


def test_score_rest_peaks_in_ideal_window() -> None:
    stats_ideal = HorseStatistics(horse_id="A", horse_name="A", past_performances=[_perf(20, 1)])
    stats_too_soon = HorseStatistics(horse_id="B", horse_name="B", past_performances=[_perf(2, 1)])
    stats_too_long = HorseStatistics(horse_id="C", horse_name="C", past_performances=[_perf(200, 1)])
    assert score_rest(stats_ideal, 14, 45) == 100.0
    assert score_rest(stats_too_soon, 14, 45) < 100.0
    assert score_rest(stats_too_long, 14, 45) < 100.0


def test_compute_score_within_bounds(settings: Settings) -> None:
    entry = RaceEntry(
        number=1, horse_id="A", horse_name="A",
        jockey=Jockey(name="J"), trainer=Trainer(name="T"), weight_kg=56.0,
    )
    stats = HorseStatistics(horse_id="A", horse_name="A", past_performances=[_perf(20, 1)])
    score = compute_score(entry, stats, 1400, TrackSurface.KUM, settings)
    assert 0 <= score.total_score <= 100
