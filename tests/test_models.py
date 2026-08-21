"""Domain modelleri icin testler: turetilmis ozellikler ve dogrulama."""
from __future__ import annotations

from datetime import date, timedelta

from atyaris.models.entities import (
    HorseStatistics,
    PastPerformance,
    TrackSurface,
)


def test_past_performance_is_win_and_placed() -> None:
    win = PastPerformance(
        race_date=date.today(), hippodrome="Ankara", distance_m=1400,
        surface=TrackSurface.KUM, finish_position=1, field_size=8,
    )
    third = PastPerformance(
        race_date=date.today(), hippodrome="Ankara", distance_m=1400,
        surface=TrackSurface.KUM, finish_position=3, field_size=8,
    )
    last = PastPerformance(
        race_date=date.today(), hippodrome="Ankara", distance_m=1400,
        surface=TrackSurface.KUM, finish_position=8, field_size=8,
    )
    assert win.is_win and win.is_placed
    assert not third.is_win and third.is_placed
    assert not last.is_win and not last.is_placed


def test_horse_statistics_rates_and_days_since_last_race() -> None:
    stats = HorseStatistics(
        horse_id="H1",
        horse_name="TEST AT",
        past_performances=[
            PastPerformance(
                race_date=date.today() - timedelta(days=20),
                hippodrome="Ankara", distance_m=1400, surface=TrackSurface.KUM,
                finish_position=1, field_size=8,
            )
        ],
        career_starts=10,
        career_wins=3,
        career_places=6,
    )
    assert stats.career_win_rate == 30.0
    assert stats.career_place_rate == 60.0
    assert stats.days_since_last_race == 20


def test_horse_statistics_recent_form_orders_by_date_desc() -> None:
    older = PastPerformance(
        race_date=date.today() - timedelta(days=60),
        hippodrome="Ankara", distance_m=1400, surface=TrackSurface.KUM, finish_position=2,
    )
    newer = PastPerformance(
        race_date=date.today() - timedelta(days=5),
        hippodrome="Ankara", distance_m=1400, surface=TrackSurface.KUM, finish_position=1,
    )
    stats = HorseStatistics(horse_id="H1", horse_name="TEST AT", past_performances=[older, newer])
    recent = stats.recent_form(5)
    assert recent[0] is newer
    assert recent[1] is older


def test_horse_statistics_empty_history_has_no_days_since_last_race() -> None:
    stats = HorseStatistics(horse_id="H1", horse_name="TEST AT")
    assert stats.days_since_last_race is None
    assert stats.career_win_rate == 0.0
