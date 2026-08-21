"""Demo ve testlerde kullanilan deterministik ornek veri kaynagi icin testler."""
from __future__ import annotations

from datetime import date

from atyaris.data_sources.base import RaceDataSource
from atyaris.data_sources.sample_source import SampleDataSource


def test_sample_source_implements_protocol() -> None:
    assert isinstance(SampleDataSource(), RaceDataSource)


def test_get_daily_races_is_deterministic() -> None:
    source = SampleDataSource()
    d = date(2026, 8, 18)
    first = source.get_daily_races(d)
    second = source.get_daily_races(d)
    assert [r.id for r in first] == [r.id for r in second]
    assert len(first) > 0


def test_get_daily_races_filters_by_city() -> None:
    source = SampleDataSource()
    races = source.get_daily_races(date.today(), city="Ankara")
    assert races
    assert all("ankara" in r.hippodrome.lower() for r in races)


def test_sample_race_uses_unique_jockeys_within_a_race() -> None:
    source = SampleDataSource()
    race = source.get_daily_races(date.today())[0]
    jockey_names = [entry.jockey.name for entry in race.entries]
    assert len(jockey_names) == len(set(jockey_names))


def test_get_horse_statistics_returns_history() -> None:
    source = SampleDataSource()
    race = source.get_daily_races(date.today())[0]
    entry = race.entries[0]
    stats = source.get_horse_statistics(entry)
    assert stats.horse_id == entry.horse_id
    assert len(stats.past_performances) > 0
