"""Paylasilan pytest fixture'lari."""
from __future__ import annotations

from datetime import date

import pytest

from atyaris.config import Settings
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.models.entities import Race


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def sample_source() -> SampleDataSource:
    return SampleDataSource()


@pytest.fixture
def sample_race(sample_source: SampleDataSource) -> Race:
    races = sample_source.get_daily_races(date.today())
    assert races, "Sample veri kaynagi her zaman demo yaris dondurmeli"
    return races[0]
