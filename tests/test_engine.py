"""Sample veri kaynagi kullanarak tahmin motoru icin uctan uca testler."""
from __future__ import annotations

from datetime import date

import pytest

from atyaris.config import Settings
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.models.entities import Race
from atyaris.prediction.engine import PredictionEngine


def test_predict_returns_ranked_and_scored_entries(
    sample_source: SampleDataSource, sample_race: Race, settings: Settings
) -> None:
    engine = PredictionEngine(sample_source, settings)
    prediction = engine.predict(sample_race)

    assert len(prediction.ranked) == len(sample_race.active_entries)
    scores = [hp.score.total_score for hp in prediction.ranked]
    assert scores == sorted(scores, reverse=True)
    assert prediction.ranked[0].tag == "Kazanan Aday"
    assert "sorumlu bahis" in prediction.disclaimer.lower()


def test_predict_generates_reasoning_bullets(
    sample_source: SampleDataSource, sample_race: Race, settings: Settings
) -> None:
    engine = PredictionEngine(sample_source, settings)
    prediction = engine.predict(sample_race)
    for hp in prediction.ranked:
        assert len(hp.reasoning) >= 5
        assert all(isinstance(b, str) and b for b in hp.reasoning)
    assert any(
        "en yuksek puana sahip" in bullet or "en ideal puana sahip" in bullet
        for hp in prediction.ranked
        for bullet in hp.reasoning
    )


def test_predict_raises_without_active_entries(sample_source: SampleDataSource, settings: Settings) -> None:
    race = sample_source.get_daily_races(date.today())[0]
    for entry in race.entries:
        entry.is_scratched = True
    engine = PredictionEngine(sample_source, settings)
    with pytest.raises(ValueError):
        engine.predict(race)
