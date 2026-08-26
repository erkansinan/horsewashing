from __future__ import annotations

from datetime import date

from atyaris.config import Settings
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.models.entities import Race
from atyaris.pipeline.backtest import build_walk_forward_backtest


class ResultsInjectedSampleSource(SampleDataSource):
    def get_daily_races(self, target_date: date, city: str | None = None) -> list[Race]:
        races = super().get_daily_races(target_date, city)
        for race in races:
            for entry in race.entries:
                entry.actual_finish_position = None
        return races

    def get_daily_race_results(self, target_date: date, city: str) -> dict[int, dict[int, int]]:  # noqa: ARG002
        races = super().get_daily_races(target_date, city)
        if not races:
            return {}
        first_race = races[0]
        return {
            first_race.race_no: {entry.number: idx for idx, entry in enumerate(first_race.active_entries, start=1)}
        }


def test_backtest_uses_external_results_and_reports_calibration_optimization() -> None:
    source = ResultsInjectedSampleSource()
    settings = Settings(backtest_optimize_calibration=True)

    metrics = build_walk_forward_backtest(
        source,
        reference_date=date.today(),
        settings=settings,
        lookback_days=2,
    )

    assert metrics.evaluated_races > 0
    assert metrics.log_loss is not None
    assert any("Kalibrasyon optimizasyonu aktif" in note for note in metrics.notes)
