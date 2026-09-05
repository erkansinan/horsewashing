from __future__ import annotations

from datetime import date

from atyaris.ml.backtest import walk_forward_backtest
from atyaris.ml.provider import SyntheticRacingDataProvider as _FixtureRacingDataProvider


def test_walk_forward_temporal_integrity_and_metrics() -> None:
    provider = _FixtureRacingDataProvider()
    data = provider.get_dataset(date(2024, 1, 1), date(2025, 3, 31))

    result = walk_forward_backtest(data, min_train_days=60)

    assert result.evaluated_days > 0
    assert result.evaluated_races > 0
    assert 0.0 <= result.top1_hit_rate <= 1.0
    assert 0.0 <= result.top2_hit_rate <= 1.0
    assert 0.0 <= result.top3_hit_rate <= 1.0
    assert result.log_loss > 0.0
    assert result.brier > 0.0

    for train_max, test_day in zip(result.fold_max_train_date, result.fold_test_date):
        assert train_max < test_day
