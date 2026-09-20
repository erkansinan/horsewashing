from __future__ import annotations

from datetime import date

from atyaris.ml.backtest import walk_forward_backtest
from atyaris.ml.provider import SyntheticRacingDataProvider as _FixtureRacingDataProvider


def test_walk_forward_temporal_integrity_and_metrics() -> None:
    provider = _FixtureRacingDataProvider()
    data = provider.get_dataset(date(2024, 1, 1), date(2025, 3, 31))

    result = walk_forward_backtest(
        data,
        min_train_days=60,
        retrain_interval_days=14,
    )

    assert result.evaluated_days > 0
    assert result.evaluated_races > 0
    assert 0.0 <= result.top1_hit_rate <= 1.0
    assert 0.0 <= result.top2_hit_rate <= 1.0
    assert 0.0 <= result.top3_hit_rate <= 1.0
    assert result.log_loss > 0.0
    assert result.brier > 0.0
    assert len(result.alpha_comparison) == 4
    assert result.paired_alpha_comparison["races"] >= 0
    assert len(result.paired_alpha_comparison["paired_difference_bootstrap_ci"]) == 2
    assert result.paired_alpha_comparison["conclusion"] in {"evidence_for_higher_alpha", "not_established"}
    assert len(result.bonferroni_paired_alpha_comparison["paired_difference_bootstrap_ci"]) == 2
    assert result.nested_alpha_validation["selected_alpha"] in {0.0, 0.2, 0.4, 0.6}
    assert result.nested_alpha_validation["selection_dates"]
    assert result.nested_alpha_validation["validation_dates"]
    assert set(result.feature_market_correlations) == {
        "jockey_change_upgrade",
        "trainer_change_upgrade",
        "class_drop_flag",
        "workout_sudden_improvement",
        "rest_optimal_fit",
    }
    assert all(-1.0 <= value <= 1.0 for value in result.feature_market_correlations.values())
    for alpha_result in result.alpha_comparison:
        assert 0.0 <= alpha_result["alpha_stage1"] <= 1.0
        assert 0.0 <= alpha_result["top4_hit_rate"] <= 1.0
        assert 0.0 <= alpha_result["longshot_top4_hit_rate"] <= 1.0
        assert len(alpha_result["top4_vs_market_bootstrap_ci"]) == 2

    for train_max, test_day in zip(result.fold_max_train_date, result.fold_test_date):
        assert train_max < test_day
