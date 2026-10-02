from __future__ import annotations

from datetime import date

import pandas as pd

from atyaris.ml.backtest import (
    _alpha_comparison,
    _harville_place_calibration_comparison,
    _market_ranking_divergence_test,
    _ranking_order_metrics,
    walk_forward_backtest,
)
from atyaris.ml.provider import SyntheticRacingDataProvider as _FixtureRacingDataProvider


def test_harville_calibration_report_compares_raw_and_gamma_top3_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "fold_test_date": ["2025-01-01"] * 4,
            "place2_gamma": [0.7] * 4,
            "place3_gamma": [0.8] * 4,
            "finish_position": [1, 2, 3, 4],
            "odds_slice": ["favorite", "middle", "longshot", "longshot"],
            "harville_place2_raw": [0.4, 0.3, 0.2, 0.1],
            "place2_probability": [0.3, 0.3, 0.25, 0.15],
            "harville_place3_raw": [0.3, 0.3, 0.25, 0.15],
            "place3_probability": [0.25, 0.3, 0.25, 0.2],
            "harville_top3_raw": [0.8, 0.75, 0.6, 0.3],
            "harville_top3_gamma": [0.7, 0.75, 0.65, 0.4],
        }
    )

    report = _harville_place_calibration_comparison(frame)

    assert report["fold_gammas"][0]["place2_gamma"] == 0.7
    assert report["positions"]["top3"]["all"]["raw"]["brier"] != report[
        "positions"
    ]["top3"]["all"]["gamma_adjusted"]["brier"]
    assert report["positions"]["top3"]["longshot"]["rows"] == 2


def test_ranking_agreement_separates_exact_order_same_set_and_different_set() -> None:
    rows = []
    cases = {
        "exact": ([0.5, 0.4, 0.3, 0.2, 0.1], [2.0, 3.0, 4.0, 5.0, 6.0]),
        "same_set": ([0.4, 0.5, 0.3, 0.2, 0.1], [2.0, 3.0, 4.0, 5.0, 6.0]),
        "different_set": ([0.2, 0.5, 0.4, 0.3, 0.6], [2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    for race_id, (probabilities, odds) in cases.items():
        rows.extend(
            {
                "race_id": race_id,
                "horse_id": f"H{index}",
                "calibrated_probability": probability,
                "odds": price,
            }
            for index, (probability, price) in enumerate(zip(probabilities, odds), start=1)
        )

    result = _ranking_order_metrics(pd.DataFrame(rows), "calibrated_probability")

    assert result["eligible_races"] == 3
    assert result["exact_order_matches"] == 1
    assert result["same_top4_set_different_order_races"] == 1
    assert result["different_top4_set_races"] == 1
    assert result["exact_order_match_rate"] == 1 / 3


def test_market_ranking_guard_blocks_production_at_ninety_percent() -> None:
    result = _market_ranking_divergence_test(
        {"exact_order_match_rate": 0.90, "eligible_races": 100}
    )

    assert result["test_name"] == "test_model_diverges_from_market_ranking"
    assert result["status"] == "fail"
    assert result["warning"] == "model_equals_odds_sort"
    assert result["production_allowed"] is False


def test_market_only_alpha_is_exactly_the_odds_sort_baseline() -> None:
    odds = [2.0, 3.0, 4.0, 5.0, 6.0]
    inverse_odds = [1.0 / price for price in odds]
    total = sum(inverse_odds)
    frame = pd.DataFrame(
        {
            "race_id": ["R1"] * 5,
            "date": [date(2025, 1, 1)] * 5,
            "horse_id": [f"H{index}" for index in range(1, 6)],
            "odds": odds,
            "market_probability_used": [value / total for value in inverse_odds],
            "odds_market_probability": [value / total for value in inverse_odds],
            "stage1_probability": [0.05, 0.10, 0.20, 0.25, 0.40],
            "is_winner": [1, 0, 0, 0, 0],
            "odds_slice": ["favorite", "favorite", "middle", "middle", "longshot"],
        }
    )

    result = _alpha_comparison(frame)
    market_only = result[0]

    assert market_only["alpha_stage1"] == 0.0
    assert market_only["ranking_exact_order_match_percent"] == 100.0
    assert market_only["ranking_same_top4_set_different_order_rate"] == 0.0
    assert market_only["ranking_different_top4_set_rate"] == 0.0
    assert market_only["top4_hit_delta_vs_odds_sort"] == 0.0
    assert market_only["top4_vs_odds_sort_evaluation"] == "descriptive_only"


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
    assert result.place_log_loss > 0.0
    assert result.place_brier > 0.0
    assert result.harville_place_calibration["fold_gammas"]
    assert set(result.harville_place_calibration["positions"]) == {"second", "third", "top3"}
    for place_name in ("second", "third", "top3"):
        assert "all" in result.harville_place_calibration["positions"][place_name]
        assert "longshot" in result.harville_place_calibration["positions"][place_name]
    assert 0.0 <= result.highest_odds_placer_metrics["model_exact_match_rate"] <= 1.0
    assert result.highest_odds_placer_metrics["eligible_races"] > 0
    assert result.highest_odds_placer_metrics["market_baseline_exact_match_rate"] >= 0.0
    assert set(result.race_highest_odds_placer_exact_match) == set(
        result.race_market_highest_odds_placer_exact_match
    )
    assert len(result.alpha_comparison) == 5
    assert result.odds_probability_vs_odds_ranking["exact_order_match_percent"] == 100.0
    assert sum(
        result.model_vs_odds_ranking[key]
        for key in (
            "exact_order_matches",
            "same_top4_set_different_order_races",
            "different_top4_set_races",
        )
    ) == result.model_vs_odds_ranking["eligible_races"]
    assert result.test_model_diverges_from_market_ranking["test_name"] == (
        "test_model_diverges_from_market_ranking"
    )
    assert result.production_readiness["production_allowed"] is False
    assert "final_model_vs_odds_sort" in result.nested_alpha_validation
    assert result.paired_alpha_comparison["races"] >= 0
    assert len(result.paired_alpha_comparison["paired_difference_bootstrap_ci"]) == 2
    assert result.paired_alpha_comparison["conclusion"] in {"evidence_for_higher_alpha", "not_established"}
    assert len(result.bonferroni_paired_alpha_comparison["paired_difference_bootstrap_ci"]) == 2
    assert result.nested_alpha_validation["selected_alpha"] in {0.0, 0.2, 0.4, 0.6, 1.0}
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
        assert 0.0 <= alpha_result["ranking_exact_order_match_percent"] <= 100.0
        assert 0.0 <= alpha_result["top4_hit_rate"] <= 1.0
        assert 0.0 <= alpha_result["longshot_top4_hit_rate"] <= 1.0
        assert alpha_result["top4_vs_odds_sort_evaluation"] == "descriptive_only"

    for train_max, test_day in zip(result.fold_max_train_date, result.fold_test_date):
        assert train_max < test_day
