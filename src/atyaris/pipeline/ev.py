"""EV katmani: model olasiliklarini oranlarla karsilastirip value bet sinyali uretir."""
from __future__ import annotations

from dataclasses import dataclass

from atyaris.models.entities import HorsePrediction, ValueBetSignal


@dataclass
class EVConfig:
    min_edge: float = 0.03
    min_ev: float = 0.02
    min_confidence: float = 0.45
    fractional_kelly: float = 0.35
    max_kelly_fraction: float = 0.25


def _kelly_fraction(probability: float, odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    value = (probability * odds - 1.0) / (odds - 1.0)
    return max(0.0, value)


def apply_ev_layer(predictions: list[HorsePrediction], config: EVConfig) -> None:
    for hp in predictions:
        odds = hp.entry.odds if hp.entry.odds and hp.entry.odds > 1.0 else None
        probability = hp.win_probability

        if odds is None or probability is None:
            hp.value_bet = ValueBetSignal(is_value_bet=False)
            continue

        implied = 1.0 / odds
        ev = probability * odds - 1.0
        edge = probability - implied

        kelly = _kelly_fraction(probability, odds)
        kelly = min(kelly, config.max_kelly_fraction)
        fractional = kelly * config.fractional_kelly

        confidence = hp.confidence_score or 0.0
        is_value = edge >= config.min_edge and ev >= config.min_ev and confidence >= config.min_confidence

        hp.value_bet = ValueBetSignal(
            implied_probability=implied,
            expected_value=ev,
            edge=edge,
            kelly_fraction=kelly,
            fractional_kelly_stake=fractional,
            is_value_bet=is_value,
        )
