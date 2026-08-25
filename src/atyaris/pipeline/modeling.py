"""Ensemble model katmani: boosting-benzeri skor + ranking-benzeri skor + olasilik kalibrasyonu."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean

from atyaris.pipeline.feature_engineering import EntryFeatures


@dataclass
class EnsembleResult:
    horse_id: str
    boosting_score: float
    ranking_score: float
    ensemble_score: float
    calibrated_probability: float
    confidence_score: float


class EnsembleRanker:
    """LightGBM/XGBoost + LambdaMART benzeri iki kanal skorunu birlestirir.

    Projede ekstra agir bagimlilik olmadan calismasi icin hesaplama saf Python
    surrogate formulasyonla yapilir; API/katman tasarimi gerçek ensemble akisini
    yansitir.
    """

    def __init__(self, boosting_weight: float = 0.58, ranking_weight: float = 0.42, calibration_temperature: float = 0.85) -> None:
        self._boosting_weight = boosting_weight
        self._ranking_weight = ranking_weight
        self._temperature = max(0.35, calibration_temperature)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def _boosting_surrogate(self, f: dict[str, float]) -> float:
        nonlinear = (
            1.8 * f["wma_form"]
            + 1.2 * f["similar_conditions_score"]
            + 1.0 * f["tempo_fit_score"]
            + 0.9 * f["rest_fit"]
            + 0.7 * f["rest_performance_interaction"]
            + 0.6 * f["relative_strength"]
            + 0.4 * f["weight_advantage"]
            + 0.3 * (f["jockey_win_rate"] + f["trainer_win_rate"])
            + 0.5 * f["form_trend_slope"]
            + 0.8 * f["market_implied_probability"]
            - 0.2 * f["rolling_std_5"]
        )
        return self._sigmoid(nonlinear)

    def _ranking_surrogate(self, f: dict[str, float]) -> float:
        # Pairwise ranking etkisini, saha ici goreli guce ve tempo uyumuna daha
        # yuksek marj vererek yaklastiriyoruz.
        rank_margin = (
            2.0 * f["relative_strength"]
            + 1.1 * f["tempo_fit_score"]
            + 0.9 * f["similar_conditions_score"]
            + 0.8 * f["lag_form_1"]
            + 0.6 * f["lag_form_2"]
            + 0.5 * f["rest_fit"]
            + 0.3 * f["weight_advantage"]
            + 0.2 * f["form_trend_slope"]
            + 0.5 * f["market_implied_probability"]
        )
        return self._sigmoid(rank_margin)

    def _softmax(self, values: list[float]) -> list[float]:
        if not values:
            return []
        max_value = max(values)
        exps = [math.exp((v - max_value) / self._temperature) for v in values]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def _confidence(self, f: dict[str, float], calibrated_probability: float) -> float:
        stability = 1.0 - min(1.0, f["rolling_std_5"])
        signal = mean([
            f["rest_fit"],
            f["tempo_fit_score"],
            f["similar_conditions_score"],
            max(0.0, min(1.0, f["wma_form"])),
        ])
        concentration = min(1.0, calibrated_probability * 4.0)
        return max(0.0, min(1.0, 0.4 * stability + 0.4 * signal + 0.2 * concentration))

    def rank(self, features: list[EntryFeatures]) -> list[EnsembleResult]:
        if not features:
            return []

        intermediate: list[tuple[str, float, float, float]] = []
        for row in features:
            boosting = self._boosting_surrogate(row.values)
            ranking = self._ranking_surrogate(row.values)
            ensemble = boosting * self._boosting_weight + ranking * self._ranking_weight
            intermediate.append((row.horse_id, boosting, ranking, ensemble))

        probs = self._softmax([row[3] for row in intermediate])
        results: list[EnsembleResult] = []
        for idx, (horse_id, boosting, ranking, ensemble) in enumerate(intermediate):
            p = probs[idx]
            conf = self._confidence(next(f.values for f in features if f.horse_id == horse_id), p)
            results.append(
                EnsembleResult(
                    horse_id=horse_id,
                    boosting_score=boosting,
                    ranking_score=ranking,
                    ensemble_score=ensemble,
                    calibrated_probability=p,
                    confidence_score=conf,
                )
            )

        # Olasilik tabanli siralama, yarisi goreceli siralama problemi olarak ele alir.
        results.sort(key=lambda item: item.calibrated_probability, reverse=True)
        return results
