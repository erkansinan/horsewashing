"""Tahmin motoru: at istatistiklerini cekme, puanlama ve siralanmis/gerekceli
bir tahmin uretme adimlarini orkestre eder.
"""
from __future__ import annotations

import logging

from atyaris.config import Settings
from atyaris.data_sources.base import RaceDataSource
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.models.entities import HorsePrediction, Race, RaceEntry, RacePrediction
from atyaris.pipeline.backtest import build_walk_forward_backtest
from atyaris.pipeline.data_layer import prepare_race_data
from atyaris.pipeline.ev import EVConfig, apply_ev_layer
from atyaris.pipeline.feature_engineering import build_entry_features
from atyaris.pipeline.modeling import EnsembleRanker
from atyaris.prediction.reasoning import build_reasoning, build_tag
from atyaris.prediction.scoring import compute_score

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Bir yaris icin at istatistiklerini toplayip agirlikli skor ve gerekce uretir."""

    def __init__(self, data_source: RaceDataSource, settings: Settings) -> None:
        self._data_source = data_source
        self._settings = settings

    @staticmethod
    def _append_component_leader_notes(ranked: list[HorsePrediction]) -> None:
        if not ranked:
            return

        component_labels = {
            "form_score": "form",
            "jockey_trainer_score": "jokey",
            "distance_surface_score": "pist/mesafe",
            "weight_score": "agirlik",
            "rest_score": "dinlenme",
        }
        comparisons = {name: max(getattr(hp.score, name) for hp in ranked) for name in component_labels}

        for hp in ranked:
            extras: list[str] = []
            for field, label in component_labels.items():
                value = getattr(hp.score, field)
                if value == comparisons[field]:
                    if field == "rest_score":
                        extras.append("Bu at dinlenme acisindan en ideal puana sahip.")
                    elif field == "form_score":
                        extras.append("Bu at form olarak en yuksek puana sahip.")
                    elif field == "jockey_trainer_score":
                        extras.append("Bu at jokey/antrenor uyumu acisindan en yuksek puana sahip.")
                    elif field == "distance_surface_score":
                        extras.append("Bu at pist/mesafe uyumu acisindan en yuksek puana sahip.")
                    elif field == "weight_score":
                        extras.append("Bu at agirlik avantaji acisindan en yuksek puana sahip.")
            hp.reasoning.extend(extras)

    def predict(
        self,
        race: Race,
        *,
        include_backtest: bool = True,
        include_detailed_reasoning: bool = True,
    ) -> RacePrediction:
        entries = race.active_entries
        if not entries:
            raise ValueError(f"Kosu {race.race_no} icin aktif at bulunamadi.")

        prepared = prepare_race_data(self._data_source, race)
        missing_ids = {entry.horse_id for entry in prepared.missing_stats_entries}
        features = build_entry_features(race, prepared.stats_by_horse_id)
        ranker = EnsembleRanker(
            boosting_weight=self._settings.ensemble_boosting_weight,
            ranking_weight=self._settings.ensemble_ranking_weight,
            calibration_temperature=self._settings.calibration_temperature,
        )
        ensemble_ranked = ranker.rank(features)
        ensemble_by_horse = {item.horse_id: item for item in ensemble_ranked}
        feature_by_horse = {row.horse_id: row.values for row in features}
        implied_market_values = [
            1.0 / entry.odds for entry in entries if entry.odds is not None and entry.odds > 0
        ]
        field_implied_mean = (
            sum(implied_market_values) / len(implied_market_values)
            if implied_market_values
            else 0.12
        )

        odds_ranked = sorted((e for e in entries if e.odds), key=lambda e: e.odds)  # type: ignore[arg-type,return-value]
        favorite_id = odds_ranked[0].horse_id if odds_ranked else None

        ranked: list[HorsePrediction] = []
        for entry in entries:
            stats = prepared.stats_by_horse_id[entry.horse_id]
            score = compute_score(
                entry,
                stats,
                race.distance_m,
                race.surface,
                self._settings,
                race_date=race.start_time.date(),
            )
            reasoning = []
            if include_detailed_reasoning:
                reasoning = build_reasoning(
                    entry,
                    stats,
                    race.distance_m,
                    race.surface,
                    race.start_time.date(),
                    self._settings.recent_form_window,
                    self._settings.ideal_rest_days_min,
                    self._settings.ideal_rest_days_max,
                )

            feature_row = feature_by_horse.get(entry.horse_id, {})
            ensemble = ensemble_by_horse.get(entry.horse_id)
            win_probability = ensemble.calibrated_probability if ensemble else None
            confidence = ensemble.confidence_score if ensemble else None

            implied_market_probability = (
                (1.0 / entry.odds) if entry.odds is not None and entry.odds > 0 else field_implied_mean
            )
            base_total = score.total_score / 100.0
            consensus = (
                0.42 * (win_probability if win_probability is not None else base_total)
                + 0.25 * (confidence if confidence is not None else 0.5)
                + 0.23 * base_total
                + 0.10 * implied_market_probability
            )
            form_volatility = float(feature_row.get("rolling_std_5", 0.0))
            stability = max(0.0, min(1.0, 1.0 - min(form_volatility, 0.6) / 0.6))
            adjusted_total = (0.65 * base_total + 0.35 * consensus) * (0.95 + 0.05 * stability)
            score.total_score = round(max(0.0, min(100.0, adjusted_total * 100.0)), 1)

            if include_detailed_reasoning:
                reasoning.extend(
                    [
                        f"Form trend egimi (son N yaris): {feature_row.get('form_trend_slope', 0.0):.3f}.",
                        f"Dinlenme-performans etkilesim puani: {feature_row.get('rest_performance_interaction', 0.0):.3f}.",
                        f"Tempo uyum skoru (erken-orta-gec): {feature_row.get('tempo_fit_score', 0.0):.3f}.",
                        f"Rakip alana gore goreli guc: {feature_row.get('relative_strength', 0.0):.3f}.",
                    ]
                )

            ranked.append(
                HorsePrediction(
                    entry=entry,
                    score=score,
                    reasoning=reasoning,
                    tag="Diger",
                    model_score=(ensemble.boosting_score if ensemble else None),
                    ranking_score=(ensemble.ranking_score if ensemble else None),
                    win_probability=win_probability,
                    confidence_score=confidence,
                    feature_snapshot={
                        "lag_form_1": round(float(feature_row.get("lag_form_1", 0.5)), 3),
                        "lag_form_2": round(float(feature_row.get("lag_form_2", 0.5)), 3),
                        "rolling_mean_3": round(float(feature_row.get("rolling_mean_3", 0.5)), 3),
                        "rolling_std_3": round(float(feature_row.get("rolling_std_3", 0.0)), 3),
                        "rolling_mean_5": round(float(feature_row.get("rolling_mean_5", 0.5)), 3),
                        "rolling_std_5": round(float(feature_row.get("rolling_std_5", 0.0)), 3),
                    },
                )
            )

        ranked.sort(key=lambda hp: hp.score.total_score, reverse=True)
        for rank, hp in enumerate(ranked, start=1):
            hp.tag = build_tag(hp.score, hp.entry, rank, hp.entry.horse_id == favorite_id)
            if hp.entry.horse_id in missing_ids:
                hp.tag = "Veri Eksik"

        apply_ev_layer(
            ranked,
            EVConfig(
                min_edge=self._settings.ev_min_edge,
                min_ev=self._settings.ev_min_value,
                min_confidence=0.45,
                fractional_kelly=self._settings.ev_fractional_kelly,
                max_kelly_fraction=self._settings.ev_max_kelly_fraction,
            ),
        )

        if include_detailed_reasoning:
            self._append_component_leader_notes(ranked)

        if not include_backtest:
            backtest = None
            backtest_note = "Bu cikti hizli modda uretildi; walk-forward backtest bu istekte calistirilmadi."
        elif isinstance(self._data_source, TJKHtmlDataSource):
            backtest = None
            backtest_note = (
                "Canli TJK kaynaginda sayfa JS/AJAX kaynakli oldugu icin walk-forward backtest bu istekte atlandi "
                "(performans ve gereksiz ag cagrilarini azaltmak icin)."
            )
        else:
            backtest = build_walk_forward_backtest(
                self._data_source,
                reference_date=race.start_time.date(),
                settings=self._settings,
                lookback_days=self._settings.backtest_lookback_days,
            )
            backtest_note = (
                "Walk-forward backtest, yalnizca etiketli gecmis sonuclarin deterministik oldugu kaynaklarda otomatik calistirilir."
            )

        model_notes = [
            "Ensemble mimari kullanildi: boosting-benzeri skor + ranking-benzeri skor birlestirildi.",
            "Cikti ham skor degil, softmax + temperature scaling ile kalibre edilmis kazanma olasiligidir.",
            "Yaris problemi mutlak regresyondan cok goreceli siralama oldugu icin ranking kanali agirliklandirildi.",
            "Nihai siralamada sapmayi azaltmak icin model olasiligi + guven + baz skor + piyasa olasiligi ile konservatif bir consensus duzeltmesi uygulandi.",
            (
                f"Backtest leakage-safe modu: acik (as_of kesiti, en yeni {self._settings.backtest_exclude_recent_races} yaris dislanir)."
                if self._settings.backtest_leakage_safe_mode
                else "Backtest leakage-safe modu: kapali (tum gecmis kayitlar kullanilir)."
            ),
            "Eksik/gurultulu veri median/notr imputasyon ile tamamlandi; notlar asagida listelendi.",
            backtest_note,
        ]

        return RacePrediction(
            race=race,
            ranked=ranked,
            model_notes=model_notes,
            imputation_notes=prepared.imputation_notes,
            backtest=backtest,
        )
