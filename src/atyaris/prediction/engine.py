"""Tahmin motoru: at istatistiklerini cekme, puanlama ve siralanmis/gerekceli
bir tahmin uretme adimlarini orkestre eder.
"""
from __future__ import annotations

import logging

from atyaris.config import Settings
from atyaris.data_sources.base import RaceDataSource
from atyaris.models.entities import HorsePrediction, Race, RaceEntry, RacePrediction, ScoreBreakdown
from atyaris.prediction.reasoning import build_reasoning, build_tag
from atyaris.prediction.scoring import compute_score

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Bir yaris icin at istatistiklerini toplayip agirlikli skor ve gerekce uretir."""

    def __init__(self, data_source: RaceDataSource, settings: Settings) -> None:
        self._data_source = data_source
        self._settings = settings

    @staticmethod
    def _normalize_low_better(value: float | None, minimum: float, maximum: float, neutral: float = 55.0) -> float:
        if value is None:
            return neutral
        span = max(maximum - minimum, 1e-9)
        return max(0.0, min(100.0, (maximum - value) / span * 100.0))

    @staticmethod
    def _normalize_high_better(value: float | None, minimum: float, maximum: float, neutral: float = 50.0) -> float:
        if value is None:
            return neutral
        span = max(maximum - minimum, 1e-9)
        return max(0.0, min(100.0, (value - minimum) / span * 100.0))

    @staticmethod
    def _form_positions_score(positions: list[int]) -> float:
        if not positions:
            return 50.0
        table = {
            1: 100.0,
            2: 86.0,
            3: 74.0,
            4: 63.0,
            5: 54.0,
            6: 46.0,
            7: 38.0,
            8: 30.0,
            9: 22.0,
        }
        vals = [table.get(pos, 14.0) for pos in positions[:6]]
        return sum(vals) / len(vals)

    def _fallback_rank_without_stats(self, race: Race, entries: list[RaceEntry]) -> list[HorsePrediction]:
        """At gecmisi alinamadiginda, mevcut kosu kartindan yedek tahmin uretir.

        Kullanilan sinyaller: son form satiri, handikap puani (HP), piyasa orani
        (ganyan, varsa) ve kilo. Bu yol yalnizca tum atlar icin
        ``get_horse_statistics`` basarisiz oldugunda devreye girer.
        """
        known_odds = [e.odds for e in entries if e.odds and e.odds > 0]
        min_odds = min(known_odds) if known_odds else 1.0
        max_odds = max(known_odds) if known_odds else 20.0

        known_weights = [e.weight_kg for e in entries if e.weight_kg > 0]
        min_weight = min(known_weights) if known_weights else 50.0
        max_weight = max(known_weights) if known_weights else 62.0

        known_hp = [e.handicap_points for e in entries if e.handicap_points is not None]
        min_hp = min(known_hp) if known_hp else 40.0
        max_hp = max(known_hp) if known_hp else 90.0

        raw: list[tuple[RaceEntry, ScoreBreakdown, list[str]]] = []
        for entry in entries:
            form_score = self._form_positions_score(entry.recent_form_positions)
            hp_score = self._normalize_high_better(entry.handicap_points, min_hp, max_hp, neutral=50.0)
            odds_score = self._normalize_low_better(entry.odds, min_odds, max_odds, neutral=55.0)
            weight_score = self._normalize_low_better(entry.weight_kg, min_weight, max_weight, neutral=50.0)

            components: list[tuple[float, float]] = []
            components.append((form_score, 0.40))
            components.append((hp_score, 0.25))
            components.append((weight_score, 0.15))
            if entry.odds and entry.odds > 0:
                components.append((odds_score, 0.20))

            total_weight = sum(w for _, w in components) or 1.0
            total = round(sum(score * w for score, w in components) / total_weight, 1)
            score = ScoreBreakdown(
                form_score=round(form_score, 1),
                jockey_trainer_score=round(hp_score, 1),
                distance_surface_score=round(odds_score, 1),
                weight_score=round(weight_score, 1),
                rest_score=50.0,
                total_score=total,
            )

            reasons = [
                "At gecmisi/istatistik verisi alinamadigi icin yedek tahmin modu kullanildi.",
                (
                    f"Son form satiri ({entry.form_raw}) orta puani: {form_score:.1f}."
                    if entry.form_raw
                    else "Son form satiri bulunamadi; form puani notr alindi."
                ),
                (
                    f"Handikap puani (HP) {entry.handicap_points:.1f}; saha icinde goreli guc sinyali olarak kullanildi."
                    if entry.handicap_points is not None
                    else "Handikap puani (HP) bulunamadi; bu bilesen notr puanlandi."
                ),
                (
                    f"Piyasa orani (ganyan) {entry.odds:.2f}; daha dusuk oran daha guclu aday kabul edilir."
                    if entry.odds
                    else "Piyasa orani bilgisi yok; bu at icin notr oran puani kullanildi."
                ),
                f"Tasidigi kilo {entry.weight_kg:.1f} kg; saha icindeki goreli kilo avantaji hesaba katildi.",
                "Mesafe/pist/jokey gecmisi olmadigindan kalan bilesenler notr kabul edildi.",
            ]
            raw.append((entry, score, reasons))

        raw.sort(key=lambda item: item[1].total_score, reverse=True)

        odds_ranked = sorted((e for e in entries if e.odds), key=lambda e: e.odds)  # type: ignore[arg-type,return-value]
        favorite_id = odds_ranked[0].horse_id if odds_ranked else None

        ranked: list[HorsePrediction] = []
        for rank, (entry, score, reasons) in enumerate(raw, start=1):
            tag = build_tag(score, entry, rank, entry.horse_id == favorite_id)
            ranked.append(HorsePrediction(entry=entry, score=score, reasoning=reasons, tag=tag))
        self._append_component_leader_notes(ranked)
        return ranked

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

    def predict(self, race: Race) -> RacePrediction:
        entries = race.active_entries
        if not entries:
            raise ValueError(f"Kosu {race.race_no} icin aktif at bulunamadi.")

        # Piyasa oranina (ganyan) gore en favori ati belirle; daha sonra
        # modelin piyasayla uyusmadigi durumlari etiketlemek icin kullanilir.
        odds_ranked = sorted((e for e in entries if e.odds), key=lambda e: e.odds)  # type: ignore[arg-type,return-value]
        favorite_id = odds_ranked[0].horse_id if odds_ranked else None

        scored = []
        failed_stats = 0
        first_failure: str | None = None
        for entry in entries:
            try:
                stats = self._data_source.get_horse_statistics(entry)
            except Exception as exc:  # noqa: BLE001
                failed_stats += 1
                if first_failure is None:
                    first_failure = str(exc)
                logger.debug("At istatistikleri alinamadi (%s): %s", entry.horse_name, exc)
                continue
            score = compute_score(
                entry,
                stats,
                race.distance_m,
                race.surface,
                self._settings,
                race_date=race.start_time.date(),
            )
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
            scored.append((entry, score, reasoning))

        if failed_stats:
            logger.warning(
                "Kosu %s icin %d/%d atin istatistigi alinamadi. Ilk hata: %s",
                race.race_no,
                failed_stats,
                len(entries),
                first_failure or "Bilinmeyen hata",
            )

        if not scored:
            logger.warning(
                "Kosu %s icin tum istatistikler eksik; yedek tahmin moduna geciliyor.",
                race.race_no,
            )
            fallback_ranked = self._fallback_rank_without_stats(race, entries)
            return RacePrediction(
                race=race,
                ranked=fallback_ranked,
                disclaimer=(
                    "Bu tahmin, at gecmis istatistikleri olmadan yedek modda uretilmistir; "
                    "kesinlik tasimaz, sorumlu bahis oynayin."
                ),
            )

        scored.sort(key=lambda item: item[1].total_score, reverse=True)

        ranked: list[HorsePrediction] = []
        for rank, (entry, score, reasoning) in enumerate(scored, start=1):
            is_market_favorite = entry.horse_id == favorite_id
            tag = build_tag(score, entry, rank, is_market_favorite)
            ranked.append(HorsePrediction(entry=entry, score=score, reasoning=reasoning, tag=tag))

        self._append_component_leader_notes(ranked)

        return RacePrediction(race=race, ranked=ranked)
