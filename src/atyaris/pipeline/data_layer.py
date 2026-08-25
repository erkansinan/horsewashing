"""Veri katmani: ham istatistikleri toplar ve eksik/gurultulu alanlari imputasyonla normalize eder."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import mean
from typing import Optional

from atyaris.data_sources.base import RaceDataSource
from atyaris.models.entities import HorseStatistics, PastPerformance, Race, RaceEntry, TrackSurface


@dataclass
class PreparedRaceData:
    race: Race
    stats_by_horse_id: dict[str, HorseStatistics]
    missing_stats_entries: list[RaceEntry]
    imputation_notes: list[str]


def _impute_past_performance(
    perf: PastPerformance,
    average_weight: float,
    fallback_field_size: int,
) -> PastPerformance:
    finish_position = perf.finish_position if perf.finish_position is not None else max(4, fallback_field_size // 2)
    field_size = perf.field_size if perf.field_size is not None else fallback_field_size
    field_size = max(field_size, finish_position)
    weight_kg = perf.weight_kg if perf.weight_kg is not None else average_weight

    # Zaman/split verisi yoksa ortalama-tempo profile dogru notr imputasyon.
    early = perf.early_pace_index if perf.early_pace_index is not None else 0.5
    mid = perf.mid_pace_index if perf.mid_pace_index is not None else 0.5
    late = perf.late_pace_index if perf.late_pace_index is not None else 0.5

    return perf.model_copy(
        update={
            "finish_position": finish_position,
            "field_size": field_size,
            "weight_kg": weight_kg,
            "early_pace_index": max(0.0, min(1.0, early)),
            "mid_pace_index": max(0.0, min(1.0, mid)),
            "late_pace_index": max(0.0, min(1.0, late)),
        }
    )


def _impute_horse_statistics(entry: RaceEntry, stats: HorseStatistics | None, race_date: date) -> tuple[HorseStatistics, list[str]]:
    notes: list[str] = []
    if stats is None:
        notes.append(
            f"{entry.horse_name}: At bazli gecmis istatistik gelmedi; form satiri, oran ve notr degerlerle fallback profil uretildi."
        )
        fallback_perfs = [
            PastPerformance(
                race_date=race_date,
                hippodrome="Bilinmiyor",
                distance_m=1400,
                surface=TrackSurface.KUM,
                finish_position=pos,
                field_size=max(8, pos),
                jockey_name=entry.jockey.name,
                trainer_name=entry.trainer.name,
                weight_kg=entry.weight_kg,
                odds=entry.odds,
            )
            for pos in (entry.recent_form_positions[:5] or [5, 5, 6])
        ]

        synthetic = HorseStatistics(
            horse_id=entry.horse_id,
            horse_name=entry.horse_name,
            past_performances=fallback_perfs,
            career_starts=len(fallback_perfs),
            career_wins=sum(1 for p in fallback_perfs if p.finish_position == 1),
            career_places=sum(1 for p in fallback_perfs if (p.finish_position or 99) <= 3),
            last_year_starts=len(fallback_perfs),
            last_year_wins=sum(1 for p in fallback_perfs if p.finish_position == 1),
            last_year_places=sum(1 for p in fallback_perfs if (p.finish_position or 99) <= 3),
            jockey_horse_combo_starts=max(1, len(fallback_perfs) // 2),
            jockey_horse_combo_wins=max(0, sum(1 for p in fallback_perfs if p.finish_position == 1) // 2),
        )
        return synthetic, notes

    perfs = sorted(stats.past_performances, key=lambda p: p.race_date, reverse=True)
    historical_weights = [p.weight_kg for p in perfs if p.weight_kg is not None]
    average_weight = mean(historical_weights) if historical_weights else entry.weight_kg
    fallback_field_size = max([p.field_size or 0 for p in perfs] + [8])

    imputed_perfs = []
    missing_counter = 0
    for perf in perfs:
        before_missing = int(perf.finish_position is None) + int(perf.field_size is None) + int(perf.weight_kg is None)
        if perf.early_pace_index is None or perf.mid_pace_index is None or perf.late_pace_index is None:
            before_missing += 1
        if before_missing:
            missing_counter += 1
        imputed_perfs.append(_impute_past_performance(perf, average_weight, fallback_field_size))

    if missing_counter:
        notes.append(
            f"{entry.horse_name}: {missing_counter} gecmis kayitta eksik/gurultulu alanlar median-notr strateji ile imput edildi."
        )

    imputed_stats = stats.model_copy(update={"past_performances": imputed_perfs})
    return imputed_stats, notes


def prepare_race_data(data_source: RaceDataSource, race: Race) -> PreparedRaceData:
    """Yaris icin tum at istatistiklerini toplayip imputasyon uygular."""
    stats_by_horse_id: dict[str, HorseStatistics] = {}
    missing_entries: list[RaceEntry] = []
    imputation_notes: list[str] = []

    for entry in race.active_entries:
        raw_stats: Optional[HorseStatistics]
        try:
            raw_stats = data_source.get_horse_statistics(entry)
        except Exception:
            raw_stats = None
            missing_entries.append(entry)

        imputed_stats, notes = _impute_horse_statistics(entry, raw_stats, race.start_time.date())
        stats_by_horse_id[entry.horse_id] = imputed_stats
        imputation_notes.extend(notes)

    return PreparedRaceData(
        race=race,
        stats_by_horse_id=stats_by_horse_id,
        missing_stats_entries=missing_entries,
        imputation_notes=imputation_notes,
    )
