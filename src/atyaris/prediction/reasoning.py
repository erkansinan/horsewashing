"""Her at icin skor kirilimina dayali dogal dilde gerekce metinleri ve ozet
etiket (tag) uretir.
"""
from __future__ import annotations

from datetime import date

from atyaris.models.entities import HorseStatistics, RaceEntry, ScoreBreakdown, TrackSurface


def _form_bullet(stats: HorseStatistics, window: int) -> str:
    recent = stats.recent_form(window)
    if not recent:
        return "Gecmis kosu verisi bulunamadi; form durumu belirsiz."
    placed = sum(1 for p in recent if p.is_placed)
    wins = sum(1 for p in recent if p.is_win)
    return f"Son {len(recent)} kosunun {placed} tanesinde ilk 3'e girdi ({wins} galibiyet dahil)."


def _jockey_trainer_bullet(stats: HorseStatistics, entry: RaceEntry) -> str:
    if stats.jockey_horse_combo_starts:
        rate = stats.jockey_horse_combo_wins / stats.jockey_horse_combo_starts * 100
        return (
            f"{entry.jockey.name}, bu atla daha once {stats.jockey_horse_combo_starts} kez "
            f"kosti ve %{rate:.0f} kazanma oranina sahip."
        )
    return f"{entry.jockey.name} ile ilk kez esleşiyor; jokey-at kombinasyon gecmisi yok."


def _distance_surface_bullet(
    stats: HorseStatistics, race_distance: int, race_surface: TrackSurface
) -> str:
    matching = [
        p
        for p in stats.past_performances
        if p.surface == race_surface and abs(p.distance_m - race_distance) <= 200
    ]
    if not matching:
        return f"Bu mesafede ({race_distance}m) ve pistte ({race_surface.value}) daha once kosmadi."
    wins = sum(1 for p in matching if p.is_win)
    return (
        f"Bu mesafe/pist kombinasyonunda ({race_distance}m, {race_surface.value}) "
        f"{len(matching)} kosuda {wins} galibiyet aldi."
    )


def _weight_bullet(stats: HorseStatistics, entry: RaceEntry) -> str:
    historical = [p.weight_kg for p in stats.past_performances if p.weight_kg]
    if not historical:
        return f"Bu kosudaki kilosu {entry.weight_kg} kg; gecmis kilo verisi yok."
    avg_weight = sum(historical) / len(historical)
    diff = round(avg_weight - entry.weight_kg, 1)
    if diff > 0.3:
        return f"Bu kosudaki kilosu gecmis ortalamasina gore {diff} kg daha hafif — avantajli."
    if diff < -0.3:
        return f"Bu kosudaki kilosu gecmis ortalamasina gore {abs(diff)} kg daha agir — dezavantajli olabilir."
    return "Bu kosudaki kilosu gecmis ortalamasiyla benzer."


def _rest_bullet(
    stats: HorseStatistics,
    ideal_min: int,
    ideal_max: int,
    race_date: date | None,
) -> str:
    if race_date is None:
        race_date = date.today()
    past_dates = [p.race_date for p in stats.past_performances if p.race_date < race_date]
    days = (race_date - max(past_dates)).days if past_dates else None
    if days is None:
        return "Son kosu tarihi bilinmiyor; dinlenme suresi degerlendirilemedi."
    if days < ideal_min:
        return f"Son kosusundan bu yana yalnizca {days} gun gecti; yorgunluk riski olabilir."
    if days > ideal_max:
        return f"Son kosusundan bu yana {days} gun gecti; uzun ara sonrasi form riski tasiyor."
    return f"Son kosusundan bu yana {days} gun gecmis; ideal dinlenme araliginda."


def build_reasoning(
    entry: RaceEntry,
    stats: HorseStatistics,
    race_distance: int,
    race_surface: TrackSurface,
    race_date: date | None,
    window: int,
    ideal_rest_min: int,
    ideal_rest_max: int,
) -> list[str]:
    """Bir at icin maddeler halinde, dogal dilde gerekce listesi olusturur."""
    return [
        _form_bullet(stats, window),
        _jockey_trainer_bullet(stats, entry),
        _distance_surface_bullet(stats, race_distance, race_surface),
        _weight_bullet(stats, entry),
        _rest_bullet(stats, ideal_rest_min, ideal_rest_max, race_date),
    ]


def build_tag(
    score: ScoreBreakdown, entry: RaceEntry, rank: int, is_market_favorite: bool
) -> str:
    """Siralamaya, piyasa oranina ve skora gore ozet bir etiket belirler."""
    if rank == 1:
        return "Kazanan Aday"
    if rank <= 3:
        return "Plase Adayi"
    if entry.odds and entry.odds >= 10 and score.total_score >= 55:
        return "Surpriz Olabilir"
    if is_market_favorite and score.total_score < 45:
        return "Kacinilmasi Onerilen Favori"
    return "Diger"
