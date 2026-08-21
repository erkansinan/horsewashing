"""Bir atin gecmis istatistiklerini 0-100 araliginda, kategori bazinda
kirilmis bir guven skoruna cevirdigi agirlikli puanlama modeli. Agirliklar
``atyaris.config.ScoringWeights`` uzerinden yapilandirilabilir.
"""
from __future__ import annotations

from datetime import date
import math

from atyaris.config import Settings
from atyaris.models.entities import HorseStatistics, RaceEntry, ScoreBreakdown, TrackSurface

_POSITION_POINTS = {1: 100, 2: 80, 3: 65}
_DEFAULT_POSITION_POINTS = 30


def _position_points(position: int | None, field_size: int | None) -> float:
    """Bitis sirasini 0-100 arasi bir puana cevirir (1. = 100, azalan skala)."""
    if position is None:
        return _DEFAULT_POSITION_POINTS
    if position in _POSITION_POINTS:
        return _POSITION_POINTS[position]
    size = field_size or 10
    # Ilk 3'un disindaki siralar icin dogrusal azalma, taban 5 puan.
    return max(5.0, 30.0 * (1 - (position - 3) / max(size - 3, 1)))


def score_form(stats: HorseStatistics, window: int) -> float:
    """Son ``window`` kosunun, yakinlik agirlikli ortalama formunu puanlar."""
    recent = stats.recent_form(window)
    if not recent:
        return 40.0  # Bilinmeyen form icin notr-dusuk bir skor.
    weights = [window - i for i in range(len(recent))]
    points = [_position_points(p.finish_position, p.field_size) for p in recent]
    return sum(w * p for w, p in zip(weights, points)) / sum(weights)


def score_jockey_trainer(stats: HorseStatistics, entry: RaceEntry) -> float:
    """Jokey-at kombinasyon basarisi ve kariyer kazanma oranini birlestirir."""
    combo_rate = None
    if stats.jockey_horse_combo_starts:
        combo_rate = stats.jockey_horse_combo_wins / stats.jockey_horse_combo_starts * 100
    base = stats.career_win_rate
    if combo_rate is not None:
        return min(100.0, 0.5 * base + 0.5 * combo_rate * 3)
    return min(100.0, base * 3)


def score_distance_surface(
    stats: HorseStatistics, race_distance: int, race_surface: TrackSurface
) -> float:
    """Ayni mesafe araligi (+/-200m) ve pist tipindeki gecmis performansi puanlar."""
    matching = [
        p
        for p in stats.past_performances
        if p.surface == race_surface and abs(p.distance_m - race_distance) <= 200
    ]
    if not matching:
        return 45.0  # Dogrudan gecmis olmadan notr bir skor.
    points = [_position_points(p.finish_position, p.field_size) for p in matching]
    return sum(points) / len(points)


def score_weight(stats: HorseStatistics, current_weight: float) -> float:
    """Bugunku kiloyu, atin gecmis ortalama kilosuna gore puanlar (hafif = avantaj)."""
    historical_weights = [p.weight_kg for p in stats.past_performances if p.weight_kg]
    if not historical_weights:
        return 50.0
    avg_weight = sum(historical_weights) / len(historical_weights)
    diff = avg_weight - current_weight  # Pozitif => ortalamadan daha hafif tasiyor.
    # Dogrusal model 0/100'e hizli saturasyon yapiyordu; tanh ile yumusak
    # bir egri kullanip puanlari 10-90 bandinda tutuyoruz.
    score = 50.0 + 40.0 * math.tanh(diff / 2.0)
    return max(10.0, min(90.0, score))


def score_rest(
    stats: HorseStatistics,
    ideal_min: int,
    ideal_max: int,
    reference_date: date | None = None,
) -> float:
    """Son kosudan bu yana gecen sureyi parcali kondisyon formuluyle puanlar.

    Kullanilan formulu (G = gun):
    - 0 < G < 14: 40 + (G * 3.5)
    - 14 <= G <= 28: 100 - (|G - 21| * 1.5)
    - 28 < G <= 90: 90 - ((G - 28) * 0.9)
    - G > 90: max(20, 35 - ((G - 90) * 0.1))

    Not: Sonuc her zaman en yakin tam sayiya yuvarlanir.
    """
    _ = (ideal_min, ideal_max)  # Geriye donuk API uyumlulugu icin korunuyor.
    if reference_date is None:
        reference_date = date.today()
    past_dates = [p.race_date for p in stats.past_performances if p.race_date < reference_date]
    days = (reference_date - max(past_dates)).days if past_dates else None
    if days is None:
        return 50.0

    if days <= 0:
        raw_score = 40.0
    elif days < 14:
        raw_score = 40.0 + (days * 3.5)
    elif days <= 28:
        raw_score = 100.0 - (abs(days - 21) * 1.5)
    elif days <= 90:
        raw_score = 90.0 - ((days - 28) * 0.9)
    else:
        raw_score = max(20.0, 35.0 - ((days - 90) * 0.1))

    rounded = math.floor(raw_score + 0.5)
    return float(max(0, min(100, rounded)))


def compute_score(
    entry: RaceEntry,
    stats: HorseStatistics,
    race_distance: int,
    race_surface: TrackSurface,
    settings: Settings,
    race_date: date | None = None,
) -> ScoreBreakdown:
    """Tum bilesenleri hesaplayip yapilandirilmis agirliklarla toplam skoru uretir."""
    weights = settings.weights
    form = score_form(stats, settings.recent_form_window)
    jockey_trainer = score_jockey_trainer(stats, entry)
    distance_surface = score_distance_surface(stats, race_distance, race_surface)
    weight = score_weight(stats, entry.weight_kg)
    rest = score_rest(stats, settings.ideal_rest_days_min, settings.ideal_rest_days_max, race_date)

    total = (
        form * weights.form
        + jockey_trainer * weights.jockey_trainer
        + distance_surface * weights.distance_surface
        + weight * weights.weight
        + rest * weights.rest
    )
    return ScoreBreakdown(
        form_score=round(form, 1),
        jockey_trainer_score=round(jockey_trainer, 1),
        distance_surface_score=round(distance_surface, 1),
        weight_score=round(weight, 1),
        rest_score=round(rest, 1),
        total_score=round(total, 1),
    )
