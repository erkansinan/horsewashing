"""Domain modelleri (Pydantic): Race, Horse/RaceEntry, Jockey, Trainer,
HorseStatistics ve tahmin ciktisi modelleri.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TrackSurface(str, Enum):
    """Pist tipi."""

    KUM = "Kum"
    CIM = "Cim"
    SENTETIK = "Sentetik"


class Jockey(BaseModel):
    name: str
    win_rate: Optional[float] = Field(default=None, ge=0, le=100)


class Trainer(BaseModel):
    name: str
    win_rate: Optional[float] = Field(default=None, ge=0, le=100)


class PastPerformance(BaseModel):
    """Bir atin gecmis bir kosudaki sonucu."""

    race_date: date
    hippodrome: str
    distance_m: int
    surface: TrackSurface
    finish_position: Optional[int] = None
    field_size: Optional[int] = None
    jockey_name: Optional[str] = None
    trainer_name: Optional[str] = None
    weight_kg: Optional[float] = None
    odds: Optional[float] = None

    @property
    def is_win(self) -> bool:
        return self.finish_position == 1

    @property
    def is_placed(self) -> bool:
        return self.finish_position is not None and self.finish_position <= 3


class HorseStatistics(BaseModel):
    """Bir atin, tahmin motoruna girdi olarak sunulan toplulastirilmis istatistikleri."""

    horse_id: str
    horse_name: str
    past_performances: list[PastPerformance] = Field(default_factory=list)
    career_starts: int = 0
    career_wins: int = 0
    career_places: int = 0
    last_year_starts: int = 0
    last_year_wins: int = 0
    last_year_places: int = 0
    jockey_horse_combo_starts: int = 0
    jockey_horse_combo_wins: int = 0

    @property
    def career_win_rate(self) -> float:
        return (self.career_wins / self.career_starts * 100) if self.career_starts else 0.0

    @property
    def career_place_rate(self) -> float:
        return (self.career_places / self.career_starts * 100) if self.career_starts else 0.0

    @property
    def days_since_last_race(self) -> Optional[int]:
        if not self.past_performances:
            return None
        last = max(p.race_date for p in self.past_performances)
        return (date.today() - last).days

    def recent_form(self, n: int = 5) -> list[PastPerformance]:
        """En yeni ``n`` kosuyu, tarihe gore azalan sirada dondurur."""
        return sorted(self.past_performances, key=lambda p: p.race_date, reverse=True)[:n]


class RaceEntry(BaseModel):
    """Belirli bir kosuya kayitli, kosu gunune ozel bilgileriyle bir at."""

    number: int
    horse_id: str
    source_horse_id: Optional[int] = None
    horse_name: str
    age: Optional[int] = None
    jockey: Jockey
    trainer: Trainer
    weight_kg: float
    odds: Optional[float] = None
    handicap_points: Optional[float] = None
    recent_form_positions: list[int] = Field(default_factory=list)
    form_raw: Optional[str] = None
    is_scratched: bool = False  # kosmaz


class Race(BaseModel):
    """Belirli bir hipodromda, belirli bir saatte kosulacak yaris."""

    id: str
    hippodrome: str
    race_no: int
    start_time: datetime
    distance_m: int
    surface: TrackSurface
    group_info: Optional[str] = None
    prize_info: Optional[str] = None
    entries: list[RaceEntry] = Field(default_factory=list)

    @property
    def active_entries(self) -> list[RaceEntry]:
        """Kosmaz (scratch) olarak isaretlenmemis atlar."""
        return [e for e in self.entries if not e.is_scratched]


class ScoreBreakdown(BaseModel):
    """Bir atin toplam skorunu olusturan bilesenler (her biri 0-100)."""

    form_score: float
    jockey_trainer_score: float
    distance_surface_score: float
    weight_score: float
    rest_score: float
    total_score: float


class HorsePrediction(BaseModel):
    """Bir at icin uretilen tahmin: skor, gerekce ve etiket."""

    entry: RaceEntry
    score: ScoreBreakdown
    reasoning: list[str]
    tag: str


class RacePrediction(BaseModel):
    """Bir yaris icin siralanmis tahmin sonucu."""

    race: Race
    ranked: list[HorsePrediction]
    disclaimer: str = (
        "Bu tahminler istatistiksel analize dayanir, kesinlik tasimaz; "
        "sorumlu bahis oynayin."
    )
