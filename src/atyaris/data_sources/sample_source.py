"""Deterministik ornek/demo veri kaynagi.

Yerel gelistirme, otomatik testler ve TJK'ya canli baglanti olmadan uctan uca
demo calistirmalari icin kullanilir (TJK scraper'in bilinen kisitlari icin
README.md > "Bilinen Kisitlar" bolumune bakin). ``TJKHtmlDataSource`` ile
ayni ``RaceDataSource`` arayuzunu uygular; bu sayede uygulamanin geri kalani
icin dogrudan yerine gecebilir (drop-in replacement).
"""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta

from atyaris.data_sources.base import RaceDataSource
from atyaris.models.entities import (
    HorseStatistics,
    Jockey,
    PastPerformance,
    Race,
    RaceEntry,
    Trainer,
    TrackSurface,
)

_HORSE_NAMES = [
    "RUZGAR GIBI", "KAGAN BEY", "YILDIZ TEPE", "ASIL DUMAN", "KRAL YAVUZ",
    "SUZME ELMAS", "BOZ ATMACA", "DELI DOLU", "MERT YIGIT", "GONUL ATESI",
    "SAHIN KANADI", "AKINCI BEY", "CESUR ADIM", "FIRTINA KIZI", "ALKAN BEY",
]
_JOCKEYS = [
    "Ahmet Celik",
    "Halis Karatas",
    "Mehmet Kaptan",
    "Onder Sahin",
    "Serkan Yildiz",
    "Murat Yildiz",
    "Aykut Ozer",
    "Berk Can",
    "Emre Yilmaz",
    "Kadir Aslan",
    "Tolga Demir",
    "Onur Kaya",
]
_TRAINERS = ["Osman Ozturk", "Necati Gur", "Kemal Aydin", "Yusuf Demir"]
_CITIES = ["Istanbul (Veliefendi)", "Ankara", "Izmir (Sirinyer)"]
_WEATHERS = ["Gunesli", "Bulutlu", "Yagmurlu", "Ruzgarli"]


def _rng_for(seed_key: str) -> random.Random:
    """Verilen anahtara gore deterministik (tekrarlanabilir) bir RNG uretir."""
    return random.Random(seed_key)


class SampleDataSource(RaceDataSource):
    """Gercekci Turkce yaris verisiyle calisan, tamamen deterministik kaynak."""

    def get_daily_races(self, target_date: date, city: str | None = None) -> list[Race]:
        cities = [c for c in _CITIES[:2] if not city or city.lower() in c.lower()]
        races: list[Race] = []
        for city_name in cities:
            rng = _rng_for(f"{city_name}-{target_date.isoformat()}")
            for race_no in range(1, 4):
                start_time = datetime.combine(target_date, datetime.min.time()) + timedelta(
                    hours=13, minutes=30 * race_no
                )
                distance = rng.choice([1200, 1400, 1600, 1800, 2000])
                surface = rng.choice(list(TrackSurface))
                entries = self._build_entries(rng, city_name, race_no)
                races.append(
                    Race(
                        id=f"{city_name}-{target_date.isoformat()}-{race_no}",
                        hippodrome=city_name,
                        race_no=race_no,
                        start_time=start_time,
                        distance_m=distance,
                        surface=surface,
                        group_info="Sartli" if race_no > 1 else "Maiden",
                        prize_info=f"{rng.randint(150, 400) * 100} TL",
                        entries=entries,
                    )
                )
        return races

    @staticmethod
    def _build_entries(rng: random.Random, city_name: str, race_no: int) -> list[RaceEntry]:
        n_horses = rng.randint(6, 9)
        names = rng.sample(_HORSE_NAMES, k=n_horses)
        jockey_names = rng.sample(_JOCKEYS, k=n_horses)
        entries = []
        for i, name in enumerate(names, start=1):
            entries.append(
                RaceEntry(
                    number=i,
                    horse_id=f"{city_name}-{race_no}-{i}",
                    horse_name=name,
                    age=rng.randint(3, 7),
                    jockey=Jockey(name=jockey_names[i - 1]),
                    trainer=Trainer(name=rng.choice(_TRAINERS)),
                    weight_kg=round(rng.uniform(52, 61), 1),
                    odds=round(rng.uniform(1.5, 25.0), 2),
                    handicap_points=round(rng.uniform(35.0, 95.0), 1),
                    recent_form_positions=[rng.randint(1, 10) for _ in range(5)],
                    form_raw="".join(str(rng.randint(1, 9)) for _ in range(5)),
                )
            )

        finish_order = list(range(1, n_horses + 1))
        rng.shuffle(finish_order)
        for idx, entry in enumerate(entries):
            entry.actual_finish_position = finish_order[idx]
        return entries

    def get_horse_statistics(self, entry: RaceEntry) -> HorseStatistics:
        rng = _rng_for(entry.horse_id)
        n_past = rng.randint(5, 12)
        performances: list[PastPerformance] = []
        cursor_date = date.today() - timedelta(days=rng.randint(10, 40))
        for _ in range(n_past):
            field_size = rng.randint(6, 10)
            performances.append(
                PastPerformance(
                    race_date=cursor_date,
                    hippodrome=rng.choice(_CITIES),
                    distance_m=rng.choice([1200, 1400, 1600, 1800, 2000]),
                    surface=rng.choice(list(TrackSurface)),
                    finish_position=rng.randint(1, field_size),
                    field_size=field_size,
                    jockey_name=entry.jockey.name,
                    trainer_name=entry.trainer.name,
                    weight_kg=round(entry.weight_kg + rng.uniform(-2, 2), 1),
                    odds=round(rng.uniform(1.5, 30.0), 2),
                    race_time_seconds=round(rng.uniform(68.0, 132.0), 2),
                    early_pace_index=round(rng.uniform(0.05, 0.95), 3),
                    mid_pace_index=round(rng.uniform(0.05, 0.95), 3),
                    late_pace_index=round(rng.uniform(0.05, 0.95), 3),
                    weather=rng.choice(_WEATHERS),
                )
            )
            cursor_date -= timedelta(days=rng.randint(14, 45))

        wins = sum(1 for p in performances if p.is_win)
        places = sum(1 for p in performances if p.is_placed)
        combo_starts = sum(1 for p in performances if p.jockey_name == entry.jockey.name)
        combo_wins = sum(
            1 for p in performances if p.jockey_name == entry.jockey.name and p.is_win
        )

        return HorseStatistics(
            horse_id=entry.horse_id,
            horse_name=entry.horse_name,
            past_performances=performances,
            career_starts=n_past + rng.randint(0, 10),
            career_wins=wins + rng.randint(0, 2),
            career_places=places + rng.randint(0, 3),
            last_year_starts=n_past,
            last_year_wins=wins,
            last_year_places=places,
            jockey_horse_combo_starts=combo_starts,
            jockey_horse_combo_wins=combo_wins,
        )
