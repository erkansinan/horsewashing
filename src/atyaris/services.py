"""CLI ve web arayuzlerinin ortak kullandigi yardimci fonksiyonlar: veri
kaynagi olusturma, tarih parse etme, gunluk yaris listesini getirme.

Bu katman sayesinde ayni is mantigi (data source secimi, "en yakin" hipodrom
onceliklendirme) hem `cli/app.py` hem de `web/app.py` tarafindan tekrar
yazilmadan kullanilir.
"""
from __future__ import annotations

from datetime import date, datetime

from atyaris.cache.sqlite_cache import SqliteTTLCache
from atyaris.config import Settings
from atyaris.data_sources.base import RaceDataSource
from atyaris.data_sources.sample_source import SampleDataSource
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.data_sources.prediction import build_prediction_data_source
from atyaris.data_sources.training import build_training_data_source
from atyaris.models.entities import Race


class InvalidSourceError(ValueError):
    """Gecersiz veri kaynagi adi ('sample'/'tjk' disinda) verildiginde firlatilir."""


def build_data_source(source: str, settings: Settings) -> RaceDataSource:
    """Kaynak adina gore ilgili ``RaceDataSource`` adaptorunu olusturur."""
    if source == "sample":
        return SampleDataSource()
    if source == "tjk":
        return build_prediction_data_source(settings)
    raise InvalidSourceError("source 'sample' veya 'tjk' olmali")


def parse_date(target_date: str | None) -> date:
    """``GG.AA.YYYY`` veya ``YYYY-MM-DD`` (HTML tarih secici) formatindaki
    tarihi parse eder; bos ise bugunu dondurur."""
    if not target_date:
        return date.today()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(target_date, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Gecersiz tarih formati: {target_date!r} (GG.AA.YYYY veya YYYY-MM-DD bekleniyor)")


def prioritize_by_city(races: list[Race], near_city: str | None) -> list[Race]:
    """Kullanicinin sectigi 'en yakin' hipodromu listenin basina alir (madde 2.1)."""
    if not near_city:
        return races
    near_lower = near_city.lower()
    return sorted(races, key=lambda r: 0 if near_lower in r.hippodrome.lower() else 1)


def fetch_races(
    data_source: RaceDataSource, target_date: date, city: str | None, near: str | None
) -> list[Race]:
    """Gunluk yaris listesini ceker; sehir filtresi yoksa 'en yakin' hipodromu one alir."""
    if city:
        races = data_source.get_daily_races(target_date, city)
    else:
        races = data_source.get_daily_races(target_date, None)
        races = prioritize_by_city(races, near)
    return races
