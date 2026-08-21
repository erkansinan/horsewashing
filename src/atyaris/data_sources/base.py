"""Veri kaynagi soyutlama katmani (Adapter/Protocol deseni).

Herhangi bir somut veri kaynagi (TJK HTML scraper, ileride cikabilecek resmi
bir API, ya da testler icin bir ornek/sahte kaynak) bu Protocol'u
uygulamalidir. Uygulamanin geri kalani (tahmin motoru, CLI) yalnizca bu
arayuze bagimlidir; bu sayede alttaki veri kaynagini degistirmek, geri kalan
kodu etkilemeden yalnizca yeni bir adapter yazmayi gerektirir.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from atyaris.models.entities import HorseStatistics, Race, RaceEntry


class DataSourceError(RuntimeError):
    """Bir veri kaynagi veri cekerken veya ayristirirken basarisiz oldugunda firlatilir."""


@runtime_checkable
class RaceDataSource(Protocol):
    """Her yaris veri saglayici adapter'in uygulamasi gereken arayuz."""

    def get_daily_races(self, target_date: date, city: str | None = None) -> list[Race]:
        """``target_date`` icin planlanmis tum yarislari dondurur (istege bagli sehir filtresiyle)."""
        ...

    def get_horse_statistics(self, entry: RaceEntry) -> HorseStatistics:
        """Bir yaris kaydindaki atin gecmis istatistiklerini/performanslarini dondurur."""
        ...
