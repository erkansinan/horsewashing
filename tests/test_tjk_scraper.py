"""TJK HTML scraper icin testler.

Canli siteye bagimli olmamak icin (bkz. checklist.md, madde 3: "Test icin
HTML/veri parsing icin mock fixture'lar kullan"), gercek TJK sayfa yapisina
benzer, sabit bir HTML pars edilir.
"""
from __future__ import annotations

from datetime import date

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource

_SAMPLE_HTML = """
<html><body>
<div>
  <h3>1. Kosu 14:10</h3>
  <p>Sartli, 3 ve Yukari Ingilizler, kg, 1400 Kum</p>
  <table>
    <tr>
      <td>1</td><td>DENIZ YILDIZI</td><td>4y a k</td><td>PEDIGRI</td>
      <td>56,5</td><td>Ahmet Celik</td><td>Sahip A.S.</td><td>Osman Ozturk</td>
      <td>1</td><td>50</td><td>1-2345</td><td>41</td><td></td><td>5,50</td><td>%10</td><td></td>
    </tr>
    <tr>
      <td>2</td><td>KOSMAZ AT(Kosmaz)</td><td>5y a k</td><td>PEDIGRI</td>
      <td>58,0</td><td>Halis Karatas</td><td>Sahip B.S.</td><td>Necati Gur</td>
      <td>2</td><td>45</td><td>3-1122</td><td>38</td><td></td><td>-</td><td>%0</td><td></td>
    </tr>
  </table>
</div>
</body></html>
"""


def test_parse_daily_program_extracts_race_and_entries() -> None:
    source = TJKHtmlDataSource()
    races = source._parse_daily_program(_SAMPLE_HTML, "Ankara", date(2026, 8, 18))

    assert len(races) == 1
    race = races[0]
    assert race.race_no == 1
    assert race.distance_m == 1400
    assert race.hippodrome == "Ankara"
    assert len(race.entries) == 2

    active = race.active_entries
    assert len(active) == 1
    assert active[0].horse_name == "DENIZ YILDIZI"
    assert active[0].weight_kg == 56.5
    assert active[0].jockey.name == "Ahmet Celik"
    assert active[0].odds == 5.5

    scratched = race.entries[1]
    assert scratched.is_scratched is True


def test_get_horse_statistics_raises_not_implemented() -> None:
    source = TJKHtmlDataSource()
    races = source._parse_daily_program(_SAMPLE_HTML, "Ankara", date(2026, 8, 18))
    entry = races[0].active_entries[0]
    try:
        source.get_horse_statistics(entry)
        assert False, "DataSourceError bekleniyordu"
    except DataSourceError:
        pass
