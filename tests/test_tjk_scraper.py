"""TJK HTML scraper icin testler.

Canli siteye bagimli olmamak icin (bkz. checklist.md, madde 3: "Test icin
HTML/veri parsing icin mock fixture'lar kullan"), gercek TJK sayfa yapisina
benzer, sabit bir HTML pars edilir.
"""
from __future__ import annotations

from datetime import date
from bs4 import BeautifulSoup

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource, _era_for_date
from atyaris.models.entities import Jockey, RaceEntry, Trainer

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


def test_historical_dates_use_tjk_past_era() -> None:
    assert _era_for_date(date(2020, 1, 1)) == "past"
    assert _era_for_date(date.today()) == "today"


def test_get_horse_statistics_parses_era_past_summary_and_history() -> None:
    source = TJKHtmlDataSource()
    entry = RaceEntry(
        number=1,
        horse_id="horse-1",
        source_horse_id=46619,
        horse_name="GUMBERGUMBER",
        jockey=Jockey(name="Ahmet Celik"),
        trainer=Trainer(name="Osman Ozturk"),
        weight_kg=56.5,
    )

    html = """
    <html><body>
      <table>
        <tr><th>TOPLAM</th><td>135</td><td>65</td><td>33</td><td>14</td><td>10</td></tr>
        <tr><th>2025 Yılı</th><td>12</td><td>4</td><td>3</td><td>2</td><td>1</td></tr>
      </table>
      <table>
        <tr><th>Tarih</th><th>Sehir</th><th>Msf</th><th>Pist</th><th>S</th><th>Derece</th><th>Siklet</th><th>Taki</th><th>Jokey</th><th>St</th><th>Gny</th><th>Grup</th><th>K. No-K. Adi</th><th>Kcins</th><th>Ant.</th><th>Sahip</th><th>HP</th><th>Ikramiye</th><th>S20</th><th>X1</th><th>X2</th></tr>
        <tr><td>01.08.2026</td><td>Ankara</td><td>1400</td><td>Çim</td><td>1</td><td>1.24.10</td><td>56,5</td><td>-</td><td>Ahmet Celik</td><td>8</td><td>2.10</td><td>KV</td><td>1-Kosu</td><td>4y</td><td>Osman Ozturk</td><td>Sahip</td><td>68</td><td>1000</td><td>0</td><td>x</td><td>x</td></tr>
        <tr><td>10.07.2026</td><td>İstanbul</td><td>1600</td><td>Kum</td><td>3</td><td>1.36.20</td><td>57,0</td><td>-</td><td>Mehmet Kaptan</td><td>9</td><td>4.50</td><td>ST</td><td>2-Kosu</td><td>4y</td><td>Necati Gur</td><td>Sahip</td><td>70</td><td>1200</td><td>0</td><td>x</td><td>x</td></tr>
      </table>
      <table>
        <tr><th>At Adı</th><th>Irk</th><th>Cins.</th><th>Yaş</th><th>1400m</th><th>1200m</th><th>1000m</th><th>800m</th><th>600m</th><th>400m</th><th>200m</th><th>Durum</th><th>İ. Tarihi</th><th>İ. Hip.</th><th>P.Dur</th><th>Pist</th><th>İ. Türü</th><th>İ. Jokeyi</th><th>Detay</th></tr>
        <tr><td>GUMBERGUMBER</td><td>İngiliz</td><td>E</td><td>4</td><td></td><td></td><td></td><td>48.20</td><td></td><td></td><td></td><td>Normal</td><td>28.08.2026</td><td>Ankara</td><td>İyi</td><td>Kum</td><td>Sprint</td><td>Ahmet Celik</td><td>-</td></tr>
      </table>
    </body></html>
    """

    source._get_html = lambda url, params: html  # type: ignore[method-assign]
    stats = source.get_horse_statistics(entry)

    assert stats.career_starts == 135
    assert stats.career_wins == 65
    assert stats.last_year_starts == 12
    assert len(stats.past_performances) == 2
    assert stats.past_performances[0].hippodrome == "Ankara"
    assert stats.past_performances[0].finish_position == 1
    assert stats.past_performances[0].trainer_name == "Osman Ozturk"
    assert stats.past_performances[0].group_info == "KV"
    assert stats.past_performances[0].race_name == "1-Kosu"
    assert len(stats.workout_records) == 1
    assert stats.workout_records[0].hippodrome == "Ankara"
    assert stats.workout_records[0].distance_m == 800


def test_get_horse_statistics_parses_history_without_summary_table() -> None:
    source = TJKHtmlDataSource()
    entry = RaceEntry(
        number=1,
        horse_id="horse-2",
        source_horse_id=103734,
        horse_name="TEST AT",
        jockey=Jockey(name="Ahmet Celik"),
        trainer=Trainer(name="Antrenor X"),
        weight_kg=56.0,
    )

    html = """
    <html><body>
      <table>
        <tr><th>Tarih</th><th>Sehir</th><th>Msf</th><th>Pist</th><th>S</th><th>Derece</th><th>Siklet</th><th>Taki</th><th>Jokey</th><th>St</th><th>Gny</th><th>Grup</th><th>K. No-K. Adi</th><th>Kcins</th><th>Ant.</th><th>Sahip</th><th>HP</th><th>Ikramiye</th><th>S20</th></tr>
        <tr><td>15.08.2026</td><td>İzmir</td><td>1200</td><td>Kum</td><td>2</td><td>1.12.34</td><td>55,0</td><td>DB</td><td>Ahmet Celik</td><td>10</td><td>3,40</td><td>Handikap</td><td>3-Kosu</td><td>4y</td><td>Antrenor X</td><td>Sahip Y</td><td>72</td><td>25000</td><td>5</td></tr>
      </table>
    </body></html>
    """

    source._get_html = lambda url, params: html  # type: ignore[method-assign]
    stats = source.get_horse_statistics(entry)

    assert len(stats.past_performances) == 1
    perf = stats.past_performances[0]
    assert perf.distance_m == 1200
    assert perf.finish_position == 2
    assert perf.equipment == "DB"
    assert perf.handicap_points == 72.0
    assert perf.prize_info == "25000"


def test_parse_entry_row_extracts_source_horse_id_from_row_markup() -> None:
    source = TJKHtmlDataSource()
    row_html = """
    <tr data-target='/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=103734&&Era=past'>
      <td>1</td><td>TEST AT</td><td>4y</td><td>-</td><td>56,0</td><td>Ahmet Celik</td><td>-</td><td>Antrenor X</td><td>1</td><td>65</td><td>1-234</td><td>40</td><td></td><td>2,50</td><td></td><td></td>
    </tr>
    """
    row = BeautifulSoup(row_html, "lxml").find("tr")
    assert row is not None
    cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]

    entry = source._parse_entry_row(cells, row)
    assert entry.source_horse_id == 103734


def test_get_daily_race_results_parses_positions() -> None:
    source = TJKHtmlDataSource()
    tabs_html = """
    <html><body>
      <ul class='gunluk-tabs'>
        <li><a id='Ankara' href='/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari?SehirId=5&QueryParameter_Tarih=18/08/2026'>Ankara</a></li>
      </ul>
    </body></html>
    """
    city_results_html = """
    <html><body>
      <h3>1. Kosu 14:00</h3>
      <table>
        <tr><td>1</td><td>DENIZ YILDIZI(4)</td><td>1.24.10</td></tr>
        <tr><td>2</td><td>ASIL DUMAN(7)</td><td>1.24.70</td></tr>
      </table>
      <h3>2. Kosu 14:30</h3>
      <table>
        <tr><td>1</td><td>MERT YIGIT(2)</td><td>1.36.10</td></tr>
      </table>
    </body></html>
    """

    calls = {"count": 0}

    def fake_get_html(url: str, params: dict[str, object]) -> str:  # noqa: ARG001
        calls["count"] += 1
        return tabs_html if calls["count"] == 1 else city_results_html

    source._get_html = fake_get_html  # type: ignore[method-assign]
    results = source.get_daily_race_results(date(2026, 8, 18), "Ankara")

    assert results[1][4] == 1
    assert results[1][7] == 2
    assert results[2][2] == 1


def test_parse_csv_results_parses_finish_order() -> None:
    csv_text = """\ufeffAnkara;(61. Yarış Günü);08/09/2026
1. Kosu :   14.00;Maiden/DHÖ
At No;At İsmi;Yaş
7;BİRİNCİ;4y
4;İKİNCİ;5y
2;ÜÇÜNCÜ;4y
GANYAN(1) :23,75 TL
2. Kosu :   14.30;Handikap
At No;At İsmi;Yaş
3;DÖRDÜNCÜ;4y
"""

    results = TJKHtmlDataSource._parse_csv_results(csv_text)

    assert results == {1: {7: 1, 4: 2, 2: 3}, 2: {3: 1}}


def test_parse_live_csv_results_maps_finish_order_by_program_name() -> None:
    csv_text = """İstanbul;(66. Yarış Günü);13/09/2026
1. Koşu : 14.00;Handikap
At No;At İsmi;Yaş
1;RICHWOOD;3y
2;NAERYS SKG SK;3y
3;JACKAL TROUBLE DB SK;3y
"""

    results = TJKHtmlDataSource._parse_csv_results(
        csv_text,
        horse_numbers_by_race_name={
            1: {
                "richwood": 7,
                "naerys skg sk": 1,
                "jackal trouble db sk": 10,
            }
        },
    )

    assert results == {1: {7: 1, 1: 2, 10: 3}}
