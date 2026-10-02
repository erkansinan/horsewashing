"""TJK HTML scraper icin testler.

Canli siteye bagimli olmamak icin (bkz. checklist.md, madde 3: "Test icin
HTML/veri parsing icin mock fixture'lar kullan"), gercek TJK sayfa yapisina
benzer, sabit bir HTML pars edilir.
"""
from __future__ import annotations

from datetime import date, datetime
import json
import httpx
from bs4 import BeautifulSoup
import pytest

from atyaris.data_sources import tjk_scraper
from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource, _era_for_date, _era_for_referer_date
from atyaris.models.entities import Jockey, Race, RaceEntry, TrackSurface, Trainer

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


def test_parse_daily_program_csv_extracts_races_and_entries() -> None:
    csv_text = """\ufeffAnkara;(55. Yarış Günü);25/08/2026
1. Kosu :   14.00;Maiden/DHÖW;3 Yaşlı Araplar;57.00kg;1200m;Çim;;;;Rekor Derece :1.16.76
İkramiye;;;;;Yetiştirici Primi;;;;
At No;At İsmi;Yaş;Orijin(Baba);Orijin(Anne);Kilo;Jokey Adı;Sahip Adı;Antrenör Adı;St;AGF;H;Son 6 Yarış;KGS;s20;EnİyiDerece;Ganyan
1;UTKANBEY KG DB SK;3y k e;BERKSOY;HABERTAY;59;M.G.ARSLAN;YILMAZ BOZKUŞ;M.ÖZYİĞİT;12;%8.46(6);31;Ç2Ç4Ç3Ç4K7;14;16;1:20.24;3,25
2;BÜYÜK PEHLİVAN KG K DB;3y d e;ÖZHABER;DENİZMERYEM;57 +1.40;M.S.ÇELİK;DİNÇER KARABULUT;F.SERİNTÜRK;4;%13.65(3);;Ç2K6;70;18;;-
2. Kosu : 14.30;Handikap 14;3 Yaşlı Araplar;kg;1200m;Kum;;;;Rekor Derece :1.20.56
At No;At İsmi;Yaş;Orijin(Baba);Orijin(Anne);Kilo;Jokey Adı;Sahip Adı;Antrenör Adı;St;AGF;H;Son 6 Yarış;KGS;s20;EnİyiDerece;Ganyan
3;ÜÇÜNCÜ;4y a e;BABA;ANNE;55;JOKEY;SAHİP;ANTRENÖR;1;%1.00(1);45;K5S1Ç4;14;19;;8,00
"""

    races = TJKHtmlDataSource._parse_daily_program_csv(
        csv_text, "Ankara", date(2026, 8, 25)
    )

    assert len(races) == 2
    first = races[0]
    assert first.race_no == 1
    assert first.start_time.strftime("%H:%M") == "14:00"
    assert first.distance_m == 1200
    assert first.surface.value == "Cim"
    assert first.group_info == "Maiden/DHÖW - 3 Yaşlı Araplar"
    assert len(first.entries) == 2
    entry = first.entries[0]
    assert entry.horse_name == "UTKANBEY"
    assert entry.age == 3
    assert entry.weight_kg == 59.0
    assert entry.jockey.name == "M.G.ARSLAN"
    assert entry.trainer.name == "M.ÖZYİĞİT"
    assert entry.handicap_points == 31.0
    assert entry.recent_form_positions == [2, 4, 3, 4, 7]
    assert entry.odds == 3.25
    assert first.entries[1].weight_kg == 57.0
    assert first.entries[1].odds is None
    assert races[1].entries[0].odds == 8.0
    assert races[1].surface.value == "Kum"


def test_parse_daily_program_csv_does_not_treat_last_unlabeled_column_as_odds() -> None:
    csv_text = """Ankara;(55. Yarış Günü);25/08/2026
1. Kosu :   14.00;Maiden;3 Yaşlı Araplar;57.00kg;1200m;Kum
At No;At İsmi;Yaş;Orijin(Baba);Orijin(Anne);Kilo;Jokey Adı;Sahip Adı;Antrenör Adı;St;AGF;H;Son 6 Yarış;KGS;s20;EnİyiDerece
1;DENEME;3y k e;BABA;ANNE;56;JOKEY;SAHİP;ANTRENÖR;1;%10(1);20;Ç2;14;16;1.75
"""

    races = TJKHtmlDataSource._parse_daily_program_csv(
        csv_text, "Ankara", date(2026, 8, 25)
    )

    assert races[0].entries[0].odds is None


def test_parse_daily_program_csv_handles_race_titles_and_shifted_track_columns() -> None:
    csv_text = """6. Kosu : VELİEFENDİ KOŞUSU 16.30;G 1/DHT;4 ve Yukarı Araplar;57.00kg;60.00 kg;2800m;Çim;;;;
  At No;At İsmi;Yaş;Orijin;Anne;Kilo;Jokey;Sahip;Antrenör;St;AGF;H;Son 6
  1;ATABEYLİ;6y;BABA;ANNE;60;JOKEY;SAHIP;ANTRENOR;6;;95;1
  7. Kosu : NURULLAH TOLON KOŞUSU 17.00;KV-7/Dişi;3 ve Yukarı İngilizler;56.50kg;60.00 kg;60.00 kg;1900m;Çim;;;;
  At No;At İsmi;Yaş;Orijin;Anne;Kilo;Jokey;Sahip;Antrenör;St;AGF;H;Son 6
  1;GOLD FLOWER;4y;BABA;ANNE;61;JOKEY;SAHIP;ANTRENOR;6;;94;1
  8. Kosu : BAHADIR GÖDEK KOŞUSU 17.30;KV-9;3 Yaşlı İngilizler;58.00kg;2000m;Sentetik;;;;
  At No;At İsmi;Yaş;Orijin;Anne;Kilo;Jokey;Sahip;Antrenör;St;AGF;H;Son 6
  1;CALVADOS;3y;BABA;ANNE;58;JOKEY;SAHIP;ANTRENOR;4;;93;1
  """

    races = TJKHtmlDataSource._parse_daily_program_csv(
      csv_text, "İstanbul", date(2026, 9, 20)
    )

    assert [(race.race_no, race.distance_m, race.surface) for race in races] == [
      (6, 2800, TrackSurface.CIM),
      (7, 1900, TrackSurface.CIM),
      (8, 2000, TrackSurface.SENTETIK),
    ]


def test_daily_program_csv_url_uses_tjk_date_and_unicode_city() -> None:
    url = TJKHtmlDataSource._daily_program_csv_url(
        date(2026, 8, 25), "İstanbul"
    )

    assert url == (
        "https://medya-cdn.tjk.org/raporftp/TJKPDF/2026/2026-08-25/CSV/"
        "GunlukYarisProgrami/25.08.2026-İstanbul-GunlukYarisProgrami-TR.csv"
    )


def test_daily_results_csv_url_uses_tjk_date_and_unicode_city() -> None:
    url = TJKHtmlDataSource._daily_results_csv_url(
        date(2026, 9, 13), "İstanbul"
    )

    assert url == (
        "https://medya-cdn.tjk.org/raporftp/TJKPDF/2026/2026-09-13/CSV/"
        "GunlukYarisSonuclari/13.09.2026-İstanbul-GunlukYarisSonuclari-TR.csv"
    )


def test_parse_entry_row_prefers_semantic_gny_cell() -> None:
    html = """
    <tr>
      <td>1</td><td>DENIZ YILDIZI</td><td>4y a k</td><td>PEDIGRI</td>
      <td>56,5</td><td>Ahmet Celik</td><td>Sahip A.S.</td><td>Osman Ozturk</td>
      <td>1</td><td>50</td><td>1-2345</td><td>41</td>
      <td class="gny">7,25</td><td>%10</td><td>ek bilgi</td>
    </tr>
    """
    row = BeautifulSoup(html, "lxml").find("tr")
    assert row is not None
    cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]

    entry = TJKHtmlDataSource._parse_entry_row(cells, row)

    assert entry.odds == 7.25


def test_parse_entries_uses_gny_header_column() -> None:
    html = """
    <table>
      <tr><th>No</th><th>At</th><th>Gny</th><th>Not</th><th>Ek</th><th>A</th><th>B</th><th>C</th></tr>
      <tr><td>1</td><td>DENIZ YILDIZI</td><td>8,40</td><td>41</td><td>9,99</td><td>56</td><td>Jokey</td><td>Trainer</td></tr>
    </table>
    """
    table = BeautifulSoup(html, "lxml").find("table")
    assert table is not None

    entries = TJKHtmlDataSource()._parse_entries(table)

    assert len(entries) == 1
    assert entries[0].odds == 8.4


def test_parse_entries_uses_headers_when_program_columns_shift() -> None:
    html = """
    <table>
      <tr><th>No</th><th>At</th><th>Gny</th><th>Jokey</th><th>Yaş</th><th>HP</th><th>Son 6</th><th>Kilo</th><th>Antrenör</th></tr>
      <tr><td>1</td><td>DENIZ YILDIZI</td><td>8,40</td><td>Ahmet Celik</td><td>4y a k</td><td>50</td><td>1-2345</td><td>56,5</td><td>Osman Ozturk</td></tr>
    </table>
    """
    table = BeautifulSoup(html, "lxml").find("table")
    assert table is not None

    entries = TJKHtmlDataSource()._parse_entries(table)

    assert len(entries) == 1
    assert entries[0].odds == 8.4
    assert entries[0].weight_kg == 56.5
    assert entries[0].jockey.name == "Ahmet Celik"
    assert entries[0].trainer.name == "Osman Ozturk"
    assert entries[0].handicap_points == 50.0
    assert entries[0].recent_form_positions == [1, 2, 3, 4, 5]


def test_parse_entry_row_extracts_trainer_id() -> None:
    html = """
    <tr>
      <td>1</td><td>DENIZ YILDIZI</td><td>4y a k</td><td>PEDIGRI</td>
      <td>56,5</td><td>Ahmet Celik</td><td>Sahip A.S.</td>
      <td><a href="/TR/YarisSever/Query/Page/AntrenorIstatistikleri?1=1&amp;QueryParameter_AntrenorId=3025">Osman Ozturk</a></td>
      <td>1</td><td>50</td><td>1-2345</td><td>41</td><td></td><td>5,50</td><td>%10</td><td></td>
    </tr>
    """
    row = BeautifulSoup(html, "lxml").find("tr")
    assert row is not None
    cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]

    entry = TJKHtmlDataSource._parse_entry_row(cells, row)

    assert entry.trainer.source_trainer_id == 3025


def test_parse_trainer_statistics_table() -> None:
    html = """
    <table>
      <tr><th>Antrenör</th><th>Koşu</th><th>1.</th><th>2.</th><th>3.</th><th>4.</th><th>5.</th><th>1.%</th><th>2.%</th><th>3.%</th><th>4.%</th><th>5.%</th></tr>
      <tr><td>Osman Ozturk</td><td>100</td><td>20</td><td>15</td><td>10</td><td>12</td><td>8</td><td>20,00</td><td>15,00</td><td>10,00</td><td>12,00</td><td>8,00</td></tr>
    </table>
    """
    source = TJKHtmlDataSource()
    source._get_html = lambda url, params: html  # type: ignore[method-assign]

    stats = source.get_trainer_statistics(3025)

    assert stats.trainer_id == 3025
    assert stats.trainer_name == "Osman Ozturk"
    assert stats.total_starts == 100
    assert stats.first_place == 20
    assert stats.third_place == 10
    assert stats.first_rate == 20.0
    assert stats.fifth_rate == 8.0


def test_get_trainer_statistics_forwards_prediction_request_policy() -> None:
    html = """
    <table>
      <tr><th>Antrenör</th><th>Koşu</th><th>1.</th><th>2.</th><th>3.</th></tr>
      <tr><td>Test Trainer</td><td>10</td><td>2</td><td>1</td><td>1</td></tr>
    </table>
    """
    source = TJKHtmlDataSource()
    calls = []
    source._get_html = lambda url, params, **kwargs: calls.append(kwargs) or html  # type: ignore[method-assign]

    source.get_trainer_statistics(3025, request_timeout=5.0, max_retries=0)

    assert calls == [{"timeout": 5.0, "max_retries": 0}]


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


def test_available_hippodromes_returns_empty_when_selected_date_has_no_city_csv() -> None:
  source = TJKHtmlDataSource()
  class EmptyResponse:
    text = ""

  source._client.get = lambda *_args, **_kwargs: EmptyResponse()  # type: ignore[method-assign]
  source._get_hippodrome_sehir_ids = lambda _date: (_ for _ in ()).throw(  # type: ignore[method-assign]
    AssertionError("CSV discovery should not request the JavaScript-loaded tab page")
  )

  hippodromes = source.get_available_hippodromes(date(2026, 9, 16))

  assert hippodromes == []


def test_available_hippodromes_falls_back_when_tjk_times_out() -> None:
  source = TJKHtmlDataSource()
  source._client.get = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
    httpx.ReadTimeout("test timeout")
  )
  with pytest.raises(DataSourceError, match="tarihli TJK hipodrom CSV listesi"):
    source.get_available_hippodromes(date(2026, 9, 23))


def test_available_hippodromes_returns_only_cities_with_selected_date_csv() -> None:
  target_date = date(2026, 9, 23)
  source = TJKHtmlDataSource()
  requests: list[tuple[str, float | None, int | None, float | None]] = []
  csv_text = """Ankara;(55. Yarış Günü);23/09/2026
1. Koşu :   14.00;Maiden/DHÖW;3 Yaşlı Araplar;57.00kg;1200m;Çim;;;;Rekor Derece :1.16.76
İkramiye;;;;;Yetiştirici Primi;;;;
At No;At İsmi;Yaş;Orijin(Baba);Orijin(Anne);Kilo;Jokey Adı;Sahip Adı;Antrenör Adı;St;AGF;H;Son 6 Yarış;KGS;s20;EnİyiDerece
1;ANKARA AT;3y a e;BERKSOY;HABERTAY;59;M.G.ARSLAN;YILMAZ BOZKUŞ;M.ÖZYİĞİT;12;%8.46(6);31;Ç2Ç4Ç3Ç4K7;14;16;1:20.24
"""

  class Response:
    def __init__(self, text: str) -> None:
      self.text = text

  def fake_get(
    url: str,
    timeout: float | None = None,
    max_retries: int | None = None,
    min_interval: float | None = None,
  ):
    requests.append((url, timeout, max_retries, min_interval))
    return Response(csv_text if "-Ankara-" in url else "")

  source._client.get = fake_get  # type: ignore[method-assign]
  source._get_hippodrome_sehir_ids = lambda _date: (_ for _ in ()).throw(  # type: ignore[method-assign]
    AssertionError("a successful dated CSV scan must not fall back to TJK tabs")
  )

  assert source.get_available_hippodromes(target_date) == ["Ankara"]
  assert len(requests) == len(tjk_scraper.KNOWN_HIPPODROMES)
  assert all(
    timeout == 5.0 and max_retries == 0 and min_interval == 0.15
    for _, timeout, max_retries, min_interval in requests
  )


def test_daily_races_uses_city_program_url_and_parameters() -> None:
  source = TJKHtmlDataSource()
  calls: list[tuple[str, dict[str, object]]] = []
  source.get_daily_races_csv = lambda target_date, city: []  # type: ignore[method-assign]
  source._get_hippodrome_sehir_ids = lambda target_date: (_ for _ in ()).throw(  # type: ignore[method-assign]
    AssertionError("known city IDs should not require discovery")
  )
  source._get_html = lambda url, params: calls.append((url, params)) or _SAMPLE_HTML  # type: ignore[method-assign]

  races = source.get_daily_races(date(2026, 9, 23), "İstanbul", resolve_missing_ids=False)

  assert len(races) == 1
  assert calls == [
    (
      "https://www.tjk.org/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
      {
        "SehirId": 3,
        "QueryParameter_Tarih": "23/09/2026",
        "SehirAdi": "İstanbul",
          "Era": "past",
      },
    )
  ]


def test_daily_races_prefers_csv_before_hippodrome_discovery() -> None:
  source = TJKHtmlDataSource()
  csv_races = TJKHtmlDataSource._parse_daily_program_csv(
    "1. Kosu : 14.00;Maiden;Araplar;kg;1200m;Kum;;;;\n"
    "At No;At İsmi;Yaş;Orijin;Anne;Kilo;Jokey;Sahip;Antrenör;St;AGF;HP;Son 6\n"
    "1;UTKANBEY;3y;BABA;ANNE;57;JOKEY;SAHIP;ANTRENOR;1;;41;1",
    "Ankara",
    date(2026, 9, 23),
  )
  source.get_daily_races_csv = lambda target_date, city: csv_races  # type: ignore[method-assign]
  source._get_hippodrome_sehir_ids = lambda target_date: (_ for _ in ()).throw(  # type: ignore[method-assign]
    AssertionError("CSV races should not require city ID discovery")
  )
  source._get_html = lambda url, params: (_ for _ in ()).throw(  # type: ignore[method-assign]
    AssertionError("CSV races should not require an HTML request")
  )  # type: ignore[method-assign]

  races = source.get_daily_races(date(2026, 9, 23), "Ankara", resolve_missing_ids=False)

  assert races == csv_races
  assert races[0].hippodrome == "Ankara"


def test_daily_races_fills_missing_csv_odds_from_city_html() -> None:
  target_date = date(2026, 9, 23)
  source = TJKHtmlDataSource()
  csv_races = TJKHtmlDataSource._parse_daily_program_csv(
    "1. Kosu : 14.00;Maiden;Araplar;kg;1200m;Kum;;;;\n"
    "At No;At İsmi;Yaş;Orijin;Anne;Kilo;Jokey;Sahip;Antrenör;St;AGF;HP;Son 6\n"
    "1;DENIZ YILDIZI DB SKG;4y;BABA;ANNE;56;JOKEY;SAHIP;ANTRENOR;1;;41;1",
    "İstanbul",
    target_date,
  )
  html_races = source._parse_daily_program(_SAMPLE_HTML, "İstanbul", target_date)
  source.get_daily_races_csv = lambda requested_date, city: csv_races  # type: ignore[method-assign]
  html_attempts = []

  def fetch_html(requested_date, city):
    html_attempts.append((requested_date, city))
    if len(html_attempts) == 1:
      request = httpx.Request("GET", "https://www.tjk.org/daily-program")
      response = httpx.Response(504, request=request)
      raise httpx.HTTPStatusError("gateway timeout", request=request, response=response)
    return html_races

  source.get_daily_races_html = fetch_html  # type: ignore[method-assign]

  races = source.get_daily_races(target_date, "İstanbul", resolve_missing_ids=False)

  assert races[0].entries[0].odds == 5.5
  assert len(html_attempts) == 2


def test_get_daily_races_html_bypasses_csv_first_path() -> None:
  source = TJKHtmlDataSource(
    request_timeout=5.0,
    max_retries=0,
    program_request_timeout=30.0,
    program_max_retries=2,
  )
  calls: list[tuple[str, dict[str, object], dict[str, object]]] = []
  source.get_daily_races_csv = lambda target_date, city: (_ for _ in ()).throw(
    AssertionError("explicit HTML lookup must not call CSV")
  )  # type: ignore[method-assign]
  source._get_html = lambda url, params, **kwargs: calls.append((url, params, kwargs)) or _SAMPLE_HTML  # type: ignore[method-assign]

  races = source.get_daily_races_html(date(2026, 9, 23), "İstanbul")

  assert len(races) == 1
  assert calls == [
    (
      "https://www.tjk.org/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
      {
        "SehirId": 3,
        "QueryParameter_Tarih": "23/09/2026",
        "SehirAdi": "İstanbul",
          "Era": "past",
      },
        {
          "timeout": 30.0,
          "max_retries": 2,
          "headers": {
            "Referer": (
              "https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisProgrami?"
              "QueryParameter_Tarih=23%2F09%2F2026&Era="
              f"{_era_for_referer_date(date(2026, 9, 23))}"
            )
          },
        },
    )
  ]


def test_referer_era_matches_tjk_historical_navigation_window() -> None:
  today = date(2026, 9, 30)

  assert _era_for_referer_date(today, today) == "today"
  assert _era_for_referer_date(date(2026, 9, 29), today) == "yesterday"
  assert _era_for_referer_date(date(2026, 9, 25), today) == "lastWeek"
  assert _era_for_referer_date(date(2026, 9, 22), today) == "past"


def test_get_daily_races_html_retries_server_error_once() -> None:
  source = TJKHtmlDataSource()
  attempts = 0

  def fetch_html(_url, _params, **kwargs):
    nonlocal attempts
    attempts += 1
    if attempts == 1:
      request = httpx.Request("GET", "https://www.tjk.org/daily-program")
      response = httpx.Response(503, request=request)
      raise httpx.HTTPStatusError("service unavailable", request=request, response=response)
    return _SAMPLE_HTML

  source._get_html = fetch_html  # type: ignore[method-assign]

  races = source.get_daily_races_html(date(2026, 9, 23), "İstanbul")

  assert len(races) == 1
  assert attempts == 2


def test_parse_daily_program_extracts_horse_id_from_program_link() -> None:
  html = """
  <h3>1. Koşu 14:10</h3>
  <p>Maiden, 4 yaşlı İngilizler, 1200 Kum</p>
  <table>
    <tr><th>No</th><th>At</th><th>Yaş</th><th>Kilo</th><th>Jokey</th><th>Antrenör</th><th>HP</th><th>Son 6</th><th>Gny</th></tr>
    <tr><td>1</td><td><a href="?QueryParameter_AtId=12345">DENIZ YILDIZI</a></td><td>4y</td><td>56</td><td>JOKEY</td><td>ANTRENOR</td><td>41</td><td>1-2</td><td>5,50</td></tr>
  </table>
  """

  races = TJKHtmlDataSource()._parse_daily_program(html, "Ankara", date(2026, 9, 26))

  assert races[0].entries[0].source_horse_id == 12345
  assert races[0].entries[0].odds == 5.5


@pytest.mark.parametrize(
  ("city", "city_id"),
  [
    ("Adana", 1),
    ("İzmir", 2),
    ("İstanbul", 3),
    ("Bursa", 4),
    ("Ankara", 5),
    ("Şanlıurfa", 6),
    ("Elazığ", 7),
    ("Diyarbakır", 8),
    ("Kocaeli", 9),
    ("Antalya", 10),
  ],
)
def test_daily_races_html_uses_known_city_id_without_discovery(city: str, city_id: int) -> None:
  source = TJKHtmlDataSource()
  calls: list[tuple[str, dict[str, object]]] = []
  source._get_hippodrome_sehir_ids = lambda _date: (_ for _ in ()).throw(  # type: ignore[method-assign]
    AssertionError("known domestic city IDs must skip tab discovery")
  )
  source._get_html = lambda url, params, **_kwargs: calls.append((url, params)) or _SAMPLE_HTML  # type: ignore[method-assign]

  races = source.get_daily_races_html(date(2026, 9, 26), city)

  assert len(races) == 1
  assert calls[0][0].endswith("/Info/Sehir/GunlukYarisProgrami")
  assert calls[0][1]["SehirId"] == city_id


def test_tjk_daily_races_merges_odds_using_direct_known_city_page() -> None:
  target_date = date(2026, 9, 26)
  source = TJKHtmlDataSource()
  csv_races = TJKHtmlDataSource._parse_daily_program_csv(
    "1. Kosu : 14.00;Maiden;Araplar;kg;1200m;Kum;;;;\n"
    "At No;At İsmi;Yaş;Orijin;Anne;Kilo;Jokey;Sahip;Antrenör;St;AGF;HP;Son 6\n"
    "1;DENIZ YILDIZI;4y;BABA;ANNE;56;JOKEY;SAHIP;ANTRENOR;1;;41;1",
    "Ankara",
    target_date,
  )
  calls: list[tuple[str, dict[str, object]]] = []
  source.get_daily_races_csv = lambda _date, _city: csv_races  # type: ignore[method-assign]
  source._get_html = lambda url, params, **_kwargs: calls.append((url, params)) or _SAMPLE_HTML  # type: ignore[method-assign]

  races = source.get_daily_races(
    target_date,
    "Ankara",
    resolve_missing_ids=False,
    include_odds=True,
    race_no=1,
  )

  assert races[0].entries[0].odds == 5.5
  assert calls[0][1]["SehirId"] == 5


def test_parse_daily_result_horse_ids() -> None:
  html = """
  <h3>2. Koşu 14:30</h3><table>
    <tr><td>1</td><td><a href="/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=12345">DENİZ YILDIZI</a></td></tr>
  </table>
  """

  assert TJKHtmlDataSource._parse_daily_result_horse_ids(html) == [
    (2, "DENİZ YILDIZI", 12345)
  ]


def test_atlar_search_resolves_only_exact_unique_name_and_age() -> None:
  html = """
  <table>
    <tr><td>3y</td><td><a href="?QueryParameter_AtId=12345">DENİZ YILDIZI</a></td></tr>
    <tr><td>4y</td><td><a href="?QueryParameter_AtId=67890">DENİZ YILDIZI</a></td></tr>
  </table>
  """
  source = TJKHtmlDataSource()
  calls: list[tuple[str, dict[str, str]]] = []
  source._post_html = lambda url, data: calls.append((url, data)) or html  # type: ignore[method-assign]

  resolution = source.find_horse_id_from_search("DENİZ YILDIZI", age=3)

  assert resolution == (12345, "resolved", [12345])
  assert calls[0][0].endswith("/TR/YarisSever/Query/Data/Atlar")
  assert calls[0][1]["QueryParameter_Yas"] == "3"


def test_atlar_search_leaves_same_name_multiple_ids_ambiguous() -> None:
  html = """
  <table>
    <tr><td><a href="?QueryParameter_AtId=12345">DENİZ YILDIZI</a></td></tr>
    <tr><td><a href="?QueryParameter_AtId=67890">DENİZ YILDIZI</a></td></tr>
  </table>
  """

  assert TJKHtmlDataSource._parse_horse_search_results(
    html, "DENİZ YILDIZI"
  ) == [
    {"resolved_at_id": 12345, "horse_name": "DENİZ YILDIZI", "age": None, "row_text": "DENİZ YILDIZI"},
    {"resolved_at_id": 67890, "horse_name": "DENİZ YILDIZI", "age": None, "row_text": "DENİZ YILDIZI"},
  ]


def test_atkosu_identity_parser_checks_visible_exact_name() -> None:
  assert TJKHtmlDataSource._parse_horse_page_identity(
    "<h1>DENİZ YILDIZI</h1>", "DENIZ YILDIZI"
  ) == ("matched", ["DENİZ YILDIZI"])
  assert TJKHtmlDataSource._parse_horse_page_identity(
    "<h1>BAŞKA AT</h1>", "DENIZ YILDIZI"
  ) == ("mismatch", ["BAŞKA AT"])


def test_daily_races_without_city_uses_city_program_urls(monkeypatch) -> None:
  from atyaris.data_sources import tjk_scraper

  source = TJKHtmlDataSource()
  monkeypatch.setattr(tjk_scraper, "KNOWN_HIPPODROMES", ["İstanbul", "Elazığ"])
  calls: list[tuple[str, dict[str, object]]] = []
  source.get_daily_races_csv = lambda target_date, city: []  # type: ignore[method-assign]
  source._get_html = lambda url, params, **_kwargs: calls.append((url, params)) or _SAMPLE_HTML  # type: ignore[method-assign]

  races = source.get_daily_races(date(2026, 9, 23), resolve_missing_ids=False)

  assert len(races) == 2
  assert [url for url, _ in calls] == [
    "https://www.tjk.org/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
    "https://www.tjk.org/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
  ]
  assert [params["SehirId"] for _, params in calls] == [3, 7]


def test_merge_source_ids_matches_race_number_and_horse_name() -> None:
    csv_races = TJKHtmlDataSource._parse_daily_program_csv(
        "1. Kosu : 14.00;Maiden;Araplar;kg;1200m;Kum;;;;\n"
        "At No;At İsmi;Yaş;Orijin;Anne;Kilo;Jokey;Sahip;Antrenör;St;AGF;HP;Son 6\n"
        "4;DENIZ YILDIZI;4y;BABA;ANNE;56;JOKEY;SAHIP;ANTRENOR;1;;41;1",
        "İstanbul",
        date(2026, 9, 23),
    )
    html_races = TJKHtmlDataSource()._parse_daily_program(
        """<h3>1. Kosu 14:00</h3><table>
        <tr data-target='/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=12345'>
          <td>8</td><td>DENIZ YILDIZI</td><td>4y a k</td><td>-</td><td>56</td>
          <td>JOKEY</td><td>SAHIP</td><td>ANTRENOR</td><td>1</td><td>41</td><td>1-2</td><td></td><td>3,00</td>
        </tr></table>""",
        "İstanbul",
        date(2026, 9, 23),
    )

    merged = TJKHtmlDataSource.merge_source_ids(csv_races, html_races)

    assert merged == 1
    assert csv_races[0].entries[0].source_horse_id == 12345


def test_tjk_bulletin_uses_local_history_when_live_id_resolution_is_disabled(tmp_path, monkeypatch) -> None:
  record = {
    "target_horse_name": "PETRIKORA",
    "source_horse_id": 12345,
    "race_date": "2026-09-01",
    "hippodrome": "Ankara",
    "distance_m": 1200,
    "surface": "Kum",
    "finish_position": 2,
    "field_size": 8,
    "jockey_name": "JOKEY",
    "weight_kg": 55.0,
    "odds": 3.4,
  }
  history_path = tmp_path / "history.jsonl"
  history_path.write_text(
    f"{json.dumps(record)}\n{json.dumps(record)}\n",
    encoding="utf-8",
  )
  source = TJKHtmlDataSource(horse_history_path=history_path)
  entry = RaceEntry(
    number=1,
    horse_id="PETRIKORA-1",
    horse_name="PETRIKORA DB SKG",
    jockey=Jockey(name="JOKEY"),
    trainer=Trainer(name="ANTRENOR"),
    weight_kg=55.0,
  )
  race = Race(
    id="Ankara-2026-09-26-1",
    hippodrome="Ankara",
    race_no=1,
    start_time=datetime(2026, 9, 26, 14, 0),
    distance_m=1200,
    surface=TrackSurface.KUM,
    entries=[entry],
  )
  monkeypatch.setattr(source, "get_daily_races_csv", lambda *_args: [race])
  monkeypatch.setattr(source, "get_daily_races_html", lambda *_args: [])
  monkeypatch.setattr(source, "_get_html", lambda *_args: pytest.fail("unexpected live history request"))

  bulletin = source.get_daily_races(date(2026, 9, 26), "Ankara", resolve_missing_ids=False)
  stats = source.get_horse_statistics(bulletin[0].entries[0])

  assert bulletin[0].entries[0].source_horse_id == 12345
  assert bulletin[0].entries[0].id_resolution_status == "local_history"
  assert stats.career_starts == 1
  assert stats.career_places == 1
  assert stats.past_performances[0].odds == 3.4


def test_get_horse_statistics_uses_matching_local_history_for_html_id(tmp_path, monkeypatch) -> None:
  record = {
    "target_horse_name": "PETRIKORA",
    "source_horse_id": 12345,
    "race_date": "2026-09-01",
    "hippodrome": "Ankara",
    "distance_m": 1200,
    "surface": "Kum",
    "finish_position": 2,
    "field_size": 8,
    "jockey_name": "JOKEY",
    "weight_kg": 55.0,
    "odds": 3.4,
  }
  history_path = tmp_path / "history.jsonl"
  history_path.write_text(f"{json.dumps(record)}\n", encoding="utf-8")
  source = TJKHtmlDataSource(horse_history_path=history_path)
  entry = RaceEntry(
    number=1,
    horse_id="PETRIKORA-1",
    source_horse_id=12345,
    horse_name="PETRIKORA",
    jockey=Jockey(name="JOKEY"),
    trainer=Trainer(name="ANTRENOR"),
    weight_kg=55.0,
  )
  monkeypatch.setattr(source, "_get_html", lambda *_args: pytest.fail("matching local history must skip network"))

  stats = source.get_horse_statistics(entry)

  assert stats.career_starts == 1
  assert stats.past_performances[0].odds == 3.4


def test_get_horse_statistics_parses_era_past_summary_and_history(tmp_path) -> None:
    source = TJKHtmlDataSource(horse_history_path=tmp_path / "history.jsonl")
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

    request_policies = []
    source._get_html = lambda url, params, **kwargs: request_policies.append(kwargs) or html  # type: ignore[method-assign]
    stats = source.get_horse_statistics(entry, request_timeout=5.0, max_retries=0)

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
    assert request_policies == [
      {"timeout": 5.0, "max_retries": 0},
      {"timeout": 5.0, "max_retries": 0},
    ]


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
    assert perf.field_size is None
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
    source.get_daily_races_csv = lambda target_date, city: []  # type: ignore[method-assign]
    source._get_csv = lambda url: None  # type: ignore[method-assign]
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

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_get_html(url: str, params: dict[str, object]) -> str:
        calls.append((url, params))
        return city_results_html

    source._get_html = fake_get_html  # type: ignore[method-assign]
    results = source.get_daily_race_results(date(2026, 9, 13), "İstanbul")

    assert results[1][4] == 1
    assert results[1][7] == 2
    assert results[2][2] == 1
    assert calls == [
      (
        "https://www.tjk.org/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari",
        {
          "SehirId": 3,
          "QueryParameter_Tarih": "13/09/2026",
          "SehirAdi": "İstanbul",
        },
      )
    ]


def test_get_daily_race_results_uses_official_csv_first() -> None:
    source = TJKHtmlDataSource()
    program_csv = """1. Kosu : 14.00;Maiden;3 Yaşlı Araplar;kg;1200m;Kum;;;;
  At No;At İsmi;Yaş;Orijin(Baba);Orijin(Anne);Kilo;Jokey Adı;Sahip Adı;Antrenör Adı;St;AGF;HP;Son 6 Yarış
  4;DENIZ YILDIZI;4y;BABA;ANNE;56,5;JOKEY A;SAHIP A;ANTRENOR A;1;%10;41;1-234
  7;ASIL DUMAN;5y;BABA;ANNE;58;JOKEY B;SAHIP B;ANTRENOR B;2;%8;38;2-345
  """
    program_races = TJKHtmlDataSource._parse_daily_program_csv(
      program_csv, "İstanbul", date(2026, 9, 13)
    )
    results_csv = """1. Kosu : 14.00;Maiden
  At No;At İsmi;Yaş
  1;DENIZ YILDIZI;4y
  2;ASIL DUMAN;5y
  """
    csv_url = TJKHtmlDataSource._daily_results_csv_url(
      date(2026, 9, 13), "İstanbul"
    )
    source.get_daily_races_csv = lambda target_date, city: program_races  # type: ignore[method-assign]
    source._get_csv = lambda url: results_csv if url == csv_url else None  # type: ignore[method-assign]
    source._get_html = lambda url, params: (_ for _ in ()).throw(  # type: ignore[method-assign]
      AssertionError("results HTML should not be requested when the CSV is available")
    )

    results = source.get_daily_race_results(date(2026, 9, 13), "İstanbul")

    assert results == {1: {4: 1, 7: 2}}


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
