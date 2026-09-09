"""Canli TJK (Turkiye Jokey Kulubu) HTML scraping adapter'i.

ONEMLI / BILINEN KISITLAR
-------------------------
TJK resmi bir REST API sunmaz. Bu adapter, tjk.org'un herkese acik,
sunucu tarafinda render edilen sayfalarini BeautifulSoup ile ayristirir.
Kullanilan DOM yapisi, canli ``GunlukYarisProgrami`` sayfasi incelenerek
tersine muhendislikle cikarilmistir ve TJK sablonu degisirse guncellenmesi
gerekebilir (bkz. README.md > "Veri Kaynagini Guncelleme").

At bazli gecmis performans sayfasi (``AtKosuBilgileri``), detay tablosunu
istemci tarafinda (AJAX/JS) yukledigi icin bu adapter tarafindan henuz
guvenilir sekilde desteklenmiyor; ``get_horse_statistics`` bu durumda acik
bir ``DataSourceError`` firlatir. Gercek TJK verisiyle at istatistigi
gerekiyorsa, tarayici gelistirici araclariyla ilgili AJAX uc noktasi tespit
edilip ayni desen bu modulde uygulanmalidir. Bu fragilite nedeniyle proje,
gelistirme/demo/test icin guvenilir bir yedek olarak ``SampleDataSource``
ile birlikte gelir.

UYARI: Gunluk bulten sayfasi (``GunlukYarisProgrami``) da benzer sekilde
buyuk olcude JavaScript/jQuery-unobtrusive-ajax ile render edilmektedir
(``Info/Page/...`` sunucu tarafinda yalnizca bir "kabuk" dondurur; asil
kosu tablosu ``Info/Data/...`` adresine yapilan, tarayicida calisan JS
tarafindan tetiklenen ic ice AJAX cagrilarla doldurulur). Bu adapter yalnizca
sunucu tarafinda dogrudan HTML icinde gelen icerigi ayristirabilir; TJK bu
sekli degistirirse (veya bazi sayfa/tarih kombinasyonlarinda sunucu tarafinda
render ederse) calisabilir, ancak genel olarak **deneyseldir ve garantili
degildir**. Guvenilir/test edilebilir bir deneyim icin varsayilan ve
onerilen kaynak ``SampleDataSource``'tur (``--source sample``).

Saygili scraping: aciklayici bir User-Agent, istekler arasi minimum bekleme
(``RateLimitedClient``) ve yanit onbelleklemesi (``SqliteTTLCache``)
kullanilarak TJK sunucularina gereksiz yuk bindirilmemesi hedeflenir.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime

from bs4 import BeautifulSoup

from atyaris.cache.sqlite_cache import SqliteTTLCache
from atyaris.data_sources.base import DataSourceError, RaceDataSource
from atyaris.data_sources.http_client import RateLimitedClient
from atyaris.models.entities import (
    HorseStatistics,
    Jockey,
    PastPerformance,
    Race,
    RaceEntry,
    Trainer,
    TrackSurface,
    WorkoutRecord,
)

logger = logging.getLogger(__name__)

# TJK'nin gunluk program sayfasi (``Info/Page/GunlukYarisProgrami``) yalnizca
# ``SehirAdi`` (hipodromun TJK'daki resmi adi) ve ``QueryParameter_Tarih``
# parametrelerini gerektirir; ayrica bir "SehirId" gerekmez (bu, sehir
# secicisindeki farkli bir uc nokta olan ``Info/Sehir/...`` icin kullanilir).
# Asagidaki liste, tjk.org'un gunluk program sayfasinda kullandigi resmi
# hipodrom adlarini (dogru Turkce karakterlerle) icerir; canli bir sayfa
# incelenerek dogrulanmistir (bkz. README.md > "Veri Kaynagini Guncelleme").
KNOWN_HIPPODROMES: list[str] = [
    "İstanbul",
    "Ankara",
    "İzmir",
    "Bursa",
    "Adana",
    "Antalya",
    "Elazığ",
    "Diyarbakır",
    "Kocaeli",
    "Şanlıurfa",
]

_TURKISH_FOLD = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ş": "s",
        "Ş": "s",
        "ğ": "g",
        "Ğ": "g",
        "ç": "c",
        "Ç": "c",
        "ö": "o",
        "Ö": "o",
        "ü": "u",
        "Ü": "u",
    }
)


def _normalize_city(text: str) -> str:
    """Turkce karakterleri ASCII benzerlerine indirger ve kucuk harfe cevirir.

    Bu sayede kullanicinin girdigi "istanbul", "İstanbul", "ISTANBUL" gibi
    farkli yazimlar, TJK'nin resmi "İstanbul" adiyla eslesir.
    """
    return text.translate(_TURKISH_FOLD).lower().strip()


def _match_hippodrome(city: str) -> str | None:
    """Kullanicinin verdigi (serbest yazimli) sehir adini, TJK'nin resmi
    hipodrom adlarindan biriyle eslestirir; eslesme yoksa ``None`` doner."""
    normalized = _normalize_city(city)
    for name in KNOWN_HIPPODROMES:
        folded = _normalize_city(name)
        if normalized == folded or normalized in folded or folded in normalized:
            return name
    return None


_DAILY_PROGRAM_PATH = "/TR/YarisSever/Info/Page/GunlukYarisProgrami"
_DAILY_PROGRAM_DATA_PATH = "/TR/YarisSever/Info/Data/GunlukYarisProgrami"
_DAILY_PROGRAM_CITY_PATH = "/TR/YarisSever/Info/Sehir/GunlukYarisProgrami"
_DAILY_RESULTS_DATA_PATH = "/TR/YarisSever/Info/Data/GunlukYarisSonuclari"
_DAILY_RESULTS_CITY_PATH = "/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari"
_HORSE_HISTORY_PATH = "/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri"
_HORSE_WORKOUT_PATH = "/TR/YarisSever/Query/Page/IdmanIstatistikleri"

_SURFACE_MAP = {
    "kum": TrackSurface.KUM,
    "cim": TrackSurface.CIM,
    "çim": TrackSurface.CIM,
    "sentetik": TrackSurface.SENTETIK,
}

_RACE_HEADER_RE = re.compile(r"(\d+)\.\s*Ko[sş]u\s+(\d{1,2}[:.]\d{2})", re.IGNORECASE)
_DISTANCE_SURFACE_RE = re.compile(r"(\d{3,4})\s*(Kum|Cim|Çim|Sentetik)", re.IGNORECASE)


class TJKHtmlDataSource(RaceDataSource):
    """tjk.org'un herkese acik gunluk program sayfalarini kazir (scrape)."""

    def __init__(
        self,
        base_url: str = "https://www.tjk.org",
        user_agent: str = "atyaris-tahmin/1.0",
        request_timeout: float = 15.0,
        min_request_interval: float = 1.0,
        cache: SqliteTTLCache | None = None,
        cache_ttl_seconds: int = 600,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = RateLimitedClient(user_agent, request_timeout, min_request_interval)
        self._cache = cache
        self._cache_ttl = cache_ttl_seconds

    def close(self) -> None:
        self._client.close()

    def _get_html(self, url: str, params: dict) -> str:
        cache_key = f"GET:{url}:{sorted(params.items())}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        response = self._client.get(url, params=params)
        html = response.text
        if self._cache is not None:
            self._cache.set(cache_key, html, self._cache_ttl)
        return html

    def _get_hippodrome_sehir_ids(self, target_date: date) -> dict[str, int]:
        """Secili tarih icin TJK sekmelerindeki resmi hipodrom->SehirId
        eslesmesini dondurur."""
        url = f"{self._base_url}{_DAILY_PROGRAM_DATA_PATH}"
        params = {
            "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
            "Era": "today",
        }
        html = self._get_html(url, params)
        soup = BeautifulSoup(html, "lxml")
        mapping: dict[str, int] = {}
        for link in soup.select("ul.gunluk-tabs li a"):
            raw = (link.get("id") or link.get_text(" ", strip=True) or "").strip()
            if not raw:
                continue
            name = re.sub(r"\s*\(.*\)\s*$", "", raw).strip()
            matched = _match_hippodrome(name)
            if not matched:
                continue
            href = link.get("href") or ""
            m = re.search(r"[?&]SehirId=(\d+)", href)
            if m:
                mapping[matched] = int(m.group(1))
        return mapping

    def _get_results_hippodrome_sehir_ids(self, target_date: date) -> dict[str, int]:
        """Gunluk yaris sonuclari sayfasindaki hipodrom->SehirId eslesmesi."""
        url = f"{self._base_url}{_DAILY_RESULTS_DATA_PATH}"
        params = {
            "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
            "Era": "today",
        }
        html = self._get_html(url, params)
        soup = BeautifulSoup(html, "lxml")
        mapping: dict[str, int] = {}
        for link in soup.select("ul.gunluk-tabs li a"):
            raw = (link.get("id") or link.get_text(" ", strip=True) or "").strip()
            if not raw:
                continue
            name = re.sub(r"\s*\(.*\)\s*$", "", raw).strip()
            matched = _match_hippodrome(name)
            if not matched:
                continue
            href = link.get("href") or ""
            m = re.search(r"[?&]SehirId=(\d+)", href)
            if m:
                mapping[matched] = int(m.group(1))
        return mapping

    def get_daily_race_results(self, target_date: date, city: str) -> dict[int, dict[int, int]]:
        """Secili hipodrom/tarih icin kosu sonucu haritasi dondurur.

        Donus: ``{race_no: {horse_number: finish_position}}``
        """
        matched = _match_hippodrome(city)
        if matched is None:
            return {}

        try:
            sehir_ids = self._get_results_hippodrome_sehir_ids(target_date)
        except Exception as exc:  # noqa: BLE001
            raise DataSourceError(f"Sonuc hipodrom/SehirId listesi cekilirken hata: {exc}") from exc

        sehir_id = sehir_ids.get(matched)
        if sehir_id is None:
            return {}

        url = f"{self._base_url}{_DAILY_RESULTS_CITY_PATH}"
        params = {
            "SehirId": sehir_id,
            "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
            "SehirAdi": matched,
            "Era": "today",
        }
        html = self._get_html(url, params)
        soup = BeautifulSoup(html, "lxml")

        results: dict[int, dict[int, int]] = {}
        for header_text in soup.find_all(string=_RACE_HEADER_RE):
            match = _RACE_HEADER_RE.search(str(header_text))
            if not match:
                continue
            race_no = int(match.group(1))
            table = header_text.parent.find_next("table")
            if table is None:
                continue

            race_map = results.setdefault(race_no, {})
            for row in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
                if len(cells) < 2:
                    continue
                finish_position = None
                horse_number = None
                for index, cell in enumerate(cells):
                    if finish_position is None:
                        position_match = re.fullmatch(r"\d+", cell)
                        if position_match:
                            finish_position = int(position_match.group(0))
                            continue
                    if finish_position is not None and index > 0:
                        horse_number_match = re.search(r"\((\d+)\)", cell)
                        if horse_number_match:
                            horse_number = int(horse_number_match.group(1))
                            break
                if finish_position is None or horse_number is None:
                    continue
                race_map[horse_number] = finish_position

        return results

    def get_available_hippodromes(self, target_date: date) -> list[str]:
        """Verilen tarihte TJK gunluk programinda listelenen hipodromlari dondurur.

        Not: Bu metod yaris detaylarini degil, yalnizca sayfada gorunen
        hipodrom sekmelerini ayristirir.
        """
        url = f"{self._base_url}{_DAILY_PROGRAM_DATA_PATH}"
        params = {
            "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
            "Era": "today",
        }
        try:
            _ = self._get_html(url, params)
            found = list(self._get_hippodrome_sehir_ids(target_date).keys())
            return found or list(KNOWN_HIPPODROMES)
        except Exception as exc:  # noqa: BLE001 - DataSourceError olarak yeniden firlatilir
            raise DataSourceError(f"Hipodrom listesi cekilirken hata: {exc}") from exc

    def get_daily_races(self, target_date: date, city: str | None = None) -> list[Race]:
        if city:
            matched = _match_hippodrome(city)
            if matched is None:
                logger.warning(
                    "Bilinmeyen sehir/hipodrom: '%s' (bilinen hipodromlar: %s)",
                    city,
                    ", ".join(KNOWN_HIPPODROMES),
                )
                return []
            try:
                sehir_ids = self._get_hippodrome_sehir_ids(target_date)
            except Exception as exc:  # noqa: BLE001
                raise DataSourceError(f"Hipodrom/SehirId listesi cekilirken hata: {exc}") from exc
            sehir_id = sehir_ids.get(matched)
            if sehir_id is None:
                logger.warning("'%s' secili tarihte aktif hipodrom listesinde bulunamadi.", matched)
                return []

            url = f"{self._base_url}{_DAILY_PROGRAM_CITY_PATH}"
            params = {
                "SehirId": sehir_id,
                "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
                "SehirAdi": matched,
                "Era": "today",
            }
            try:
                html = self._get_html(url, params)
                races = self._parse_daily_program(html, matched, target_date)
            except Exception as exc:  # noqa: BLE001 - DataSourceError olarak yeniden firlatilir
                raise DataSourceError(f"'{matched}' bulteni cekilirken hata: {exc}") from exc
            if not races:
                logger.warning(
                    "'%s' icin hicbir kosu ayristirilamadi; TJK sayfayi JavaScript/AJAX ile "
                    "render ediyor olabilir (bkz. README.md > 'Bilinen Kisitlar') veya bu "
                    "hipodromda o tarihte yaris yoktur.",
                    matched,
                )
            return races
        else:
            hippodromes = list(KNOWN_HIPPODROMES)

        races: list[Race] = []
        missing: list[str] = []
        for name in hippodromes:
            url = f"{self._base_url}{_DAILY_PROGRAM_PATH}"
            params = {
                "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
                "SehirAdi": name,
            }
            try:
                html = self._get_html(url, params)
                parsed = self._parse_daily_program(html, name, target_date)
            except Exception as exc:  # noqa: BLE001 - DataSourceError olarak yeniden firlatilir
                raise DataSourceError(f"'{name}' bulteni cekilirken hata: {exc}") from exc
            if not parsed:
                missing.append(name)
            races.extend(parsed)

        if city and missing:
            logger.warning(
                "'%s' icin hicbir kosu ayristirilamadi; TJK sayfayi JavaScript/AJAX ile "
                "render ediyor olabilir (bkz. README.md > 'Bilinen Kisitlar') veya bu "
                "hipodromda o tarihte yaris yoktur.",
                missing[0],
            )
        elif not city and missing and len(missing) == len(hippodromes):
            logger.warning(
                "TJK gunluk bulteninden hicbir hipodrom icin kosu ayristirilamadi (%d/%d). "
                "Muhtemel neden: sayfa istemci tarafinda JavaScript/AJAX ile render ediliyor "
                "(bkz. README.md > 'Bilinen Kisitlar').",
                len(missing),
                len(hippodromes),
            )
        return races



    def _parse_daily_program(self, html: str, hippodrome: str, target_date: date) -> list[Race]:
        soup = BeautifulSoup(html, "lxml")
        races_by_no: dict[int, Race] = {}
        headers = list(soup.find_all(string=_RACE_HEADER_RE))
        for header_text in headers:
            match = _RACE_HEADER_RE.search(str(header_text))
            if not match:
                continue
            race_no = int(match.group(1))
            time_str = match.group(2).replace(".", ":")
            header_tag = header_text.parent
            table = header_tag.find_next("table")
            if table is None:
                continue

            distance = 0
            surface = TrackSurface.KUM
            info_text = ""
            probe = header_tag.find_next(string=_DISTANCE_SURFACE_RE)
            if probe:
                info_text = str(probe)
                dm = _DISTANCE_SURFACE_RE.search(info_text)
                if dm:
                    distance = int(dm.group(1))
                    surface = _SURFACE_MAP.get(dm.group(2).lower(), TrackSurface.KUM)

            entries = self._parse_entries(table)
            try:
                start_time = datetime.combine(
                    target_date, datetime.strptime(time_str, "%H:%M").time()
                )
            except ValueError:
                start_time = datetime.combine(target_date, datetime.min.time())

            race = Race(
                id=f"{hippodrome}-{target_date.isoformat()}-{race_no}",
                hippodrome=hippodrome,
                race_no=race_no,
                start_time=start_time,
                distance_m=distance,
                surface=surface,
                group_info=info_text.strip() or None,
                entries=entries,
            )

            # TJK sehir sayfasinda ayni kosu basligi birden fazla DOM bolumunde
            # tekrar edebildigi icin, ayni kosu numarasinda son goruleni esas al.
            races_by_no[race_no] = race

        return [races_by_no[key] for key in sorted(races_by_no)]

    def _parse_entries(self, table) -> list[RaceEntry]:  # type: ignore[no-untyped-def]
        entries: list[RaceEntry] = []
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if len(cells) < 8:
                continue
            try:
                entries.append(self._parse_entry_row(cells, row))
            except (ValueError, IndexError) as exc:
                logger.debug("Satir ayristirilamadi, atlaniyor: %s (%s)", cells, exc)
        return entries

    @staticmethod
    def _parse_entry_row(cells: list[str], row=None) -> RaceEntry:  # type: ignore[no-untyped-def]
        def _first_float(text: str) -> float | None:
            if not text:
                return None
            normalized = text.replace(",", ".")
            m = re.search(r"\d+(?:\.\d+)?", normalized)
            if not m:
                return None
            try:
                return float(m.group(0))
            except ValueError:
                return None

        number_idx = 0
        for i in range(min(3, len(cells))):
            if re.fullmatch(r"\d+", cells[i] or ""):
                number_idx = i
                break

        number = int(re.sub(r"\D", "", cells[number_idx]) or 0)
        name_idx = min(number_idx + 1, len(cells) - 1)
        raw_name = cells[name_idx]
        is_scratched = "Kosmaz" in raw_name or "Koşmaz" in raw_name
        horse_name = re.sub(r"\(.*?\)", "", raw_name).replace("Kosmaz", "").replace("Koşmaz", "")
        horse_name = re.sub(r"\bt\s*\d[\d\.,]*\s*TL\b", "", horse_name, flags=re.IGNORECASE)
        horse_name = horse_name.split("Kapalı gözlük", 1)[0].split("Dilinin bağlanacağını", 1)[0]
        horse_name = horse_name.split("Ring mahalinden", 1)[0]
        horse_name = re.sub(r"\s+(?:KG|DB|SK|K|D|B|GKR|KB|KBB)+\s*$", "", horse_name).strip()

        age = None
        age_idx = min(number_idx + 2, len(cells) - 1)
        age_match = re.search(r"(\d+)\s*y", cells[age_idx], flags=re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))

        weight_idx = min(number_idx + 4, len(cells) - 1)
        jockey_idx = min(number_idx + 5, len(cells) - 1)
        trainer_idx = min(number_idx + 7, len(cells) - 1)
        hp_idx = min(number_idx + 9, len(cells) - 1)
        form_idx = min(number_idx + 10, len(cells) - 1)

        weight = _first_float(cells[weight_idx]) or 0.0
        jockey_name = cells[jockey_idx] or "Bilinmiyor"
        trainer_name = cells[trainer_idx] or "Bilinmiyor"

        handicap_points = None
        hp_raw = cells[hp_idx]
        if hp_raw and hp_raw != "-":
            handicap_points = _first_float(hp_raw)

        form_raw = cells[form_idx] if cells[form_idx] and cells[form_idx] != "-" else None
        recent_form_positions: list[int] = []
        if form_raw:
            chunks = re.split(r"\s+|-|/", form_raw.strip())
            for chunk in chunks:
                token = chunk.strip()
                if not token:
                    continue
                if token.isdigit() and len(token) > 1 and " " not in form_raw and "-" in form_raw:
                    recent_form_positions.extend(int(ch) for ch in token if ch.isdigit())
                    continue
                m = re.match(r"\d+", token)
                if m:
                    recent_form_positions.append(int(m.group(0)))

        odds_raw = cells[-3] if len(cells) >= 3 else None
        odds = None
        if odds_raw and odds_raw not in {"-", ""}:
            try:
                odds = float(odds_raw.replace(",", "."))
            except ValueError:
                odds = None

        source_horse_id = None
        if row is not None:
            link = row.find("a", href=re.compile(r"QueryParameter_AtId=\d+"))
            if link is not None:
                href = link.get("href") or ""
                m = re.search(r"QueryParameter_AtId=(\d+)", href)
                if m:
                    source_horse_id = int(m.group(1))
            if source_horse_id is None:
                # Bazi TJK satirlarinda AtId href yerine onclick/data-* icinde gelebilir.
                row_markup = str(row)
                m = re.search(r"QueryParameter_AtId=(\d+)", row_markup)
                if m:
                    source_horse_id = int(m.group(1))

        return RaceEntry(
            number=number,
            horse_id=f"{horse_name}-{number}",
            source_horse_id=source_horse_id,
            horse_name=horse_name,
            age=age,
            jockey=Jockey(name=jockey_name),
            trainer=Trainer(name=trainer_name),
            weight_kg=weight,
            odds=odds,
            handicap_points=handicap_points,
            recent_form_positions=recent_form_positions,
            form_raw=form_raw,
            is_scratched=is_scratched,
        )

    def get_horse_statistics(self, entry: RaceEntry) -> HorseStatistics:
        if entry.source_horse_id is None:
            raise DataSourceError(
                "At istatistikleri icin gerekli QueryParameter_AtId bulunamadi."
            )

        url = f"{self._base_url}{_HORSE_HISTORY_PATH}"
        params = {
            "QueryParameter_AtId": str(entry.source_horse_id),
            "Era": "past",
        }
        html = self._get_html(url, params)
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        if not tables:
            raise DataSourceError("AtKosuBilgileri sayfasinda tablo bulunamadi.")

        summary_table = None
        history_table = None
        date_re = re.compile(r"\d{2}\.\d{2}\.\d{4}")

        best_history_rows = 0
        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue
            row_texts = [
                [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
                for row in rows
            ]
            joined = " | ".join(" | ".join(cells) for cells in row_texts if cells)
            if "TOPLAM" in joined:
                summary_table = table
            matching_rows = [cells for cells in row_texts if cells and date_re.fullmatch(cells[0] or "")]
            if matching_rows and len(matching_rows) > best_history_rows:
                history_table = table
                best_history_rows = len(matching_rows)

        if history_table is None:
            raise DataSourceError("AtKosuBilgileri gecmis kosu tablosu bulunamadi.")

        def _parse_int(text: str) -> int:
            m = re.search(r"\d+", text or "")
            return int(m.group(0)) if m else 0

        def _parse_float(text: str) -> float | None:
            txt = (text or "").strip()
            if not txt or txt == "-":
                return None
            txt = txt.replace(".", "").replace(",", ".") if txt.count(",") == 1 and txt.count(".") > 1 else txt.replace(",", ".")
            m = re.search(r"\d+(?:\.\d+)?", txt)
            if not m:
                return None
            try:
                return float(m.group(0))
            except ValueError:
                return None

        career_starts = 0
        career_wins = 0
        career_places = 0
        last_year_starts = 0
        last_year_wins = 0
        last_year_places = 0

        if summary_table is not None:
            year_rows: list[tuple[int, list[str]]] = []
            for row in summary_table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
                if len(cells) < 5:
                    continue
                if cells[0] == "TOPLAM":
                    career_starts = _parse_int(cells[1])
                    career_wins = _parse_int(cells[2])
                    career_places = _parse_int(cells[2]) + _parse_int(cells[3]) + _parse_int(cells[4])
                ym = re.match(r"(\d{4})\s+Yılı", cells[0])
                if ym:
                    year_rows.append((int(ym.group(1)), cells))

            if year_rows:
                year_rows.sort(key=lambda x: x[0], reverse=True)
                latest = year_rows[0][1]
                last_year_starts = _parse_int(latest[1])
                last_year_wins = _parse_int(latest[2])
                last_year_places = _parse_int(latest[2]) + _parse_int(latest[3]) + _parse_int(latest[4])

        def _normalize_header(text: str) -> str:
            folded = text.lower().strip().replace(" ", "")
            folded = (
                folded.replace("ı", "i")
                .replace("İ", "i")
                .replace("ş", "s")
                .replace("Ş", "s")
                .replace("ğ", "g")
                .replace("Ğ", "g")
                .replace("ç", "c")
                .replace("Ç", "c")
                .replace("ö", "o")
                .replace("Ö", "o")
                .replace("ü", "u")
                .replace("Ü", "u")
            )
            return re.sub(r"[^a-z0-9]", "", folded)

        def _parse_time_to_seconds(text: str) -> float | None:
            raw = (text or "").strip()
            if not raw or raw == "-":
                return None
            m = re.search(r"(\d+)[\.,:](\d{2})(?:[\.,:](\d{2}))?", raw)
            if not m:
                return None
            minutes = int(m.group(1))
            seconds = int(m.group(2))
            hundredths = int(m.group(3) or 0)
            return float(minutes * 60 + seconds + (hundredths / 100.0))

        def _parse_split_time_to_seconds(text: str) -> float | None:
            raw = (text or "").strip()
            if not raw or raw == "-":
                return None
            normalized = raw.replace(",", ".")
            m = re.match(r"^(\d+)[\.:](\d{1,2})(?:[\.:](\d{1,2}))?$", normalized)
            if m:
                first = int(m.group(1))
                second = int(m.group(2))
                third = m.group(3)
                if third is not None:
                    hundredths = int(third)
                    return float(first * 60 + second + (hundredths / 100.0))
                if first <= 4 and second <= 59:
                    return float(first * 60 + second)
            try:
                return float(normalized)
            except ValueError:
                return None

        header_map: dict[str, int] = {}
        for row in history_table.find_all("tr"):
            headers = [c.get_text(" ", strip=True) for c in row.find_all("th")]
            if not headers:
                continue
            normalized = [_normalize_header(h) for h in headers]
            if "tarih" in normalized and ("msf" in normalized or "mesafe" in normalized):
                for idx, name in enumerate(normalized):
                    header_map[name] = idx
                break

        def _cell(cells: list[str], *aliases: str) -> str:
            for alias in aliases:
                idx = header_map.get(alias)
                if idx is not None and idx < len(cells):
                    return cells[idx]
            return ""

        past_performances: list[PastPerformance] = []
        for row in history_table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if not cells:
                continue

            date_text = _cell(cells, "tarih") or (cells[0] if cells else "")
            if not date_re.fullmatch(date_text):
                continue

            try:
                race_date = datetime.strptime(date_text, "%d.%m.%Y").date()
            except ValueError:
                continue

            hippodrome = _cell(cells, "sehir") or _cell(cells, "şehir") or (cells[1] if len(cells) > 1 else "")
            distance_m = _parse_int(_cell(cells, "msf", "mesafe") or (cells[2] if len(cells) > 2 else ""))
            surface_info = _cell(cells, "pist") or (cells[3] if len(cells) > 3 else "")
            if "Ç:" in surface_info or "Çim" in surface_info:
                surface = TrackSurface.CIM
            elif "S:" in surface_info or "Sentetik" in surface_info:
                surface = TrackSurface.SENTETIK
            else:
                surface = TrackSurface.KUM

            finish_position = _parse_int(_cell(cells, "s", "sira") or (cells[4] if len(cells) > 4 else "")) or None
            race_time_seconds = _parse_time_to_seconds(_cell(cells, "derece") or (cells[5] if len(cells) > 5 else ""))
            weight_kg = _parse_float(_cell(cells, "siklet") or (cells[6] if len(cells) > 6 else ""))
            equipment = _cell(cells, "taki") or (cells[7] if len(cells) > 7 else None)
            jockey_name = _cell(cells, "jokey") or (cells[8] if len(cells) > 8 else None)
            field_size = _parse_int(_cell(cells, "st") or (cells[9] if len(cells) > 9 else "")) or None
            odds = _parse_float(_cell(cells, "gny") or (cells[10] if len(cells) > 10 else ""))
            group_info = _cell(cells, "grup") or (cells[11] if len(cells) > 11 else None)
            race_name = _cell(cells, "k.no-k.adi") or _cell(cells, "k.no-k.adi") or (cells[12] if len(cells) > 12 else None)
            race_class = _cell(cells, "kcins") or (cells[13] if len(cells) > 13 else None)
            trainer_name = _cell(cells, "ant") or _cell(cells, "ant.") or (cells[14] if len(cells) > 14 else None)
            owner_name = _cell(cells, "sahip") or (cells[15] if len(cells) > 15 else None)
            handicap_points = _parse_float(_cell(cells, "hp") or (cells[16] if len(cells) > 16 else ""))
            prize_info = _cell(cells, "ikramiye") or (cells[17] if len(cells) > 17 else None)
            s20 = _cell(cells, "s20") or (cells[18] if len(cells) > 18 else None)

            past_performances.append(
                PastPerformance(
                    race_date=race_date,
                    hippodrome=hippodrome,
                    distance_m=distance_m,
                    surface=surface,
                    finish_position=finish_position,
                    field_size=field_size,
                    jockey_name=jockey_name,
                    trainer_name=trainer_name,
                    equipment=equipment,
                    group_info=group_info,
                    race_name=race_name,
                    race_class=race_class,
                    owner_name=owner_name,
                    handicap_points=handicap_points,
                    prize_info=prize_info,
                    s20=s20,
                    weight_kg=weight_kg,
                    odds=odds,
                    race_time_seconds=race_time_seconds,
                    early_pace_index=None,
                    mid_pace_index=None,
                    late_pace_index=None,
                    weather=None,
                )
            )

        combo_starts = 0
        combo_wins = 0
        jockey_norm = entry.jockey.name.strip().lower()
        for p in past_performances:
            if (p.jockey_name or "").strip().lower() == jockey_norm:
                combo_starts += 1
                if p.finish_position == 1:
                    combo_wins += 1

        if not past_performances:
            raise DataSourceError(
                "AtKosuBilgileri gecmis kosu tablosu bulundu fakat parse edilebilen satir yok."
            )

        if not career_starts:
            career_starts = len(past_performances)
        if not career_wins:
            career_wins = sum(1 for p in past_performances if p.finish_position == 1)
        if not career_places:
            career_places = sum(1 for p in past_performances if p.finish_position and p.finish_position <= 3)

        workout_records: list[WorkoutRecord] = []
        workout_url = f"{self._base_url}{_HORSE_WORKOUT_PATH}"
        workout_params = {"QueryParameter_AtId": str(entry.source_horse_id)}
        try:
            workout_html = self._get_html(workout_url, workout_params)
            workout_soup = BeautifulSoup(workout_html, "lxml")
            workout_tables = workout_soup.find_all("table")
            workout_table = None
            for table in workout_tables:
                headers = [
                    c.get_text(" ", strip=True)
                    for c in table.find_all("tr")[0].find_all(["th", "td"])
                ] if table.find_all("tr") else []
                normalized = [_normalize_header(h) for h in headers]
                if "itarihi" in normalized and "ihip" in normalized:
                    workout_table = table
                    break

            if workout_table is not None:
                header_map: dict[str, int] = {}
                for row in workout_table.find_all("tr"):
                    headers = [c.get_text(" ", strip=True) for c in row.find_all("th")]
                    if not headers:
                        continue
                    normalized = [_normalize_header(h) for h in headers]
                    if "itarihi" in normalized and "ihip" in normalized:
                        for idx, name in enumerate(normalized):
                            header_map[name] = idx
                        break

                def _wcell(cells: list[str], *aliases: str) -> str:
                    for alias in aliases:
                        idx = header_map.get(alias)
                        if idx is not None and idx < len(cells):
                            return cells[idx]
                    return ""

                split_distances = [1400, 1200, 1000, 800, 600, 400, 200]
                for row in workout_table.find_all("tr"):
                    cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
                    if not cells:
                        continue
                    workout_date_text = _wcell(cells, "itarihi")
                    if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", workout_date_text or ""):
                        continue

                    workout_date = None
                    try:
                        workout_date = datetime.strptime(workout_date_text, "%d.%m.%Y").date()
                    except ValueError:
                        pass

                    picked_distance = None
                    picked_time = None
                    for dist in split_distances:
                        value = _wcell(cells, str(dist) + "m")
                        if not value:
                            continue
                        parsed = _parse_split_time_to_seconds(value)
                        if parsed is not None:
                            picked_distance = dist
                            picked_time = parsed
                            break

                    workout_records.append(
                        WorkoutRecord(
                            workout_date=workout_date,
                            hippodrome=_wcell(cells, "ihip") or None,
                            surface=_wcell(cells, "pist") or None,
                            workout_type=_wcell(cells, "ituru") or None,
                            workout_jockey=_wcell(cells, "ijokeyi") or None,
                            status=_wcell(cells, "durum") or None,
                            ranking_status=_wcell(cells, "pdur") or None,
                            detail=_wcell(cells, "detay") or None,
                            distance_m=picked_distance,
                            time_seconds=picked_time,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("IdmanIstatistikleri parse edilemedi (%s): %s", entry.source_horse_id, exc)

        return HorseStatistics(
            horse_id=entry.horse_id,
            horse_name=entry.horse_name,
            past_performances=past_performances,
            workout_records=workout_records,
            career_starts=career_starts,
            career_wins=career_wins,
            career_places=career_places,
            last_year_starts=last_year_starts or min(len(past_performances), 10),
            last_year_wins=last_year_wins,
            last_year_places=last_year_places,
            jockey_horse_combo_starts=combo_starts,
            jockey_horse_combo_wins=combo_wins,
        )
