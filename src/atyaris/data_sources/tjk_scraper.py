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

import csv
import io
import json
import logging
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup

from atyaris.cache.sqlite_cache import SqliteTTLCache
from atyaris.data_sources.base import DataSourceError, RaceDataSource
from atyaris.data_sources.http_client import RateLimitedClient, manual_check_url
from atyaris.models.entities import (
    HorseStatistics,
    Jockey,
    PastPerformance,
    Race,
    RaceEntry,
    Trainer,
    TrainerStatistics,
    TrackSurface,
    WorkoutRecord,
)

logger = logging.getLogger(__name__)

# TJK'nin sehir bazli gunluk program ve sonuc sayfalari SehirId, tarih ve
# resmi SehirAdi parametrelerini kullanir. Sehir ID'si bilinmeyen hipodromlar
# gunluk program/sonuc sekmelerinden dinamik olarak cozulur.
# Asagidaki liste, tjk.org'un kullandigi resmi hipodrom adlarini icerir.
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

_KNOWN_HIPPODROME_SEHIR_IDS = {
    "Adana": 1,
    "İzmir": 2,
    "İstanbul": 3,
    "Bursa": 4,
    "Ankara": 5,
    "Şanlıurfa": 6,
    "Elazığ": 7,
    "Diyarbakır": 8,
    "Kocaeli": 9,
    "Antalya": 10,
}

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
_HORSE_SEARCH_PATH = "/TR/YarisSever/Query/Data/Atlar"
_HORSE_WORKOUT_PATH = "/TR/YarisSever/Query/Page/IdmanIstatistikleri"
_TRAINER_STATISTICS_PATH = "/TR/YarisSever/Query/Page/AntrenorIstatistikleri"
_DAILY_PROGRAM_CSV_BASE_URL = "https://medya-cdn.tjk.org/raporftp/TJKPDF"

_SURFACE_MAP = {
    "kum": TrackSurface.KUM,
    "cim": TrackSurface.CIM,
    "çim": TrackSurface.CIM,
    "sentetik": TrackSurface.SENTETIK,
}

_RACE_HEADER_RE = re.compile(r"(\d+)\.\s*Ko[sş]u\s+(\d{1,2}[:.]\d{2})", re.IGNORECASE)
_CSV_RACE_HEADER_RE = re.compile(r"^\s*(\d+)\.\s*Ko[sş]u\b", re.IGNORECASE)
_CSV_PROGRAM_RACE_HEADER_RE = re.compile(
    r"^\s*(\d+)\.\s*Ko[sş]u\s*:\s*.*?(\d{1,2})[.:](\d{2})\s*$", re.IGNORECASE
)
_DISTANCE_SURFACE_RE = re.compile(r"(\d{3,4})\s*(Kum|Cim|Çim|Sentetik)", re.IGNORECASE)


def _normalize_horse_name(value: str) -> str:
    value = re.sub(r"\bK Kulaklık takılacağını ifade eder\.?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def _horse_identity_key(value: str) -> str:
    cleaned = re.sub(r"\bK Kulaklık takılacağını ifade eder\.?", "", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _normalize_city(cleaned).casefold()


_HORSE_EQUIPMENT_SUFFIXES = {
    "DB", "G", "GKR", "K", "KG", "KGK", "SK", "SKG", "SGKR", "Y", "YP", "ÖG"
}


def _local_horse_name_key(value: str) -> str:
    tokens = value.split()
    while len(tokens) > 1 and tokens[-1].upper() in _HORSE_EQUIPMENT_SUFFIXES:
        tokens.pop()
    return _horse_identity_key(" ".join(tokens))


_LOCAL_HISTORY_INDEX_LOCK = threading.Lock()
_LOCAL_HISTORY_INDEX_REFRESH_SECONDS = 60.0
_LOCAL_HISTORY_OFFSETS: dict[str, tuple[int, int, float, dict[str, list[int]]]] = {}


def _local_history_offsets(path: Path) -> dict[str, list[int]]:
    try:
        resolved_path = str(path.resolve())
        stat = path.stat()
    except OSError:
        return {}

    now = time.monotonic()
    cached = _LOCAL_HISTORY_OFFSETS.get(resolved_path)
    if cached is not None and (
        (cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size)
        or now - cached[2] < _LOCAL_HISTORY_INDEX_REFRESH_SECONDS
    ):
        return cached[3]

    with _LOCAL_HISTORY_INDEX_LOCK:
        try:
            stat = path.stat()
        except OSError:
            return {}
        now = time.monotonic()
        cached = _LOCAL_HISTORY_OFFSETS.get(resolved_path)
        if cached is not None and (
            (cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size)
            or now - cached[2] < _LOCAL_HISTORY_INDEX_REFRESH_SECONDS
        ):
            return cached[3]

        offsets: dict[str, list[int]] = {}
        try:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(record, dict):
                        continue
                    horse_name = record.get("target_horse_name")
                    if isinstance(horse_name, str):
                        offsets.setdefault(_local_horse_name_key(horse_name), []).append(offset)
        except OSError as exc:
            logger.debug("Yerel TJK at gecmisi indekslenemedi | path=%s | hata=%s", path, exc)
            return {}

        _LOCAL_HISTORY_OFFSETS[resolved_path] = (stat.st_mtime_ns, stat.st_size, now, offsets)
        return offsets


def _era_for_date(target_date: date) -> str:
    """Use TJK's historical view for dates before the current day."""
    return "today" if target_date >= date.today() else "past"


def _era_for_referer_date(target_date: date, today: date | None = None) -> str:
    """Match the daily-program Referer era to TJK's date navigation window."""
    current_date = today or date.today()
    days_ago = (current_date - target_date).days
    if days_ago <= 0:
        return "today"
    if days_ago == 1:
        return "yesterday"
    if days_ago <= 7:
        return "lastWeek"
    return "past"


class TJKHtmlDataSource(RaceDataSource):
    """tjk.org'un herkese acik gunluk program sayfalarini kazir (scrape)."""

    def __init__(
        self,
        base_url: str = "https://www.tjk.org",
        user_agent: str = "atyaris-tahmin/1.0",
        request_timeout: float = 30.0,
        min_request_interval: float = 1.0,
        cache: SqliteTTLCache | None = None,
        cache_ttl_seconds: int = 600,
        horse_history_path: str | Path = "data/raw/tjk_horse_history.jsonl",
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        program_request_timeout: float | None = None,
        program_max_retries: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = RateLimitedClient(
            user_agent,
            request_timeout,
            min_request_interval,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._cache = cache
        self._cache_ttl = cache_ttl_seconds
        self._program_request_timeout = request_timeout if program_request_timeout is None else program_request_timeout
        self._program_max_retries = max_retries if program_max_retries is None else program_max_retries
        self._horse_search_timeout = min(request_timeout, 8.0)
        self._horse_search_unavailable = False
        self._horse_history_path = Path(horse_history_path)
        self._local_history_by_name: dict[str, list[dict[str, object]]] = {}
        self._local_history_loaded_names: set[str] = set()
        self._local_stats_cache: dict[tuple[str, int], HorseStatistics] = {}

    def close(self) -> None:
        self._client.close()

    def _prime_local_horse_history(self, entries: list[RaceEntry]) -> None:
        name_keys = {_local_horse_name_key(entry.horse_name) for entry in entries}
        pending = name_keys - self._local_history_loaded_names
        if not pending:
            return

        matches: dict[str, list[dict[str, object]]] = {key: [] for key in pending}
        offsets_by_name = _local_history_offsets(self._horse_history_path)
        try:
            with self._horse_history_path.open("rb") as handle:
                for name_key in pending:
                    for offset in offsets_by_name.get(name_key, []):
                        handle.seek(offset)
                        line = handle.readline()
                        try:
                            record = json.loads(line)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if isinstance(record, dict):
                            matches[name_key].append(record)
        except OSError as exc:
            logger.debug("Yerel TJK at gecmisi okunamadi | path=%s | hata=%s", self._horse_history_path, exc)

        self._local_history_by_name.update(matches)
        self._local_history_loaded_names.update(pending)

    def _local_horse_statistics(self, entry: RaceEntry) -> HorseStatistics | None:
        name_key = _local_horse_name_key(entry.horse_name)
        self._prime_local_horse_history([entry])
        records = self._local_history_by_name.get(name_key, [])
        source_ids: set[int] = set()
        for record in records:
            try:
                if record.get("source_horse_id") is not None:
                    source_ids.add(int(record["source_horse_id"]))
            except (TypeError, ValueError):
                continue

        source_id = entry.source_horse_id
        if source_id is None:
            if len(source_ids) != 1:
                return None
            source_id = next(iter(source_ids))
            entry.source_horse_id = source_id
            entry.id_unresolved = False
            entry.id_resolution_status = "local_history"
        elif source_ids and source_id not in source_ids:
            return None

        cache_key = (name_key, source_id)
        if cache_key in self._local_stats_cache:
            return self._local_stats_cache[cache_key]

        performances: dict[tuple[object, ...], PastPerformance] = {}
        for record in records:
            try:
                if int(record.get("source_horse_id")) != source_id:
                    continue
                performance = PastPerformance.model_validate(record)
            except (TypeError, ValueError):
                continue
            key = (
                performance.race_date,
                performance.hippodrome,
                performance.distance_m,
                performance.surface,
                performance.finish_position,
                performance.field_size,
                performance.race_time_seconds,
                performance.weight_kg,
                performance.jockey_name,
                performance.race_name,
            )
            performances[key] = performance

        past_performances = sorted(
            performances.values(), key=lambda performance: performance.race_date, reverse=True
        )
        if not past_performances:
            return None

        latest_year = max(performance.race_date.year for performance in past_performances)
        last_year = [performance for performance in past_performances if performance.race_date.year == latest_year]
        jockey_key = _horse_identity_key(entry.jockey.name)
        combo = [
            performance
            for performance in past_performances
            if performance.jockey_name and _horse_identity_key(performance.jockey_name) == jockey_key
        ]
        stats = HorseStatistics(
            horse_id=entry.horse_id,
            horse_name=entry.horse_name,
            past_performances=past_performances,
            career_starts=len(past_performances),
            career_wins=sum(performance.is_win for performance in past_performances),
            career_places=sum(performance.is_placed for performance in past_performances),
            last_year_starts=len(last_year),
            last_year_wins=sum(performance.is_win for performance in last_year),
            last_year_places=sum(performance.is_placed for performance in last_year),
            jockey_horse_combo_starts=len(combo),
            jockey_horse_combo_wins=sum(performance.is_win for performance in combo),
        )
        self._local_stats_cache[cache_key] = stats
        return stats

    def _get_html(
        self,
        url: str,
        params: dict,
        timeout: float | None = None,
        max_retries: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        cache_key = f"GET:{url}:{sorted(params.items())}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            request_options = {}
            if timeout is not None:
                request_options["timeout"] = timeout
            if max_retries is not None:
                request_options["max_retries"] = max_retries
            if headers is not None:
                request_options["headers"] = headers
            response = self._client.get(url, params=params, **request_options)
        except (httpx.HTTPError, OSError) as exc:
            logger.error(
                "TJK sayfasi alinamadi | error=%s: %s | manuel kontrol URL: %s",
                type(exc).__name__,
                exc,
                manual_check_url(url, params),
            )
            raise DataSourceError(
                f"TJK sayfasi alinamadi | {type(exc).__name__}: {exc} | "
                f"Manuel kontrol URL: {manual_check_url(url, params)}"
            ) from exc
        html = response.text
        if self._cache is not None:
            self._cache.set(cache_key, html, self._cache_ttl)
        return html

    def _post_html(self, url: str, data: dict[str, str]) -> str:
        cache_key = f"POST:{url}:{sorted(data.items())}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        if self._horse_search_unavailable:
            raise DataSourceError("TJK Atlar aramasi bu oturumda kullanilamiyor.")
        try:
            response = self._client.post(
                url,
                data,
                timeout=self._horse_search_timeout,
                max_retries=0,
            )
        except (httpx.HTTPError, OSError) as exc:
            self._horse_search_unavailable = True
            raise DataSourceError(
                f"TJK arama sayfasi alinamadi | {type(exc).__name__}: {exc} | URL: {url}"
            ) from exc
        html = response.text
        if self._cache is not None:
            self._cache.set(cache_key, html, self._cache_ttl)
        return html

    @staticmethod
    def _parse_horse_search_results(
        html: str, horse_name: str, age: int | None = None
    ) -> list[dict[str, int | str | None]]:
        expected_name = _horse_identity_key(horse_name)
        soup = BeautifulSoup(html, "lxml")
        candidates: dict[int, dict[str, int | str | None]] = {}
        for link in soup.select('a[href*="QueryParameter_AtId="]'):
            match = re.search(r"QueryParameter_AtId=(\d+)", str(link.get("href") or ""))
            linked_name = link.get_text(" ", strip=True)
            row = link.find_parent("tr")
            row_text = row.get_text(" ", strip=True) if row is not None else linked_name
            if match is None or _horse_identity_key(linked_name) != expected_name:
                continue
            age_match = re.search(r"\b(\d{1,2})\s*(?:y|yas|yaş)\b", row_text, re.IGNORECASE)
            candidate_age = int(age_match.group(1)) if age_match else None
            if age is not None and candidate_age is not None and candidate_age != age:
                continue
            source_id = int(match.group(1))
            candidates[source_id] = {
                "resolved_at_id": source_id,
                "horse_name": linked_name,
                "age": candidate_age,
                "row_text": row_text,
            }
        return list(candidates.values())

    def find_horse_id_from_search(
        self, horse_name: str, age: int | None = None
    ) -> tuple[int | None, str, list[int]]:
        """Resolve only an exact, unique Atlar result; never guess between namesakes."""
        data = {
            "QueryParameter_AtIsmi": horse_name,
            "QueryParameter_IrkId": "",
            "QueryParameter_CinsiyetId": "",
            "QueryParameter_Yas": str(age) if age is not None else "",
            "QueryParameter_BabaId": "",
            "QueryParameter_AnneId": "",
            "QueryParameter_UzerineKosanSahipId": "",
            "QueryParameter_YetistiricAdi": "",
            "QueryParameter_AntronorId": "",
            "QueryParameter_UlkeId": "",
            "QueryParameter_OLDUFLG": "0",
            "Era": "past",
            "Sort": "",
        }
        html = self._post_html(f"{self._base_url}{_HORSE_SEARCH_PATH}", data)
        candidates = self._parse_horse_search_results(html, horse_name, age)
        ids = sorted(int(item["resolved_at_id"]) for item in candidates)
        if len(ids) == 1:
            return ids[0], "resolved", ids
        if ids:
            return None, "ambiguous", ids
        return None, "unresolved", []

    @staticmethod
    def _daily_program_csv_url(target_date: date, city: str) -> str:
        return (
            f"{_DAILY_PROGRAM_CSV_BASE_URL}/{target_date:%Y}/{target_date:%Y-%m-%d}/CSV/"
            f"GunlukYarisProgrami/{target_date:%d.%m.%Y}-{city}-GunlukYarisProgrami-TR.csv"
        )

    @staticmethod
    def _daily_results_csv_url(target_date: date, city: str) -> str:
        return (
            f"{_DAILY_PROGRAM_CSV_BASE_URL}/{target_date:%Y}/{target_date:%Y-%m-%d}/CSV/"
            f"GunlukYarisSonuclari/{target_date:%d.%m.%Y}-{city}-GunlukYarisSonuclari-TR.csv"
        )

    def _get_csv(self, url: str) -> str | None:
        cache_key = f"GET:{url}:[]"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        try:
            response = self._client.get(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise DataSourceError(
                f"TJK CSV dosyasi alinamadi | HTTP {exc.response.status_code} | URL: {url}"
            ) from exc
        except (httpx.HTTPError, OSError) as exc:
            raise DataSourceError(
                f"TJK CSV dosyasi alinamadi | {type(exc).__name__}: {exc} | URL: {url}"
            ) from exc
        csv_text = response.text
        if self._cache is not None:
            self._cache.set(cache_key, csv_text, self._cache_ttl)
        return csv_text

    def get_daily_races_csv(self, target_date: date, city: str) -> list[Race]:
        """Download and parse the official daily-program CSV for one city."""
        matched = _match_hippodrome(city)
        if matched is None:
            return []
        url = self._daily_program_csv_url(target_date, matched)
        csv_text = self._get_csv(url)
        if csv_text is None:
            return []
        races = self._parse_daily_program_csv(csv_text, matched, target_date)
        if not races:
            raise DataSourceError(f"TJK CSV dosyasinda kosu bulunamadi: {url}")
        return races

    def get_daily_races_html(
        self,
        target_date: date,
        city: str,
        race_no: int | None = None,
    ) -> list[Race]:
        """Fetch the city HTML program explicitly, bypassing CSV-first lookup."""
        matched = _match_hippodrome(city)
        if matched is None:
            return []
        sehir_id = _KNOWN_HIPPODROME_SEHIR_IDS.get(matched)
        if sehir_id is None:
            sehir_id = self._get_hippodrome_sehir_ids(target_date).get(matched)
        if sehir_id is None:
            return []
        url = f"{self._base_url}{_DAILY_PROGRAM_CITY_PATH}"
        params = {
            "SehirId": sehir_id,
            "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
            "SehirAdi": matched,
            "Era": _era_for_date(target_date),
        }
        program_request_options = {}
        if self._program_request_timeout != self._request_timeout:
            program_request_options["timeout"] = self._program_request_timeout
        if self._program_max_retries != self._max_retries:
            program_request_options["max_retries"] = self._program_max_retries
        referer_era = _era_for_referer_date(target_date)
        program_request_options["headers"] = {
            "Referer": (
                f"{self._base_url}{_DAILY_PROGRAM_PATH}?"
                f"{urlencode({'QueryParameter_Tarih': target_date.strftime('%d/%m/%Y'), 'Era': referer_era})}"
            )
        }
        try:
            html = self._get_html(url, params, **program_request_options)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                raise
            logger.warning(
                "TJK program HTML'i HTTP %s dondu; bir kez daha denenecek | tarih=%s | hipodrom=%s",
                exc.response.status_code,
                target_date.isoformat(),
                matched,
            )
            html = self._get_html(url, params, **program_request_options)
        races = self._parse_daily_program(html, matched, target_date)
        if race_no is not None:
            races = [race for race in races if race.race_no == race_no]
        self._prime_local_horse_history([entry for race in races for entry in race.entries])
        return races

    def get_daily_result_horse_ids_html(
        self, target_date: date, city: str
    ) -> list[tuple[int, str, int]]:
        """Return (race number, horse name, TJK horse ID) from result HTML links."""
        matched = _match_hippodrome(city)
        if matched is None:
            return []
        sehir_id = _KNOWN_HIPPODROME_SEHIR_IDS.get(matched)
        if sehir_id is None:
            sehir_id = self._get_results_hippodrome_sehir_ids(target_date).get(matched)
        if sehir_id is None:
            return []
        url = f"{self._base_url}{_DAILY_RESULTS_CITY_PATH}"
        params = {
            "SehirId": sehir_id,
            "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
            "SehirAdi": matched,
        }
        html = self._get_html(url, params)
        return self._parse_daily_result_horse_ids(html)

    @staticmethod
    def _parse_daily_result_horse_ids(html: str) -> list[tuple[int, str, int]]:
        soup = BeautifulSoup(html, "lxml")
        observations: list[tuple[int, str, int]] = []
        for header_text in soup.find_all(string=_RACE_HEADER_RE):
            match = _RACE_HEADER_RE.search(str(header_text))
            if match is None:
                continue
            race_no = int(match.group(1))
            table = header_text.parent.find_next("table")
            if table is None:
                continue
            for row in table.find_all("tr"):
                link = row.find("a", href=re.compile(r"QueryParameter_AtId=\d+"))
                if link is None:
                    continue
                id_match = re.search(r"QueryParameter_AtId=(\d+)", str(link.get("href") or ""))
                horse_name = link.get_text(" ", strip=True)
                if id_match and horse_name:
                    observations.append((race_no, horse_name, int(id_match.group(1))))
        return observations

    @staticmethod
    def merge_market_odds(csv_races: list[Race], html_races: list[Race]) -> int:
        """Join TJK HTML Gny values onto the matching daily CSV entries."""
        html_odds: dict[tuple[int, str], set[float]] = {}
        for race in html_races:
            for entry in race.entries:
                if entry.odds is not None:
                    key = (race.race_no, _local_horse_name_key(entry.horse_name))
                    html_odds.setdefault(key, set()).add(entry.odds)

        merged = 0
        for race in csv_races:
            for entry in race.entries:
                if entry.odds is not None:
                    continue
                key = (race.race_no, _local_horse_name_key(entry.horse_name))
                candidate_odds = html_odds.get(key, set())
                if len(candidate_odds) == 1:
                    entry.odds = next(iter(candidate_odds))
                    merged += 1
        return merged

    @staticmethod
    def merge_source_ids(csv_races: list[Race], html_races: list[Race]) -> int:
        """Join HTML IDs onto CSV rows using the complete race event and horse name."""
        html_entries: dict[tuple[date, str, int, str], set[int]] = {}
        html_trainers: dict[tuple[date, str, int, str], int] = {}
        for race in html_races:
            event_prefix = (
                race.start_time.date(),
                _normalize_city(race.hippodrome),
                race.race_no,
            )
            for entry in race.entries:
                key = (*event_prefix, _horse_identity_key(entry.horse_name))
                if entry.source_horse_id is not None:
                    html_entries.setdefault(key, set()).add(entry.source_horse_id)
                if entry.trainer.source_trainer_id is not None:
                    html_trainers[key] = entry.trainer.source_trainer_id
        merged = 0
        for race in csv_races:
            event_prefix = (
                race.start_time.date(),
                _normalize_city(race.hippodrome),
                race.race_no,
            )
            for entry in race.entries:
                key = (*event_prefix, _horse_identity_key(entry.horse_name))
                candidate_ids = html_entries.get(key, set())
                if len(candidate_ids) != 1:
                    if len(candidate_ids) > 1:
                        logger.warning(
                            "Ayni kosu ve at adi icin celiskili TJK AtId degerleri | tarih=%s | hipodrom=%s | kosu=%s | at=%s | ids=%s",
                            event_prefix[0], event_prefix[1], event_prefix[2],
                            entry.horse_name, sorted(candidate_ids),
                        )
                    continue
                html_id = next(iter(candidate_ids))
                if entry.source_horse_id not in (None, html_id):
                    logger.warning(
                        "CSV/HTML AtId uyusmazligi | tarih=%s | hipodrom=%s | kosu=%s | at=%s | csv=%s | html=%s",
                        event_prefix[0], event_prefix[1], event_prefix[2],
                        entry.horse_name, entry.source_horse_id, html_id,
                    )
                    continue
                entry.source_horse_id = html_id
                if entry.trainer.source_trainer_id is None:
                    entry.trainer.source_trainer_id = html_trainers.get(key)
                merged += 1
        return merged

    @staticmethod
    def _parse_daily_program_csv(csv_text: str, hippodrome: str, target_date: date) -> list[Race]:
        races: list[Race] = []
        current_race: Race | None = None
        race_entries: list[RaceEntry] = []
        program_odds_index: int | None = None

        def save_current_race() -> None:
            if current_race is not None and race_entries:
                current_race.entries = list(race_entries)
                races.append(current_race)

        for row in csv.reader(io.StringIO(csv_text.lstrip("\ufeff")), delimiter=";"):
            first_cell = row[0].strip() if row else ""
            race_match = _CSV_PROGRAM_RACE_HEADER_RE.match(first_cell)
            if race_match:
                save_current_race()
                program_odds_index = None
                race_no = int(race_match.group(1))
                start_time = datetime.combine(
                    target_date,
                    datetime.strptime(
                        f"{race_match.group(2)}:{race_match.group(3)}", "%H:%M"
                    ).time(),
                )
                row_details = " ".join(row[1:])
                distance_match = re.search(r"(\d{3,4})\s*m\b", row_details, re.I)
                distance = int(distance_match.group(1)) if distance_match else 0
                surface_match = re.search(r"\b(Kum|Cim|Çim|Sentetik)\b", row_details, re.I)
                surface = (
                    _SURFACE_MAP.get(surface_match.group(1).casefold(), TrackSurface.KUM)
                    if surface_match
                    else TrackSurface.KUM
                )
                group_info = " - ".join(
                    value.strip() for value in row[1:3] if value.strip()
                ) or None
                current_race = Race(
                    id=f"{hippodrome}-{target_date.isoformat()}-{race_no}",
                    hippodrome=hippodrome,
                    race_no=race_no,
                    start_time=start_time,
                    distance_m=distance,
                    surface=surface,
                    group_info=group_info,
                )
                race_entries = []
                continue

            if current_race is None or not row:
                continue
            if first_cell.casefold() == "at no":
                program_odds_index = next(
                    (
                        index
                        for index, header in enumerate(row)
                        if re.sub(r"[^a-z0-9]", "", header.casefold()) in {"gny", "ganyan", "odds"}
                    ),
                    None,
                )
                continue
            if not first_cell.isdigit():
                continue
            if len(row) < 9:
                continue
            try:
                entry = TJKHtmlDataSource._parse_entry_row(
                    row,
                    odds_index=program_odds_index,
                    column_indices={
                        "no": 0,
                        "at": 1,
                        "yas": 2,
                        "kilo": 5,
                        "jokey": 6,
                        "antrenor": 8,
                        "hp": 11,
                        "son6": 12,
                    },
                    fallback_odds=False,
                )
            except (ValueError, IndexError) as exc:
                logger.debug("TJK CSV satiri ayristirilamadi, atlaniyor: %s (%s)", row, exc)
                continue
            race_entries.append(entry)

        save_current_race()
        return races

    def _get_hippodrome_sehir_ids(self, target_date: date) -> dict[str, int]:
        """Secili tarih icin TJK sekmelerindeki resmi hipodrom->SehirId
        eslesmesini dondurur."""
        url = f"{self._base_url}{_DAILY_PROGRAM_DATA_PATH}"
        params = {
            "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
            "Era": _era_for_date(target_date),
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
            "Era": _era_for_date(target_date),
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
            program_races = self.get_daily_races_csv(target_date, matched)
            if program_races:
                horse_numbers = {
                    race.race_no: {
                        _normalize_horse_name(entry.horse_name): entry.number
                        for entry in race.entries
                    }
                    for race in program_races
                }
                csv_results = self._parse_csv_results(
                    self._get_csv(self._daily_results_csv_url(target_date, matched)) or "",
                    horse_numbers_by_race_name=horse_numbers,
                )
                if csv_results:
                    return csv_results
        except DataSourceError as exc:
            logger.warning(
                "TJK CSV sonuclari alinamadi; sehir sonucu sayfasi denenecek | tarih=%s | hipodrom=%s | hata=%s",
                target_date.isoformat(),
                matched,
                exc,
            )

        sehir_id = _KNOWN_HIPPODROME_SEHIR_IDS.get(matched)
        if sehir_id is None:
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
        }
        html = self._get_html(url, params)
        csv_link = BeautifulSoup(html, "lxml").select_one("a#CSVBulten[href]")
        if csv_link is not None:
            csv_url = csv_link.get("href")
            if csv_url:
                program_races = self.get_daily_races_csv(target_date, matched)
                horse_numbers = {
                    race.race_no: {
                        _normalize_horse_name(entry.horse_name): entry.number
                        for entry in race.entries
                    }
                    for race in program_races
                }
                csv_results = self._parse_csv_results(
                    self._get_html(str(csv_url), {}),
                    horse_numbers_by_race_name=horse_numbers,
                )
                if csv_results:
                    return csv_results

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

    @staticmethod
    def _parse_csv_results(
        csv_text: str,
        horse_numbers_by_race_name: dict[int, dict[str, int]] | None = None,
    ) -> dict[int, dict[int, int]]:
        results: dict[int, dict[int, int]] = {}
        current_race_no: int | None = None
        in_entries = False
        for row in csv.reader(io.StringIO(csv_text.lstrip("\ufeff")), delimiter=";"):
            first_cell = row[0].strip() if row else ""
            race_match = _CSV_RACE_HEADER_RE.match(first_cell)
            if race_match:
                current_race_no = int(race_match.group(1))
                results.setdefault(current_race_no, {})
                in_entries = False
                continue
            if first_cell.casefold() == "at no":
                in_entries = True
                continue
            if not in_entries or current_race_no is None or not first_cell.isdigit():
                continue
            race_results = results[current_race_no]
            finish_position = len(race_results) + 1
            if horse_numbers_by_race_name is None:
                horse_number = int(first_cell)
            else:
                horse_name = row[1] if len(row) > 1 else ""
                normalized_name = _normalize_horse_name(horse_name)
                race_names = horse_numbers_by_race_name.get(current_race_no, {})
                horse_number = race_names.get(normalized_name)
                if horse_number is None:
                    matches = [
                        number
                        for program_name, number in race_names.items()
                        if program_name.startswith(normalized_name)
                        or normalized_name.startswith(program_name)
                    ]
                    if len(set(matches)) == 1:
                        horse_number = matches[0]
                if horse_number is None:
                    continue
            race_results.setdefault(horse_number, finish_position)
        return {race_no: race_results for race_no, race_results in results.items() if race_results}

    def get_available_hippodromes(self, target_date: date) -> list[str]:
        """Return domestic hippodromes with a program CSV for the selected date."""
        availability_cache_key = f"GET:available_hippodromes:{target_date.isoformat()}"
        if self._cache is not None:
            cached = self._cache.get(availability_cache_key)
            if isinstance(cached, list):
                return [str(city) for city in cached]

        available: list[str] = []
        request_failed = False
        for city in KNOWN_HIPPODROMES:
            url = self._daily_program_csv_url(target_date, city)
            cache_key = f"GET:{url}:[]"
            if self._cache is not None:
                csv_text = self._cache.get(cache_key)
            else:
                csv_text = None
            try:
                if csv_text is None:
                    response = self._client.get(
                        url,
                        timeout=5.0,
                        max_retries=0,
                        min_interval=0.15,
                    )
                    csv_text = response.text
                    if self._cache is not None:
                        self._cache.set(cache_key, csv_text, self._cache_ttl)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    request_failed = True
                    logger.debug(
                        "Hipodrom CSV kontrolu basarisiz | tarih=%s | hipodrom=%s | HTTP=%s",
                        target_date.isoformat(), city, exc.response.status_code,
                    )
                continue
            except (httpx.HTTPError, OSError) as exc:
                request_failed = True
                logger.debug(
                    "Hipodrom CSV kontrolu basarisiz | tarih=%s | hipodrom=%s | hata=%s",
                    target_date.isoformat(), city, exc,
                )
                continue
            if csv_text and self._parse_daily_program_csv(csv_text, city, target_date):
                available.append(city)

        if available:
            if self._cache is not None:
                self._cache.set(availability_cache_key, available, self._cache_ttl)
            return available
        if request_failed:
            raise DataSourceError(
                f"{target_date.isoformat()} tarihli TJK hipodrom CSV listesi alinamadi."
            )
        if self._cache is not None:
            self._cache.set(availability_cache_key, [], self._cache_ttl)
        return []

    def get_daily_races(
        self,
        target_date: date,
        city: str | None = None,
        resolve_missing_ids: bool = True,
        include_odds: bool = True,
        race_no: int | None = None,
    ) -> list[Race]:
        """Gunun kosu programini dondurur.

        ``resolve_missing_ids=True`` (varsayilan) oldugunda, CSV bultenin
        tasimadigi TJK ``AtId`` degerleri, ayni tarih/hipodrom icin program/
        sonuc HTML sayfalarindaki isim->AtId baglantilari ve gerekirse
        'Atlar' arama sayfasi ile doldurulmaya calisilir (bkz.
        ``_enrich_source_horse_ids``). Bu adim opsiyoneldir ve basarisiz
        olursa CSV verisini bozmaz; yalnizca ``source_horse_id`` alani
        bos kalabilir.
        """
        if city:
            matched = _match_hippodrome(city)
            if matched is None:
                logger.warning(
                    "Bilinmeyen sehir/hipodrom: '%s' (bilinen hipodromlar: %s)",
                    city,
                    ", ".join(KNOWN_HIPPODROMES),
                )
                return []
            csv_error: DataSourceError | None = None
            try:
                races = self.get_daily_races_csv(target_date, matched)
                if race_no is not None and races:
                    races = [race for race in races if race.race_no == race_no]
            except DataSourceError as exc:
                csv_error = exc
                logger.warning(
                    "TJK CSV programi alinamadi; sehir sayfasi denenecek | tarih=%s | hipodrom=%s | hata=%s",
                    target_date.isoformat(),
                    matched,
                    exc,
                )
            else:
                if races:
                    if include_odds and any(
                        entry.odds is None
                        for race in races
                        for entry in race.entries
                    ):
                        try:
                            try:
                                html_races = self.get_daily_races_html(target_date, matched)
                            except httpx.HTTPStatusError as exc:
                                if exc.response.status_code < 500:
                                    raise
                                logger.warning(
                                    "TJK program HTML'i HTTP %s dondu; Gny icin bir kez daha denenecek | tarih=%s | hipodrom=%s",
                                    exc.response.status_code,
                                    target_date.isoformat(),
                                    matched,
                                )
                                html_races = self.get_daily_races_html(target_date, matched)
                        except Exception as exc:  # noqa: BLE001 - CSV verisi kullanilabilir kalir
                            html_races = []
                            logger.warning(
                                "TJK program HTML'inden Gny oranlari alinamadi | tarih=%s | hipodrom=%s | hata=%s",
                                target_date.isoformat(),
                                matched,
                                exc,
                            )
                        if html_races:
                            merged_odds = self.merge_market_odds(races, html_races)
                            merged_ids = self.merge_source_ids(races, html_races)
                            logger.info(
                                "TJK program HTML'inden %s Gny orani ve %s AtId eslesti | tarih=%s | hipodrom=%s",
                                merged_odds,
                                merged_ids,
                                target_date.isoformat(),
                                matched,
                            )
                    self._prime_local_horse_history([entry for race in races for entry in race.entries])
                    if resolve_missing_ids:
                        races = self._enrich_source_horse_ids(
                            races, target_date, matched, resolve_missing_ids=True
                        )
                    return races

            sehir_id = _KNOWN_HIPPODROME_SEHIR_IDS.get(matched)
            if sehir_id is None:
                try:
                    sehir_ids = self._get_hippodrome_sehir_ids(target_date)
                except Exception as exc:  # noqa: BLE001
                    if csv_error is not None:
                        raise csv_error
                    logger.warning(
                        "TJK SehirId listesi alinamadi; CSV programinda da yaris bulunamadi | tarih=%s | hipodrom=%s | hata=%s",
                        target_date.isoformat(),
                        matched,
                        exc,
                    )
                    return []
                sehir_id = sehir_ids.get(matched)
            if sehir_id is None:
                logger.warning(
                    "'%s' icin TJK SehirId bulunamadi ve CSV programinda yaris yok.", matched
                )
                if csv_error is not None:
                    raise csv_error
                return []

            url = f"{self._base_url}{_DAILY_PROGRAM_CITY_PATH}"
            params = {
                "SehirId": sehir_id,
                "QueryParameter_Tarih": target_date.strftime("%d/%m/%Y"),
                "SehirAdi": matched,
                "Era": _era_for_date(target_date),
            }
            try:
                html = self._get_html(url, params)
                races = self._parse_daily_program(html, matched, target_date)
                if race_no is not None:
                    races = [race for race in races if race.race_no == race_no]
            except Exception as exc:  # noqa: BLE001 - DataSourceError olarak yeniden firlatilir
                if csv_error is not None:
                    raise csv_error
                logger.warning(
                    "TJK sehir programi alinamadi ve CSV programinda yaris yok | tarih=%s | hipodrom=%s | hata=%s",
                    target_date.isoformat(),
                    matched,
                    exc,
                )
                return []
            if races:
                self._prime_local_horse_history([entry for race in races for entry in race.entries])
                if resolve_missing_ids:
                    races = self._enrich_source_horse_ids(
                        races, target_date, matched, resolve_missing_ids=True
                    )
                return races
            logger.warning(
                "'%s' icin hicbir kosu ayristirilamadi; TJK sayfayi JavaScript/AJAX ile "
                "render ediyor olabilir (bkz. README.md > 'Bilinen Kisitlar') veya bu "
                "hipodromda o tarihte yaris yoktur.",
                matched,
            )
            if csv_error is not None:
                raise csv_error
            return []
        races: list[Race] = []
        for name in KNOWN_HIPPODROMES:
            races.extend(
                self.get_daily_races(target_date, name, resolve_missing_ids=resolve_missing_ids)
            )
        return races

    def _enrich_source_horse_ids(
        self,
        races: list[Race],
        target_date: date,
        city: str,
        resolve_missing_ids: bool = True,
    ) -> list[Race]:
        """CSV bultenden gelen kosu girdilerine mumkun oldugunca TJK
        ``AtId`` (source_horse_id) degerini ekler.

        KOK NEDEN: TJK'nin gunluk program/sonuc CSV dosyalari yalnizca at
        ISMINI tasir, ID tasimaz; ID yalnizca HTML sayfalarindaki
        ``QueryParameter_AtId=...`` baglantilarinda bulunur. Bu adim
        calisturulmadan CSV kaynakli girdiler icin ``source_horse_id``
        daima ``None`` kalir ve ``get_horse_statistics`` gibi ID gerektiren
        cagrilar basarisiz olur.

        Sira (guvenilirlik azalan, maliyet artan sirayla):
          1) Ayni tarih/hipodrom icin gunluk PROGRAM HTML sayfasi -
             ``merge_source_ids`` ile (tarih, hipodrom, kosu no, at adi)
             anahtariyla birebir/celismesiz eslesenler kabul edilir.
          2) Ayni tarih/hipodrom icin gunluk SONUC HTML sayfasi (kosu
             sonuclanmissa) - ayni anahtar mantigiyla.
          3) Hala eksik kalanlar icin 'Atlar' arama sayfasi - YALNIZCA tek
             ve kesin (adas olmayan) eslesme varsa ID atanir; birden fazla
             aday varsa (isim celismesi/adas at riski) ID atanMAZ, uyari
             loglanir.

        Her adim best-effort'tur: HTML/arama sayfalarindan biri alinamazsa
        (agdan dusme, sayfa yapisi degisti, arama devre disi vb.) CSV
        verisi bozulmadan, yalnizca ilgili adim atlanarak devam edilir.
        """
        if not races:
            return races

        entries = [entry for race in races for entry in race.entries]
        self._prime_local_horse_history(entries)
        for entry in entries:
            if entry.source_horse_id is not None:
                continue
            source_ids: set[int] = set()
            for record in self._local_history_by_name.get(_local_horse_name_key(entry.horse_name), []):
                try:
                    if record.get("source_horse_id") is not None:
                        source_ids.add(int(record["source_horse_id"]))
                except (TypeError, ValueError):
                    continue
            if len(source_ids) == 1:
                entry.source_horse_id = next(iter(source_ids))
                entry.id_unresolved = False
                entry.id_resolution_status = "local_history"

        def _missing_count() -> int:
            return sum(
                1 for race in races for entry in race.entries if entry.source_horse_id is None
            )

        before = _missing_count()
        if before == 0:
            return races

        # 1) Gunluk program HTML'i.
        try:
            html_program_races = self.get_daily_races_html(target_date, city)
        except Exception as exc:  # noqa: BLE001 - eslesme opsiyonel, ana veriyi bozmaz
            html_program_races = []
            logger.debug(
                "TJK program HTML'i AtId eslesmesi icin alinamadi | tarih=%s | hipodrom=%s | hata=%s",
                target_date.isoformat(),
                city,
                exc,
            )
        if html_program_races:
            merged = self.merge_source_ids(races, html_program_races)
            if merged:
                logger.info(
                    "TJK program HTML'inden %s at icin AtId eslesti | tarih=%s | hipodrom=%s",
                    merged,
                    target_date.isoformat(),
                    city,
                )

        # 2) Gunluk sonuc HTML'i (kosu sonuclanmissa).
        if _missing_count() > 0:
            try:
                result_ids = self.get_daily_result_horse_ids_html(target_date, city)
            except Exception as exc:  # noqa: BLE001
                result_ids = []
                logger.debug(
                    "TJK sonuc HTML'i AtId eslesmesi icin alinamadi | tarih=%s | hipodrom=%s | hata=%s",
                    target_date.isoformat(),
                    city,
                    exc,
                )
            if result_ids:
                candidates_by_key: dict[tuple[int, str], set[int]] = {}
                for race_no, horse_name, horse_id in result_ids:
                    key = (race_no, _horse_identity_key(horse_name))
                    candidates_by_key.setdefault(key, set()).add(horse_id)
                merged = 0
                for race in races:
                    for entry in race.entries:
                        if entry.source_horse_id is not None:
                            continue
                        key = (race.race_no, _horse_identity_key(entry.horse_name))
                        ids = candidates_by_key.get(key, set())
                        if len(ids) == 1:
                            entry.source_horse_id = next(iter(ids))
                            merged += 1
                        elif len(ids) > 1:
                            logger.warning(
                                "Sonuc HTML'inde ayni kosu/at adi icin celiskili AtId | "
                                "tarih=%s | hipodrom=%s | kosu=%s | at=%s | ids=%s",
                                target_date.isoformat(),
                                city,
                                race.race_no,
                                entry.horse_name,
                                sorted(ids),
                            )
                if merged:
                    logger.info(
                        "TJK sonuc HTML'inden %s at icin AtId eslesti | tarih=%s | hipodrom=%s",
                        merged,
                        target_date.isoformat(),
                        city,
                    )

        # 3) Kalanlar icin 'Atlar' arama sayfasi - yalnizca kesin eslesme.
        if resolve_missing_ids and _missing_count() > 0:
            for race in races:
                for entry in race.entries:
                    if entry.source_horse_id is not None:
                        continue
                    try:
                        resolved_id, status, candidate_ids = self.find_horse_id_from_search(
                            entry.horse_name, entry.age
                        )
                    except DataSourceError as exc:
                        logger.debug(
                            "TJK Atlar aramasi bu oturumda kullanilamiyor | at=%s | hata=%s",
                            entry.horse_name,
                            exc,
                        )
                        break
                    if status == "resolved" and resolved_id is not None:
                        entry.source_horse_id = resolved_id
                    elif status == "ambiguous":
                        logger.warning(
                            "TJK Atlar aramasinda adas atlar bulundu, ID atanmadi "
                            "(yanlis at ile eslesme riski) | at=%s | yas=%s | adaylar=%s",
                            entry.horse_name,
                            entry.age,
                            candidate_ids,
                        )
                    else:
                        logger.debug(
                            "TJK Atlar aramasinda at bulunamadi | at=%s | yas=%s",
                            entry.horse_name,
                            entry.age,
                        )
                else:
                    continue
                break

        after = _missing_count()
        if after:
            logger.info(
                "TJK AtId eslesmesi tamamlandi: %s/%s girdi icin ID bulunamadi "
                "(kalanlar source_horse_id=None olarak birakildi) | tarih=%s | hipodrom=%s",
                after,
                before,
                target_date.isoformat(),
                city,
            )
        elif before:
            logger.info(
                "TJK AtId eslesmesi tamamlandi: %s girdinin tamami icin ID bulundu | "
                "tarih=%s | hipodrom=%s",
                before,
                target_date.isoformat(),
                city,
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
        column_indices: dict[str, int] = {}
        for header_row in table.find_all("tr"):
            headers = header_row.find_all("th")
            if not headers:
                continue
            for index, header in enumerate(headers):
                header_text = re.sub(r"[^a-z0-9]", "", header.get_text(" ", strip=True).casefold())
                if header_text:
                    column_indices.setdefault(header_text, index)
            if column_indices:
                break
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all("td")]
            if len(cells) < 8:
                continue
            try:
                entries.append(self._parse_entry_row(cells, row, column_indices=column_indices))
            except (ValueError, IndexError) as exc:
                logger.debug("Satir ayristirilamadi, atlaniyor: %s (%s)", cells, exc)
        return entries

    @staticmethod
    def _parse_entry_row(
        cells: list[str],
        row=None,
        odds_index: int | None = None,
        column_indices: dict[str, int] | None = None,
        fallback_odds: bool = True,
    ) -> RaceEntry:  # type: ignore[no-untyped-def]
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

        column_indices = column_indices or {}

        def _header_index(*aliases: str) -> int | None:
            for alias in aliases:
                normalized = re.sub(r"[^a-z0-9]", "", alias.casefold())
                if normalized in column_indices:
                    return column_indices[normalized]
            return None

        number_idx = _header_index("no", "atno") or 0
        for i in range(min(3, len(cells))):
            if re.fullmatch(r"\d+", cells[i] or ""):
                number_idx = i
                break

        number = int(re.sub(r"\D", "", cells[number_idx]) or 0)
        name_idx = _header_index("at", "atadi", "horse", "horsename")
        if name_idx is None:
            name_idx = min(number_idx + 1, len(cells) - 1)
        raw_name = cells[name_idx]
        is_scratched = "Kosmaz" in raw_name or "Koşmaz" in raw_name
        horse_name = re.sub(r"\(.*?\)", "", raw_name).replace("Kosmaz", "").replace("Koşmaz", "")
        horse_name = re.sub(r"\bt\s*\d[\d\.,]*\s*TL\b", "", horse_name, flags=re.IGNORECASE)
        horse_name = horse_name.split("Kapalı gözlük", 1)[0].split("Dilinin bağlanacağını", 1)[0]
        horse_name = horse_name.split("Ring mahalinden", 1)[0]
        horse_name = re.sub(
            r"(?:\s+(?:KG|DB|SK|K|D|B|GKR|KB|KBB))+\s*$", "", horse_name
        ).strip()

        age = None
        age_idx = _header_index("yas", "age")
        if age_idx is None:
            age_idx = min(number_idx + 2, len(cells) - 1)
        age_match = re.search(r"(\d+)\s*y", cells[age_idx], flags=re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))

        weight_idx = _header_index("kilo", "siklet", "weight")
        jockey_idx = _header_index("jokey", "jockey")
        trainer_idx = _header_index("antrenor", "antrenör", "trainer")
        hp_idx = _header_index("hp", "handikap", "handicappoints")
        form_idx = _header_index("son6", "son6kosu", "form", "recentform")
        weight_idx = min(number_idx + 4, len(cells) - 1) if weight_idx is None else weight_idx
        jockey_idx = min(number_idx + 5, len(cells) - 1) if jockey_idx is None else jockey_idx
        trainer_idx = min(number_idx + 7, len(cells) - 1) if trainer_idx is None else trainer_idx
        hp_idx = min(number_idx + 9, len(cells) - 1) if hp_idx is None else hp_idx
        form_idx = min(number_idx + 10, len(cells) - 1) if form_idx is None else form_idx

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
            if not recent_form_positions:
                recent_form_positions = [
                    int(value) for value in re.findall(r"\d+", form_raw)
                ]

        header_odds_index = _header_index("gny", "ganyan", "odds")
        odds_index = header_odds_index if header_odds_index is not None else odds_index
        odds_cell_index = (
            odds_index if odds_index is not None and odds_index >= 0
            else len(cells) + odds_index if odds_index is not None
            else None
        )
        odds_raw = (
            cells[odds_cell_index]
            if odds_cell_index is not None and 0 <= odds_cell_index < len(cells)
            else None
        )
        if row is not None:
            for cell in row.find_all("td"):
                marker_values: list[str] = []
                for attribute in ("class", "id", "data-field", "data-column", "title"):
                    value = cell.get(attribute, "")
                    if isinstance(value, (list, tuple)):
                        marker_values.extend(str(item) for item in value)
                    else:
                        marker_values.append(str(value))
                markers = " ".join(marker_values).casefold()
                if re.search(r"(?:^|[\s_-])(gny|odds|ganyan)(?:$|[\s_-])", markers):
                    odds_raw = cell.get_text(" ", strip=True)
                    break
        if fallback_odds and odds_raw is None and len(cells) >= 3:
            odds_raw = cells[-3]
        odds = None
        if odds_raw and odds_raw not in {"-", ""}:
            try:
                odds = float(odds_raw.replace(",", "."))
            except ValueError:
                odds = None

        source_horse_id = None
        source_trainer_id = None
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
            trainer_link = row.find("a", href=re.compile(r"QueryParameter_AntrenorId=\d+", re.IGNORECASE))
            trainer_markup = str(trainer_link or row)
            trainer_match = re.search(r"QueryParameter_AntrenorId=(\d+)", trainer_markup, re.IGNORECASE)
            if trainer_match:
                source_trainer_id = int(trainer_match.group(1))

        return RaceEntry(
            number=number,
            horse_id=f"{horse_name}-{number}",
            source_horse_id=source_horse_id,
            horse_name=horse_name,
            age=age,
            jockey=Jockey(name=jockey_name),
            trainer=Trainer(name=trainer_name, source_trainer_id=source_trainer_id),
            weight_kg=weight,
            odds=odds,
            handicap_points=handicap_points,
            recent_form_positions=recent_form_positions,
            form_raw=form_raw,
            is_scratched=is_scratched,
        )

    def get_trainer_statistics(
        self,
        trainer_id: int,
        request_timeout: float | None = None,
        max_retries: int | None = None,
    ) -> TrainerStatistics:
        """TJK antrenor ozet tablosunu baslik adlarina gore parse eder."""
        if trainer_id <= 0:
            raise DataSourceError("Gecerli bir QueryParameter_AntrenorId bulunamadi.")
        url = f"{self._base_url}{_TRAINER_STATISTICS_PATH}"
        params = {"1": "1", "QueryParameter_AntrenorId": str(trainer_id)}
        if request_timeout is None and max_retries is None:
            html = self._get_html(url, params)
        else:
            html = self._get_html(url, params, timeout=request_timeout, max_retries=max_retries)
        soup = BeautifulSoup(html, "lxml")

        def normalize(value: str) -> str:
            folded = value.casefold().translate(_TURKISH_FOLD)
            return re.sub(r"[^a-z0-9%]", "", folded)

        def number(value: str) -> float:
            text = (value or "").replace("%", "").strip()
            text = text.replace(".", "").replace(",", ".") if "," in text and "." in text else text.replace(",", ".")
            match = re.search(r"\d+(?:\.\d+)?", text)
            return float(match.group(0)) if match else 0.0

        aliases = {
            "trainer_name": {"antrenor", "trainer"},
            "total_starts": {"kosu", "kosusayisi", "starts"},
            "first_place": {"1"}, "second_place": {"2"}, "third_place": {"3"},
            "fourth_place": {"4"}, "fifth_place": {"5"},
            "first_rate": {"1%"}, "second_rate": {"2%"}, "third_rate": {"3%"},
            "fourth_rate": {"4%"}, "fifth_rate": {"5%"},
        }
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            header_cells = rows[0].find_all(["th", "td"])
            header_map = {normalize(cell.get_text(" ", strip=True)): index for index, cell in enumerate(header_cells)}
            if not ({"kosu", "1", "2", "3"} <= set(header_map)):
                continue
            for row in rows[1:]:
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                if len(cells) < len(header_map):
                    continue
                values: dict[str, float | str] = {}
                for field, field_aliases in aliases.items():
                    index = next((header_map[alias] for alias in field_aliases if alias in header_map), None)
                    if index is None or index >= len(cells):
                        continue
                    values[field] = cells[index] if field == "trainer_name" else number(cells[index])
                if not values.get("total_starts"):
                    continue
                return TrainerStatistics(
                    trainer_id=trainer_id,
                    trainer_name=str(values.get("trainer_name", "")),
                    **{field: int(value) if field in {"total_starts", "first_place", "second_place", "third_place", "fourth_place", "fifth_place"} else float(value) for field, value in values.items() if field != "trainer_name"},
                )
        raise DataSourceError("AntrenorIstatistikleri tablosu bulunamadi veya parse edilemedi.")

    @staticmethod
    def _parse_horse_page_identity(html: str, expected_name: str) -> tuple[str, list[str]]:
        soup = BeautifulSoup(html, "lxml")
        expected = _horse_identity_key(expected_name)
        visible_text = [text.strip() for text in soup.stripped_strings if text.strip()]
        exact_matches = [
            text for text in visible_text
            if _horse_identity_key(text) == expected
        ]
        if exact_matches:
            return "matched", exact_matches
        headings = [
            node.get_text(" ", strip=True)
            for node in soup.select("h1, h2, h3, [id*='AtAdi'], [class*='AtAdi']")
            if node.get_text(" ", strip=True)
        ]
        return ("mismatch", headings) if headings else ("unverifiable", [])

    def validate_horse_id_name(self, horse_id: int, expected_name: str) -> tuple[str, list[str]]:
        url = f"{self._base_url}{_HORSE_HISTORY_PATH}"
        params = {"QueryParameter_AtId": str(horse_id), "Era": "past"}
        cache_key = f"GET:{url}:{sorted(params.items())}"
        html = self._cache.get(cache_key) if self._cache is not None else None
        if html is None:
            try:
                response = self._client.get(
                    url,
                    params,
                    timeout=min(self._horse_search_timeout, 8.0),
                    max_retries=0,
                )
            except (httpx.HTTPError, OSError) as exc:
                raise DataSourceError(
                    f"TJK AtKosuBilgileri dogrulamasi alinamadi | {type(exc).__name__}: {exc}"
                ) from exc
            html = response.text
            if self._cache is not None:
                self._cache.set(cache_key, html, self._cache_ttl)
        return self._parse_horse_page_identity(html, expected_name)

    def get_horse_statistics(
        self,
        entry: RaceEntry,
        include_workouts: bool = True,
        request_timeout: float | None = None,
        max_retries: int | None = None,
    ) -> HorseStatistics:
        local_stats = self._local_horse_statistics(entry)
        if local_stats is not None:
            return local_stats

        if entry.source_horse_id is None:
            if local_stats is not None:
                return local_stats
            raise DataSourceError(
                "At istatistikleri icin gerekli QueryParameter_AtId bulunamadi."
            )

        url = f"{self._base_url}{_HORSE_HISTORY_PATH}"
        params = {
            "QueryParameter_AtId": str(entry.source_horse_id),
            "Era": "past",
        }
        try:
            if request_timeout is None and max_retries is None:
                html = self._get_html(url, params)
            else:
                html = self._get_html(url, params, timeout=request_timeout, max_retries=max_retries)
        except DataSourceError:
            if local_stats is not None:
                return local_stats
            raise
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        if not tables:
            if local_stats is not None:
                return local_stats
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
            if local_stats is not None:
                return local_stats
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
            field_size = None
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
            if local_stats is not None:
                return local_stats
            raise DataSourceError(
                "AtKosuBilgileri gecmis kosu tablosu bulundu fakat parse edilebilen satir yok."
            )

        if not career_starts:
            career_starts = len(past_performances)
        if not career_wins:
            career_wins = sum(1 for p in past_performances if p.finish_position == 1)
        if not career_places:
            career_places = sum(1 for p in past_performances if p.finish_position and p.finish_position <= 3)

        if not include_workouts:
            return HorseStatistics(
                horse_id=entry.horse_id,
                horse_name=entry.horse_name,
                past_performances=past_performances,
                workout_records=[],
                career_starts=career_starts,
                career_wins=career_wins,
                career_places=career_places,
                last_year_starts=last_year_starts or min(len(past_performances), 10),
                last_year_wins=last_year_wins,
                last_year_places=last_year_places,
                jockey_horse_combo_starts=combo_starts,
                jockey_horse_combo_wins=combo_wins,
            )

        workout_records: list[WorkoutRecord] = []
        workout_url = f"{self._base_url}{_HORSE_WORKOUT_PATH}"
        workout_params = {"QueryParameter_AtId": str(entry.source_horse_id)}
        try:
            if request_timeout is None and max_retries is None:
                workout_html = self._get_html(workout_url, workout_params)
            else:
                workout_html = self._get_html(
                    workout_url,
                    workout_params,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
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