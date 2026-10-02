"""Hiz sinirlamali (rate-limited) HTTP istemci sarmalayicisi.

TJK'nin herkese acik web sayfalarina karsi kibar/makul bir istek deseni
saglar: ozel bir User-Agent, mantikli bir timeout ve ardisik istekler
arasinda minimum bir bekleme suresi (basit bir "token bucket of one"
hiz sinirlayici).
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)


def manual_check_url(url: str, params: dict | None = None) -> str:
    """Return the exact GET URL that can be opened manually in a browser."""
    url = url.replace(
        "/TR/YarisSever/Info/Data/GunlukYarisProgrami",
        "/TR/YarisSever/Info/Page/GunlukYarisProgrami",
    )
    if not params:
        return url
    return f"{url}?{urlencode(params, safe='/')}"


class RateLimitedClient:
    """Ardisik istekler arasinda minimum bekleme uygulayan httpx sarmalayicisi."""

    def __init__(
        self,
        user_agent: str,
        timeout: float = 30.0,
        min_interval: float = 1.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Language": "tr-TR,tr;q=0.9"},
            timeout=timeout,
            follow_redirects=True,
        )
        self._min_interval = min_interval
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def get(
        self,
        url: str,
        params: dict | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        min_interval: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        retry_count = self._max_retries if max_retries is None else max(0, max_retries)
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            request_interval = self._min_interval if min_interval is None else max(0.0, min_interval)
            wait = request_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            logger.info(
                "TJK GET basliyor | url=%s | params=%s | timeout=%.1fs | max_retry=%d",
                url,
                params or {},
                timeout or self._client.timeout.read,
                retry_count,
            )
            for attempt in range(retry_count + 1):
                try:
                    logger.info(
                        "TJK GET deneniyor | attempt=%d/%d | url=%s | params=%s",
                        attempt + 1,
                        retry_count + 1,
                        url,
                        params or {},
                    )
                    request_options = {}
                    if timeout is not None:
                        request_options["timeout"] = timeout
                    if headers is not None:
                        request_options["headers"] = headers
                    response = self._client.get(url, params=params, **request_options)
                    logger.info(
                        "TJK GET yanitlandi | attempt=%d/%d | status=%d | url=%s | params=%s",
                        attempt + 1,
                        retry_count + 1,
                        response.status_code,
                        url,
                        params or {},
                    )
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    logger.warning(
                        "TJK GET hatasi | attempt=%d/%d | error=%s: %s | url=%s | params=%s",
                        attempt + 1,
                        retry_count + 1,
                        type(exc).__name__,
                        exc,
                        url,
                        params or {},
                    )
                    if attempt >= retry_count:
                        logger.error(
                            "TJK GET basarisiz | manuel kontrol URL: %s",
                            manual_check_url(url, params),
                        )
                        raise
                    delay = self._retry_backoff_seconds * (attempt + 1)
                    logger.warning(
                        "TJK GET tekrar denenecek | retry=%d/%d | delay=%.1fs | url=%s | params=%s",
                        attempt + 1,
                        retry_count,
                        delay,
                        url,
                        params or {},
                    )
                    if delay:
                        time.sleep(delay)
            self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def post(
        self,
        url: str,
        data: dict[str, str],
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> httpx.Response:
        retry_count = self._max_retries if max_retries is None else max(0, max_retries)
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            logger.info(
                "TJK POST basliyor | url=%s | timeout=%.1fs | max_retry=%d",
                url,
                timeout or self._client.timeout.read,
                retry_count,
            )
            for attempt in range(retry_count + 1):
                try:
                    if timeout is None:
                        response = self._client.post(url, data=data)
                    else:
                        response = self._client.post(url, data=data, timeout=timeout)
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    logger.warning(
                        "TJK POST hatasi | attempt=%d/%d | error=%s: %s | url=%s",
                        attempt + 1,
                        retry_count + 1,
                        type(exc).__name__,
                        exc,
                        url,
                    )
                    if attempt >= retry_count:
                        raise
                    delay = self._retry_backoff_seconds * (attempt + 1)
                    if delay:
                        time.sleep(delay)
            self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RateLimitedClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
