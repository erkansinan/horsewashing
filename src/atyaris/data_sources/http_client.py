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

_RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})
_MAX_RETRY_AFTER_SECONDS = 30.0
_MAX_PENALTY = 8.0
_BLOCK_SIGNAL_STATUSES = frozenset({403, 429, 503})


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
        self._last_start_at = 0.0
        self._cooldown_until = 0.0
        # Sunucu yavaslama/engel belirtisi verirse aralik bu carpanla uzar, basarida yavasca doner.
        self._penalty = 1.0

    def _reserve_slot(self, request_interval: float) -> float:
        """Reserve the next request start time; spacing is start-to-start, so
        several workers can overlap network latency without raising the rate."""
        with self._lock:
            now = time.monotonic()
            start = max(
                now,
                self._last_start_at + request_interval * self._penalty,
                self._cooldown_until,
            )
            self._last_start_at = start
            return start - now

    def _record_success(self) -> None:
        with self._lock:
            self._penalty = max(1.0, self._penalty * 0.95)

    def _record_pressure(self, request_interval: float, severe: bool) -> None:
        with self._lock:
            self._penalty = min(_MAX_PENALTY, self._penalty * 2.0)
            if severe:
                cooldown = request_interval * 10.0 * self._penalty
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + cooldown)

    def get(
        self,
        url: str,
        params: dict | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        min_interval: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return self._request(
            "GET",
            url,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
            min_interval=min_interval,
            headers=headers,
        )

    def post(
        self,
        url: str,
        data: dict[str, str],
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> httpx.Response:
        return self._request("POST", url, data=data, timeout=timeout, max_retries=max_retries)

    def _retry_delay(self, attempt: int, response: httpx.Response | None = None) -> float:
        delay = self._retry_backoff_seconds * (attempt + 1)
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), _MAX_RETRY_AFTER_SECONDS))
            except ValueError:
                pass
        return delay

    def _request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        data: dict[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        min_interval: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        retry_count = self._max_retries if max_retries is None else max(0, max_retries)
        request_options: dict[str, object] = {}
        if params is not None:
            request_options["params"] = params
        if data is not None:
            request_options["data"] = data
        if timeout is not None:
            request_options["timeout"] = timeout
        if headers is not None:
            request_options["headers"] = headers
        request_interval = self._min_interval if min_interval is None else max(0.0, min_interval)
        for attempt in range(retry_count + 1):
            wait = self._reserve_slot(request_interval)
            if wait > 0:
                time.sleep(wait)
            logger.info(
                "TJK %s basliyor | url=%s | params=%s | timeout=%.1fs | max_retry=%d",
                method,
                url,
                params or {},
                timeout or self._client.timeout.read,
                retry_count,
            )
            try:
                response = self._client.request(method, url, **request_options)
                response.raise_for_status()
                self._record_success()
                logger.info(
                    "TJK %s yanitlandi | attempt=%d/%d | status=%d | url=%s | params=%s",
                    method,
                    attempt + 1,
                    retry_count + 1,
                    response.status_code,
                    url,
                    params or {},
                )
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status >= 500 or status in _RETRYABLE_CLIENT_STATUSES or status in _BLOCK_SIGNAL_STATUSES:
                    self._record_pressure(request_interval, severe=status in _BLOCK_SIGNAL_STATUSES)
                retryable = status >= 500 or status in _RETRYABLE_CLIENT_STATUSES
                if not retryable or attempt >= retry_count:
                    logger.error(
                        "TJK %s HTTP hatasi | attempt=%d/%d | status=%d | url=%s | params=%s",
                        method,
                        attempt + 1,
                        retry_count + 1,
                        status,
                        url,
                        params or {},
                    )
                    raise
                delay = self._retry_delay(attempt, exc.response)
                logger.warning(
                    "TJK %s tekrar denenecek | retry=%d/%d | status=%d | delay=%.1fs | url=%s | params=%s",
                    method,
                    attempt + 1,
                    retry_count,
                    status,
                    delay,
                    url,
                    params or {},
                )
                if delay:
                    time.sleep(delay)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self._record_pressure(request_interval, severe=False)
                logger.warning(
                    "TJK %s hatasi | attempt=%d/%d | error=%s: %s | url=%s | params=%s",
                    method,
                    attempt + 1,
                    retry_count + 1,
                    type(exc).__name__,
                    exc,
                    url,
                    params or {},
                )
                if attempt >= retry_count:
                    logger.error(
                        "TJK %s basarisiz | manuel kontrol URL: %s",
                        method,
                        manual_check_url(url, params),
                    )
                    raise
                delay = self._retry_delay(attempt)
                if delay:
                    time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RateLimitedClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
