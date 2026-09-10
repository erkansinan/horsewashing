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

import httpx

logger = logging.getLogger(__name__)


class RateLimitedClient:
    """Ardisik istekler arasinda minimum bekleme uygulayan httpx sarmalayicisi."""

    def __init__(
        self,
        user_agent: str,
        timeout: float = 15.0,
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

    def get(self, url: str, params: dict | None = None) -> httpx.Response:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            logger.debug("GET %s params=%s", url, params)
            for attempt in range(self._max_retries + 1):
                try:
                    response = self._client.get(url, params=params)
                    break
                except (httpx.ReadTimeout, httpx.NetworkError):
                    if attempt >= self._max_retries:
                        raise
                    delay = self._retry_backoff_seconds * (attempt + 1)
                    logger.warning(
                        "GET network error (%d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        url,
                    )
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
