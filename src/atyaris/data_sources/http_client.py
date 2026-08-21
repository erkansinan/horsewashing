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

    def __init__(self, user_agent: str, timeout: float = 15.0, min_interval: float = 1.0) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Language": "tr-TR,tr;q=0.9"},
            timeout=timeout,
            follow_redirects=True,
        )
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def get(self, url: str, params: dict | None = None) -> httpx.Response:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            logger.debug("GET %s params=%s", url, params)
            response = self._client.get(url, params=params)
            self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RateLimitedClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
