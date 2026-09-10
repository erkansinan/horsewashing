from __future__ import annotations

import httpx
import pytest

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.http_client import RateLimitedClient


def test_get_retries_read_timeout() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("The read operation timed out", request=request)
        return httpx.Response(200, text="ok", request=request)

    client = RateLimitedClient(
        "test-agent",
        min_interval=0.0,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        response = client.get("https://example.test/hippodromes")
    finally:
        client.close()

    assert response.text == "ok"
    assert attempts["count"] == 2


def test_get_retries_connection_reset() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(200, text="ok", request=request)

    client = RateLimitedClient(
        "test-agent",
        min_interval=0.0,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        response = client.get("https://example.test/hippodromes")
    finally:
        client.close()

    assert response.text == "ok"
    assert attempts["count"] == 2


def test_tjk_adapter_normalizes_dns_failure_to_data_source_error() -> None:
    from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource

    source = TJKHtmlDataSource()

    def fail_get(url: str, params: dict[str, object]) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")

    source._client.get = fail_get  # type: ignore[method-assign]
    try:
        with pytest.raises(DataSourceError, match="TJK sayfasi alinamadi"):
            source._get_html("https://www.tjk.org/test", {})
    finally:
        source.close()
