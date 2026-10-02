from __future__ import annotations

import httpx
import pytest

from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.http_client import RateLimitedClient
from atyaris.config import Settings
from atyaris.services import build_data_source
from atyaris.data_sources.prediction import build_prediction_data_source
from atyaris.data_sources.training import build_training_data_source


def test_prediction_and_training_factories_have_isolated_request_policies() -> None:
    settings = Settings(
        request_timeout_seconds=20.0,
        min_request_interval_seconds=1.25,
        prediction_data_request_timeout_seconds=3.0,
        prediction_data_min_request_interval_seconds=0.4,
        prediction_data_request_max_retries=0,
        prediction_program_request_timeout_seconds=25.0,
        prediction_program_request_max_retries=1,
    )
    prediction = build_prediction_data_source(settings)
    training = build_training_data_source(settings)
    generic_tjk = build_data_source("tjk", settings)

    try:
        assert prediction._client is not training._client
        assert prediction._client._max_retries == 0
        assert prediction._client._min_interval == 0.4
        assert prediction._client._client.timeout.read == 3.0
        assert prediction._program_request_timeout == 25.0
        assert prediction._program_max_retries == 1
        assert training._client._max_retries == 2
        assert training._client._min_interval == 1.25
        assert training._client._client.timeout.read == 20.0
        assert generic_tjk._client._max_retries == 0
        assert generic_tjk._client._client.timeout.read == 3.0
    finally:
        prediction.close()
        training.close()
        generic_tjk.close()


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


def test_get_prediction_policy_uses_short_timeout_without_retry() -> None:
    observed_timeouts = []
    attempts = 0

    def always_timeout(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        observed_timeouts.append(request.extensions["timeout"])
        raise httpx.ReadTimeout("prediction request timed out", request=request)

    client = RateLimitedClient(
        "test-agent",
        min_interval=0.0,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(always_timeout))

    try:
        with pytest.raises(httpx.ReadTimeout):
            client.get(
                "https://example.test/prediction-data",
                timeout=0.25,
                max_retries=0,
            )
    finally:
        client.close()

    assert attempts == 1
    assert observed_timeouts[0]["read"] == 0.25


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


def test_post_retries_timeout_and_preserves_form_data() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        assert request.method == "POST"
        assert request.content == b"QueryParameter_AtIsmi=TEST"
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, text="ok", request=request)

    client = RateLimitedClient(
        "test-agent",
        min_interval=0.0,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))

    try:
        response = client.post(
            "https://example.test/horses", {"QueryParameter_AtIsmi": "TEST"}
        )
    finally:
        client.close()

    assert response.text == "ok"
    assert attempts["count"] == 2


def test_get_final_timeout_logs_manual_check_url(caplog: pytest.LogCaptureFixture) -> None:
    def always_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("The read operation timed out", request=request)

    client = RateLimitedClient(
        "test-agent",
        min_interval=0.0,
        max_retries=0,
        retry_backoff_seconds=0.0,
    )
    client._client = httpx.Client(transport=httpx.MockTransport(always_timeout))

    try:
        with pytest.raises(httpx.ReadTimeout):
            client.get(
                "https://www.tjk.org/TR/YarisSever/Info/Data/GunlukYarisProgrami",
                {"QueryParameter_Tarih": "16/09/2026", "Era": "past"},
            )
    finally:
        client.close()

    assert (
        "https://www.tjk.org/TR/YarisSever/Info/Page/GunlukYarisProgrami?"
        "QueryParameter_Tarih=16/09/2026&Era=past"
    ) in caplog.text


def test_tjk_adapter_normalizes_dns_failure_to_data_source_error() -> None:
    from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource

    source = TJKHtmlDataSource()

    def fail_get(url: str, params: dict[str, object]) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")

    source._client.get = fail_get  # type: ignore[method-assign]
    try:
        with pytest.raises(DataSourceError, match="Manuel kontrol URL: https://www.tjk.org/test"):
            source._get_html("https://www.tjk.org/test", {})
    finally:
        source.close()
