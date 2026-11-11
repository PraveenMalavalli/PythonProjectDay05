import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from app.exceptions import LLMClientError, LLMHTTPError
from app.services.llm_client import LLMClient
from tests.conftest import completion


MESSAGES = [{"role": "user", "content": "Return valid JSON"}]


def test_chat_openai_compatible_request_and_token_usage(options):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=completion('{"ok":true}'))

    with LLMClient(options, transport=httpx.MockTransport(handler)) as client:
        result = client.chat(MESSAGES, temperature=0.2, seed=42)

    assert requests[0].url.path == "/v1/chat/completions"
    sent = json.loads(requests[0].content)
    assert sent["model"] == "qwen3.5:2b"
    assert sent["temperature"] == 0.2
    assert sent["seed"] == 42
    assert sent["response_format"] == {"type": "json_object"}
    assert requests[0].headers["Authorization"] == "Bearer ollama"
    assert result.usage.input_tokens == 120
    assert result.usage.output_tokens == 60
    assert result.usage.total_tokens == 180
    assert result.attempts == 1


def test_every_request_has_timeout(options):
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.extensions["timeout"])
        return httpx.Response(200, json=completion('{}'))

    with LLMClient(options, transport=httpx.MockTransport(handler)) as client:
        client.chat(MESSAGES)
    assert captured
    assert all(value == 4.0 for value in captured[0].values())


def test_retry_429_respects_retry_after(options):
    attempts = []
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json=completion('{}'))

    with LLMClient(
        options, transport=httpx.MockTransport(handler), sleep=delays.append
    ) as client:
        result = client.chat(MESSAGES)

    assert len(attempts) == 2
    assert result.attempts == 2
    assert delays == [3.0]


def test_retry_after_is_not_shortened_by_backoff_cap(options):
    delays = []
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "30"})
        return httpx.Response(200, json=completion('{}'))

    with LLMClient(options, transport=httpx.MockTransport(handler), sleep=delays.append) as client:
        client.chat(MESSAGES)
    assert delays == [30.0]  # honor the server, even if our backoff cap is only 8s


def test_retry_after_http_date(options):
    now = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
    wait_until = format_datetime(now + timedelta(seconds=6))
    delays = []
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"Retry-After": wait_until})
        return httpx.Response(200, json=completion('{}'))

    with LLMClient(
        options, transport=httpx.MockTransport(handler),
        now=lambda: now, sleep=delays.append
    ) as client:
        client.chat(MESSAGES)
    assert delays == [6.0]


def test_retry_503_exponential_backoff_and_jitter(options):
    attempts = 0
    delays = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return httpx.Response(503)
        return httpx.Response(200, json=completion('{}'))

    with LLMClient(
        options, transport=httpx.MockTransport(handler),
        sleep=delays.append, jitter=lambda: 1.0
    ) as client:
        result = client.chat(MESSAGES)
    assert attempts == 3
    assert delays == [0.5, 1.0]
    assert result.attempts == 3


def test_full_jitter_can_reduce_backoff_to_zero(options):
    calls = 0
    delays = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500) if calls < 2 else httpx.Response(200, json=completion('{}'))

    with LLMClient(
        options, transport=httpx.MockTransport(handler),
        sleep=delays.append, jitter=lambda: 0.0
    ) as client:
        client.chat(MESSAGES)
    assert delays == [0]


def test_fallback_to_second_model_after_503(options):
    model_names = []
    config = replace(options, models=("primary", "backup"), max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        model_names.append(model)
        return (
            httpx.Response(503)
            if model == "primary" else httpx.Response(200, json=completion("{}", "backup"))
        )

    with LLMClient(config, transport=httpx.MockTransport(handler)) as client:
        result = client.chat(MESSAGES)
    assert model_names == ["primary", "backup"]
    assert result.model == "backup"
    assert result.attempts == 2


def test_fallback_if_primary_not_installed(options):
    config = replace(options, models=("missing", "backup"), max_retries=0)
    names = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        names.append(model)
        return (
            httpx.Response(404)
            if model == "missing" else httpx.Response(200, json=completion("{}", "backup"))
        )

    with LLMClient(config, transport=httpx.MockTransport(handler)) as client:
        result = client.chat(MESSAGES)
    assert names == ["missing", "backup"]
    assert result.model == "backup"


def test_max_retries_then_error(options):
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500)

    with LLMClient(
        options, transport=httpx.MockTransport(handler),
        sleep=lambda _: None
    ) as client:
        with pytest.raises(LLMClientError, match="All configured models failed"):
            client.chat(MESSAGES)
    assert attempts == 3  # first + 2 retries


def test_timeout_handled_as_transient_then_error(options):
    config = replace(options, max_retries=1)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("simulated timeout", request=request)

    with LLMClient(config, transport=httpx.MockTransport(handler),
                   sleep=lambda _: None) as client:
        with pytest.raises(LLMClientError, match="transport failure"):
            client.chat(MESSAGES)
    assert attempts == 2


def test_bad_request_not_retried_or_fallback(options):
    config = replace(options, models=("one", "two"))
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400)

    with LLMClient(config, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMHTTPError) as failure:
            client.chat(MESSAGES)
    assert failure.value.status_code == 400
    assert calls == 1


def test_bad_json_response_structure_fails(options):
    with LLMClient(options, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"unknown": []})
    )) as client:
        with pytest.raises(LLMClientError, match="response structure"):
            client.chat(MESSAGES)


def test_missing_token_counts_stay_unknown(options):
    payload = completion("{}")
    payload.pop("usage")
    with LLMClient(options, transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload)
    )) as client:
        result = client.chat(MESSAGES)
    assert result.usage.total_tokens is None
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
