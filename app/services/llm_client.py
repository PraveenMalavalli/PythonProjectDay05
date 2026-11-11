"""HTTPX-powered OpenAI-compatible LLM client with retries, fallback and token tracking.

Works with Ollama's http://localhost:11434/v1 endpoint and optional hosted providers.
All injected HTTP transports in tests are offline httpx.MockTransport instances.
"""

import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.config import ClientOptions
from app.exceptions import LLMClientError, LLMHTTPError


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_json(cls, data: Any) -> "TokenUsage":
        if not isinstance(data, dict):
            return cls()
        prompt = data.get("prompt_tokens")
        output = data.get("completion_tokens")
        total = data.get("total_tokens")
        return cls(
            input_tokens=prompt if isinstance(prompt, int) else None,
            output_tokens=output if isinstance(output, int) else None,
            total_tokens=total if isinstance(total, int) else None,
        )

    def add(self, other: "TokenUsage") -> "TokenUsage":
        def combine(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)

        return TokenUsage(
            combine(self.input_tokens, other.input_tokens),
            combine(self.output_tokens, other.output_tokens),
            combine(self.total_tokens, other.total_tokens),
        )


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    elapsed_seconds: float
    attempts: int  # including unsuccessful HTTP requests and fallback calls


class LLMClient:
    """OpenAI-compatible chat completions using httpx with explicit request timeouts."""

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        options: ClientOptions,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not options.models:
            raise ValueError("At least one model must be configured")
        self.options = options
        self._sleep = sleep
        self._jitter = jitter
        self._now = now
        self._http = httpx.Client(
            base_url=options.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {options.api_key}"},
            timeout=options.timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            value = response.headers.get("Retry-After")
            if value:
                try:
                    return max(float(value), 0)
                except ValueError:
                    try:
                        retry_time = parsedate_to_datetime(value)
                        if retry_time.tzinfo is None:
                            retry_time = retry_time.replace(tzinfo=timezone.utc)
                        difference = (retry_time - self._now()).total_seconds()
                        return max(difference, 0)
                    except (TypeError, ValueError, OverflowError):
                        pass
        ceiling = min(self.options.backoff_cap, self.options.backoff_base * 2**attempt)
        return ceiling * max(0.0, min(1.0, float(self._jitter())))

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0,
        seed: int | None = None,
        json_mode: bool = True,
        model: str | None = None,
    ) -> LLMResponse:
        """Try preferred then fallback models for retryable errors, respecting Retry-After.

        Does not retry non-transient 4xx responses or malformed provider data.
        """
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not messages:
            raise ValueError("messages cannot be empty")

        models = (model,) if model else self.options.models
        started = time.perf_counter()
        attempts = 0
        last_error: Exception | None = None

        for name in models:
            request_data: dict[str, Any] = {
                "model": name,
                "messages": [dict(message) for message in messages],
                "temperature": temperature,
                "stream": False,
            }
            if seed is not None:
                request_data["seed"] = seed
            if json_mode:
                request_data["response_format"] = {"type": "json_object"}

            for attempt in range(self.options.max_retries + 1):
                attempts += 1
                try:
                    response = self._http.post(
                        "chat/completions", json=request_data, timeout=self.options.timeout
                    )
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    last_error = LLMClientError(f"Model {name} transport failure: {exc}")
                    if attempt < self.options.max_retries:
                        self._sleep(self._retry_delay(None, attempt))
                        continue
                    break  # transient failure, try next fallback model

                if response.status_code in self.RETRYABLE_STATUS:
                    last_error = LLMHTTPError(
                        response.status_code, f"Temporary error from model {name}"
                    )
                    if attempt < self.options.max_retries:
                        self._sleep(self._retry_delay(response, attempt))
                        continue
                    break  # retries exhausted: try fallback

                if response.status_code == 404 and len(models) > 1:
                    last_error = LLMHTTPError(404, f"Model {name} was not found")
                    break  # a fallback may be installed even if the primary is not

                if response.is_error:
                    raise LLMHTTPError(
                        response.status_code,
                        "Non-retryable API error (check model, credentials or request)",
                    )

                try:
                    body = response.json()
                    choice = body["choices"][0]
                    content = choice["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Empty or non-text assistant message")
                    used_model = body.get("model") or name
                    if not isinstance(used_model, str):
                        used_model = name
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    raise LLMClientError("Invalid OpenAI-compatible response structure") from exc
                return LLMResponse(
                    content=content,
                    model=used_model,
                    usage=TokenUsage.from_json(body.get("usage")),
                    elapsed_seconds=time.perf_counter() - started,
                    attempts=attempts,
                )

        raise LLMClientError(
            f"All configured models failed after {attempts} attempts: {last_error}"
        ) from last_error


class OllamaNativeClient:
    """Ollama-native API used ONLY for the num_ctx experiment.

    Ollama's OpenAI-compatible endpoint does not expose num_ctx per request;
    /api/chat accepts options.num_ctx. The primary client above remains OpenAI-compatible.
    """

    def __init__(
        self,
        openai_base_url: str,
        *,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base = openai_base_url.rstrip("/")
        if not base.endswith("/v1"):
            raise ValueError("Ollama local base URL must end in /v1")
        self._http = httpx.Client(base_url=base[:-3], timeout=timeout, transport=transport)
        self.timeout = timeout

    def __enter__(self) -> "OllamaNativeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self._http.close()

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        num_ctx: int | None = None,
        temperature: float = 0,
        seed: int | None = None,
    ) -> LLMResponse:
        if num_ctx is not None and num_ctx <= 0:
            raise ValueError("num_ctx must be positive")
        options: dict[str, int | float] = {"temperature": temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if seed is not None:
            options["seed"] = seed
        started = time.perf_counter()
        try:
            response = self._http.post(
                "api/chat",
                json={
                    "model": model,
                    "messages": [dict(message) for message in messages],
                    "stream": False,
                    "format": "json",
                    "options": options,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Invalid Ollama response message")
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            raise LLMClientError(f"Ollama native request failed: {exc}") from exc
        prompt = body.get("prompt_eval_count")
        generated = body.get("eval_count")
        usage = TokenUsage(
            input_tokens=prompt if isinstance(prompt, int) else None,
            output_tokens=generated if isinstance(generated, int) else None,
            total_tokens=(prompt + generated)
            if isinstance(prompt, int) and isinstance(generated, int)
            else None,
        )
        return LLMResponse(
            content=content,
            model=str(body.get("model", model)),
            usage=usage,
            elapsed_seconds=time.perf_counter() - started,
            attempts=1,
        )
