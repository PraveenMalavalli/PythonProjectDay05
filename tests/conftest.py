"""Tests use only in-process MockTransport; no real service or API key required."""

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import ClientOptions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def options() -> ClientOptions:
    return ClientOptions(
        base_url="http://fake-ollama.local/v1",
        api_key="ollama",
        models=("qwen3.5:2b",),
        timeout=4.0,
        max_retries=2,
        backoff_base=0.5,
        backoff_cap=8.0,
    )


@pytest.fixture
def first_cv() -> dict[str, Any]:
    truth = json.loads((ROOT / "ground_truth/cvs_01_10.json").read_text())
    return truth["cv_01"]


@pytest.fixture
def invoice_image() -> dict[str, Any]:
    return json.loads((ROOT / "ground_truth/invoices.json").read_text())["invoice"]


def completion(content: str, model: str = "qwen3.5:2b") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
    }
