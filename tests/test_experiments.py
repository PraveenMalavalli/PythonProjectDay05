import json

import httpx

from app.services.experiments import run_context_experiment, run_temperature_experiment
from app.services.extractor import StructuredExtractor
from app.services.llm_client import LLMClient, OllamaNativeClient
from tests.conftest import ROOT, completion


def test_temperature_runs_three_zero_three_high_and_one_seed(options, first_cv):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(json.dumps(first_cv)))

    with LLMClient(options, transport=httpx.MockTransport(handler)) as client:
        rows = run_temperature_experiment(
            StructuredExtractor(client), ROOT / "day5_samples/cvs/cv_01.txt",
            high_temperature=1.2, seed=99,
        )
    assert len(rows) == 7
    assert [row["temperature"] for row in rows] == [0, 0, 0, 1.2, 1.2, 1.2, 1.2]
    assert [row["seed"] for row in rows] == [None] * 6 + [99]
    assert [req.get("seed") for req in requests] == [None] * 6 + [99]
    assert len({row["output_sha256"] for row in rows}) == 1


def test_native_ollama_context_default_and_explicit_num_ctx():
    captures = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        captures.append(body)
        return httpx.Response(200, json={
            "model": "qwen3.5:2b", "message": {"content": json.dumps({
                "email": "asha.alder@example.com", "final_marker": "CONTEXT-END-73941"
            })}, "prompt_eval_count": 2200, "eval_count": 42,
        })

    with OllamaNativeClient(
        "http://fake-ollama.local/v1",
        transport=httpx.MockTransport(handler)
    ) as client:
        rows = run_context_experiment(
            client, model="qwen3.5:2b", cv_text="Asha Alder\nEmail: asha.alder@example.com",
            num_ctx=8192, repeats=4,
        )
    assert len(rows) == 2
    assert "num_ctx" not in captures[0]["options"]
    assert captures[1]["options"]["num_ctx"] == 8192
    assert rows[0]["final_marker_correct"] is True
    assert rows[1]["usage"]["total_tokens"] == 2242
