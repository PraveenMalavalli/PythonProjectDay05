# Day 5 — Ollama LLM client, CV extractor and vision invoices

**Complete runnable Python project using the actual uploaded Day 5 files:** 20 fictional CVs, invoice PNG, scanned invoice PDF. The additional `mock_api (1).zip` uploaded with the materials belongs to the earlier REST API exercise and is not needed here.

### What the project does

- **OpenAI-compatible LLM client** built on HTTPX (`POST /v1/chat/completions`): env-driven settings, timeouts on every request, retries on HTTP 429/5xx and transient transport failures, exponential backoff with jitter, `Retry-After` support, configurable fallback model names, actual server-reported token usage.
- **CV extraction** into Pydantic `CVData` for all 20 CVs. Versioned prompt files. Source CV text is JSON-quoted into a *user* message, never concatenated into the system instructions. One and only one repair request for invalid JSON/schema. Each CV's status, model, latency, retry count and tokens are recorded.
- **Ground-truth evaluation**: first 10 real sample CVs labeled for eight fields; `results/ground_truth_review.csv` is a manual-review worksheet. Local-vs-hosted comparison outputs 80 field-level rows and measured summaries. Missing/failed CVs score zero, not skipped.
- **Temperature/seed experiment**: same CV 3× at temperature 0, 3× at high temperature, once with fixed seed.
- **Context experiment**: an artificially long document with known content at both ends, tested with Ollama's default context and explicit `num_ctx` via the **native** `/api/chat` endpoint. (`num_ctx` is not part of the OpenAI compatibility API.)
- **Vision extraction**: sends the actual invoice PNG or an image-rendered PDF page as a base64 `image_url` to local vision model `qwen3.5:2b`, then validates against Pydantic `InvoiceData`. No OCR required, and document text is untrusted.
- **Offline Pytest suite**: `httpx.MockTransport` is used for all HTTP simulations; the tests do not require Ollama or internet access.

## Requirements

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- Local [Ollama](https://ollama.com/) available from the shell / WSL distro
- An installed vision-capable Ollama model, by default `qwen3.5:2b` (about 3 GB download; memory requirements depend on context size). You may override `OLLAMA_MODEL` to any installed compatible vision model.

**Terminal 1** (only if Ollama is not already running as a service):

```bash
ollama serve
```

**Terminal 2**, from this project's root:

```bash
ollama pull qwen3.5:2b
cp .env.example .env
uv sync --group dev
uv run python -m app.main doctor
```

If Ollama is running on Windows and the Python project runs inside WSL, make sure the `OLLAMA_BASE_URL` points to an address reachable from WSL. Test with `curl http://127.0.0.1:11434/api/tags`. Never expose Ollama to the public internet without access controls.

## Exercise commands

**1. Extract one CV:**

```bash
uv run python -m app.main extract-cv --file day5_samples/cvs/cv_01.txt
```

**2. Extract all 20 CVs** (one JSON object per line in the result file):

```bash
uv run python -m app.main extract-all --limit 20 --output results/local_20.jsonl
```

**3. Extract exactly the first 10 with the local model, then score them:**

```bash
uv run python -m app.main extract-all --limit 10 --output results/local_10.jsonl
uv run python -m app.main evaluate --predictions results/local_10.jsonl --output results/local_accuracy.json
```

Read `docs/EVALUATION_GUIDE.md` and verify ground-truth labels in `results/ground_truth_review.csv` by hand. The score is out of **80** fields (8 per CV).

**4. Optional hosted comparison required for the full assignment:** Set `HOSTED_API_KEY` and an actual available `HOSTED_MODEL` in `.env`. Example `HOSTED_BASE_URL` points to OpenRouter. You may need provider credits or a free-model allowance. This is **not needed to run locally**:

```bash
uv run python -m app.main extract-all --limit 10 --provider hosted --output results/hosted_10.jsonl
uv run python -m app.main compare --local results/local_10.jsonl --hosted results/hosted_10.jsonl --output results/comparison.csv
```

This produces `results/comparison.csv` and `results/comparison_summary.json`. Both models use the same CVs, prompt version and Pydantic schema. Actual accuracy, failures and timings are only generated after inference.

**5. Temperature and seed (7 local runs):**

```bash
uv run python -m app.main temperature --file day5_samples/cvs/cv_01.txt --high-temp 1.2 --seed 42 --output results/temperature.json
```

Compare `output_sha256`, `data`, `status`, token usage and duration for the three zero-temperature runs, the three high-temperature runs, and the seeded run. Higher temperature does not guarantee differences; a fixed seed does not guarantee identical outputs across hardware/runtime versions.

**6. Long context experiment:**

```bash
uv run python -m app.main context --file day5_samples/cvs/cv_01.txt --num-ctx 8192 --repeats 420 --output results/context.json
```

Compares the server default (no explicit `num_ctx`) against `num_ctx=8192` using `/api/chat`. It asks for CV email at the beginning and a marker at the end. Inspect both answers, speed and token counts. Increase `--repeats` if the default window handles it easily. Larger contexts require additional memory. **Do not assume the server default is 4096** without checking your version/configuration.

**7. Vision: actual invoice image** (image is embedded and passed to the model):

```bash
uv run python -m app.main extract-invoice --file day5_samples/invoice_image/invoice.png > results/invoice_image_result.json
```

**8. Vision: actual scanned PDF page** (rendered by PyMuPDF; no embedded searchable text):

```bash
uv run python -m app.main extract-invoice --file day5_samples/scanned_page/invoice_scan.pdf --page 0 > results/invoice_scan_result.json
```

Check against `ground_truth/invoices.json`. Both invoices have **no explicit currency**; the correct response is `null`, not an invented currency.

**9. Run the complete offline test suite and code checks:**

```bash
uv run python -m pytest -v
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Folder map

```text
app/
  main.py                  # argparse CLI
  config.py                # Pydantic settings; .env
  models/cv.py             # 8-field structured CV schema
  models/invoice.py        # invoice and line-item schema
  services/llm_client.py   # OpenAI-compatible HTTPX + Ollama-native num_ctx adapter
  services/extractor.py    # prompt loading, JSON validation, one repair retry
  services/document_reader.py # CV text, PNG/PDF -> base64 images
  services/evaluation.py  # accuracy and per-field comparison CSV
  services/experiments.py # temperature, seed and long context
prompts/                    # versioned files, NEVER put CV text in system prompts
day5_samples/               # your 20 CVs and two original invoices
ground_truth/               # actual source-based first-10 CV labels; two invoice labels
results/                    # manual review and empty accuracy template; runs appear here
tests/                      # mocked offline Pytest tests
```

## Error handling and limitations

- The model must already be downloaded. The project **cannot install Ollama or pull models for you**.
- Invalid model JSON/schema causes *one* repair attempt. HTTP retries are separate, for transient failures; parse/schema mistakes do not re-run indefinitely.
- Errors include validation failures in the saved JSONL. A failed extraction counts as eight incorrect fields for evaluation, rather than disappearing from the denominator.
- Fallback names work only for installed or otherwise accessible model IDs. A nonexistent local fallback is not automatically downloaded.
- Some models might not understand vision or OpenAI-compatible JSON output mode. The bundled `qwen3.5:2b` default is listed as vision-capable; use a newer Ollama build if your instance does not support that model or JSON mode.
- Caution: `model_validate_json` guarantees **structure**, not **factual accuracy**. That is the reason for the 10-CV ground truth and hand-reviewed table.
- The code treats document text as untrusted, places it in a lower-priority message, and never executes document instructions. This decreases but does not eliminate prompt-injection risk.
- No actual hosted/local accuracy, model-run speed or model failure measurements have been fabricated into this repository. Generate them using the commands above.
# PythonProjectDay05
