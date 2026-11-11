# Day 5 exercise completion checklist

| Exercise requirement | Implementation | Verification |
|---|---|---|
| OpenAI-compatible `llm_client` | `app/services/llm_client.py` | `test_chat_openai_compatible_request_and_token_usage` |
| Settings-driven | `app/config.py`, `.env.example` | Change `OLLAMA_MODEL`, `LLM_TIMEOUT_SECONDS`; run `doctor` |
| Per-call timeout | `LLMClient.chat()` and `OllamaNativeClient.chat()` | `test_every_request_has_timeout`, `test_timeout_handled_as_transient_then_error` |
| Retries with exponential backoff and jitter | `LLMClient._retry_delay()`, `.chat()` | `test_retry_503_exponential_backoff_and_jitter`, `test_full_jitter_can_reduce_backoff_to_zero` |
| Rate limit `Retry-After` | `_retry_delay` | `test_retry_429_respects_retry_after`, HTTP-date, large-seconds tests |
| Model fallback | `ClientOptions.models`, `.env` fallbacks | `test_fallback_to_second_model_after_503`, `test_fallback_if_primary_not_installed` |
| Token usage | `TokenUsage`, response and JSONL output | `test_chat_openai_compatible_request_and_token_usage` |
| 20 synthetic CVs | `day5_samples/cvs/*.txt` | `extract-all --limit 20`, `test_actual_samples_are_20_cv_files` |
| Pydantic schema | `app/models/cv.py` | `test_pydantic_rejects_missing_cv_fields`, invalid-years test |
| One repair retry | `StructuredExtractor._execute()` | `test_invalid_schema_triggers_one_repair`, `test_bad_record_fails_after_one_repair` |
| Versioned prompts | `prompts/cv_extract_v1.txt`, `cv_extract_v2.txt`, `repair_v1.txt` | `test_prompt_file_v2_selected` |
| Document treated as untrusted | document passed as user data; instructions in system | `test_doc_instructions_are_lower_priority_untrusted_user_data` |
| First 10 CVs manually checked | `ground_truth/cvs_01_10.json`, `results/ground_truth_review.csv` | User completes human-verification column |
| Field accuracy and failures | `app/services/evaluation.py` | `evaluate --predictions ...`, evaluation tests |
| Hosted/local comparison | `.env` optional hosted, `compare` command | Run both models on first 10 and generate `results/comparison.csv` |
| Temperature experiment | `app/services/experiments.py` | `temperature`, `test_temperature_runs_three_zero_three_high_and_one_seed` |
| Fixed seed | same experiment, final run | Inspect seven run outputs |
| Default vs tuned `num_ctx` | `OllamaNativeClient.chat()` | `context`, `test_native_ollama_context_default_and_explicit_num_ctx` |
| Vision invoice PNG | `extract-invoice --file ...invoice.png` | `test_invoice_png_uses_image_url_not_text_extraction` |
| Vision scanned PDF | PyMuPDF PDF page -> data URI -> model | `test_pdf_scanned_page_is_image_not_ocr` |
| Offline unit tests | `tests/`, `httpx.MockTransport` | `uv run pytest -v` |

**Not automatically complete without your local Ollama runtime:** actual CV inference on 20 files, actual hosted/local measurement on the same 10 CVs, human verification of accuracy, observed temperature/seed/context results, and real vision inference. The project provides commands, source files, schemas, tests, and empty score sheets; it does not invent model outputs.
