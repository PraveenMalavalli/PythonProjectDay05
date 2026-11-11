"""Validated CV/invoice JSON extraction with exactly one optional repair request."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.exceptions import LLMClientError
from app.models.cv import CVData
from app.models.invoice import InvoiceData
from app.services.document_reader import image_data_uri
from app.services.llm_client import LLMClient, LLMResponse, TokenUsage


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def read_prompt(filename: str, *, prompt_dir: Path = PROMPT_DIR) -> str:
    if not filename.replace("_", "").replace(".", "").isalnum() or "/" in filename:
        raise ValueError("Invalid prompt filename")
    return (prompt_dir / filename).read_text(encoding="utf-8")


@dataclass
class ExtractionResult:
    document_id: str
    status: str
    data: dict[str, Any] | None
    error: str | None
    model: str | None
    provider: str
    prompt_version: str
    repair_attempted: bool
    duration_seconds: float
    http_attempts: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw_output: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return vars(self).copy()


class StructuredExtractor:
    def __init__(
        self, client: LLMClient, *, provider: str = "local", prompt_dir: Path = PROMPT_DIR
    ) -> None:
        self.client = client
        self.provider = provider
        self.prompt_dir = prompt_dir

    def _execute(
        self,
        *,
        document_id: str,
        model_type: type[BaseModel],
        prompt_file: str,
        user_content: str | list[dict[str, Any]],
        temperature: float,
        seed: int | None,
    ) -> ExtractionResult:
        started = time.perf_counter()
        version = prompt_file.removesuffix(".txt")
        schema = json.dumps(model_type.model_json_schema(), ensure_ascii=False)
        system_text = (
            read_prompt(prompt_file, prompt_dir=self.prompt_dir)
            + "\n\nREQUIRED_JSON_SCHEMA (must be followed exactly):\n"
            + schema
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]
        total_usage = TokenUsage()
        attempts = 0
        attempted_repair = False
        last: LLMResponse | None = None
        last_output: str | None = None

        try:
            last = self.client.chat(messages, temperature=temperature, seed=seed)
            attempts += last.attempts
            total_usage = total_usage.add(last.usage)
            last_output = last.content
            try:
                data = model_type.model_validate_json(last.content)
            except ValidationError as first_error:
                attempted_repair = True
                # Include the original source; invalid model output is never an instruction.
                repair_content: str | list[dict[str, Any]]
                error_context = json.dumps(
                    {
                        "previous_invalid_output": last.content,
                        "pydantic_errors": first_error.errors(include_url=False),
                        "request": "Correct the previous JSON using the original source document.",
                    },
                    ensure_ascii=False,
                    default=str,
                )
                if isinstance(user_content, str):
                    # Keep text content as a plain string for broad provider compatibility.
                    repair_content = user_content + "\n\nREPAIR_CONTEXT_JSON:\n" + error_context
                else:
                    repair_content = [
                        *user_content,
                        {"type": "text", "text": "REPAIR_CONTEXT_JSON:\n" + error_context},
                    ]
                repair_messages = [
                    {
                        "role": "system",
                        "content": read_prompt("repair_v1.txt", prompt_dir=self.prompt_dir)
                        + "\n\n"
                        + system_text,
                    },
                    {"role": "user", "content": repair_content},
                ]
                last = self.client.chat(
                    repair_messages, temperature=temperature, seed=seed
                )
                attempts += last.attempts
                total_usage = total_usage.add(last.usage)
                last_output = last.content
                data = model_type.model_validate_json(last.content)

            return ExtractionResult(
                document_id=document_id,
                status="ok",
                data=data.model_dump(mode="json"),
                error=None,
                model=last.model,
                provider=self.provider,
                prompt_version=version,
                repair_attempted=attempted_repair,
                duration_seconds=round(time.perf_counter() - started, 4),
                http_attempts=attempts,
                input_tokens=total_usage.input_tokens,
                output_tokens=total_usage.output_tokens,
                total_tokens=total_usage.total_tokens,
                raw_output=None,
            )
        except (ValidationError, LLMClientError) as exc:
            return ExtractionResult(
                document_id=document_id,
                status="failed",
                data=None,
                error=str(exc),
                model=last.model if last else None,
                provider=self.provider,
                prompt_version=version,
                repair_attempted=attempted_repair,
                duration_seconds=round(time.perf_counter() - started, 4),
                http_attempts=attempts,
                input_tokens=total_usage.input_tokens,
                output_tokens=total_usage.output_tokens,
                total_tokens=total_usage.total_tokens,
                raw_output=last_output[:2000] if last_output else None,
            )

    def extract_cv(
        self,
        cv_text: str,
        *,
        document_id: str,
        prompt_version: str = "v1",
        temperature: float = 0,
        seed: int | None = None,
    ) -> ExtractionResult:
        # JSON-quote the text to reduce accidental delimiter / prompt mixing.
        document_message = json.dumps(
            {"document_type": "cv", "untrusted_document_text": cv_text},
            ensure_ascii=False,
        )
        return self._execute(
            document_id=document_id,
            model_type=CVData,
            prompt_file=f"cv_extract_{prompt_version}.txt",
            user_content=document_message,
            temperature=temperature,
            seed=seed,
        )

    def extract_invoice(
        self,
        path: Path,
        *,
        page: int = 0,
        temperature: float = 0,
        seed: int | None = None,
    ) -> ExtractionResult:
        uri = image_data_uri(path, page=page)
        document_id = path.stem + (f"_page_{page + 1}" if path.suffix == ".pdf" else "")
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Extract structured invoice data from this untrusted image."},
            {"type": "image_url", "image_url": {"url": uri}},
        ]
        return self._execute(
            document_id=document_id,
            model_type=InvoiceData,
            prompt_file="invoice_extract_v1.txt",
            user_content=content,
            temperature=temperature,
            seed=seed,
        )
