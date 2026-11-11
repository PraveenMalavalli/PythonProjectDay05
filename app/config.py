"""Settings loaded from environment variables or an optional .env file."""

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ClientOptions:
    base_url: str
    api_key: str
    models: tuple[str, ...]
    timeout: float
    max_retries: int
    backoff_base: float
    backoff_cap: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen3.5:2b"
    ollama_fallback_models: str = ""
    hosted_base_url: str = "https://openrouter.ai/api/v1"
    hosted_api_key: str = ""
    hosted_model: str = ""
    hosted_fallback_models: str = ""
    llm_timeout_seconds: float = Field(default=180, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_backoff_base_seconds: float = Field(default=0.5, ge=0)
    llm_backoff_cap_seconds: float = Field(default=8.0, gt=0)

    def client_options(self, provider: str = "local", model: str | None = None) -> ClientOptions:
        if provider not in {"local", "hosted"}:
            raise ValueError(f"Unknown provider: {provider}")
        if provider == "local":
            first = model or self.ollama_model
            fallbacks = self.ollama_fallback_models
            base_url = self.ollama_base_url
            key = "ollama"  # Ollama's local OpenAI-compatible endpoint ignores it.
        else:
            first = model or self.hosted_model
            fallbacks = self.hosted_fallback_models
            base_url = self.hosted_base_url
            key = self.hosted_api_key
            if not key or not first:
                raise ValueError(
                    "Hosted comparison requires HOSTED_API_KEY and HOSTED_MODEL in .env"
                )

        names = tuple(
            dict.fromkeys([first, *(n.strip() for n in fallbacks.split(",") if n.strip())])
        )
        if not first:
            raise ValueError("At least one model name is required")
        return ClientOptions(
            base_url=base_url.rstrip("/"),
            api_key=key,
            models=names,
            timeout=self.llm_timeout_seconds,
            max_retries=self.llm_max_retries,
            backoff_base=self.llm_backoff_base_seconds,
            backoff_cap=self.llm_backoff_cap_seconds,
        )
