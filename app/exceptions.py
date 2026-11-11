"""User-oriented errors for model transport and extraction failures."""


class LLMClientError(Exception):
    """Unable to obtain a usable model completion."""


class LLMHTTPError(LLMClientError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class ExtractionError(Exception):
    """Document could not be extracted or validated."""
