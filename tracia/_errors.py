"""Error handling for the Tracia SDK."""

from __future__ import annotations

import re
from enum import Enum


class TraciaErrorCode(str, Enum):
    """Error codes for Tracia SDK errors."""

    UNAUTHORIZED = "UNAUTHORIZED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    MISSING_VARIABLES = "MISSING_VARIABLES"
    INVALID_REQUEST = "INVALID_REQUEST"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    ABORTED = "ABORTED"
    UNKNOWN = "UNKNOWN"
    MISSING_PROVIDER_SDK = "MISSING_PROVIDER_SDK"
    MISSING_PROVIDER_API_KEY = "MISSING_PROVIDER_API_KEY"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"


class TraciaError(Exception):
    """Exception raised for Tracia SDK errors."""

    def __init__(
        self,
        code: TraciaErrorCode,
        message: str,
        status_code: int | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.code.value}] {self.message} (status: {self.status_code})"
        return f"[{self.code.value}] {self.message}"

    def __repr__(self) -> str:
        return f"TraciaError(code={self.code!r}, message={self.message!r}, status_code={self.status_code!r})"


# Patterns for sanitizing sensitive data from error messages
_SENSITIVE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI API keys
    re.compile(r"sk-ant-[a-zA-Z0-9-]{20,}"),  # Anthropic API keys
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+"),  # Bearer tokens
    re.compile(r"Basic\s+[a-zA-Z0-9+/=]+"),  # Basic auth
    re.compile(r"Authorization:\s*[^\s]+", re.IGNORECASE),  # Auth headers
]


def sanitize_error_message(message: str) -> str:
    """Remove sensitive data from error messages.

    Args:
        message: The error message to sanitize.

    Returns:
        The sanitized error message.
    """
    result = message
    for pattern in _SENSITIVE_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def map_api_error_code(code: str) -> TraciaErrorCode:
    """Map an API error code string to a TraciaErrorCode.

    Args:
        code: The error code from the API response.

    Returns:
        The corresponding TraciaErrorCode.
    """
    try:
        return TraciaErrorCode(code)
    except ValueError:
        return TraciaErrorCode.UNKNOWN
