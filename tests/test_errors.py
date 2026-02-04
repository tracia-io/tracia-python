"""Tests for error handling."""

import pytest

from tracia import TraciaError, TraciaErrorCode
from tracia._errors import map_api_error_code, sanitize_error_message


class TestTraciaError:
    """Tests for TraciaError class."""

    def test_error_creation(self) -> None:
        """Test creating a TraciaError."""
        error = TraciaError(
            TraciaErrorCode.UNAUTHORIZED,
            "Access denied",
            401,
        )
        assert error.code == TraciaErrorCode.UNAUTHORIZED
        assert error.message == "Access denied"
        assert error.status_code == 401

    def test_error_str_with_status(self) -> None:
        """Test string representation with status code."""
        error = TraciaError(TraciaErrorCode.NOT_FOUND, "Not found", 404)
        assert str(error) == "[NOT_FOUND] Not found (status: 404)"

    def test_error_str_without_status(self) -> None:
        """Test string representation without status code."""
        error = TraciaError(TraciaErrorCode.NETWORK_ERROR, "Connection failed")
        assert str(error) == "[NETWORK_ERROR] Connection failed"

    def test_error_is_exception(self) -> None:
        """Test that TraciaError is an exception."""
        error = TraciaError(TraciaErrorCode.UNKNOWN, "Unknown error")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self) -> None:
        """Test that TraciaError can be raised and caught."""
        with pytest.raises(TraciaError) as exc_info:
            raise TraciaError(TraciaErrorCode.TIMEOUT, "Timed out")
        assert exc_info.value.code == TraciaErrorCode.TIMEOUT


class TestErrorCodeMapping:
    """Tests for API error code mapping."""

    def test_map_known_codes(self) -> None:
        """Test mapping of known error codes."""
        assert map_api_error_code("UNAUTHORIZED") == TraciaErrorCode.UNAUTHORIZED
        assert map_api_error_code("NOT_FOUND") == TraciaErrorCode.NOT_FOUND
        assert map_api_error_code("CONFLICT") == TraciaErrorCode.CONFLICT
        assert map_api_error_code("INVALID_REQUEST") == TraciaErrorCode.INVALID_REQUEST

    def test_map_unknown_code(self) -> None:
        """Test mapping of unknown error codes."""
        assert map_api_error_code("SOME_UNKNOWN_CODE") == TraciaErrorCode.UNKNOWN
        assert map_api_error_code("") == TraciaErrorCode.UNKNOWN


class TestErrorSanitization:
    """Tests for error message sanitization."""

    def test_sanitize_openai_key(self) -> None:
        """Test sanitization of OpenAI API keys."""
        msg = "Invalid API key: sk-1234567890abcdefghijklmnop"
        result = sanitize_error_message(msg)
        assert "sk-" not in result
        assert "[REDACTED]" in result

    def test_sanitize_bearer_token(self) -> None:
        """Test sanitization of Bearer tokens."""
        msg = "Auth failed: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = sanitize_error_message(msg)
        assert "Bearer" not in result or "eyJ" not in result
        assert "[REDACTED]" in result

    def test_sanitize_trace_id(self) -> None:
        """Test sanitization of trace IDs."""
        msg = "Span tr_1234567890abcdef not found"
        result = sanitize_error_message(msg)
        assert "tr_1234567890abcdef" not in result
        assert "[REDACTED]" in result

    def test_sanitize_span_id(self) -> None:
        """Test sanitization of span IDs."""
        msg = "Span sp_abcdef1234567890 not found"
        result = sanitize_error_message(msg)
        assert "sp_abcdef1234567890" not in result
        assert "[REDACTED]" in result

    def test_sanitize_no_sensitive_data(self) -> None:
        """Test that safe messages are unchanged."""
        msg = "Something went wrong"
        result = sanitize_error_message(msg)
        assert result == msg

    def test_sanitize_multiple_patterns(self) -> None:
        """Test sanitization of multiple sensitive patterns."""
        # Use a longer API key that matches the pattern (20+ chars)
        msg = "Key sk-abc123def456ghi789jklmnop and trace tr_1234567890abcdef failed"
        result = sanitize_error_message(msg)
        assert "sk-abc123def456ghi789jklmnop" not in result
        assert "tr_1234567890abcdef" not in result
        assert result.count("[REDACTED]") == 2
