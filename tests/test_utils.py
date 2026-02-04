"""Tests for utility functions."""

import pytest

from tracia._utils import (
    generate_span_id,
    generate_trace_id,
    interpolate_message_content,
    interpolate_variables,
    is_valid_span_id_format,
    is_valid_trace_id_format,
)


class TestIdGeneration:
    """Tests for ID generation functions."""

    def test_generate_span_id_format(self) -> None:
        """Test that generated span IDs have correct format."""
        span_id = generate_span_id()
        assert span_id.startswith("sp_")
        assert len(span_id) == 19  # sp_ + 16 hex chars
        assert is_valid_span_id_format(span_id)

    def test_generate_span_id_uniqueness(self) -> None:
        """Test that generated span IDs are unique."""
        ids = {generate_span_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_trace_id_format(self) -> None:
        """Test that generated trace IDs have correct format."""
        trace_id = generate_trace_id()
        assert trace_id.startswith("tr_")
        assert len(trace_id) == 19  # tr_ + 16 hex chars
        assert is_valid_trace_id_format(trace_id)

    def test_generate_trace_id_uniqueness(self) -> None:
        """Test that generated trace IDs are unique."""
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100


class TestIdValidation:
    """Tests for ID validation functions."""

    def test_valid_span_id_sp_format(self) -> None:
        """Test validation of sp_ format span IDs."""
        assert is_valid_span_id_format("sp_1234567890abcdef")
        assert is_valid_span_id_format("sp_ABCDEF1234567890")

    def test_valid_span_id_legacy_tr_format(self) -> None:
        """Test validation of legacy tr_ format span IDs."""
        assert is_valid_span_id_format("tr_1234567890abcdef")

    def test_invalid_span_id(self) -> None:
        """Test rejection of invalid span IDs."""
        assert not is_valid_span_id_format("")
        assert not is_valid_span_id_format("invalid")
        assert not is_valid_span_id_format("sp_123")  # Too short
        assert not is_valid_span_id_format("sp_12345678901234567890")  # Too long
        assert not is_valid_span_id_format("xx_1234567890abcdef")  # Wrong prefix

    def test_valid_trace_id(self) -> None:
        """Test validation of valid trace IDs."""
        assert is_valid_trace_id_format("tr_1234567890abcdef")

    def test_invalid_trace_id(self) -> None:
        """Test rejection of invalid trace IDs."""
        assert not is_valid_trace_id_format("")
        assert not is_valid_trace_id_format("invalid")
        assert not is_valid_trace_id_format("sp_1234567890abcdef")  # Wrong prefix
        assert not is_valid_trace_id_format("tr_123")  # Too short


class TestVariableInterpolation:
    """Tests for variable interpolation functions."""

    def test_interpolate_variables_simple(self) -> None:
        """Test simple variable interpolation."""
        text = "Hello {{name}}!"
        result = interpolate_variables(text, {"name": "World"})
        assert result == "Hello World!"

    def test_interpolate_variables_multiple(self) -> None:
        """Test multiple variable interpolation."""
        text = "{{greeting}} {{name}}, welcome to {{place}}!"
        result = interpolate_variables(
            text, {"greeting": "Hello", "name": "User", "place": "Tracia"}
        )
        assert result == "Hello User, welcome to Tracia!"

    def test_interpolate_variables_missing(self) -> None:
        """Test that missing variables are preserved."""
        text = "Hello {{name}}, your ID is {{id}}"
        result = interpolate_variables(text, {"name": "User"})
        assert result == "Hello User, your ID is {{id}}"

    def test_interpolate_variables_empty(self) -> None:
        """Test interpolation with no variables."""
        text = "Hello World!"
        result = interpolate_variables(text, {})
        assert result == "Hello World!"

    def test_interpolate_message_content_string(self) -> None:
        """Test interpolation of string content."""
        content = "Hello {{name}}!"
        result = interpolate_message_content(content, {"name": "World"})
        assert result == "Hello World!"

    def test_interpolate_message_content_list(self) -> None:
        """Test interpolation of list content."""
        content = [
            {"type": "text", "text": "Hello {{name}}!"},
            {"type": "tool_call", "id": "1", "name": "test", "arguments": {}},
        ]
        result = interpolate_message_content(content, {"name": "World"})
        assert result[0]["text"] == "Hello World!"
        assert result[1] == content[1]  # Tool call unchanged

    def test_interpolate_message_content_none_variables(self) -> None:
        """Test that None variables returns content unchanged."""
        content = "Hello {{name}}!"
        result = interpolate_message_content(content, None)
        assert result == content
