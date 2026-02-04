"""Utility functions for the Tracia SDK."""

from __future__ import annotations

import re
import secrets
from typing import Any

from ._constants import SPAN_ID_PREFIX, TRACE_ID_PREFIX


def generate_span_id() -> str:
    """Generate a new span ID.

    Returns:
        A span ID in the format 'sp_' followed by 16 hex characters.
    """
    random_part = secrets.token_hex(8)
    return f"{SPAN_ID_PREFIX}{random_part}"


def generate_trace_id() -> str:
    """Generate a new trace ID.

    Returns:
        A trace ID in the format 'tr_' followed by 16 hex characters.
    """
    random_part = secrets.token_hex(8)
    return f"{TRACE_ID_PREFIX}{random_part}"


# Validation patterns
_SPAN_ID_PATTERN = re.compile(r"^sp_[a-f0-9]{16}$", re.IGNORECASE)
_LEGACY_SPAN_ID_PATTERN = re.compile(r"^tr_[a-f0-9]{16}$", re.IGNORECASE)
_TRACE_ID_PATTERN = re.compile(r"^tr_[a-f0-9]{16}$")


def is_valid_span_id_format(span_id: str) -> bool:
    """Check if a span ID has a valid format.

    Accepts both 'sp_' format and legacy 'tr_' format.

    Args:
        span_id: The span ID to validate.

    Returns:
        True if the span ID is valid, False otherwise.
    """
    return bool(
        _SPAN_ID_PATTERN.match(span_id) or _LEGACY_SPAN_ID_PATTERN.match(span_id)
    )


def is_valid_trace_id_format(trace_id: str) -> bool:
    """Check if a trace ID has a valid format.

    Args:
        trace_id: The trace ID to validate.

    Returns:
        True if the trace ID is valid, False otherwise.
    """
    return bool(_TRACE_ID_PATTERN.match(trace_id))


# Variable interpolation pattern: {{variable_name}}
_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def interpolate_variables(text: str, variables: dict[str, str]) -> str:
    """Interpolate variables into text using {{variable_name}} syntax.

    Args:
        text: The text containing variable placeholders.
        variables: A dictionary of variable names to values.

    Returns:
        The text with variables interpolated.
    """
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return variables.get(var_name, match.group(0))

    return _VARIABLE_PATTERN.sub(replace, text)


def interpolate_message_content(
    content: str | list[Any],
    variables: dict[str, str] | None,
) -> str | list[Any]:
    """Interpolate variables in message content.

    Args:
        content: The message content (string or list of content parts).
        variables: Optional dictionary of variable names to values.

    Returns:
        The content with variables interpolated.
    """
    if variables is None:
        return content

    if isinstance(content, str):
        return interpolate_variables(content, variables)

    # Handle list of content parts
    result = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
            result.append({
                **part,
                "text": interpolate_variables(part["text"], variables),
            })
        else:
            result.append(part)
    return result
