"""Tests for LLM wrapper functionality."""

import pytest

from tracia import LLMProvider, LocalPromptMessage, ToolCall, ToolDefinition, ToolParameters
from tracia._errors import TraciaError, TraciaErrorCode
from tracia._llm import (
    build_assistant_message,
    convert_messages,
    convert_tool_choice,
    convert_tools,
    parse_finish_reason,
    resolve_provider,
)
from tracia._types import JsonSchemaProperty, TextPart, ToolCallPart


class TestResolveProvider:
    """Tests for provider resolution."""

    def test_explicit_provider_used(self) -> None:
        """Test that explicit provider is used."""
        provider = resolve_provider("some-model", LLMProvider.ANTHROPIC)
        assert provider == LLMProvider.ANTHROPIC

    def test_openai_model_detection(self) -> None:
        """Test detection of OpenAI models."""
        assert resolve_provider("gpt-4o", None) == LLMProvider.OPENAI
        assert resolve_provider("gpt-3.5-turbo", None) == LLMProvider.OPENAI
        assert resolve_provider("o1", None) == LLMProvider.OPENAI
        assert resolve_provider("o3-mini", None) == LLMProvider.OPENAI

    def test_anthropic_model_detection(self) -> None:
        """Test detection of Anthropic models."""
        assert resolve_provider("claude-3-opus-20240229", None) == LLMProvider.ANTHROPIC
        assert resolve_provider("claude-sonnet-4-20250514", None) == LLMProvider.ANTHROPIC

    def test_google_model_detection(self) -> None:
        """Test detection of Google models."""
        assert resolve_provider("gemini-2.0-flash", None) == LLMProvider.GOOGLE
        assert resolve_provider("gemini-2.5-pro", None) == LLMProvider.GOOGLE

    def test_unknown_model_raises(self) -> None:
        """Test that unknown model raises error."""
        with pytest.raises(TraciaError) as exc_info:
            resolve_provider("unknown-model", None)
        assert exc_info.value.code == TraciaErrorCode.UNSUPPORTED_MODEL


class TestConvertMessages:
    """Tests for message conversion."""

    def test_convert_simple_message(self) -> None:
        """Test converting a simple message."""
        messages = [LocalPromptMessage(role="user", content="Hello")]
        result = convert_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_convert_developer_to_system(self) -> None:
        """Test that developer role is converted to system."""
        messages = [LocalPromptMessage(role="developer", content="Instructions")]
        result = convert_messages(messages)
        assert result[0]["role"] == "system"

    def test_convert_tool_message(self) -> None:
        """Test converting a tool message."""
        messages = [
            LocalPromptMessage(
                role="tool", content="Result", tool_call_id="call_123"
            )
        ]
        result = convert_messages(messages)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == "Result"

    def test_convert_assistant_with_tool_calls(self) -> None:
        """Test converting assistant message with tool calls."""
        messages = [
            LocalPromptMessage(
                role="assistant",
                content=[
                    TextPart(text="Let me search"),
                    ToolCallPart(
                        id="call_1", name="search", arguments={"query": "test"}
                    ),
                ],
            )
        ]
        result = convert_messages(messages)
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"


class TestConvertTools:
    """Tests for tool conversion."""

    def test_convert_tools_none(self) -> None:
        """Test converting None tools."""
        result = convert_tools(None)
        assert result is None

    def test_convert_tools_empty(self) -> None:
        """Test converting empty tools list."""
        result = convert_tools([])
        assert result is None

    def test_convert_tool_definition(self) -> None:
        """Test converting a tool definition."""
        tools = [
            ToolDefinition(
                name="search",
                description="Search for information",
                parameters=ToolParameters(
                    properties={
                        "query": JsonSchemaProperty(type="string")
                    }
                ),
            )
        ]
        result = convert_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search"


class TestConvertToolChoice:
    """Tests for tool choice conversion."""

    def test_convert_auto(self) -> None:
        """Test converting 'auto' tool choice."""
        assert convert_tool_choice("auto") == "auto"

    def test_convert_none_choice(self) -> None:
        """Test converting 'none' tool choice."""
        assert convert_tool_choice("none") == "none"

    def test_convert_required(self) -> None:
        """Test converting 'required' tool choice."""
        assert convert_tool_choice("required") == "required"

    def test_convert_specific_tool(self) -> None:
        """Test converting specific tool choice."""
        result = convert_tool_choice({"tool": "search"})
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"

    def test_convert_none_value(self) -> None:
        """Test converting None value."""
        assert convert_tool_choice(None) is None


class TestParseFinishReason:
    """Tests for finish reason parsing."""

    def test_parse_tool_calls(self) -> None:
        """Test parsing tool_calls finish reason."""
        assert parse_finish_reason("tool_calls") == "tool_calls"

    def test_parse_length(self) -> None:
        """Test parsing length finish reason."""
        assert parse_finish_reason("length") == "max_tokens"

    def test_parse_stop(self) -> None:
        """Test parsing stop finish reason."""
        assert parse_finish_reason("stop") == "stop"

    def test_parse_unknown(self) -> None:
        """Test parsing unknown finish reason defaults to stop."""
        assert parse_finish_reason("unknown") == "stop"
        assert parse_finish_reason(None) == "stop"


class TestBuildAssistantMessage:
    """Tests for building assistant messages."""

    def test_build_text_only(self) -> None:
        """Test building text-only message."""
        msg = build_assistant_message("Hello!", [])
        assert msg.role == "assistant"
        assert msg.content == "Hello!"

    def test_build_with_tool_calls(self) -> None:
        """Test building message with tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="search", arguments={"query": "test"})
        ]
        msg = build_assistant_message("Searching...", tool_calls)
        assert msg.role == "assistant"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        # First part is text
        assert msg.content[0].type == "text"
        # Second part is tool call
        assert msg.content[1].type == "tool_call"
