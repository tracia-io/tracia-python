"""Tests for type definitions."""

import pytest
from pydantic import ValidationError

from tracia import (
    ContentPart,
    CreateSpanPayload,
    EvaluateOptions,
    JsonSchemaProperty,
    LLMProvider,
    LocalPromptMessage,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolDefinition,
    ToolParameters,
)


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_create_with_aliases(self) -> None:
        """Test creating TokenUsage with camelCase aliases."""
        usage = TokenUsage(inputTokens=100, outputTokens=50, totalTokens=150)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_create_with_snake_case(self) -> None:
        """Test creating TokenUsage with snake_case."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.input_tokens == 100

    def test_serialize_with_aliases(self) -> None:
        """Test serialization uses aliases."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        data = usage.model_dump(by_alias=True)
        assert data["inputTokens"] == 100
        assert data["outputTokens"] == 50
        assert data["totalTokens"] == 150


class TestLocalPromptMessage:
    """Tests for LocalPromptMessage model."""

    def test_create_simple_message(self) -> None:
        """Test creating a simple text message."""
        msg = LocalPromptMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_create_tool_message(self) -> None:
        """Test creating a tool message."""
        msg = LocalPromptMessage(
            role="tool",
            content="Result",
            tool_call_id="call_123",
            tool_name="my_tool",
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_create_with_content_parts(self) -> None:
        """Test creating a message with content parts."""
        msg = LocalPromptMessage(
            role="assistant",
            content=[
                TextPart(text="Here's the result"),
                ToolCallPart(id="1", name="search", arguments={"query": "test"}),
            ],
        )
        assert len(msg.content) == 2

    def test_serialize_with_aliases(self) -> None:
        """Test serialization uses aliases."""
        msg = LocalPromptMessage(
            role="tool", content="Result", tool_call_id="call_123"
        )
        data = msg.model_dump(by_alias=True, exclude_none=True)
        assert data["toolCallId"] == "call_123"


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_create_tool(self) -> None:
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="search",
            description="Search for information",
            parameters=ToolParameters(
                properties={
                    "query": JsonSchemaProperty(
                        type="string", description="Search query"
                    )
                },
                required=["query"],
            ),
        )
        assert tool.name == "search"
        assert tool.description == "Search for information"
        assert "query" in tool.parameters.properties


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self) -> None:
        """Test creating a tool call."""
        call = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
        )
        assert call.id == "call_123"
        assert call.name == "search"
        assert call.arguments["query"] == "test"


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_provider_values(self) -> None:
        """Test provider enum values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"

    def test_provider_from_string(self) -> None:
        """Test creating provider from string."""
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC


class TestCreateSpanPayload:
    """Tests for CreateSpanPayload model."""

    def test_create_span_payload(self) -> None:
        """Test creating a span payload."""
        payload = CreateSpanPayload(
            spanId="sp_1234567890abcdef",
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            input={"messages": []},
            status="SUCCESS",
            latencyMs=100,
            inputTokens=50,
            outputTokens=25,
            totalTokens=75,
        )
        assert payload.span_id == "sp_1234567890abcdef"
        assert payload.status == "SUCCESS"

    def test_serialize_with_aliases(self) -> None:
        """Test serialization uses aliases."""
        payload = CreateSpanPayload(
            span_id="sp_1234567890abcdef",
            model="gpt-4o",
            provider=LLMProvider.OPENAI,
            input={"messages": []},
            status="SUCCESS",
            latency_ms=100,
            input_tokens=50,
            output_tokens=25,
            total_tokens=75,
        )
        data = payload.model_dump(by_alias=True)
        assert data["spanId"] == "sp_1234567890abcdef"
        assert data["latencyMs"] == 100


class TestEvaluateOptions:
    """Tests for EvaluateOptions model."""

    def test_create_evaluate_options(self) -> None:
        """Test creating evaluate options."""
        options = EvaluateOptions(
            evaluator="quality",
            value=1,
            note="Good response",
        )
        assert options.evaluator == "quality"
        assert options.value == 1
        assert options.note == "Good response"

    def test_create_without_note(self) -> None:
        """Test creating evaluate options without note."""
        options = EvaluateOptions(evaluator="quality", value=0)
        assert options.note is None
