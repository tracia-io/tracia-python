"""Type definitions for the Tracia SDK."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AMAZON_BEDROCK = "amazon_bedrock"


class TokenUsage(BaseModel):
    """Token usage statistics."""

    model_config = ConfigDict(populate_by_name=True)

    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    total_tokens: int = Field(alias="totalTokens")


# Content Parts


class TextPart(BaseModel):
    """Text content part."""

    type: Literal["text"] = "text"
    text: str


class ToolCallPart(BaseModel):
    """Tool call content part."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


ContentPart = Union[TextPart, ToolCallPart]


# Tool Definitions


class JsonSchemaProperty(BaseModel):
    """JSON schema property definition."""

    type: Literal["string", "number", "integer", "boolean", "array", "object"]
    description: str | None = None
    enum: list[str | int] | None = None
    items: "JsonSchemaProperty | None" = None
    properties: dict[str, "JsonSchemaProperty"] | None = None
    required: list[str] | None = None


class ToolParameters(BaseModel):
    """Tool parameter schema."""

    type: Literal["object"] = "object"
    properties: dict[str, JsonSchemaProperty]
    required: list[str] | None = None


class ToolDefinition(BaseModel):
    """Tool definition for function calling."""

    name: str
    description: str
    parameters: ToolParameters


class ToolCall(BaseModel):
    """A tool call made by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


ToolChoice = Union[Literal["auto", "none", "required"], dict[str, str]]


# Response Format


class ResponseFormatJsonSchema(BaseModel):
    """JSON schema response format for structured outputs."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["json"] = "json"
    schema_: dict[str, Any] = Field(alias="schema")
    name: str | None = None
    description: str | None = None


ResponseFormat = Union[
    dict[str, Any],
    ResponseFormatJsonSchema,
]


# Messages


class LocalPromptMessage(BaseModel):
    """A message in a local prompt."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[ContentPart]
    tool_call_id: str | None = Field(default=None, alias="toolCallId")
    tool_name: str | None = Field(default=None, alias="toolName")

    model_config = ConfigDict(populate_by_name=True)


FinishReason = Literal["stop", "tool_calls", "max_tokens"]


# Run Local Types


class RunLocalInput(BaseModel):
    """Input for run_local method."""

    model_config = ConfigDict(populate_by_name=True)

    messages: list[LocalPromptMessage]
    model: str
    stream: bool = False
    provider: LLMProvider | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    top_p: float | None = Field(default=None, alias="topP")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    custom_options: dict[str, Any] | None = Field(default=None, alias="customOptions")
    variables: dict[str, str] | None = None
    provider_api_key: str | None = Field(default=None, alias="providerApiKey")
    tags: list[str] | None = None
    user_id: str | None = Field(default=None, alias="userId")
    session_id: str | None = Field(default=None, alias="sessionId")
    send_trace: bool | None = Field(default=None, alias="sendTrace")
    span_id: str | None = Field(default=None, alias="spanId")
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = Field(default=None, alias="toolChoice")
    response_format: ResponseFormat | None = Field(default=None, alias="responseFormat")
    trace_id: str | None = Field(default=None, alias="traceId")
    parent_span_id: str | None = Field(default=None, alias="parentSpanId")


class RunLocalResult(BaseModel):
    """Result from run_local method."""

    model_config = ConfigDict(populate_by_name=True)

    text: str
    span_id: str = Field(alias="spanId")
    trace_id: str = Field(alias="traceId")
    latency_ms: int = Field(alias="latencyMs")
    usage: TokenUsage
    cost: float | None = None
    provider: LLMProvider
    model: str
    tool_calls: list[ToolCall] = Field(default_factory=list, alias="toolCalls")
    finish_reason: FinishReason = Field(alias="finishReason")
    message: LocalPromptMessage


class StreamResult(RunLocalResult):
    """Result from a streaming run_local call."""

    aborted: bool = False


# Run Responses Types (OpenAI Responses API)


class ResponsesInputMessage(BaseModel):
    """User or developer message for Responses API."""

    role: Literal["developer", "user"]
    content: str


class ResponsesFunctionCallOutput(BaseModel):
    """Function call output for Responses API."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(alias="callId")
    output: str

    model_config = ConfigDict(populate_by_name=True)


class ResponsesFunctionCall(BaseModel):
    """Function call for Responses API."""

    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str = Field(alias="callId")
    name: str
    arguments: str

    model_config = ConfigDict(populate_by_name=True)


class ResponsesMessage(BaseModel):
    """Message output for Responses API."""

    type: Literal["message"] = "message"
    role: Literal["assistant"]
    content: list[dict[str, Any]]


ResponsesOutputItem = Union[ResponsesFunctionCall, ResponsesMessage]
ResponsesInputItem = Union[
    ResponsesInputMessage, ResponsesFunctionCallOutput, ResponsesOutputItem
]


class RunResponsesInput(BaseModel):
    """Input for run_responses method."""

    model_config = ConfigDict(populate_by_name=True)

    model: str
    input: list[ResponsesInputItem]
    stream: bool = False
    tools: list[ToolDefinition] | None = None
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    provider_api_key: str | None = Field(default=None, alias="providerApiKey")
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")
    send_trace: bool | None = Field(default=None, alias="sendTrace")
    span_id: str | None = Field(default=None, alias="spanId")
    tags: list[str] | None = None
    user_id: str | None = Field(default=None, alias="userId")
    session_id: str | None = Field(default=None, alias="sessionId")
    trace_id: str | None = Field(default=None, alias="traceId")
    parent_span_id: str | None = Field(default=None, alias="parentSpanId")


class ResponsesToolCall(BaseModel):
    """Tool call from Responses API."""

    id: str
    call_id: str = Field(alias="callId")
    name: str
    arguments: dict[str, Any]

    model_config = ConfigDict(populate_by_name=True)


class RunResponsesResult(BaseModel):
    """Result from run_responses method."""

    model_config = ConfigDict(populate_by_name=True)

    text: str
    span_id: str = Field(alias="spanId")
    trace_id: str = Field(alias="traceId")
    latency_ms: int = Field(alias="latencyMs")
    usage: TokenUsage
    output_items: list[ResponsesOutputItem] = Field(
        default_factory=list, alias="outputItems"
    )
    tool_calls: list[ResponsesToolCall] = Field(default_factory=list, alias="toolCalls")
    aborted: bool = False


# Responses Events


class TextDeltaEvent(BaseModel):
    """Text delta event during streaming."""

    type: Literal["text_delta"] = "text_delta"
    data: str


class TextEvent(BaseModel):
    """Complete text event."""

    type: Literal["text"] = "text"
    data: str


class ReasoningEvent(BaseModel):
    """Reasoning/thinking content event."""

    type: Literal["reasoning"] = "reasoning"
    content: str


class ToolCallEvent(BaseModel):
    """Tool call event."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["tool_call"] = "tool_call"
    id: str
    call_id: str = Field(alias="callId")
    name: str
    arguments: dict[str, Any]


class DoneEvent(BaseModel):
    """Stream completion event."""

    type: Literal["done"] = "done"
    usage: TokenUsage


ResponsesEvent = Union[
    TextDeltaEvent, TextEvent, ReasoningEvent, ToolCallEvent, DoneEvent
]


# Span Types


class CreateSpanPayload(BaseModel):
    """Payload for creating a span."""

    model_config = ConfigDict(populate_by_name=True)

    span_id: str = Field(alias="spanId")
    model: str
    provider: LLMProvider
    input: dict[str, Any]
    variables: dict[str, str] | None = None
    output: str | None = None
    status: Literal["SUCCESS", "ERROR"]
    error: str | None = None
    latency_ms: int = Field(alias="latencyMs")
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    total_tokens: int = Field(alias="totalTokens")
    tags: list[str] | None = None
    user_id: str | None = Field(default=None, alias="userId")
    session_id: str | None = Field(default=None, alias="sessionId")
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    top_p: float | None = Field(default=None, alias="topP")
    tools: list[ToolDefinition] | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, alias="toolCalls")
    trace_id: str | None = Field(default=None, alias="traceId")
    parent_span_id: str | None = Field(default=None, alias="parentSpanId")


class CreateSpanResult(BaseModel):
    """Result from creating a span."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    span_id: str = Field(alias="spanId")
    trace_id: str = Field(alias="traceId")


class SpanListItem(BaseModel):
    """A span item from the list endpoint (reduced fields)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    span_id: str = Field(alias="spanId")
    trace_id: str | None = Field(default=None, alias="traceId")
    prompt_slug: str | None = Field(default=None, alias="promptSlug")
    model: str
    status: Literal["SUCCESS", "ERROR"]
    latency_ms: int = Field(alias="latencyMs")
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    total_tokens: int = Field(alias="totalTokens")
    cost: float | None = None
    created_at: datetime = Field(alias="createdAt")


class Span(BaseModel):
    """A span from the API (full detail)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    span_id: str = Field(alias="spanId")
    trace_id: str | None = Field(default=None, alias="traceId")
    parent_span_id: str | None = Field(default=None, alias="parentSpanId")
    prompt_slug: str | None = Field(default=None, alias="promptSlug")
    prompt_version: int | None = Field(default=None, alias="promptVersion")
    model: str
    provider: str
    input: dict[str, Any]
    variables: dict[str, str] | None = None
    output: str | None = None
    status: Literal["SUCCESS", "ERROR"]
    error: str | None = None
    latency_ms: int = Field(alias="latencyMs")
    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    total_tokens: int = Field(alias="totalTokens")
    cost: float | None = None
    tags: list[str] = Field(default_factory=list)
    user_id: str | None = Field(default=None, alias="userId")
    session_id: str | None = Field(default=None, alias="sessionId")
    created_at: datetime = Field(alias="createdAt")


class ListSpansOptions(BaseModel):
    """Options for listing spans."""

    model_config = ConfigDict(populate_by_name=True)

    prompt_slug: str | None = Field(default=None, alias="promptSlug")
    status: Literal["SUCCESS", "ERROR"] | None = None
    start_date: datetime | None = Field(default=None, alias="startDate")
    end_date: datetime | None = Field(default=None, alias="endDate")
    user_id: str | None = Field(default=None, alias="userId")
    session_id: str | None = Field(default=None, alias="sessionId")
    tags: list[str] | None = None
    limit: int | None = None
    cursor: str | None = None


class ListSpansResult(BaseModel):
    """Result from listing spans."""

    model_config = ConfigDict(populate_by_name=True)

    spans: list[SpanListItem]
    cursor: str | None = None
    has_more: bool = Field(default=False, alias="hasMore")


# Evaluate Types


class EvaluateOptions(BaseModel):
    """Options for evaluating a span."""

    evaluator: str
    value: int | float
    note: str | None = None


class EvaluateResult(BaseModel):
    """Result from evaluating a span."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    evaluator_key: str = Field(alias="evaluatorKey")
    evaluator_name: str = Field(alias="evaluatorName")
    value: float
    source: str
    note: str | None = None
    created_at: datetime = Field(alias="createdAt")


# Prompt Types


class PromptMessage(BaseModel):
    """A message in a prompt template."""

    id: str
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class PromptVersion(BaseModel):
    """A version of a prompt."""

    model_config = ConfigDict(populate_by_name=True)

    version: int
    messages: list[PromptMessage]
    model: str
    provider: LLMProvider
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    top_p: float | None = Field(default=None, alias="topP")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")
    created_at: datetime = Field(alias="createdAt")


class Prompt(BaseModel):
    """A prompt from the API."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    slug: str
    name: str
    description: str | None = None
    current_version: int = Field(alias="currentVersion")
    versions: list[PromptVersion] = Field(default_factory=list)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class PromptListItem(BaseModel):
    """A prompt item in the list response."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    slug: str
    name: str
    description: str | None = None
    current_version: int = Field(alias="currentVersion")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class CreatePromptOptions(BaseModel):
    """Options for creating a prompt."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    slug: str | None = None
    description: str | None = None
    content: list[PromptMessage]
    model: str | None = None
    provider: LLMProvider | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    top_p: float | None = Field(default=None, alias="topP")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")


class UpdatePromptOptions(BaseModel):
    """Options for updating a prompt."""

    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    slug: str | None = None
    description: str | None = None
    content: list[PromptMessage] | None = None
    model: str | None = None
    provider: LLMProvider | None = None
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, alias="maxOutputTokens")
    top_p: float | None = Field(default=None, alias="topP")
    stop_sequences: list[str] | None = Field(default=None, alias="stopSequences")


class RunOptions(BaseModel):
    """Options for running a prompt."""

    model_config = ConfigDict(populate_by_name=True)

    model: str | None = None
    version: int | None = None
    tags: list[str] | None = None
    user_id: str | None = Field(default=None, alias="userId")
    session_id: str | None = Field(default=None, alias="sessionId")
    trace_id: str | None = Field(default=None, alias="traceId")
    parent_span_id: str | None = Field(default=None, alias="parentSpanId")
    messages: list[LocalPromptMessage] | None = None


class RunResult(BaseModel):
    """Result from running a prompt (via API)."""

    model_config = ConfigDict(populate_by_name=True)

    text: str | None = None
    span_id: str = Field(alias="spanId")
    trace_id: str = Field(alias="traceId")
    prompt_version: int = Field(alias="promptVersion")
    latency_ms: int = Field(alias="latencyMs")
    usage: TokenUsage
    cost: float
    finish_reason: FinishReason | None = Field(default=None, alias="finishReason")
    tool_calls: list[ToolCall] | None = Field(default=None, alias="toolCalls")
    structured_output: dict[str, Any] | None = Field(default=None, alias="structuredOutput")
    messages: list[LocalPromptMessage] | None = None
