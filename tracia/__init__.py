"""
Tracia - LLM prompt management and tracing SDK

A Python SDK for managing LLM prompts and tracing LLM interactions.
Uses LiteLLM as the unified provider abstraction layer.

Example usage:
    ```python
    from tracia import Tracia

    client = Tracia(api_key="your_api_key")

    # Run a local prompt
    result = client.run_local(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(result.text)

    # Run with streaming
    stream = client.run_local(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True
    )
    for chunk in stream:
        print(chunk, end="")

    # Use prompts API
    prompts = client.prompts.list()
    prompt = client.prompts.get("my-prompt")

    # Use spans API
    spans = client.spans.list()
    client.spans.evaluate("sp_xxx", EvaluateOptions(evaluator="quality", value=1))

    # Create a session for multi-turn conversations
    session = client.create_session()
    r1 = session.run_local(model="gpt-4o", messages=[...])
    r2 = session.run_local(model="gpt-4o", messages=[...])  # Linked

    # Clean up
    client.close()
    ```
"""

from ._client import Tracia
from ._constants import SDK_VERSION, Eval
from ._errors import TraciaError, TraciaErrorCode
from ._session import TraciaSession
from ._streaming import AsyncLocalStream, LocalStream
from ._types import (
    ContentPart,
    CreatePromptOptions,
    CreateSpanPayload,
    CreateSpanResult,
    EvaluateOptions,
    EvaluateResult,
    FinishReason,
    JsonSchemaProperty,
    ListSpansOptions,
    ListSpansResult,
    LLMProvider,
    LocalPromptMessage,
    Prompt,
    PromptListItem,
    PromptMessage,
    PromptVersion,
    ResponseFormatJsonSchema,
    ResponsesEvent,
    ResponsesFunctionCall,
    ResponsesFunctionCallOutput,
    ResponsesInputItem,
    ResponsesInputMessage,
    ResponsesMessage,
    ResponsesOutputItem,
    ResponsesToolCall,
    RunLocalInput,
    RunLocalResult,
    RunOptions,
    RunResponsesInput,
    RunResponsesResult,
    RunResult,
    Span,
    SpanListItem,
    StreamResult,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolParameters,
    UpdatePromptOptions,
)

__version__ = SDK_VERSION
__all__ = [
    # Main client
    "Tracia",
    "TraciaSession",
    # Errors
    "TraciaError",
    "TraciaErrorCode",
    # Constants
    "Eval",
    # Streaming
    "LocalStream",
    "AsyncLocalStream",
    # Types - Core
    "LLMProvider",
    "TokenUsage",
    "FinishReason",
    # Types - Messages
    "LocalPromptMessage",
    "ContentPart",
    "TextPart",
    "ToolCallPart",
    # Types - Tools
    "ToolDefinition",
    "ToolParameters",
    "JsonSchemaProperty",
    "ToolCall",
    "ToolChoice",
    # Types - Response Format
    "ResponseFormatJsonSchema",
    # Types - Run Local
    "RunLocalInput",
    "RunLocalResult",
    "StreamResult",
    # Types - Run Responses (OpenAI Responses API)
    "RunResponsesInput",
    "RunResponsesResult",
    "ResponsesInputItem",
    "ResponsesInputMessage",
    "ResponsesFunctionCallOutput",
    "ResponsesFunctionCall",
    "ResponsesMessage",
    "ResponsesOutputItem",
    "ResponsesToolCall",
    "ResponsesEvent",
    # Types - Spans
    "CreateSpanPayload",
    "CreateSpanResult",
    "Span",
    "SpanListItem",
    "ListSpansOptions",
    "ListSpansResult",
    "EvaluateOptions",
    "EvaluateResult",
    # Types - Prompts
    "Prompt",
    "PromptListItem",
    "PromptMessage",
    "PromptVersion",
    "CreatePromptOptions",
    "UpdatePromptOptions",
    "RunOptions",
    "RunResult",
]
