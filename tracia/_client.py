"""Main Tracia client implementation."""

from __future__ import annotations

import asyncio
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock
from typing import Any, Callable, Literal, overload

from ._constants import (
    BASE_URL,
    DEFAULT_TIMEOUT_MS,
    MAX_PENDING_SPANS,
    SPAN_RETRY_ATTEMPTS,
    SPAN_RETRY_DELAY_MS,
    SPAN_STATUS_ERROR,
    SPAN_STATUS_SUCCESS,
)
from ._errors import TraciaError, TraciaErrorCode, sanitize_error_message
from ._http import AsyncHttpClient, HttpClient
from ._llm import LLMClient, build_assistant_message, convert_response_format, resolve_provider
from ._session import TraciaSession
from ._streaming import AsyncLocalStream, LocalStream
from ._types import (
    CreateSpanPayload,
    LocalPromptMessage,
    LLMProvider,
    ResponseFormat,
    RunLocalResult,
    StreamResult,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolChoice,
)
from ._utils import (
    generate_span_id,
    generate_trace_id,
    interpolate_message_content,
    is_valid_span_id_format,
)
from .resources import Prompts, Spans


class Tracia:
    """Main Tracia client for LLM prompt management and tracing.

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

        # Create a session for multi-turn conversations
        session = client.create_session()
        r1 = session.run_local(model="gpt-4o", messages=[...])
        r2 = session.run_local(model="gpt-4o", messages=[...])  # Linked

        # Clean up
        client.close()
        ```
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        on_span_error: Callable[[Exception, str], None] | None = None,
    ) -> None:
        """Initialize the Tracia client.

        Args:
            api_key: Your Tracia API key.
            base_url: The base URL for the Tracia API.
            on_span_error: Optional callback for span creation errors.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._on_span_error = on_span_error

        # HTTP clients
        self._http_client = HttpClient(api_key, base_url)
        self._async_http_client: AsyncHttpClient | None = None

        # LLM client
        self._llm_client = LLMClient()

        # Resources
        self.prompts = Prompts(self._http_client)
        self.spans = Spans(self._http_client)

        # Pending spans management
        self._pending_spans: dict[str, Future[None]] = {}
        self._pending_spans_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="tracia-span-")
        self._closed = False

    def _get_async_http_client(self) -> AsyncHttpClient:
        """Get or create the async HTTP client."""
        if self._async_http_client is None:
            self._async_http_client = AsyncHttpClient(self._api_key, self._base_url)
            # Update resources with async client
            self.prompts._async_client = self._async_http_client
            self.spans._async_client = self._async_http_client
        return self._async_http_client

    def _validate_run_local_input(
        self,
        messages: list[dict[str, Any] | LocalPromptMessage],
        model: str,
        span_id: str | None,
    ) -> None:
        """Validate run_local input parameters."""
        if not model or not model.strip():
            raise TraciaError(
                code=TraciaErrorCode.INVALID_REQUEST,
                message="model is required and cannot be empty",
            )

        if not messages or len(messages) == 0:
            raise TraciaError(
                code=TraciaErrorCode.INVALID_REQUEST,
                message="messages is required and cannot be empty",
            )

        if span_id is not None and not is_valid_span_id_format(span_id):
            raise TraciaError(
                code=TraciaErrorCode.INVALID_REQUEST,
                message=f"Invalid span_id format: {span_id}. Expected sp_XXXXXXXXXXXXXXXX or tr_XXXXXXXXXXXXXXXX",
            )

        # Validate tool messages have tool_call_id
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "tool" and not msg.get("tool_call_id") and not msg.get("toolCallId"):
                    raise TraciaError(
                        code=TraciaErrorCode.INVALID_REQUEST,
                        message="Tool messages must have tool_call_id",
                    )
            elif isinstance(msg, LocalPromptMessage):
                if msg.role == "tool" and not msg.tool_call_id:
                    raise TraciaError(
                        code=TraciaErrorCode.INVALID_REQUEST,
                        message="Tool messages must have tool_call_id",
                    )

    def _convert_messages(
        self,
        messages: list[dict[str, Any] | LocalPromptMessage],
        variables: dict[str, str] | None,
    ) -> list[LocalPromptMessage]:
        """Convert dict messages to LocalPromptMessage and interpolate variables."""
        result: list[LocalPromptMessage] = []

        for msg in messages:
            if isinstance(msg, LocalPromptMessage):
                pm = msg
            else:
                pm = LocalPromptMessage(
                    role=msg["role"],
                    content=msg.get("content", ""),
                    tool_call_id=msg.get("tool_call_id") or msg.get("toolCallId"),
                    tool_name=msg.get("tool_name") or msg.get("toolName"),
                )

            # Interpolate variables (skip tool messages)
            if pm.role != "tool" and variables:
                pm = LocalPromptMessage(
                    role=pm.role,
                    content=interpolate_message_content(pm.content, variables),
                    tool_call_id=pm.tool_call_id,
                    tool_name=pm.tool_name,
                )

            result.append(pm)

        return result

    def _schedule_span_creation(
        self,
        payload: CreateSpanPayload,
    ) -> None:
        """Schedule span creation in the background with retry logic."""
        span_id = payload.span_id

        def create_span_with_retry() -> None:
            last_error: Exception | None = None

            for attempt in range(SPAN_RETRY_ATTEMPTS + 1):
                try:
                    self._http_client.post(
                        "/api/v1/spans",
                        payload.model_dump(by_alias=True, exclude_none=True),
                    )
                    return
                except Exception as e:
                    last_error = e
                    if attempt < SPAN_RETRY_ATTEMPTS:
                        delay = SPAN_RETRY_DELAY_MS * (attempt + 1) / 1000.0
                        time.sleep(delay)

            # All retries failed
            if self._on_span_error and last_error:
                try:
                    self._on_span_error(last_error, span_id)
                except Exception:
                    pass

        # Evict old spans if at capacity
        with self._pending_spans_lock:
            if len(self._pending_spans) >= MAX_PENDING_SPANS:
                # Remove the oldest span
                oldest_key = next(iter(self._pending_spans))
                del self._pending_spans[oldest_key]

            future = self._executor.submit(create_span_with_retry)
            self._pending_spans[span_id] = future

            # Clean up when done
            def on_done(f: Future[None]) -> None:
                with self._pending_spans_lock:
                    self._pending_spans.pop(span_id, None)

            future.add_done_callback(on_done)

    @overload
    def run_local(
        self,
        *,
        messages: list[dict[str, Any] | LocalPromptMessage],
        model: str,
        stream: Literal[False] = False,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_ms: int | None = None,
        variables: dict[str, str] | None = None,
        provider_api_key: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        send_trace: bool | None = None,
        span_id: str | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> RunLocalResult: ...

    @overload
    def run_local(
        self,
        *,
        messages: list[dict[str, Any] | LocalPromptMessage],
        model: str,
        stream: Literal[True],
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_ms: int | None = None,
        variables: dict[str, str] | None = None,
        provider_api_key: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        send_trace: bool | None = None,
        span_id: str | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> LocalStream: ...

    def run_local(
        self,
        *,
        messages: list[dict[str, Any] | LocalPromptMessage],
        model: str,
        stream: bool = False,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_ms: int | None = None,
        variables: dict[str, str] | None = None,
        provider_api_key: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        send_trace: bool | None = None,
        span_id: str | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> RunLocalResult | LocalStream:
        """Run a local prompt with an LLM.

        Args:
            messages: The messages to send.
            model: The model name (e.g., "gpt-4o", "claude-sonnet-4-20250514").
            stream: Whether to stream the response.
            provider: The LLM provider (auto-detected if not specified).
            temperature: Sampling temperature.
            max_output_tokens: Maximum output tokens.
            top_p: Top-p sampling parameter.
            stop_sequences: Stop sequences.
            timeout_ms: Request timeout in milliseconds.
            variables: Variables to interpolate into messages.
            provider_api_key: API key for the LLM provider.
            tags: Tags for the span.
            user_id: User ID for the span.
            session_id: Session ID for the span.
            send_trace: Whether to send trace data (default True).
            span_id: Custom span ID.
            tools: Tool definitions for function calling.
            tool_choice: Tool choice setting.
            trace_id: Trace ID for linking spans.
            parent_span_id: Parent span ID for nested spans.

        Returns:
            RunLocalResult for non-streaming, LocalStream for streaming.

        Raises:
            TraciaError: If the request fails.
        """
        # Validate input
        self._validate_run_local_input(messages, model, span_id)

        # Convert and interpolate messages
        prompt_messages = self._convert_messages(messages, variables)

        # Generate IDs
        effective_span_id = span_id or generate_span_id()
        effective_trace_id = trace_id or generate_trace_id()
        should_send_trace = send_trace is not False

        # Calculate timeout
        timeout_seconds = (timeout_ms or DEFAULT_TIMEOUT_MS) / 1000.0

        litellm_response_format = convert_response_format(response_format)

        if stream:
            return self._run_local_streaming(
                messages=prompt_messages,
                model=model,
                provider=provider,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences,
                timeout=timeout_seconds,
                provider_api_key=provider_api_key,
                tags=tags,
                user_id=user_id,
                session_id=session_id,
                send_trace=should_send_trace,
                span_id=effective_span_id,
                trace_id=effective_trace_id,
                parent_span_id=parent_span_id,
                tools=tools,
                tool_choice=tool_choice,
                variables=variables,
                response_format=litellm_response_format,
            )
        else:
            return self._run_local_non_streaming(
                messages=prompt_messages,
                model=model,
                provider=provider,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences,
                timeout=timeout_seconds,
                provider_api_key=provider_api_key,
                tags=tags,
                user_id=user_id,
                session_id=session_id,
                send_trace=should_send_trace,
                span_id=effective_span_id,
                trace_id=effective_trace_id,
                parent_span_id=parent_span_id,
                tools=tools,
                tool_choice=tool_choice,
                variables=variables,
                response_format=litellm_response_format,
            )

    def _run_local_non_streaming(
        self,
        *,
        messages: list[LocalPromptMessage],
        model: str,
        provider: LLMProvider | None,
        temperature: float | None,
        max_output_tokens: int | None,
        top_p: float | None,
        stop_sequences: list[str] | None,
        timeout: float,
        provider_api_key: str | None,
        tags: list[str] | None,
        user_id: str | None,
        session_id: str | None,
        send_trace: bool,
        span_id: str,
        trace_id: str,
        parent_span_id: str | None,
        tools: list[ToolDefinition] | None,
        tool_choice: ToolChoice | None,
        variables: dict[str, str] | None,
        response_format: dict[str, Any] | None = None,
    ) -> RunLocalResult:
        """Run local prompt without streaming."""
        start_time = time.time()
        error_message: str | None = None
        result_text = ""
        result_tool_calls: list[ToolCall] = []
        result_usage = TokenUsage(inputTokens=0, outputTokens=0, totalTokens=0)
        result_provider = resolve_provider(model, provider)
        finish_reason = "stop"

        try:
            completion = self._llm_client.complete(
                model=model,
                messages=messages,
                provider=result_provider,
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
                api_key=provider_api_key,
                timeout=timeout,
                response_format=response_format,
            )

            result_text = completion.text
            result_tool_calls = completion.tool_calls
            result_provider = completion.provider
            finish_reason = completion.finish_reason
            result_usage = TokenUsage(
                inputTokens=completion.input_tokens,
                outputTokens=completion.output_tokens,
                totalTokens=completion.total_tokens,
            )

        except TraciaError:
            raise
        except Exception as e:
            error_message = sanitize_error_message(str(e))
            raise TraciaError(
                code=TraciaErrorCode.PROVIDER_ERROR,
                message=f"LLM provider error: {error_message}",
            ) from e
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            if send_trace:
                payload = CreateSpanPayload(
                    spanId=span_id,
                    model=model,
                    provider=result_provider,
                    input={"messages": [m.model_dump(by_alias=True, exclude_none=True) for m in messages]},
                    variables=variables,
                    output=result_text if not error_message else None,
                    status=SPAN_STATUS_ERROR if error_message else SPAN_STATUS_SUCCESS,
                    error=error_message,
                    latencyMs=latency_ms,
                    inputTokens=result_usage.input_tokens,
                    outputTokens=result_usage.output_tokens,
                    totalTokens=result_usage.total_tokens,
                    tags=tags,
                    userId=user_id,
                    sessionId=session_id,
                    temperature=temperature,
                    maxOutputTokens=max_output_tokens,
                    topP=top_p,
                    tools=tools,
                    toolCalls=result_tool_calls if result_tool_calls else None,
                    traceId=trace_id,
                    parentSpanId=parent_span_id,
                )
                self._schedule_span_creation(payload)

        return RunLocalResult(
            text=result_text,
            spanId=span_id,
            traceId=trace_id,
            latencyMs=latency_ms,
            usage=result_usage,
            cost=None,
            provider=result_provider,
            model=model,
            toolCalls=result_tool_calls,
            finishReason=finish_reason,
            message=build_assistant_message(result_text, result_tool_calls),
        )

    def _run_local_streaming(
        self,
        *,
        messages: list[LocalPromptMessage],
        model: str,
        provider: LLMProvider | None,
        temperature: float | None,
        max_output_tokens: int | None,
        top_p: float | None,
        stop_sequences: list[str] | None,
        timeout: float,
        provider_api_key: str | None,
        tags: list[str] | None,
        user_id: str | None,
        session_id: str | None,
        send_trace: bool,
        span_id: str,
        trace_id: str,
        parent_span_id: str | None,
        tools: list[ToolDefinition] | None,
        tool_choice: ToolChoice | None,
        variables: dict[str, str] | None,
        response_format: dict[str, Any] | None = None,
    ) -> LocalStream:
        """Run local prompt with streaming."""
        start_time = time.time()
        abort_event = Event()
        result_future: Future[StreamResult] = Future()

        chunks_iter, result_holder, resolved_provider = self._llm_client.stream(
            model=model,
            messages=messages,
            provider=provider,
            temperature=temperature,
            max_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            api_key=provider_api_key,
            timeout=timeout,
            response_format=response_format,
        )

        def wrapped_chunks():
            error_message: str | None = None
            aborted = False

            try:
                for chunk in chunks_iter:
                    if abort_event.is_set():
                        aborted = True
                        break
                    yield chunk
            except Exception as e:
                error_message = sanitize_error_message(str(e))
                raise
            finally:
                latency_ms = int((time.time() - start_time) * 1000)

                # Get completion result
                completion = result_holder[0] if result_holder else None

                result_text = completion.text if completion else ""
                result_tool_calls = completion.tool_calls if completion else []
                result_usage = TokenUsage(
                    inputTokens=completion.input_tokens if completion else 0,
                    outputTokens=completion.output_tokens if completion else 0,
                    totalTokens=completion.total_tokens if completion else 0,
                )
                finish_reason = completion.finish_reason if completion else "stop"

                if send_trace:
                    payload = CreateSpanPayload(
                        spanId=span_id,
                        model=model,
                        provider=resolved_provider,
                        input={"messages": [m.model_dump(by_alias=True, exclude_none=True) for m in messages]},
                        variables=variables,
                        output=result_text if not error_message else None,
                        status=SPAN_STATUS_ERROR if error_message else SPAN_STATUS_SUCCESS,
                        error=error_message,
                        latencyMs=latency_ms,
                        inputTokens=result_usage.input_tokens,
                        outputTokens=result_usage.output_tokens,
                        totalTokens=result_usage.total_tokens,
                        tags=tags,
                        userId=user_id,
                        sessionId=session_id,
                        temperature=temperature,
                        maxOutputTokens=max_output_tokens,
                        topP=top_p,
                        tools=tools,
                        toolCalls=result_tool_calls if result_tool_calls else None,
                        traceId=trace_id,
                        parentSpanId=parent_span_id,
                    )
                    self._schedule_span_creation(payload)

                # Set the result
                stream_result = StreamResult(
                    text=result_text,
                    spanId=span_id,
                    traceId=trace_id,
                    latencyMs=latency_ms,
                    usage=result_usage,
                    cost=None,
                    provider=resolved_provider,
                    model=model,
                    toolCalls=result_tool_calls,
                    finishReason=finish_reason,
                    message=build_assistant_message(result_text, result_tool_calls),
                    aborted=aborted,
                )
                result_future.set_result(stream_result)

        return LocalStream(
            span_id=span_id,
            trace_id=trace_id,
            chunks=wrapped_chunks(),
            result_holder=result_holder,
            result_future=result_future,
            abort_event=abort_event,
        )

    @overload
    async def arun_local(
        self,
        *,
        messages: list[dict[str, Any] | LocalPromptMessage],
        model: str,
        stream: Literal[False] = False,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_ms: int | None = None,
        variables: dict[str, str] | None = None,
        provider_api_key: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        send_trace: bool | None = None,
        span_id: str | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> RunLocalResult: ...

    @overload
    async def arun_local(
        self,
        *,
        messages: list[dict[str, Any] | LocalPromptMessage],
        model: str,
        stream: Literal[True],
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_ms: int | None = None,
        variables: dict[str, str] | None = None,
        provider_api_key: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        send_trace: bool | None = None,
        span_id: str | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> AsyncLocalStream: ...

    async def arun_local(
        self,
        *,
        messages: list[dict[str, Any] | LocalPromptMessage],
        model: str,
        stream: bool = False,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        top_p: float | None = None,
        stop_sequences: list[str] | None = None,
        timeout_ms: int | None = None,
        variables: dict[str, str] | None = None,
        provider_api_key: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        send_trace: bool | None = None,
        span_id: str | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> RunLocalResult | AsyncLocalStream:
        """Run a local prompt asynchronously.

        See run_local for parameter documentation.
        """
        # Validate input
        self._validate_run_local_input(messages, model, span_id)

        # Convert and interpolate messages
        prompt_messages = self._convert_messages(messages, variables)

        # Generate IDs
        effective_span_id = span_id or generate_span_id()
        effective_trace_id = trace_id or generate_trace_id()
        should_send_trace = send_trace is not False

        # Calculate timeout
        timeout_seconds = (timeout_ms or DEFAULT_TIMEOUT_MS) / 1000.0

        litellm_response_format = convert_response_format(response_format)

        if stream:
            return await self._arun_local_streaming(
                messages=prompt_messages,
                model=model,
                provider=provider,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences,
                timeout=timeout_seconds,
                provider_api_key=provider_api_key,
                tags=tags,
                user_id=user_id,
                session_id=session_id,
                send_trace=should_send_trace,
                span_id=effective_span_id,
                trace_id=effective_trace_id,
                parent_span_id=parent_span_id,
                tools=tools,
                tool_choice=tool_choice,
                variables=variables,
                response_format=litellm_response_format,
            )
        else:
            return await self._arun_local_non_streaming(
                messages=prompt_messages,
                model=model,
                provider=provider,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stop_sequences=stop_sequences,
                timeout=timeout_seconds,
                provider_api_key=provider_api_key,
                tags=tags,
                user_id=user_id,
                session_id=session_id,
                send_trace=should_send_trace,
                span_id=effective_span_id,
                trace_id=effective_trace_id,
                parent_span_id=parent_span_id,
                tools=tools,
                tool_choice=tool_choice,
                variables=variables,
                response_format=litellm_response_format,
            )

    async def _arun_local_non_streaming(
        self,
        *,
        messages: list[LocalPromptMessage],
        model: str,
        provider: LLMProvider | None,
        temperature: float | None,
        max_output_tokens: int | None,
        top_p: float | None,
        stop_sequences: list[str] | None,
        timeout: float,
        provider_api_key: str | None,
        tags: list[str] | None,
        user_id: str | None,
        session_id: str | None,
        send_trace: bool,
        span_id: str,
        trace_id: str,
        parent_span_id: str | None,
        tools: list[ToolDefinition] | None,
        tool_choice: ToolChoice | None,
        variables: dict[str, str] | None,
        response_format: dict[str, Any] | None = None,
    ) -> RunLocalResult:
        """Run local prompt without streaming (async)."""
        start_time = time.time()
        error_message: str | None = None
        result_text = ""
        result_tool_calls: list[ToolCall] = []
        result_usage = TokenUsage(inputTokens=0, outputTokens=0, totalTokens=0)
        result_provider = resolve_provider(model, provider)
        finish_reason = "stop"
        latency_ms = 0

        try:
            completion = await self._llm_client.acomplete(
                model=model,
                messages=messages,
                provider=result_provider,
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=top_p,
                stop=stop_sequences,
                tools=tools,
                tool_choice=tool_choice,
                api_key=provider_api_key,
                timeout=timeout,
                response_format=response_format,
            )

            result_text = completion.text
            result_tool_calls = completion.tool_calls
            result_provider = completion.provider
            finish_reason = completion.finish_reason
            result_usage = TokenUsage(
                inputTokens=completion.input_tokens,
                outputTokens=completion.output_tokens,
                totalTokens=completion.total_tokens,
            )

        except TraciaError:
            raise
        except Exception as e:
            error_message = sanitize_error_message(str(e))
            raise TraciaError(
                code=TraciaErrorCode.PROVIDER_ERROR,
                message=f"LLM provider error: {error_message}",
            ) from e
        finally:
            latency_ms = int((time.time() - start_time) * 1000)

            if send_trace:
                payload = CreateSpanPayload(
                    spanId=span_id,
                    model=model,
                    provider=result_provider,
                    input={"messages": [m.model_dump(by_alias=True, exclude_none=True) for m in messages]},
                    variables=variables,
                    output=result_text if not error_message else None,
                    status=SPAN_STATUS_ERROR if error_message else SPAN_STATUS_SUCCESS,
                    error=error_message,
                    latencyMs=latency_ms,
                    inputTokens=result_usage.input_tokens,
                    outputTokens=result_usage.output_tokens,
                    totalTokens=result_usage.total_tokens,
                    tags=tags,
                    userId=user_id,
                    sessionId=session_id,
                    temperature=temperature,
                    maxOutputTokens=max_output_tokens,
                    topP=top_p,
                    tools=tools,
                    toolCalls=result_tool_calls if result_tool_calls else None,
                    traceId=trace_id,
                    parentSpanId=parent_span_id,
                )
                self._schedule_span_creation(payload)

        return RunLocalResult(
            text=result_text,
            spanId=span_id,
            traceId=trace_id,
            latencyMs=latency_ms,
            usage=result_usage,
            cost=None,
            provider=result_provider,
            model=model,
            toolCalls=result_tool_calls,
            finishReason=finish_reason,
            message=build_assistant_message(result_text, result_tool_calls),
        )

    async def _arun_local_streaming(
        self,
        *,
        messages: list[LocalPromptMessage],
        model: str,
        provider: LLMProvider | None,
        temperature: float | None,
        max_output_tokens: int | None,
        top_p: float | None,
        stop_sequences: list[str] | None,
        timeout: float,
        provider_api_key: str | None,
        tags: list[str] | None,
        user_id: str | None,
        session_id: str | None,
        send_trace: bool,
        span_id: str,
        trace_id: str,
        parent_span_id: str | None,
        tools: list[ToolDefinition] | None,
        tool_choice: ToolChoice | None,
        variables: dict[str, str] | None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncLocalStream:
        """Run local prompt with streaming (async)."""
        start_time = time.time()
        abort_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[StreamResult] = loop.create_future()

        chunks_iter, result_holder, resolved_provider = await self._llm_client.astream(
            model=model,
            messages=messages,
            provider=provider,
            temperature=temperature,
            max_tokens=max_output_tokens,
            top_p=top_p,
            stop=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            api_key=provider_api_key,
            timeout=timeout,
            response_format=response_format,
        )

        async def wrapped_chunks():
            error_message: str | None = None
            aborted = False

            try:
                async for chunk in chunks_iter:
                    if abort_event.is_set():
                        aborted = True
                        break
                    yield chunk
            except Exception as e:
                error_message = sanitize_error_message(str(e))
                raise
            finally:
                latency_ms = int((time.time() - start_time) * 1000)

                # Get completion result
                completion = result_holder[0] if result_holder else None

                result_text = completion.text if completion else ""
                result_tool_calls = completion.tool_calls if completion else []
                result_usage = TokenUsage(
                    inputTokens=completion.input_tokens if completion else 0,
                    outputTokens=completion.output_tokens if completion else 0,
                    totalTokens=completion.total_tokens if completion else 0,
                )
                finish_reason = completion.finish_reason if completion else "stop"

                if send_trace:
                    payload = CreateSpanPayload(
                        spanId=span_id,
                        model=model,
                        provider=resolved_provider,
                        input={"messages": [m.model_dump(by_alias=True, exclude_none=True) for m in messages]},
                        variables=variables,
                        output=result_text if not error_message else None,
                        status=SPAN_STATUS_ERROR if error_message else SPAN_STATUS_SUCCESS,
                        error=error_message,
                        latencyMs=latency_ms,
                        inputTokens=result_usage.input_tokens,
                        outputTokens=result_usage.output_tokens,
                        totalTokens=result_usage.total_tokens,
                        tags=tags,
                        userId=user_id,
                        sessionId=session_id,
                        temperature=temperature,
                        maxOutputTokens=max_output_tokens,
                        topP=top_p,
                        tools=tools,
                        toolCalls=result_tool_calls if result_tool_calls else None,
                        traceId=trace_id,
                        parentSpanId=parent_span_id,
                    )
                    self._schedule_span_creation(payload)

                # Set the result
                stream_result = StreamResult(
                    text=result_text,
                    spanId=span_id,
                    traceId=trace_id,
                    latencyMs=latency_ms,
                    usage=result_usage,
                    cost=None,
                    provider=resolved_provider,
                    model=model,
                    toolCalls=result_tool_calls,
                    finishReason=finish_reason,
                    message=build_assistant_message(result_text, result_tool_calls),
                    aborted=aborted,
                )
                if not result_future.done():
                    result_future.set_result(stream_result)

        return AsyncLocalStream(
            span_id=span_id,
            trace_id=trace_id,
            chunks=wrapped_chunks(),
            result_holder=result_holder,
            result_future=result_future,
            abort_event=abort_event,
        )

    def create_session(
        self,
        *,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> TraciaSession:
        """Create a new session for linked runs.

        Args:
            trace_id: Optional initial trace ID.
            parent_span_id: Optional initial parent span ID.

        Returns:
            A new TraciaSession instance.
        """
        return TraciaSession(self, trace_id, parent_span_id)

    def flush(self) -> None:
        """Wait for all pending span creations to complete."""
        with self._pending_spans_lock:
            futures = list(self._pending_spans.values())

        for future in futures:
            try:
                future.result(timeout=30.0)
            except Exception:
                pass

    async def aflush(self) -> None:
        """Wait for all pending span creations to complete (async)."""
        # In async context, we still use the sync executor
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.flush)

    def close(self) -> None:
        """Close the client and release resources."""
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._http_client.close()
        if self._async_http_client:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_http_client.aclose())
            except RuntimeError:
                # No running loop - run synchronously
                try:
                    asyncio.run(self._async_http_client.aclose())
                except Exception:
                    pass
        self._executor.shutdown(wait=False)

    def __del__(self) -> None:
        try:
            if not self._closed:
                warnings.warn(
                    "Unclosed Tracia client. Use 'client.close()' or 'with Tracia(...) as client:'.",
                    ResourceWarning,
                    stacklevel=1,
                )
        except Exception:
            pass

    async def aclose(self) -> None:
        """Close the client and release resources (async)."""
        if self._closed:
            return
        self._closed = True
        await self.aflush()
        self._http_client.close()
        if self._async_http_client:
            await self._async_http_client.aclose()
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "Tracia":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> "Tracia":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
