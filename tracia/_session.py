"""Session management for the Tracia SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from ._streaming import AsyncLocalStream, LocalStream
from ._types import RunEmbeddingResult, RunLocalInput, RunLocalResult, StreamResult
from ._utils import generate_trace_id

if TYPE_CHECKING:
    from ._client import Tracia


class TraciaSession:
    """Session for managing related traces.

    A session automatically links related runs by managing trace IDs and
    parent span IDs. This is useful for multi-turn conversations or
    related operations that should be grouped together.
    """

    def __init__(
        self,
        tracia: "Tracia",
        initial_trace_id: str | None = None,
        initial_parent_span_id: str | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            tracia: The Tracia client instance.
            initial_trace_id: Optional initial trace ID.
            initial_parent_span_id: Optional initial parent span ID.
        """
        self._tracia = tracia
        self._trace_id = initial_trace_id
        self._last_span_id = initial_parent_span_id

    @property
    def trace_id(self) -> str | None:
        """Get the current trace ID."""
        return self._trace_id

    @property
    def last_span_id(self) -> str | None:
        """Get the last span ID."""
        return self._last_span_id

    def reset(self) -> None:
        """Reset the session, clearing trace and span IDs."""
        self._trace_id = None
        self._last_span_id = None

    def _update_from_result(
        self, trace_id: str, span_id: str
    ) -> None:
        """Update session state from a result.

        Args:
            trace_id: The trace ID from the result.
            span_id: The span ID from the result.
        """
        if self._trace_id is None:
            self._trace_id = trace_id
        self._last_span_id = span_id

    @overload
    def run_local(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = ...,
        **kwargs: Any,
    ) -> RunLocalResult: ...

    @overload
    def run_local(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = True,
        **kwargs: Any,
    ) -> LocalStream: ...

    def run_local(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> RunLocalResult | LocalStream:
        """Run a local prompt with session context.

        Automatically includes trace_id and parent_span_id from the session.

        Args:
            messages: The messages to send.
            model: The model name.
            stream: Whether to stream the response.
            **kwargs: Additional arguments for run_local.

        Returns:
            The result or stream.
        """
        # Ensure trace_id is set
        if self._trace_id is None:
            self._trace_id = generate_trace_id()

        # Build input with session context
        kwargs["trace_id"] = self._trace_id
        if self._last_span_id is not None:
            kwargs["parent_span_id"] = self._last_span_id

        result = self._tracia.run_local(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs,
        )

        if stream:
            # For streaming, wrap to capture result
            return self._wrap_stream(result)

        # Update session state from result
        self._update_from_result(result.trace_id, result.span_id)
        return result

    def _wrap_stream(self, stream: LocalStream) -> LocalStream:
        """Wrap a stream to capture the result for session state.

        Args:
            stream: The original stream.

        Returns:
            The wrapped stream.
        """
        # Update state when stream is consumed
        original_future = stream._result_future

        def on_result_ready() -> None:
            if original_future.done():
                try:
                    result = original_future.result()
                    self._update_from_result(result.trace_id, result.span_id)
                except Exception:
                    pass

        # Register callback
        original_future.add_done_callback(lambda _: on_result_ready())
        return stream

    @overload
    async def arun_local(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = ...,
        **kwargs: Any,
    ) -> RunLocalResult: ...

    @overload
    async def arun_local(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = True,
        **kwargs: Any,
    ) -> AsyncLocalStream: ...

    async def arun_local(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> RunLocalResult | AsyncLocalStream:
        """Run a local prompt with session context asynchronously.

        Automatically includes trace_id and parent_span_id from the session.

        Args:
            messages: The messages to send.
            model: The model name.
            stream: Whether to stream the response.
            **kwargs: Additional arguments for run_local.

        Returns:
            The result or stream.
        """
        # Ensure trace_id is set
        if self._trace_id is None:
            self._trace_id = generate_trace_id()

        # Build input with session context
        kwargs["trace_id"] = self._trace_id
        if self._last_span_id is not None:
            kwargs["parent_span_id"] = self._last_span_id

        result = await self._tracia.arun_local(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs,
        )

        if stream:
            # For streaming, wrap to capture result
            return self._wrap_async_stream(result)

        # Update session state from result
        self._update_from_result(result.trace_id, result.span_id)
        return result

    def run_embedding(self, **kwargs: Any) -> RunEmbeddingResult:
        """Generate embeddings with session context.

        Automatically includes trace_id and parent_span_id from the session.

        Args:
            **kwargs: Arguments for run_embedding.

        Returns:
            The embedding result.
        """
        if self._trace_id is None:
            self._trace_id = generate_trace_id()

        kwargs["trace_id"] = self._trace_id
        if self._last_span_id is not None:
            kwargs["parent_span_id"] = self._last_span_id

        result = self._tracia.run_embedding(**kwargs)
        self._update_from_result(result.trace_id, result.span_id)
        return result

    async def arun_embedding(self, **kwargs: Any) -> RunEmbeddingResult:
        """Generate embeddings with session context asynchronously.

        Automatically includes trace_id and parent_span_id from the session.

        Args:
            **kwargs: Arguments for arun_embedding.

        Returns:
            The embedding result.
        """
        if self._trace_id is None:
            self._trace_id = generate_trace_id()

        kwargs["trace_id"] = self._trace_id
        if self._last_span_id is not None:
            kwargs["parent_span_id"] = self._last_span_id

        result = await self._tracia.arun_embedding(**kwargs)
        self._update_from_result(result.trace_id, result.span_id)
        return result

    def _wrap_async_stream(self, stream: AsyncLocalStream) -> AsyncLocalStream:
        """Wrap an async stream to capture the result for session state.

        Args:
            stream: The original stream.

        Returns:
            The wrapped stream.
        """
        # Update state when stream is consumed
        original_future = stream._result_future

        def on_result_ready(future: Any) -> None:
            if future.done():
                try:
                    result = future.result()
                    self._update_from_result(result.trace_id, result.span_id)
                except Exception:
                    pass

        # Register callback
        original_future.add_done_callback(on_result_ready)
        return stream
