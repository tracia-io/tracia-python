"""Streaming classes for the Tracia SDK."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Event
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from ._types import StreamResult

if TYPE_CHECKING:
    from ._types import RunLocalResult


class LocalStream:
    """Synchronous stream wrapper for run_local with streaming."""

    def __init__(
        self,
        span_id: str,
        trace_id: str,
        chunks: Iterator[str],
        result_holder: list[Any],
        result_future: "Future[StreamResult]",
        abort_event: Event,
    ) -> None:
        """Initialize the stream.

        Args:
            span_id: The span ID.
            trace_id: The trace ID.
            chunks: Iterator yielding text chunks.
            result_holder: List that will hold the completion result.
            result_future: Future that resolves to the stream result.
            abort_event: Event to signal abort.
        """
        self._span_id = span_id
        self._trace_id = trace_id
        self._chunks = chunks
        self._result_holder = result_holder
        self._result_future = result_future
        self._abort_event = abort_event
        self._consumed = False

    @property
    def span_id(self) -> str:
        """Get the span ID."""
        return self._span_id

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace_id

    @property
    def result(self) -> "Future[StreamResult]":
        """Get the future that resolves to the stream result."""
        return self._result_future

    def __iter__(self) -> Iterator[str]:
        """Iterate over text chunks."""
        if self._consumed:
            raise RuntimeError("Stream already consumed")
        self._consumed = True

        for chunk in self._chunks:
            if self._abort_event.is_set():
                break
            yield chunk

    def abort(self) -> None:
        """Abort the stream."""
        self._abort_event.set()


class AsyncLocalStream:
    """Asynchronous stream wrapper for run_local with streaming."""

    def __init__(
        self,
        span_id: str,
        trace_id: str,
        chunks: AsyncIterator[str],
        result_holder: list[Any],
        result_future: "asyncio.Future[StreamResult]",
        abort_event: asyncio.Event,
    ) -> None:
        """Initialize the async stream.

        Args:
            span_id: The span ID.
            trace_id: The trace ID.
            chunks: Async iterator yielding text chunks.
            result_holder: List that will hold the completion result.
            result_future: Future that resolves to the stream result.
            abort_event: Event to signal abort.
        """
        self._span_id = span_id
        self._trace_id = trace_id
        self._chunks = chunks
        self._result_holder = result_holder
        self._result_future = result_future
        self._abort_event = abort_event
        self._consumed = False

    @property
    def span_id(self) -> str:
        """Get the span ID."""
        return self._span_id

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace_id

    @property
    def result(self) -> "asyncio.Future[StreamResult]":
        """Get the future that resolves to the stream result."""
        return self._result_future

    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over text chunks asynchronously."""
        if self._consumed:
            raise RuntimeError("Stream already consumed")
        self._consumed = True

        async for chunk in self._chunks:
            if self._abort_event.is_set():
                break
            yield chunk

    def abort(self) -> None:
        """Abort the stream."""
        self._abort_event.set()
