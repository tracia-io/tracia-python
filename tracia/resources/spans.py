"""Spans API resource."""

from __future__ import annotations

from urllib.parse import quote
from typing import TYPE_CHECKING, Any

from .._types import (
    CreateSpanPayload,
    CreateSpanResult,
    EvaluateOptions,
    EvaluateResult,
    ListSpansOptions,
    ListSpansResult,
    Span,
)

if TYPE_CHECKING:
    from .._http import AsyncHttpClient, HttpClient


class Spans:
    """Spans API resource for managing trace spans."""

    def __init__(
        self,
        client: "HttpClient",
        async_client: "AsyncHttpClient | None" = None,
    ) -> None:
        """Initialize the Spans resource.

        Args:
            client: The synchronous HTTP client.
            async_client: Optional async HTTP client.
        """
        self._client = client
        self._async_client = async_client

    def create(self, payload: CreateSpanPayload) -> CreateSpanResult:
        """Create a new span.

        Args:
            payload: The span creation payload.

        Returns:
            The created span result.

        Raises:
            TraciaError: If the request fails.
        """
        data = self._client.post(
            "/api/v1/spans",
            payload.model_dump(by_alias=True, exclude_none=True),
        )
        return CreateSpanResult.model_validate(data)

    async def acreate(self, payload: CreateSpanPayload) -> CreateSpanResult:
        """Create a new span asynchronously.

        Args:
            payload: The span creation payload.

        Returns:
            The created span result.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        data = await self._async_client.post(
            "/api/v1/spans",
            payload.model_dump(by_alias=True, exclude_none=True),
        )
        return CreateSpanResult.model_validate(data)

    def get(self, span_id: str) -> Span:
        """Get a span by ID.

        Args:
            span_id: The span ID.

        Returns:
            The span.

        Raises:
            TraciaError: If the request fails.
        """
        data = self._client.get(f"/api/v1/spans/{quote(span_id, safe='')}")
        return Span.model_validate(data)

    async def aget(self, span_id: str) -> Span:
        """Get a span by ID asynchronously.

        Args:
            span_id: The span ID.

        Returns:
            The span.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        data = await self._async_client.get(f"/api/v1/spans/{quote(span_id, safe='')}")
        return Span.model_validate(data)

    def list(self, options: ListSpansOptions | None = None) -> ListSpansResult:
        """List spans with optional filtering.

        Args:
            options: Optional filtering and pagination options.

        Returns:
            The list of spans.

        Raises:
            TraciaError: If the request fails.
        """
        params = self._build_list_params(options)
        data = self._client.get("/api/v1/spans", params=params)
        return ListSpansResult.model_validate(data)

    async def alist(self, options: ListSpansOptions | None = None) -> ListSpansResult:
        """List spans asynchronously with optional filtering.

        Args:
            options: Optional filtering and pagination options.

        Returns:
            The list of spans.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        params = self._build_list_params(options)
        data = await self._async_client.get("/api/v1/spans", params=params)
        return ListSpansResult.model_validate(data)

    def _build_list_params(
        self, options: ListSpansOptions | None
    ) -> dict[str, Any] | None:
        """Build query parameters for list endpoint."""
        if options is None:
            return None

        params: dict[str, Any] = {}

        if options.prompt_slug is not None:
            params["promptSlug"] = options.prompt_slug
        if options.status is not None:
            params["status"] = options.status
        if options.start_date is not None:
            params["startDate"] = options.start_date.isoformat()
        if options.end_date is not None:
            params["endDate"] = options.end_date.isoformat()
        if options.user_id is not None:
            params["userId"] = options.user_id
        if options.session_id is not None:
            params["sessionId"] = options.session_id
        if options.tags is not None and len(options.tags) > 0:
            params["tags"] = ",".join(options.tags)
        if options.limit is not None:
            params["limit"] = options.limit
        if options.cursor is not None:
            params["cursor"] = options.cursor

        return params if params else None

    def evaluate(self, span_id: str, options: EvaluateOptions) -> EvaluateResult:
        """Evaluate a span.

        Args:
            span_id: The span ID to evaluate.
            options: The evaluation options.

        Returns:
            The evaluation result.

        Raises:
            TraciaError: If the request fails.
        """
        body = {
            "evaluatorKey": options.evaluator,
            "value": options.value,
        }
        if options.note is not None:
            body["note"] = options.note

        data = self._client.post(f"/api/v1/spans/{quote(span_id, safe='')}/evaluations", body)
        return EvaluateResult.model_validate(data)

    async def aevaluate(
        self, span_id: str, options: EvaluateOptions
    ) -> EvaluateResult:
        """Evaluate a span asynchronously.

        Args:
            span_id: The span ID to evaluate.
            options: The evaluation options.

        Returns:
            The evaluation result.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        body = {
            "evaluatorKey": options.evaluator,
            "value": options.value,
        }
        if options.note is not None:
            body["note"] = options.note

        data = await self._async_client.post(
            f"/api/v1/spans/{quote(span_id, safe='')}/evaluations", body
        )
        return EvaluateResult.model_validate(data)
