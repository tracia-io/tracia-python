"""Prompts API resource."""

from __future__ import annotations

from urllib.parse import quote
from typing import TYPE_CHECKING, Any

from .._types import (
    CreatePromptOptions,
    LocalPromptMessage,
    Prompt,
    PromptListItem,
    RunOptions,
    RunResult,
    UpdatePromptOptions,
)

if TYPE_CHECKING:
    from .._http import AsyncHttpClient, HttpClient


class Prompts:
    """Prompts API resource for managing prompt templates."""

    def __init__(
        self,
        client: "HttpClient",
        async_client: "AsyncHttpClient | None" = None,
    ) -> None:
        """Initialize the Prompts resource.

        Args:
            client: The synchronous HTTP client.
            async_client: Optional async HTTP client.
        """
        self._client = client
        self._async_client = async_client

    def list(self) -> list[PromptListItem]:
        """List all prompts.

        Returns:
            The list of prompts.

        Raises:
            TraciaError: If the request fails.
        """
        data = self._client.get("/api/v1/prompts")
        return [PromptListItem.model_validate(item) for item in data["prompts"]]

    async def alist(self) -> list[PromptListItem]:
        """List all prompts asynchronously.

        Returns:
            The list of prompts.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        data = await self._async_client.get("/api/v1/prompts")
        return [PromptListItem.model_validate(item) for item in data["prompts"]]

    def get(self, slug: str) -> Prompt:
        """Get a prompt by slug.

        Args:
            slug: The prompt slug.

        Returns:
            The prompt.

        Raises:
            TraciaError: If the request fails.
        """
        data = self._client.get(f"/api/v1/prompts/{quote(slug, safe='')}")
        return Prompt.model_validate(data)

    async def aget(self, slug: str) -> Prompt:
        """Get a prompt by slug asynchronously.

        Args:
            slug: The prompt slug.

        Returns:
            The prompt.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        data = await self._async_client.get(f"/api/v1/prompts/{quote(slug, safe='')}")
        return Prompt.model_validate(data)

    def create(self, options: CreatePromptOptions) -> Prompt:
        """Create a new prompt.

        Args:
            options: The prompt creation options.

        Returns:
            The created prompt.

        Raises:
            TraciaError: If the request fails.
        """
        data = self._client.post(
            "/api/v1/prompts",
            options.model_dump(by_alias=True, exclude_none=True),
        )
        return Prompt.model_validate(data)

    async def acreate(self, options: CreatePromptOptions) -> Prompt:
        """Create a new prompt asynchronously.

        Args:
            options: The prompt creation options.

        Returns:
            The created prompt.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        data = await self._async_client.post(
            "/api/v1/prompts",
            options.model_dump(by_alias=True, exclude_none=True),
        )
        return Prompt.model_validate(data)

    def update(self, slug: str, options: UpdatePromptOptions) -> Prompt:
        """Update an existing prompt.

        Args:
            slug: The prompt slug.
            options: The update options.

        Returns:
            The updated prompt.

        Raises:
            TraciaError: If the request fails.
        """
        data = self._client.put(
            f"/api/v1/prompts/{quote(slug, safe='')}",
            options.model_dump(by_alias=True, exclude_none=True),
        )
        return Prompt.model_validate(data)

    async def aupdate(self, slug: str, options: UpdatePromptOptions) -> Prompt:
        """Update an existing prompt asynchronously.

        Args:
            slug: The prompt slug.
            options: The update options.

        Returns:
            The updated prompt.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        data = await self._async_client.put(
            f"/api/v1/prompts/{quote(slug, safe='')}",
            options.model_dump(by_alias=True, exclude_none=True),
        )
        return Prompt.model_validate(data)

    def delete(self, slug: str) -> None:
        """Delete a prompt.

        Args:
            slug: The prompt slug.

        Raises:
            TraciaError: If the request fails.
        """
        self._client.delete(f"/api/v1/prompts/{quote(slug, safe='')}")

    async def adelete(self, slug: str) -> None:
        """Delete a prompt asynchronously.

        Args:
            slug: The prompt slug.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        await self._async_client.delete(f"/api/v1/prompts/{quote(slug, safe='')}")

    def run(
        self,
        slug: str,
        variables: dict[str, str] | None = None,
        options: RunOptions | None = None,
    ) -> RunResult:
        """Run a prompt template.

        Args:
            slug: The prompt slug.
            variables: Optional variables to interpolate.
            options: Optional run options.

        Returns:
            The run result.

        Raises:
            TraciaError: If the request fails.
        """
        body = self._build_run_body(variables, options)
        data = self._client.post(f"/api/v1/prompts/{quote(slug, safe='')}/run", body)
        return RunResult.model_validate(data)

    async def arun(
        self,
        slug: str,
        variables: dict[str, str] | None = None,
        options: RunOptions | None = None,
    ) -> RunResult:
        """Run a prompt template asynchronously.

        Args:
            slug: The prompt slug.
            variables: Optional variables to interpolate.
            options: Optional run options.

        Returns:
            The run result.

        Raises:
            TraciaError: If the request fails.
        """
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")

        body = self._build_run_body(variables, options)
        data = await self._async_client.post(f"/api/v1/prompts/{quote(slug, safe='')}/run", body)
        return RunResult.model_validate(data)

    def _build_run_body(
        self,
        variables: dict[str, str] | None,
        options: RunOptions | None,
    ) -> dict[str, Any]:
        """Build the request body for the run endpoint."""
        body: dict[str, Any] = {}

        if variables:
            body["variables"] = variables

        if options:
            if options.model is not None:
                body["model"] = options.model
            if options.version is not None:
                body["version"] = options.version
            if options.tags is not None:
                body["tags"] = options.tags
            if options.user_id is not None:
                body["userId"] = options.user_id
            if options.session_id is not None:
                body["sessionId"] = options.session_id
            if options.trace_id is not None:
                body["traceId"] = options.trace_id
            if options.parent_span_id is not None:
                body["parentSpanId"] = options.parent_span_id
            if options.messages is not None:
                body["messages"] = [
                    msg.model_dump(by_alias=True, exclude_none=True)
                    for msg in options.messages
                ]

        return body
