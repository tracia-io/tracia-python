"""HTTP client for the Tracia API."""

from __future__ import annotations

from typing import Any, TypeVar

import httpx

from ._constants import BASE_URL, DEFAULT_TIMEOUT_MS, SDK_VERSION
from ._errors import TraciaError, TraciaErrorCode, map_api_error_code

T = TypeVar("T")


class HttpClient:
    """Synchronous HTTP client for the Tracia API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            api_key: The Tracia API key.
            base_url: The base URL for the API.
            timeout_ms: Request timeout in milliseconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_ms / 1000.0  # Convert to seconds

        self._client = httpx.Client(
            base_url=self._base_url,
            headers=self._get_headers(),
            timeout=httpx.Timeout(self._timeout),
        )

    def _get_headers(self) -> dict[str, str]:
        """Get the default headers for requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"tracia-sdk-python/{SDK_VERSION}",
        }

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle the API response and raise errors if needed."""
        if response.status_code >= 400:
            try:
                data = response.json()
                error_data = data.get("error", {})
                code = map_api_error_code(error_data.get("code", "UNKNOWN"))
                message = error_data.get("message", "Unknown error")
            except Exception:
                code = TraciaErrorCode.UNKNOWN
                message = response.text or f"HTTP {response.status_code}"

            raise TraciaError(
                code=code,
                message=message,
                status_code=response.status_code,
            )

        if response.status_code == 204:
            return None

        return response.json()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request.

        Args:
            path: The API path.
            params: Optional query parameters.

        Returns:
            The JSON response.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = self._client.get(path, params=params)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    def post(self, path: str, body: Any = None) -> Any:
        """Make a POST request.

        Args:
            path: The API path.
            body: The request body.

        Returns:
            The JSON response.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = self._client.post(path, json=body)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    def put(self, path: str, body: Any = None) -> Any:
        """Make a PUT request.

        Args:
            path: The API path.
            body: The request body.

        Returns:
            The JSON response.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = self._client.put(path, json=body)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    def delete(self, path: str) -> Any:
        """Make a DELETE request.

        Args:
            path: The API path.

        Returns:
            The JSON response (or None for 204).

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = self._client.delete(path)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncHttpClient:
    """Asynchronous HTTP client for the Tracia API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = BASE_URL,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        """Initialize the async HTTP client.

        Args:
            api_key: The Tracia API key.
            base_url: The base URL for the API.
            timeout_ms: Request timeout in milliseconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_ms / 1000.0  # Convert to seconds

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._get_headers(),
            timeout=httpx.Timeout(self._timeout),
        )

    def _get_headers(self) -> dict[str, str]:
        """Get the default headers for requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": f"tracia-sdk-python/{SDK_VERSION}",
        }

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle the API response and raise errors if needed."""
        if response.status_code >= 400:
            try:
                data = response.json()
                error_data = data.get("error", {})
                code = map_api_error_code(error_data.get("code", "UNKNOWN"))
                message = error_data.get("message", "Unknown error")
            except Exception:
                code = TraciaErrorCode.UNKNOWN
                message = response.text or f"HTTP {response.status_code}"

            raise TraciaError(
                code=code,
                message=message,
                status_code=response.status_code,
            )

        if response.status_code == 204:
            return None

        return response.json()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request.

        Args:
            path: The API path.
            params: Optional query parameters.

        Returns:
            The JSON response.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = await self._client.get(path, params=params)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    async def post(self, path: str, body: Any = None) -> Any:
        """Make a POST request.

        Args:
            path: The API path.
            body: The request body.

        Returns:
            The JSON response.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = await self._client.post(path, json=body)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    async def put(self, path: str, body: Any = None) -> Any:
        """Make a PUT request.

        Args:
            path: The API path.
            body: The request body.

        Returns:
            The JSON response.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = await self._client.put(path, json=body)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    async def delete(self, path: str) -> Any:
        """Make a DELETE request.

        Args:
            path: The API path.

        Returns:
            The JSON response (or None for 204).

        Raises:
            TraciaError: If the request fails.
        """
        try:
            response = await self._client.delete(path)
            return self._handle_response(response)
        except httpx.TimeoutException as e:
            raise TraciaError(
                code=TraciaErrorCode.TIMEOUT,
                message=f"Request timed out after {int(self._timeout * 1000)}ms",
            ) from e
        except httpx.RequestError as e:
            raise TraciaError(
                code=TraciaErrorCode.NETWORK_ERROR,
                message=f"Network error: {e}",
            ) from e

    async def aclose(self) -> None:
        """Close the async HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncHttpClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
