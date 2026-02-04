"""Tests for the main Tracia client."""

import pytest

from tracia import Tracia, TraciaError, TraciaErrorCode


class TestClientInit:
    """Tests for client initialization."""

    def test_create_client(self) -> None:
        """Test creating a client."""
        client = Tracia(api_key="test_key")
        assert client is not None
        client.close()

    def test_client_has_resources(self) -> None:
        """Test that client has prompts and spans resources."""
        client = Tracia(api_key="test_key")
        assert hasattr(client, "prompts")
        assert hasattr(client, "spans")
        client.close()

    def test_client_context_manager(self) -> None:
        """Test using client as context manager."""
        with Tracia(api_key="test_key") as client:
            assert client is not None


class TestClientSession:
    """Tests for session management."""

    def test_create_session(self) -> None:
        """Test creating a session."""
        client = Tracia(api_key="test_key")
        session = client.create_session()
        assert session is not None
        assert session.trace_id is None  # No trace yet
        assert session.last_span_id is None
        client.close()

    def test_create_session_with_trace_id(self) -> None:
        """Test creating a session with initial trace ID."""
        client = Tracia(api_key="test_key")
        session = client.create_session(trace_id="tr_1234567890abcdef")
        assert session.trace_id == "tr_1234567890abcdef"
        client.close()

    def test_session_reset(self) -> None:
        """Test resetting a session."""
        client = Tracia(api_key="test_key")
        session = client.create_session(trace_id="tr_1234567890abcdef")
        session.reset()
        assert session.trace_id is None
        assert session.last_span_id is None
        client.close()


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_model_rejected(self) -> None:
        """Test that empty model is rejected."""
        client = Tracia(api_key="test_key")
        with pytest.raises(TraciaError) as exc_info:
            client.run_local(messages=[{"role": "user", "content": "Hi"}], model="")
        assert exc_info.value.code == TraciaErrorCode.INVALID_REQUEST
        client.close()

    def test_empty_messages_rejected(self) -> None:
        """Test that empty messages is rejected."""
        client = Tracia(api_key="test_key")
        with pytest.raises(TraciaError) as exc_info:
            client.run_local(messages=[], model="gpt-4o")
        assert exc_info.value.code == TraciaErrorCode.INVALID_REQUEST
        client.close()

    def test_invalid_span_id_rejected(self) -> None:
        """Test that invalid span ID is rejected."""
        client = Tracia(api_key="test_key")
        with pytest.raises(TraciaError) as exc_info:
            client.run_local(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o",
                span_id="invalid",
            )
        assert exc_info.value.code == TraciaErrorCode.INVALID_REQUEST
        client.close()

    def test_tool_message_without_tool_call_id_rejected(self) -> None:
        """Test that tool message without tool_call_id is rejected."""
        client = Tracia(api_key="test_key")
        with pytest.raises(TraciaError) as exc_info:
            client.run_local(
                messages=[{"role": "tool", "content": "Result"}],
                model="gpt-4o",
            )
        assert exc_info.value.code == TraciaErrorCode.INVALID_REQUEST
        client.close()
