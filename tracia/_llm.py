"""LiteLLM wrapper for unified LLM access."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from ._constants import ENV_VAR_MAP
from ._errors import TraciaError, TraciaErrorCode, sanitize_error_message
from ._types import (
    ContentPart,
    FinishReason,
    LLMProvider,
    LocalPromptMessage,
    ResponseFormatJsonSchema,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
)

if TYPE_CHECKING:
    from litellm import ModelResponse


@dataclass
class CompletionResult:
    """Result from an LLM completion."""

    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    provider: LLMProvider


# Model to provider mapping for common models
_MODEL_PROVIDER_MAP: dict[str, LLMProvider] = {
    # OpenAI
    "gpt-3.5-turbo": LLMProvider.OPENAI,
    "gpt-4": LLMProvider.OPENAI,
    "gpt-4-turbo": LLMProvider.OPENAI,
    "gpt-4o": LLMProvider.OPENAI,
    "gpt-4o-mini": LLMProvider.OPENAI,
    "gpt-4.1": LLMProvider.OPENAI,
    "gpt-4.1-mini": LLMProvider.OPENAI,
    "gpt-4.1-nano": LLMProvider.OPENAI,
    "gpt-4.5-preview": LLMProvider.OPENAI,
    "gpt-5": LLMProvider.OPENAI,
    "o1": LLMProvider.OPENAI,
    "o1-mini": LLMProvider.OPENAI,
    "o1-preview": LLMProvider.OPENAI,
    "o3": LLMProvider.OPENAI,
    "o3-mini": LLMProvider.OPENAI,
    "o4-mini": LLMProvider.OPENAI,
    # Anthropic
    "claude-3-haiku-20240307": LLMProvider.ANTHROPIC,
    "claude-3-sonnet-20240229": LLMProvider.ANTHROPIC,
    "claude-3-opus-20240229": LLMProvider.ANTHROPIC,
    "claude-3-5-haiku-20241022": LLMProvider.ANTHROPIC,
    "claude-3-5-sonnet-20241022": LLMProvider.ANTHROPIC,
    "claude-sonnet-4-20250514": LLMProvider.ANTHROPIC,
    "claude-opus-4-20250514": LLMProvider.ANTHROPIC,
    # Google
    "gemini-2.0-flash": LLMProvider.GOOGLE,
    "gemini-2.0-flash-lite": LLMProvider.GOOGLE,
    "gemini-2.5-pro": LLMProvider.GOOGLE,
    "gemini-2.5-flash": LLMProvider.GOOGLE,
    # Amazon Bedrock - Anthropic Claude (via Bedrock)
    "anthropic.claude-haiku-4-5-20251001-v1:0": LLMProvider.AMAZON_BEDROCK,
    "anthropic.claude-sonnet-4-20250514-v1:0": LLMProvider.AMAZON_BEDROCK,
    "anthropic.claude-sonnet-4-5-20250929-v1:0": LLMProvider.AMAZON_BEDROCK,
    "anthropic.claude-opus-4-5-20251101-v1:0": LLMProvider.AMAZON_BEDROCK,
    "anthropic.claude-opus-4-6-v1": LLMProvider.AMAZON_BEDROCK,
    # Amazon Bedrock - Amazon Nova
    "amazon.nova-micro-v1:0": LLMProvider.AMAZON_BEDROCK,
    "amazon.nova-lite-v1:0": LLMProvider.AMAZON_BEDROCK,
    "amazon.nova-pro-v1:0": LLMProvider.AMAZON_BEDROCK,
    # Amazon Bedrock - Mistral
    "mistral.pixtral-large-2502-v1:0": LLMProvider.AMAZON_BEDROCK,
}


_BEDROCK_VENDOR_PREFIXES = ("anthropic.", "amazon.", "meta.", "mistral.", "cohere.", "deepseek.")
_KNOWN_REGION_PREFIXES = ("us", "eu", "ap", "sa", "ca", "me", "af")


def _is_bedrock_model(model: str) -> bool:
    if any(model.startswith(prefix) for prefix in _BEDROCK_VENDOR_PREFIXES):
        return True
    dot = model.find(".")
    if dot > 0 and model[:dot] in _KNOWN_REGION_PREFIXES:
        after_region = model[dot + 1 :]
        return any(after_region.startswith(prefix) for prefix in _BEDROCK_VENDOR_PREFIXES)
    return False


def apply_bedrock_region_prefix(model: str, region: str) -> str:
    """Prepend the region shorthand to a Bedrock model ID.

    Newer Bedrock models require a region prefix (e.g. ``eu.anthropic.claude-...``
    for ``eu-central-1``). If the model already has a region prefix it is replaced.
    """
    prefix = region.split("-")[0]
    dot = model.find(".")
    if dot == -1:
        return f"{prefix}.{model}"
    before_dot = model[:dot]
    if before_dot in _KNOWN_REGION_PREFIXES:
        return f"{prefix}.{model[dot + 1:]}"
    return f"{prefix}.{model}"


def resolve_provider(model: str, explicit_provider: LLMProvider | None) -> LLMProvider:
    """Resolve the provider for a model.

    Args:
        model: The model name.
        explicit_provider: Explicitly specified provider.

    Returns:
        The resolved provider.

    Raises:
        TraciaError: If the provider cannot be determined.
    """
    if explicit_provider is not None:
        return explicit_provider

    # Check the model registry
    if model in _MODEL_PROVIDER_MAP:
        return _MODEL_PROVIDER_MAP[model]

    # Try prefix-based detection
    if _is_bedrock_model(model):
        return LLMProvider.AMAZON_BEDROCK
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return LLMProvider.OPENAI
    if model.startswith("claude-"):
        return LLMProvider.ANTHROPIC
    if model.startswith("gemini-"):
        return LLMProvider.GOOGLE

    raise TraciaError(
        code=TraciaErrorCode.UNSUPPORTED_MODEL,
        message=f"Cannot determine provider for model '{model}'. Please specify the provider explicitly.",
    )


def get_litellm_model(model: str, provider: LLMProvider) -> str:
    """Get the litellm-compatible model name.

    LiteLLM requires a ``gemini/`` prefix to route Google AI Studio models
    correctly. Without it, litellm defaults to the Vertex AI path which
    requires Application Default Credentials instead of an API key.

    Args:
        model: The user-facing model name (e.g. ``gemini-2.0-flash``).
        provider: The resolved provider.

    Returns:
        The model string suitable for ``litellm.completion()``.
    """
    if provider == LLMProvider.AMAZON_BEDROCK:
        region = os.environ.get("AWS_REGION", "eu-central-1")
        return f"bedrock/{apply_bedrock_region_prefix(model, region)}"
    if provider == LLMProvider.GOOGLE and not model.startswith("gemini/"):
        return f"gemini/{model}"
    return model


def get_provider_api_key(
    provider: LLMProvider,
    provider_api_key: str | None = None,
) -> str | None:
    """Get the API key for a provider.

    Args:
        provider: The LLM provider.
        provider_api_key: Explicitly provided API key.

    Returns:
        The API key, or None for Bedrock when relying on the AWS credential chain.

    Raises:
        TraciaError: If no API key is found (except for Bedrock).
    """
    if provider_api_key:
        return provider_api_key

    env_var = ENV_VAR_MAP.get(provider.value)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key

    # Bedrock can use the AWS SDK credential chain (IAM roles, env vars, etc.)
    if provider == LLMProvider.AMAZON_BEDROCK:
        return None

    raise TraciaError(
        code=TraciaErrorCode.MISSING_PROVIDER_API_KEY,
        message=f"No API key found for provider '{provider.value}'. "
        f"Set the {ENV_VAR_MAP.get(provider.value, 'PROVIDER_API_KEY')} environment variable "
        "or pass provider_api_key parameter.",
    )


def convert_messages(
    messages: list[LocalPromptMessage],
) -> list[dict[str, Any]]:
    """Convert Tracia messages to LiteLLM/OpenAI format.

    Args:
        messages: The Tracia messages.

    Returns:
        Messages in LiteLLM format.
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        # Handle tool role
        if msg.role == "tool":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content,
            })
            continue

        # Handle developer role (map to system)
        role = "system" if msg.role == "developer" else msg.role

        # Handle string content
        if isinstance(msg.content, str):
            result.append({"role": role, "content": msg.content})
            continue

        # Handle list content (text parts and tool calls)
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []

        for part in msg.content:
            if isinstance(part, TextPart) or (isinstance(part, dict) and part.get("type") == "text"):
                text = part.text if isinstance(part, TextPart) else part.get("text", "")
                content_parts.append({"type": "text", "text": text})
            elif isinstance(part, ToolCallPart) or (isinstance(part, dict) and part.get("type") == "tool_call"):
                if isinstance(part, ToolCallPart):
                    tc_id = part.id
                    tc_name = part.name
                    tc_args = part.arguments
                else:
                    tc_id = part.get("id", "")
                    tc_name = part.get("name", "")
                    tc_args = part.get("arguments", {})

                tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc_name,
                        "arguments": json.dumps(tc_args) if isinstance(tc_args, dict) else tc_args,
                    },
                })

        # Build the message
        msg_dict: dict[str, Any] = {"role": role}

        if content_parts:
            # If only text parts, we can simplify
            if len(content_parts) == 1 and not tool_calls:
                msg_dict["content"] = content_parts[0]["text"]
            else:
                msg_dict["content"] = content_parts
        elif not tool_calls:
            msg_dict["content"] = ""

        if tool_calls:
            msg_dict["tool_calls"] = tool_calls

        result.append(msg_dict)

    return result


def convert_tools(tools: list[ToolDefinition] | None) -> list[dict[str, Any]] | None:
    """Convert tool definitions to LiteLLM format.

    Args:
        tools: The tool definitions.

    Returns:
        Tools in LiteLLM format.
    """
    if not tools:
        return None

    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump(exclude_none=True),
            },
        })
    return result


def convert_tool_choice(tool_choice: ToolChoice | None) -> str | dict[str, Any] | None:
    """Convert tool choice to LiteLLM format.

    Args:
        tool_choice: The tool choice.

    Returns:
        Tool choice in LiteLLM format.
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, dict) and "tool" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice["tool"]}}

    return None


def convert_response_format(
    response_format: dict[str, Any] | ResponseFormatJsonSchema | None,
) -> dict[str, Any] | None:
    """Convert a Tracia response format to LiteLLM format.

    Args:
        response_format: The Tracia response format.

    Returns:
        Response format in LiteLLM/OpenAI format, or None.
    """
    if response_format is None:
        return None

    if isinstance(response_format, ResponseFormatJsonSchema):
        schema = response_format.schema_
        name = response_format.name or "response"
        json_schema: dict[str, Any] = {"name": name, "schema": schema}
        if response_format.description is not None:
            json_schema["description"] = response_format.description
        return {"type": "json_schema", "json_schema": json_schema}

    # Plain dict — check for our simplified format
    if isinstance(response_format, dict):
        fmt_type = response_format.get("type")
        schema = response_format.get("schema")

        if fmt_type == "json" and schema is not None:
            name = response_format.get("name", "response")
            json_schema: dict[str, Any] = {"name": name, "schema": schema}
            description = response_format.get("description")
            if description is not None:
                json_schema["description"] = description
            return {"type": "json_schema", "json_schema": json_schema}

        if fmt_type == "json":
            return {"type": "json_object"}

        return response_format

    return None


def parse_finish_reason(reason: str | None) -> FinishReason:
    """Parse the finish reason from LiteLLM response.

    Args:
        reason: The raw finish reason.

    Returns:
        The normalized finish reason.
    """
    if reason == "tool_calls":
        return "tool_calls"
    if reason == "length":
        return "max_tokens"
    return "stop"


def extract_tool_calls(response: "ModelResponse") -> list[ToolCall]:
    """Extract tool calls from a LiteLLM response.

    Args:
        response: The LiteLLM response.

    Returns:
        The extracted tool calls.
    """
    tool_calls: list[ToolCall] = []

    choices = getattr(response, "choices", [])
    if not choices:
        return tool_calls

    message = getattr(choices[0], "message", None)
    if not message:
        return tool_calls

    raw_tool_calls = getattr(message, "tool_calls", None)
    if not raw_tool_calls:
        return tool_calls

    for tc in raw_tool_calls:
        func = getattr(tc, "function", None)
        if func:
            try:
                args = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
            except json.JSONDecodeError:
                args = {}

            tool_calls.append(ToolCall(
                id=tc.id,
                name=func.name,
                arguments=args,
            ))

    return tool_calls


class LLMClient:
    """Client for making LLM calls via LiteLLM."""

    def complete(
        self,
        model: str,
        messages: list[LocalPromptMessage],
        *,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Make a synchronous completion request.

        Args:
            model: The model name.
            messages: The messages to send.
            provider: The LLM provider.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.
            stop: Stop sequences.
            tools: Tool definitions.
            tool_choice: Tool choice setting.
            api_key: Provider API key.
            timeout: Request timeout in seconds.

        Returns:
            The completion result.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            import litellm
        except ImportError as e:
            raise TraciaError(
                code=TraciaErrorCode.MISSING_PROVIDER_SDK,
                message="litellm is not installed. Install it with: pip install litellm",
            ) from e

        resolved_provider = resolve_provider(model, provider)
        resolved_api_key = get_provider_api_key(resolved_provider, api_key)

        # Build the request
        litellm_messages = convert_messages(messages)
        litellm_tools = convert_tools(tools)
        litellm_tool_choice = convert_tool_choice(tool_choice)

        request_kwargs: dict[str, Any] = {
            "model": get_litellm_model(model, resolved_provider),
            "messages": litellm_messages,
        }
        if resolved_api_key is not None:
            request_kwargs["api_key"] = resolved_api_key

        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop is not None:
            request_kwargs["stop"] = stop
        if litellm_tools is not None:
            request_kwargs["tools"] = litellm_tools
        if litellm_tool_choice is not None:
            request_kwargs["tool_choice"] = litellm_tool_choice
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        try:
            response = litellm.completion(**request_kwargs)
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            raise TraciaError(
                code=TraciaErrorCode.PROVIDER_ERROR,
                message=f"LLM provider error: {error_msg}",
            ) from e

        # Extract result
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", [])
        message = choices[0].message if choices else None
        content = getattr(message, "content", "") or ""
        finish_reason = choices[0].finish_reason if choices else "stop"

        return CompletionResult(
            text=content,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            tool_calls=extract_tool_calls(response),
            finish_reason=parse_finish_reason(finish_reason),
            provider=resolved_provider,
        )

    async def acomplete(
        self,
        model: str,
        messages: list[LocalPromptMessage],
        *,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Make an asynchronous completion request.

        Args:
            model: The model name.
            messages: The messages to send.
            provider: The LLM provider.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.
            stop: Stop sequences.
            tools: Tool definitions.
            tool_choice: Tool choice setting.
            api_key: Provider API key.
            timeout: Request timeout in seconds.

        Returns:
            The completion result.

        Raises:
            TraciaError: If the request fails.
        """
        try:
            import litellm
        except ImportError as e:
            raise TraciaError(
                code=TraciaErrorCode.MISSING_PROVIDER_SDK,
                message="litellm is not installed. Install it with: pip install litellm",
            ) from e

        resolved_provider = resolve_provider(model, provider)
        resolved_api_key = get_provider_api_key(resolved_provider, api_key)

        # Build the request
        litellm_messages = convert_messages(messages)
        litellm_tools = convert_tools(tools)
        litellm_tool_choice = convert_tool_choice(tool_choice)

        request_kwargs: dict[str, Any] = {
            "model": get_litellm_model(model, resolved_provider),
            "messages": litellm_messages,
        }
        if resolved_api_key is not None:
            request_kwargs["api_key"] = resolved_api_key

        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop is not None:
            request_kwargs["stop"] = stop
        if litellm_tools is not None:
            request_kwargs["tools"] = litellm_tools
        if litellm_tool_choice is not None:
            request_kwargs["tool_choice"] = litellm_tool_choice
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        try:
            response = await litellm.acompletion(**request_kwargs)
        except Exception as e:
            error_msg = sanitize_error_message(str(e))
            raise TraciaError(
                code=TraciaErrorCode.PROVIDER_ERROR,
                message=f"LLM provider error: {error_msg}",
            ) from e

        # Extract result
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", [])
        message = choices[0].message if choices else None
        content = getattr(message, "content", "") or ""
        finish_reason = choices[0].finish_reason if choices else "stop"

        return CompletionResult(
            text=content,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            tool_calls=extract_tool_calls(response),
            finish_reason=parse_finish_reason(finish_reason),
            provider=resolved_provider,
        )

    def stream(
        self,
        model: str,
        messages: list[LocalPromptMessage],
        *,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[Iterator[str], list[CompletionResult], LLMProvider]:
        """Make a streaming completion request.

        Args:
            model: The model name.
            messages: The messages to send.
            provider: The LLM provider.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.
            stop: Stop sequences.
            tools: Tool definitions.
            tool_choice: Tool choice setting.
            api_key: Provider API key.
            timeout: Request timeout in seconds.

        Returns:
            A tuple of (chunk iterator, result holder list, provider).

        Raises:
            TraciaError: If the request fails.
        """
        try:
            import litellm
        except ImportError as e:
            raise TraciaError(
                code=TraciaErrorCode.MISSING_PROVIDER_SDK,
                message="litellm is not installed. Install it with: pip install litellm",
            ) from e

        resolved_provider = resolve_provider(model, provider)
        resolved_api_key = get_provider_api_key(resolved_provider, api_key)

        # Build the request
        litellm_messages = convert_messages(messages)
        litellm_tools = convert_tools(tools)
        litellm_tool_choice = convert_tool_choice(tool_choice)

        request_kwargs: dict[str, Any] = {
            "model": get_litellm_model(model, resolved_provider),
            "messages": litellm_messages,
            "stream": True,
        }
        if resolved_api_key is not None:
            request_kwargs["api_key"] = resolved_api_key

        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop is not None:
            request_kwargs["stop"] = stop
        if litellm_tools is not None:
            request_kwargs["tools"] = litellm_tools
        if litellm_tool_choice is not None:
            request_kwargs["tool_choice"] = litellm_tool_choice
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        result_holder: list[CompletionResult] = []

        def generate_chunks() -> Iterator[str]:
            full_text = ""
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            tool_calls: list[ToolCall] = []
            finish_reason: FinishReason = "stop"
            tool_call_chunks: dict[int, dict[str, Any]] = {}

            try:
                response = litellm.completion(**request_kwargs)

                for chunk in response:
                    choices = getattr(chunk, "choices", [])
                    if not choices:
                        continue

                    delta = getattr(choices[0], "delta", None)
                    if delta:
                        content = getattr(delta, "content", None)
                        if content:
                            full_text += content
                            yield content

                        # Handle streaming tool calls
                        delta_tool_calls = getattr(delta, "tool_calls", None)
                        if delta_tool_calls:
                            for tc in delta_tool_calls:
                                idx = tc.index
                                if idx not in tool_call_chunks:
                                    tool_call_chunks[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    }
                                if tc.id:
                                    tool_call_chunks[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_call_chunks[idx]["name"] = tc.function.name
                                    if tc.function.arguments:
                                        tool_call_chunks[idx]["arguments"] += tc.function.arguments

                    chunk_finish = getattr(choices[0], "finish_reason", None)
                    if chunk_finish:
                        finish_reason = parse_finish_reason(chunk_finish)

                    # Extract usage from final chunk
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "prompt_tokens", 0)
                        output_tokens = getattr(usage, "completion_tokens", 0)
                        total_tokens = getattr(usage, "total_tokens", 0)

                # Convert accumulated tool calls
                for idx in sorted(tool_call_chunks.keys()):
                    tc_data = tool_call_chunks[idx]
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args,
                    ))

                result_holder.append(CompletionResult(
                    text=full_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    provider=resolved_provider,
                ))

            except Exception as e:
                error_msg = sanitize_error_message(str(e))
                raise TraciaError(
                    code=TraciaErrorCode.PROVIDER_ERROR,
                    message=f"LLM provider error: {error_msg}",
                ) from e

        return generate_chunks(), result_holder, resolved_provider

    async def astream(
        self,
        model: str,
        messages: list[LocalPromptMessage],
        *,
        provider: LLMProvider | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[AsyncIterator[str], list[CompletionResult], LLMProvider]:
        """Make an async streaming completion request.

        Args:
            model: The model name.
            messages: The messages to send.
            provider: The LLM provider.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            top_p: Top-p sampling.
            stop: Stop sequences.
            tools: Tool definitions.
            tool_choice: Tool choice setting.
            api_key: Provider API key.
            timeout: Request timeout in seconds.

        Returns:
            A tuple of (async chunk iterator, result holder list, provider).

        Raises:
            TraciaError: If the request fails.
        """
        try:
            import litellm
        except ImportError as e:
            raise TraciaError(
                code=TraciaErrorCode.MISSING_PROVIDER_SDK,
                message="litellm is not installed. Install it with: pip install litellm",
            ) from e

        resolved_provider = resolve_provider(model, provider)
        resolved_api_key = get_provider_api_key(resolved_provider, api_key)

        # Build the request
        litellm_messages = convert_messages(messages)
        litellm_tools = convert_tools(tools)
        litellm_tool_choice = convert_tool_choice(tool_choice)

        request_kwargs: dict[str, Any] = {
            "model": get_litellm_model(model, resolved_provider),
            "messages": litellm_messages,
            "stream": True,
        }
        if resolved_api_key is not None:
            request_kwargs["api_key"] = resolved_api_key

        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if stop is not None:
            request_kwargs["stop"] = stop
        if litellm_tools is not None:
            request_kwargs["tools"] = litellm_tools
        if litellm_tool_choice is not None:
            request_kwargs["tool_choice"] = litellm_tool_choice
        if timeout is not None:
            request_kwargs["timeout"] = timeout
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        result_holder: list[CompletionResult] = []

        async def generate_chunks() -> AsyncIterator[str]:
            full_text = ""
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            tool_calls: list[ToolCall] = []
            finish_reason: FinishReason = "stop"
            tool_call_chunks: dict[int, dict[str, Any]] = {}

            try:
                response = await litellm.acompletion(**request_kwargs)

                async for chunk in response:
                    choices = getattr(chunk, "choices", [])
                    if not choices:
                        continue

                    delta = getattr(choices[0], "delta", None)
                    if delta:
                        content = getattr(delta, "content", None)
                        if content:
                            full_text += content
                            yield content

                        # Handle streaming tool calls
                        delta_tool_calls = getattr(delta, "tool_calls", None)
                        if delta_tool_calls:
                            for tc in delta_tool_calls:
                                idx = tc.index
                                if idx not in tool_call_chunks:
                                    tool_call_chunks[idx] = {
                                        "id": "",
                                        "name": "",
                                        "arguments": "",
                                    }
                                if tc.id:
                                    tool_call_chunks[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_call_chunks[idx]["name"] = tc.function.name
                                    if tc.function.arguments:
                                        tool_call_chunks[idx]["arguments"] += tc.function.arguments

                    chunk_finish = getattr(choices[0], "finish_reason", None)
                    if chunk_finish:
                        finish_reason = parse_finish_reason(chunk_finish)

                    # Extract usage from final chunk
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "prompt_tokens", 0)
                        output_tokens = getattr(usage, "completion_tokens", 0)
                        total_tokens = getattr(usage, "total_tokens", 0)

                # Convert accumulated tool calls
                for idx in sorted(tool_call_chunks.keys()):
                    tc_data = tool_call_chunks[idx]
                    try:
                        args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args,
                    ))

                result_holder.append(CompletionResult(
                    text=full_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    provider=resolved_provider,
                ))

            except Exception as e:
                error_msg = sanitize_error_message(str(e))
                raise TraciaError(
                    code=TraciaErrorCode.PROVIDER_ERROR,
                    message=f"LLM provider error: {error_msg}",
                ) from e

        return generate_chunks(), result_holder, resolved_provider


def build_assistant_message(
    text: str,
    tool_calls: list[ToolCall],
) -> LocalPromptMessage:
    """Build an assistant message from completion result.

    Args:
        text: The text content.
        tool_calls: Any tool calls made.

    Returns:
        The assistant message.
    """
    if not tool_calls:
        return LocalPromptMessage(role="assistant", content=text)

    content: list[ContentPart] = []

    if text:
        content.append(TextPart(type="text", text=text))

    for tc in tool_calls:
        content.append(ToolCallPart(
            type="tool_call",
            id=tc.id,
            name=tc.name,
            arguments=tc.arguments,
        ))

    return LocalPromptMessage(role="assistant", content=content)
