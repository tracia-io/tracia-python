# Tracia

**LLM prompt management and tracing SDK for Python**

[![PyPI version](https://img.shields.io/pypi/v/tracia)](https://pypi.org/project/tracia/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is Tracia?

Tracia is a modern LLM prompt management and tracing platform. This Python SDK provides:

- **Unified LLM Access** - Call OpenAI, Anthropic, Google, and 100+ providers through a single interface (powered by LiteLLM)
- **Automatic Tracing** - Every LLM call is automatically traced with latency, token usage, and cost
- **Prompt Management** - Store, version, and manage your prompts in the cloud
- **Session Linking** - Easily link related calls for multi-turn conversations

## Installation

```bash
pip install tracia
```

You'll also need API keys for the LLM providers you want to use:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Quick Start

```python
from tracia import Tracia

# Initialize the client
client = Tracia(api_key="your_tracia_api_key")

# Run a local prompt
result = client.run_local(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(result.text)
print(f"Tokens: {result.usage.total_tokens}")
```

## Streaming

```python
# Stream the response
stream = client.run_local(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    print(chunk, end="", flush=True)

# Get the final result (stream.result is a Future[StreamResult])
final = stream.result.result()
print(f"\nTotal tokens: {final.usage.total_tokens}")
```

## Multi-turn Conversations with Sessions

```python
# Create a session for linked conversations
session = client.create_session()

# First message
r1 = session.run_local(
    model="gpt-4o",
    messages=[{"role": "user", "content": "My name is Alice"}]
)

# Follow-up - automatically linked to the same trace
r2 = session.run_local(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": r1.text},
        {"role": "user", "content": "What's my name?"}
    ]
)
```

## Function Calling

```python
from tracia import ToolDefinition, ToolParameters, JsonSchemaProperty

# Define a tool
tools = [
    ToolDefinition(
        name="get_weather",
        description="Get the current weather",
        parameters=ToolParameters(
            properties={
                "location": JsonSchemaProperty(
                    type="string",
                    description="City name"
                )
            },
            required=["location"]
        )
    )
]

result = client.run_local(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

if result.tool_calls:
    for call in result.tool_calls:
        print(f"Tool: {call.name}, Args: {call.arguments}")
```

## Variable Interpolation

```python
result = client.run_local(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant named {{name}}."},
        {"role": "user", "content": "Hello!"}
    ],
    variables={"name": "Claude"}
)
```

## Prompts API

```python
# List all prompts
prompts = client.prompts.list()

# Get a specific prompt
prompt = client.prompts.get("my-prompt")

# Run a prompt template
result = client.prompts.run(
    "my-prompt",
    variables={"name": "World"}
)
```

## Spans API

```python
from tracia import Eval, EvaluateOptions

# List spans
spans = client.spans.list()

# Evaluate a span
client.spans.evaluate(
    "sp_xxx",
    EvaluateOptions(
        evaluator="quality",
        value=Eval.POSITIVE,  # or Eval.NEGATIVE
        note="Great response!",
    ),
)
```

## Async Support

All methods have async variants:

```python
import asyncio

async def main():
    async with Tracia(api_key="...") as client:
        result = await client.arun_local(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(result.text)

asyncio.run(main())
```

## Supported Providers

Via LiteLLM, Tracia supports 100+ providers including:

- **OpenAI**: gpt-4o, gpt-4, gpt-3.5-turbo, o1, o3
- **Anthropic**: claude-3-opus, claude-sonnet-4, claude-3-haiku
- **Google**: gemini-2.0-flash, gemini-2.5-pro
- And many more...

## Error Handling

```python
from tracia import TraciaError, TraciaErrorCode

try:
    result = client.run_local(...)
except TraciaError as e:
    if e.code == TraciaErrorCode.MISSING_PROVIDER_API_KEY:
        print("Please set your API key")
    elif e.code == TraciaErrorCode.PROVIDER_ERROR:
        print(f"LLM error: {e.message}")
```

## Configuration Options

```python
client = Tracia(
    api_key="...",
    base_url="https://app.tracia.io",  # Custom API URL
    on_span_error=lambda e, span_id: print(f"Span error: {e}")
)

result = client.run_local(
    model="gpt-4o",
    messages=[...],
    temperature=0.7,
    max_output_tokens=1000,
    timeout_ms=30000,
    tags=["production"],
    user_id="user_123",
    session_id="session_456",
    send_trace=True,  # Set to False to disable tracing
)
```

## Learn More

- Website: [tracia.io](https://tracia.io)
- Documentation: [docs.tracia.io](https://docs.tracia.io)
- GitHub: [github.com/tracia](https://github.com/tracia)

## License

MIT
