# Gemini API SDK

A Python SDK for interacting with the Google Gemini API, with a focus on clean, modern, and asynchronous-first design. This SDK provides a simple interface for content generation, including robust support for streaming responses.

## Features

-   **Simple Interface**: A straightforward client for generating content.
-   **Model Selection**: Easily switch between Gemini models (`1.5 Pro`, `1.5 Flash`).
-   **Streaming Support**: Both synchronous and asynchronous streaming with callbacks.
-   **Error Handling**: Custom exceptions for common API errors.
-   **Type-Safe**: Uses Pydantic for robust data modeling and validation.

## Installation

This SDK is part of a larger project and is not a standalone package. To use it, ensure you have the required dependencies installed:

```bash
pip install google-generativeai pydantic python-dotenv
```

## Authentication

The SDK requires a Gemini API key. You can provide it in one of two ways:

1.  **Environment Variable (Recommended)**:
    Set the `GEMINI_API_KEY` environment variable in a `.env` file in your project root.

    ```
    GEMINI_API_KEY="your_api_key_here"
    ```

2.  **Directly in Code**:
    Pass the API key when initializing the `GeminiClient`.

    ```python
    from ml_models.gemini import GeminiClient

    client = GeminiClient(api_key="your_api_key_here")
    ```

## Usage

### Basic Usage

Generate text from a simple prompt:

```python
from ml_models.gemini import GeminiClient, GeminiModel

# Initialize the client (assumes API key is in .env)
client = GeminiClient()

# Generate text
response_text = client.generate_text(
    model=GeminiModel.GEMINI_2_5_FLASH,
    prompt="Explain the importance of asynchronous programming in 50 words."
)

print(response_text)
```

### Streaming Usage

The SDK provides powerful streaming capabilities for both synchronous and asynchronous applications.

#### Synchronous Streaming

You can iterate over the streaming response to get content as it becomes available.

```python
from ml_models.gemini import GeminiClient, GeminiModel

client = GeminiClient()

# Start a streaming request
response_stream = client.stream_content(
    model=GeminiModel.GEMINI_2_5_FLASH,
    prompt="Write a short story about a robot who discovers music."
)

# Iterate over the stream
for delta in response_stream:
    if delta.text:
        print(delta.text, end="", flush=True)

# Or collect the full response at the end
full_text = response_stream.collect()
print(full_text)
```

#### Synchronous Streaming with a Callback

You can also provide a callback function to process each delta as it arrives.

```python
from ml_models.gemini import GeminiClient, GeminiModel, StreamingDelta

def my_callback(delta: StreamingDelta):
    """A simple callback to print streaming text."""
    if delta.text:
        print(f"(Callback): {delta.text}")

client = GeminiClient()
response_stream = client.stream_content(
    model=GeminiModel.GEMINI_2_5_FLASH,
    prompt="What are the best practices for writing clean code?",
    callback=my_callback
)

# You still need to consume the iterator for the callbacks to fire
list(response_stream)
```

#### Asynchronous Streaming

For `asyncio`-based applications, use the `astream_content` method.

```python
import asyncio
from ml_models.gemini import GeminiClient, GeminiModel

async def main():
    client = GeminiClient()

    # Start an async streaming request
    response_stream = await client.astream_content(
        model=GeminiModel.GEMINI_2_5_FLASH,
        prompt="Write a haiku about Python."
    )

    # Asynchronously iterate over the stream
    async for delta in response_stream:
        if delta.text:
            print(delta.text, end="", flush=True)

    # Or collect the full response at the end
    full_text = await response_stream.acollect()
    print(full_text)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Asynchronous Streaming with a Callback

Provide an `async` callback function for asynchronous processing.

```python
import asyncio
from ml_models.gemini import GeminiClient, GeminiModel, StreamingDelta

async def my_async_callback(delta: StreamingDelta):
    """An async callback to process streaming text."""
    if delta.text:
        print(f"(Async Callback): {delta.text}")

async def main():
    client = GeminiClient()
    response_stream = await client.astream_content(
        model=GeminiModel.GEMINI_2_5_FLASH,
        prompt="What is the event loop in asyncio?",
        callback=my_async_callback
    )

    # Consume the async iterator for callbacks to fire
    async for _ in response_stream:
        pass

if __name__ == "__main__":
    asyncio.run(main())
```

### Async Convenience Methods

For simple text generation in an async context, you can use `agenerate_text`:

```python
import asyncio
from ml_models.gemini import GeminiClient, GeminiModel

async def main():
    client = GeminiClient()
    response_text = await client.agenerate_text(
        model=GeminiModel.GEMINI_2_5_FLASH,
        prompt="What is a decorator in Python?"
    )
    print(response_text)

if __name__ == "__main__":
    asyncio.run(main())
```
