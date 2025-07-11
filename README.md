# ML Models Integration Layer

A comprehensive Python library for interacting with various machine learning model providers, designed with a focus on clean architecture, type safety, and developer experience.

## 🌟 Features

- **Unified Interface**: Consistent API across multiple ML model providers
- **Asynchronous Support**: First-class async/await support for high-performance applications
- **Type Safety**: Built with Python type hints and Pydantic models
- **Extensible**: Easy to add new model providers or customize existing ones
- **Testing**: Comprehensive test suite with both unit and integration tests
- **Configuration**: Environment-based configuration with sensible defaults

## 🚀 Supported Providers

| Provider | Text | Image | Audio | Sync | Async | Streaming |
|---|---|---|---|---|---|---|
| [OpenAI](openai/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Google Gemini](gemini/) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [Anthropic Claude](claude/) | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| [DeepSeek](deepseek/) | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| [HuggingFace](huggingface/) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| [Pollinations](pollinations/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (simulated) |

## 📖 Usage

Here’s how to use the clients to generate content. All examples assume you have the necessary API keys set as environment variables (e.g., `HUGGINGFACE_API_KEY`).

### Text Generation (Async)

This example shows how to generate a simple text completion.

```python
import asyncio
from src.ml_models.huggingface import HuggingFaceClient

async def main():
    # Initialize the client
    # Make sure to use a model that supports text generation
    client = HuggingFaceClient(model="mistralai/Mistral-7B-Instruct-v0.2")

    # Generate text
    prompt = "Explain the importance of asynchronous programming in 50 words."
    response = await client.generate_text_async(prompt, max_new_tokens=60)
    
    print("--- Async Text Generation ---")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Text Generation (Streaming)

For long-form content or real-time applications, you can stream the response token by token.

```python
import asyncio
from src.ml_models.huggingface import HuggingFaceClient

async def main():
    # Initialize the client
    client = HuggingFaceClient(model="mistralai/Mistral-7B-Instruct-v0.2")

    # Stream text
    prompt = "Write a short story about a robot who discovers music."
    stream = client.generate_text_stream(prompt, max_new_tokens=100)
    
    print("--- Streaming Text Generation ---")
    full_response = ""
    async for event in stream:
        if event.event_type == "CONTENT":
            token = event.data
            full_response += token
            print(token, end="", flush=True)

    print("\n--- End of Stream ---")

if __name__ == "__main__":
    asyncio.run(main())
```

### Image Generation (Async)

This example demonstrates how to generate an image from a text prompt.

```python
import asyncio
from src.ml_models.huggingface import HuggingFaceClient

async def main():
    # Initialize the client with a text-to-image model
    client = HuggingFaceClient(model="stabilityai/stable-diffusion-2-1")

    # Generate an image
    prompt = "An astronaut riding a horse on the moon, digital art"
    image_bytes = await client.generate_image_async(prompt)
    
    # Save the image
    with open("generated_image.png", "wb") as f:
        f.write(image_bytes)
        
    print("--- Image Generation ---")
    print("Image saved to generated_image.png")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

The library follows a clean architecture with these main components:

### Base Generator

[`base_generator.py`](base_generator.py) provides an abstract base class that all model generators inherit from. It handles:
- Environment variable loading
- Configuration management
- API key validation
- Common utilities

### Provider Implementations

Each provider (OpenAI, Gemini, etc.) implements the base interface with provider-specific logic:
- API client initialization
- Request/response handling
- Error handling and retries
- Rate limiting

## 📦 Installation

```bash
# Install with pip (from project root)
pip install -e .

## ⚙️ Configuration

API keys are managed via environment variables. Create a `.env` file in the project root and add your keys:

```.env
OPENAI_API_KEY="your-openai-key"
HUGGINGFACE_API_KEY="your-huggingface-key"
DEEPSEEK_API_KEY="your-deepseek-key"
# ...and so on for other providers
```

The clients will automatically load these variables.

## ⚠️ Error Handling

Each provider has custom exceptions that inherit from a base error class (e.g., `HuggingFaceError`). This allows for granular error handling.

```python
import asyncio
from src.ml_models.huggingface import HuggingFaceClient
from src.ml_models.huggingface.exceptions import HuggingFaceApiError, HuggingFaceAuthError

async def main():
    try:
        client = HuggingFaceClient(model="invalid-model")
        await client.generate_text_async("test prompt")
    except HuggingFaceAuthError as e:
        print(f"Authentication error: {e}")
    except HuggingFaceApiError as e:
        print(f"API error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🤝 Contributing

Adding a new provider is straightforward:

1.  **Create a new module** in `src/ml_models/` (e.g., `src/ml_models/new_provider/`).
2.  **Implement the Client**: Create a `NewProviderClient` class that inherits from `BaseGenerator` and/or `BaseStreamingGenerator`.
3.  **Implement the abstract methods**: Provide concrete implementations for `generate_text_async`, `generate_image_async`, etc.
4.  **Add Custom Exceptions**: Define custom exceptions in `exceptions.py`.
5.  **Add Tests**: Create a new test file in `src/ml_models/_tests/ai/`.
6.  **Update the README**: Add the new provider to the supported providers table.

## ✅ Testing

To run the test suite for the ML models, use `pytest` from the project root:

```bash
# Run all tests for the ml_models directory
pytest src/ml_models/_tests/ai/

# Run tests for a specific client
pytest src/ml_models/_tests/ai/test_huggingface_client.py
```

# Or with poetry
poetry install
```

## 🔧 Configuration

1. Copy `.env.example` to `.env`
2. Add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   CLAUDE_API_KEY=your_claude_key
   DEEPSEEK_API_KEY=your_deepseek_key
   HUGGINGFACE_API_KEY=your_hf_key
   ```

## 🚀 Quick Start

### Using OpenAI

```python
from ml_models.openai import OpenAIClient

# Sync client
client = OpenAIClient()
response = client.generate(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Async client
async def main():
    async with OpenAIClient() as client:
        response = await client.agenerate(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.choices[0].message.content)
```

### Using Gemini

```python
from ml_models.gemini import GeminiClient, GeminiModel

# Sync client
client = GeminiClient()
response = client.generate_content(
    model=GeminiModel.GEMINI_2_5_FLASH,
    prompt="Tell me a joke"
)
print(response.text)

# Async streaming
async def main():
    client = GeminiClient()
    async for chunk in await client.astream_content(
        model=GeminiModel.GEMINI_2_5_FLASH,
        prompt="Write a short story"
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/ml_models/test_gemini.py -v
```

Run with coverage:
```bash
pytest --cov=ml_models tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- All the amazing ML model providers for their APIs
- The Python community for awesome open source tools
- All contributors who help improve this project
