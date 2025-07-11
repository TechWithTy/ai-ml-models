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
| [HuggingFace](huggingface/) | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| [Pollinations](pollinations/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (simulated) |

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
