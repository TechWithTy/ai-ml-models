# AI/ML Models Integration

This directory contains implementations for various AI/ML model integrations used in the project. Each model is implemented as a separate module with a consistent interface for easy swapping and testing.

## Available Models

### 1. Gemini
- **Path**: `gemini/`
- **Description**: Google's Gemini AI model integration with support for text, image, and multimodal content generation.
- **Features**:
  - Synchronous and asynchronous API
  - Streaming support
  - Type safety with Pydantic
  - Comprehensive error handling

### 2. OpenAI
- **Path**: `openai/`
- **Description**: Integration with OpenAI's GPT models.
- **Features**:
  - Support for chat completions
  - Function calling
  - Streaming responses

### 3. Claude
- **Path**: `claude/`
- **Description**: Anthropic's Claude model integration.
- **Features**:
  - Long context support
  - Structured outputs
  - Safety features

### 4. DeepSeek
- **Path**: `deepseek/`
- **Description**: Integration with DeepSeek models.
- **Features**:
  - Code generation
  - Long-context understanding
  - High accuracy on technical tasks

### 5. HuggingFace
- **Path**: `huggingface/`
- **Description**: Interface for HuggingFace's model hub.
- **Features**:
  - Support for thousands of pre-trained models
  - Easy model loading and inference
  - Custom pipeline support

### 6. OpenRouter
- **Path**: `open_router/`
- **Description**: Unified API for multiple AI models through OpenRouter.
- **Features**:
  - Single interface for multiple providers
  - Model fallback support
  - Usage tracking

### 7. Pollinations
- **Path**: `pollinations/`
- **Description**: Integration for AI-generated media.
- **Features**:
  - Image generation
  - Video synthesis
  - Media manipulation

## Common Interface

All model implementations follow a common interface pattern:

```python
class BaseGenerator:
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate text from a prompt."""
        pass
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream generated text."""
        pass
    
    async def astream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Asynchronously stream generated text."""
        pass
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   poetry install
   ```

2. **Set Environment Variables**:
   ```bash
   # For Gemini
   export GEMINI_API_KEY="your-api-key"
   
   # For OpenAI
   export OPENAI_API_KEY="your-api-key"
   
   # For other models, set their respective API keys
   ```

3. **Usage Example**:
   ```python
   from ml_models.gemini import GeminiClient
   from ml_models.models import GeminiModel
   
   # Initialize client
   client = GeminiClient()
   
   # Generate text
   response = client.generate_text(
       model=GeminiModel.GEMINI_2_5_FLASH,
       prompt="Hello, world!"
   )
   print(response)
   ```

## Testing

Run tests with pytest:

```bash
pytest tests/ml_models/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
