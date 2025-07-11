import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from ml_models.gemini import (
    GeminiClient,
    GeminiModel,
    AuthenticationError,
    APIError,
    RateLimitError,
    StreamingDelta,
    StreamingResponseState
)

@patch('ml_models.gemini.gemini_client.get_gemini_api_key', return_value='test_api_key')
class TestGeminiClient:

    def test_client_initialization_success(self, mock_get_key):
        """Tests successful initialization of the GeminiClient."""
        with patch('google.generativeai.configure') as mock_configure:
            client = GeminiClient()
            mock_get_key.assert_called_once()
            mock_configure.assert_called_once_with(api_key='test_api_key')
            assert client.api_key == 'test_api_key'

    def test_client_initialization_failure(self, mock_get_key):
        """Tests that AuthenticationError is raised if the API key is not found."""
        mock_get_key.side_effect = AuthenticationError('API key not found')
        with pytest.raises(AuthenticationError):
            GeminiClient()

    @patch('google.generativeai.GenerativeModel')
    def test_generate_content_success(self, mock_generative_model, mock_get_key):
        """Tests successful content generation."""
        mock_response = MagicMock()
        mock_response.text = 'Hello, world!'
        mock_generative_model.return_value.generate_content.return_value = mock_response

        client = GeminiClient()
        response = client.generate_content(GeminiModel.GEMINI_2_5_FLASH, 'test prompt')

        assert response.text == 'Hello, world!'
        mock_generative_model.return_value.generate_content.assert_called_once_with('test prompt')

    @patch('google.generativeai.GenerativeModel')
    def test_generate_text_convenience_method(self, mock_generative_model, mock_get_key):
        """Tests the generate_text convenience method."""
        mock_response = MagicMock()
        mock_response.text = 'Convenience works!'
        mock_generative_model.return_value.generate_content.return_value = mock_response

        client = GeminiClient()
        text = client.generate_text(GeminiModel.GEMINI_2_5_FLASH, 'test prompt')

        assert text == 'Convenience works!'

    @patch('google.generativeai.GenerativeModel')
    def test_api_error_handling(self, mock_generative_model, mock_get_key):
        """Tests that APIError is raised for general API failures."""
        mock_generative_model.return_value.generate_content.side_effect = Exception('Generic API error')

        client = GeminiClient()
        with pytest.raises(APIError, match='An unexpected error occurred: Generic API error'):
            client.generate_content(GeminiModel.GEMINI_2_5_FLASH, 'test prompt')

    @patch('google.generativeai.GenerativeModel')
    def test_rate_limit_error_handling(self, mock_generative_model, mock_get_key):
        """Tests that RateLimitError is raised when a rate limit message is received."""
        mock_generative_model.return_value.generate_content.side_effect = Exception('429 rate limit exceeded')

        client = GeminiClient()
        with pytest.raises(RateLimitError, match='Rate limit exceeded: 429 rate limit exceeded'):
            client.generate_content(GeminiModel.GEMINI_2_5_FLASH, 'test prompt')

    def test_stream_content_success(self, mock_get_key):
        """Tests successful streaming content generation."""
        # Setup mock response with chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.parts = [MagicMock()]
        mock_chunk1.text = "Hello"
        mock_chunk1.to_dict.return_value = {"text": "Hello"}
        
        mock_chunk2 = MagicMock()
        mock_chunk2.parts = [MagicMock()]
        mock_chunk2.text = ", world!"
        mock_chunk2.to_dict.return_value = {"text": ", world!"}
        
        mock_generative_model = MagicMock()
        mock_generative_model.return_value.generate_content.return_value = [mock_chunk1, mock_chunk2]
        
        with patch('google.generativeai.GenerativeModel', mock_generative_model):
            client = GeminiClient()
            response = client.stream_content(GeminiModel.GEMINI_2_5_FLASH, "Hello")
            
            # Test iteration
            deltas = list(response)
            assert len(deltas) == 4  # START + 2 content + COMPLETE
            assert deltas[0].state == StreamingResponseState.START
            assert deltas[1].text == "Hello"
            assert deltas[2].text == ", world!"
            assert deltas[3].state == StreamingResponseState.END
            
            # Test collect()
            response.stream = (delta for delta in [
                StreamingDelta(text="Hello", state=StreamingResponseState.CONTENT, raw={}),
                StreamingDelta(text=", world!", state=StreamingResponseState.CONTENT, raw={}),
            ])
            assert response.collect() == "Hello, world!"

    def test_stream_content_with_callback(self, mock_get_key):
        """Tests streaming with a callback function."""
        mock_chunk = MagicMock()
        mock_chunk.parts = [MagicMock()]
        mock_chunk.text = "Test"
        mock_chunk.to_dict.return_value = {"text": "Test"}
        
        mock_generative_model = MagicMock()
        mock_generative_model.return_value.generate_content.return_value = [mock_chunk]
        
        callback = MagicMock()
        
        with patch('google.generativeai.GenerativeModel', mock_generative_model):
            client = GeminiClient()
            response = client.stream_content(
                GeminiModel.GEMINI_2_5_FLASH,
                "Test prompt",
                callback=callback
            )
            
            # Consume the generator to trigger callbacks
            list(response)
            
            # Verify callbacks were made
            assert callback.call_count >= 2  # At least START and one content
            assert any(call[0][0].state == StreamingResponseState.START for call in callback.call_args_list)
            assert any(call[0][0].text == "Test" for call in callback.call_args_list)
            assert any(call[0][0].state == StreamingResponseState.END for call in callback.call_args_list)

    @pytest.mark.asyncio
    async def test_astream_content_success(self, mock_get_key):
        """Tests successful async streaming content generation."""
        # Setup mock response with chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.parts = [MagicMock()]
        mock_chunk1.text = "Async"
        mock_chunk1.to_dict.return_value = {"text": "Async"}
        
        mock_chunk2 = MagicMock()
        mock_chunk2.parts = [MagicMock()]
        mock_chunk2.text = " streaming"
        mock_chunk2.to_dict.return_value = {"text": " streaming"}
        
        mock_generative_model = MagicMock()
        mock_generative_model.return_value.generate_content.return_value = [mock_chunk1, mock_chunk2]
        
        with patch('google.generativeai.GenerativeModel', mock_generative_model):
            client = GeminiClient()
            response = await client.astream_content(GeminiModel.GEMINI_2_5_FLASH, "Async test")
            
            # Test async iteration
            deltas = [delta async for delta in response]
            assert len(deltas) == 4  # START + 2 content + COMPLETE
            assert deltas[0].state == StreamingResponseState.START
            assert deltas[1].text == "Async"
            assert deltas[2].text == " streaming"
            assert deltas[3].state == StreamingResponseState.END
            
            # Test acollect()
            async def mock_async_gen():
                yield StreamingDelta(text="Async", state=StreamingResponseState.CONTENT, raw={})
                yield StreamingDelta(text=" streaming", state=StreamingResponseState.CONTENT, raw={})
            
            response.stream = mock_async_gen()
            assert await response.acollect() == "Async streaming"

    @pytest.mark.asyncio
    async def test_astream_content_with_async_callback(self, mock_get_key):
        """Tests async streaming with an async callback function."""
        mock_chunk = MagicMock()
        mock_chunk.parts = [MagicMock()]
        mock_chunk.text = "Async test"
        mock_chunk.to_dict.return_value = {"text": "Async test"}
        
        mock_generative_model = MagicMock()
        mock_generative_model.return_value.generate_content.return_value = [mock_chunk]
        
        mock_callback = AsyncMock()
        
        with patch('google.generativeai.GenerativeModel', mock_generative_model):
            client = GeminiClient()
            response = await client.astream_content(
                GeminiModel.GEMINI_2_5_FLASH,
                "Async test",
                callback=mock_callback
            )
            
            # Consume the generator to trigger callbacks
            async for _ in response:
                pass
            
            # Verify callbacks were made
            assert mock_callback.await_count >= 2  # At least START and one content
            
            # Check that at least one call was with a content delta
            assert any(call_args[0][0].text == "Async test" 
                      for call_args in mock_callback.await_args_list)

    @pytest.mark.asyncio
    async def test_agenerate_content_success(self, mock_get_key):
        """Tests successful async content generation."""
        # Mock the streaming chunks for the async generator
        mock_chunk = MagicMock()
        mock_chunk.parts = [MagicMock()]
        mock_chunk.text = "Async response"
        mock_chunk.to_dict.return_value = {"text": "Async response"}

        mock_generative_model = MagicMock()
        mock_generative_model.return_value.generate_content.return_value = [mock_chunk]
        
        with patch('google.generativeai.GenerativeModel', mock_generative_model):
            client = GeminiClient()
            
            async def test():
                response = await client.agenerate_content(GeminiModel.GEMINI_2_5_FLASH, "Test")
                return response.text
            
            result = await test()
            assert result == "Async response"

    @pytest.mark.asyncio
    async def test_agenerate_text_success(self, mock_get_key):
        """Tests successful async text generation."""
        mock_chunk = MagicMock()
        mock_chunk.parts = [MagicMock()]
        mock_chunk.text = "Async text"
        mock_chunk.to_dict.return_value = {"text": "Async text"}

        mock_generative_model = MagicMock()
        mock_generative_model.return_value.generate_content.return_value = [mock_chunk]
        
        with patch('google.generativeai.GenerativeModel', mock_generative_model):
            client = GeminiClient()
            
            async def test():
                return await client.agenerate_text(GeminiModel.GEMINI_2_5_FLASH, "Test")
            
            result = await test()
            assert result == "Async text"
