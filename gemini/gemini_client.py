"""
Main client for interacting with the Google Gemini API.
"""
import asyncio
import google.generativeai as genai
from typing import Optional, AsyncGenerator, Generator

from .models import GeminiModel
from .types import (
    GenerateContentResponse,
    StreamingDelta,
    StreamingResponse,
    StreamingResponseState,
    StreamConfig,
    StreamingCallback,
    AsyncStreamingCallback
)
from .utils import get_gemini_api_key, get_gemini_logger, handle_api_error
from .exceptions import AuthenticationError

# Configure logging
logger = get_gemini_logger(__name__)

class GeminiClient:
    """! A client for the Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initializes the Gemini client.

        Args:
            api_key: The API key for the Gemini API. If not provided, it will be
                     retrieved from the GEMINI_API_KEY environment variable.
        
        Raises:
            AuthenticationError: If the API key is not provided or found.
        """
        self.api_key = api_key or get_gemini_api_key()
        if not self.api_key:
            raise AuthenticationError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
        genai.configure(api_key=self.api_key)

    def generate_content(
        self, 
        model: GeminiModel, 
        prompt: any,
        media_path: Optional[str] = None,
        media_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        tools: Optional[list] = None
    ) -> GenerateContentResponse:
        """Generates content using the specified model and prompt, optionally with media/tools.

        Args:
            model: The Gemini model to use.
            prompt: The prompt to send to the model.
            media_path: Path to media file (image/video/audio).
            media_mime_type: MIME type of the media.
            system_instruction: System instructions for the model.
            json_mode: If True, enforces application/json output.
            tools: List of tools (functions) to bind to the model.

        Returns:
            A GenerateContentResponse object with the generated text.
        """
        try:
            # Configure generation config
            generation_config = {}
            if json_mode:
                generation_config["response_mime_type"] = "application/json"
            
            # Initialize model with system instructions and tools if provided
            model_instance = genai.GenerativeModel(
                model_name=model.value,
                system_instruction=system_instruction,
                tools=tools
            )
            
            content_parts = [prompt]
            
            if media_path:
                if not media_mime_type:
                    # Simple extension-based guess if not provided
                    ext = media_path.split(".")[-1].lower()
                    if ext in ["jpg", "jpeg", "png"]: media_mime_type = f"image/{ext}"
                    elif ext == "mp4": media_mime_type = "video/mp4"
                    elif ext == "mp3": media_mime_type = "audio/mp3"
                    elif ext == "wav": media_mime_type = "audio/wav"
                
                try:
                    uploaded_file = genai.upload_file(path=media_path, mime_type=media_mime_type)
                    
                    # For video, we might need to wait for processing? 
                    # The SDK suggests waiting for state=ACTIVE generally for video.
                    import time
                    while uploaded_file.state.name == "PROCESSING":
                        time.sleep(1)
                        uploaded_file = genai.get_file(uploaded_file.name)
                        
                    if uploaded_file.state.name == "FAILED":
                         raise ValueError("Gemini File API processing failed.")
                         
                    content_parts.append(uploaded_file)
                    
                except Exception as upload_err:
                     raise upload_err

            response = model_instance.generate_content(
                content_parts, 
                generation_config=generation_config
            )
            return GenerateContentResponse(text=response.text, raw_response=response)
        except Exception as e:
            handle_api_error(e)

    def generate_text(self, model: GeminiModel, prompt: str) -> Optional[str]:
        """A convenience method to generate text and return it directly."""
        response = self.generate_content(model, prompt)
        return response.text

    def stream_content(
        self, 
        model: GeminiModel, 
        prompt: any, 
        config: Optional[StreamConfig] = None,
        callback: Optional[StreamingCallback] = None
    ) -> StreamingResponse:
        """Streams content from the model.

        Args:
            model: The Gemini model to use.
            prompt: The prompt to send to the model.
            config: Streaming configuration options.
            callback: A callback function to handle each streaming delta.

        Returns:
            A StreamingResponse object that can be iterated over.
        """
        def generate() -> Generator[StreamingDelta, None, None]:
            try:
                model_instance = genai.GenerativeModel(model.value)
                response = model_instance.generate_content(prompt, stream=True)
                
                # Yield start of stream
                delta = StreamingDelta(text=None, state=StreamingResponseState.START, raw={"model": model.value})
                yield delta
                if callback:
                    callback(delta)

                # Stream chunks
                for chunk in response:
                    if not hasattr(chunk, 'parts') or not chunk.parts:
                        continue
                    text = chunk.text if hasattr(chunk, 'text') else ''
                    if text:
                        delta = StreamingDelta(
                            text=text, 
                            state=StreamingResponseState.CONTENT, 
                            raw=chunk.to_dict() if hasattr(chunk, 'to_dict') else str(chunk)
                        )
                        yield delta
                        if callback:
                            callback(delta)

            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                delta = StreamingDelta(text=None, state=StreamingResponseState.ERROR, raw={"error": str(e)})
                yield delta
                if callback:
                    callback(delta)
                handle_api_error(e)
            finally:
                # Yield end of stream
                delta = StreamingDelta(text=None, state=StreamingResponseState.END, raw={})
                yield delta
                if callback:
                    callback(delta)

        return StreamingResponse(stream=generate(), is_async=False)

    async def astream_content(
        self, 
        model: GeminiModel, 
        prompt: any, 
        config: Optional[StreamConfig] = None,
        callback: Optional[AsyncStreamingCallback] = None
    ) -> StreamingResponse:
        """Asynchronously streams content from the model."""
        async def generate() -> AsyncGenerator[StreamingDelta, None]:
            try:
                model_instance = genai.GenerativeModel(model.value)
                response = await asyncio.to_thread(
                    model_instance.generate_content,
                    prompt,
                    stream=True
                )

                # Yield start of stream
                delta = StreamingDelta(text=None, state=StreamingResponseState.START, raw={"model": model.value})
                yield delta
                if callback:
                    await callback(delta)

                # Stream chunks
                for chunk in response:
                    if not hasattr(chunk, 'parts') or not chunk.parts:
                        continue
                    text = chunk.text if hasattr(chunk, 'text') else ''
                    if text:
                        delta = StreamingDelta(
                            text=text, 
                            state=StreamingResponseState.CONTENT, 
                            raw=chunk.to_dict() if hasattr(chunk, 'to_dict') else str(chunk)
                        )
                        yield delta
                        if callback:
                            await callback(delta)

            except Exception as e:
                logger.error(f"Error in async streaming: {e}")
                delta = StreamingDelta(text=None, state=StreamingResponseState.ERROR, raw={"error": str(e)})
                yield delta
                if callback:
                    await callback(delta)
                handle_api_error(e)
            finally:
                # Yield end of stream
                delta = StreamingDelta(text=None, state=StreamingResponseState.END, raw={})
                yield delta
                if callback:
                    await callback(delta)

        return StreamingResponse(stream=generate(), is_async=True)

    async def agenerate_content(self, model: GeminiModel, prompt: any) -> GenerateContentResponse:
        """A convenience method to generate content asynchronously."""
        response_stream = await self.astream_content(model, prompt)
        full_text = await response_stream.acollect()
        return GenerateContentResponse(text=full_text, raw_response=None)

    async def agenerate_text(self, model: GeminiModel, prompt: str) -> Optional[str]:
        """A convenience method to generate text asynchronously and return it directly."""
        response = await self.agenerate_content(model, prompt)
        return response.text
