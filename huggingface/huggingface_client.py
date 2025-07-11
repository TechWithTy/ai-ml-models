#!/usr/bin/env python
# src/ml_models/huggingface/huggingface_client.py

import asyncio
import os
from typing import AsyncGenerator, Dict, Any, Optional
from io import BytesIO

from PIL import Image
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.inference._generated.types import ChatCompletionOutput, ChatCompletionStreamOutput
from huggingface_hub.utils import HfHubError

from ..base_generator import BaseGenerator
from ..base_streaming_generator import BaseStreamingGenerator
from ..utils.stream_event import StreamEvent
from .exceptions import HuggingFaceApiError, HuggingFaceAuthError


class HuggingFaceClient(BaseGenerator, BaseStreamingGenerator):
    """A Hugging Face client for generating text and images asynchronously and with streaming."""

    def __init__(self, model: str, token: Optional[str] = None):
        self.model = model
        self.token = token or os.getenv("HUGGINGFACE_API_KEY")
        if not self.token:
            raise HuggingFaceAuthError("Hugging Face API key not found. Please set HUGGINGFACE_API_KEY environment variable.")
        
        try:
            self.async_client = AsyncInferenceClient(model=self.model, token=self.token)
        except HfHubError as e:
            raise HuggingFaceAuthError(f"Failed to initialize Hugging Face client: {e}")

    async def generate_text_async(self, prompt: str, **kwargs: Any) -> str:
        """Generates text asynchronously."""
        messages = [{"role": "user", "content": prompt}]
        try:
            response: ChatCompletionOutput = await self.async_client.chat_completion(
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except HfHubError as e:
            raise HuggingFaceApiError(f"Hugging Face API error: {e}")

    async def generate_text_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[StreamEvent, None]:
        """Generates text in a stream."""
        yield StreamEvent(event_type="START", data={})
        messages = [{"role": "user", "content": prompt}]
        try:
            stream = self.async_client.chat_completion_stream(
                messages=messages,
                **kwargs
            )
            async for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield StreamEvent(event_type="CONTENT", data=content)
        except HfHubError as e:
            yield StreamEvent(event_type="ERROR", data={"error": str(e)})
        finally:
            yield StreamEvent(event_type="DONE", data={})

    async def generate_image_async(self, prompt: str, **kwargs: Any) -> bytes:
        """Generates an image asynchronously."""
        try:
            image: Image.Image = await self.async_client.text_to_image(prompt, **kwargs)
            with BytesIO() as output:
                image.save(output, format="PNG")
                return output.getvalue()
        except HfHubError as e:
            raise HuggingFaceApiError(f"Hugging Face API error: {e}")

    async def generate_audio_async(self, text: str, **kwargs: Any) -> bytes:
        """Generates audio asynchronously. Not yet implemented for Hugging Face."""
        raise NotImplementedError("Audio generation is not yet implemented for Hugging Face.")
