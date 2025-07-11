import httpx
import asyncio
import urllib.parse
from typing import Optional, Dict, Any

from typing import AsyncGenerator
from ..base_generator import BaseGenerator
from ..base_streaming_generator import BaseStreamingGenerator, StreamEvent, StreamEventType

class PollinationsClient(BaseGenerator, BaseStreamingGenerator):
    """Client for interacting with the Pollinations API."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(provider="pollinations", api_key=api_key, config_section="Pollinations", **kwargs)
        self.base_image_url = self.config.get("base_image_url", "https://image.pollinations.ai")
        self.base_text_url = self.config.get("base_text_url", "https://text.pollinations.ai")
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_backoff = self.config.get("retry_backoff_seconds", 1.5)

    async def _fetch_with_retries(self, url: str, client: httpx.AsyncClient) -> Optional[httpx.Response]:
        for attempt in range(self.max_retries):
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return response
                print(f"[Pollinations] Attempt {attempt + 1} failed ({response.status_code}), retrying...")
            except httpx.RequestError as e:
                print(f"[Pollinations] Request error on attempt {attempt + 1}: {e}")
            await asyncio.sleep(self.retry_backoff)
        print("[Pollinations] All retries failed.")
        return None

    async def generate_image(
        self, 
        prompt: str, 
        model: str = "stable-diffusion",
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        nologo: bool = True,
        private: bool = True,
        enhance: bool = False,
        safe: bool = True
    ) -> Dict[str, Any]:
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"{self.base_image_url}/prompt/{encoded_prompt}"
        params = {
            "model": model,
            "width": width,
            "height": height,
            "seed": seed,
            "nologo": str(nologo).lower(),
            "private": str(private).lower(),
            "enhance": str(enhance).lower(),
            "safe": str(safe).lower()
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                return {"status": "success", "url": str(response.url)}
            except httpx.HTTPStatusError as e:
                return {"status": "error", "message": f"HTTP Error: {e.response.status_code}", "details": e.response.text}
            except Exception as e:
                return {"status": "error", "message": "An unexpected error occurred", "details": str(e)}

    async def generate_text(
        self, 
        prompt: str, 
        model: str = "gpt-3.5-turbo"
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "prompt": prompt
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_text_url, json=payload, timeout=30.0)
                response.raise_for_status()
                return {"status": "success", "response": response.text}
            except httpx.HTTPStatusError as e:
                return {"status": "error", "message": f"HTTP Error: {e.response.status_code}", "details": e.response.text}
            except Exception as e:
                return {"status": "error", "message": "An unexpected error occurred", "details": str(e)}

    async def generate_audio(
        self, 
        prompt: str, 
        voice: str = "nova"
    ) -> Dict[str, Any]:
        url = f"{self.base_text_url}/{urllib.parse.quote(prompt)}?model=openai-audio&voice={voice}"
        async with httpx.AsyncClient() as client:
            try:
                response = await self._fetch_with_retries(url, client)
                if response and "audio/mpeg" in response.headers.get("Content-Type", ""):
                    return {"status": "success", "url": url, "content_type": response.headers.get("Content-Type")}
                elif response:
                    return {"status": "error", "message": "Unexpected content type", "details": response.text}
                else:
                    return {"status": "error", "message": "Failed to generate audio after retries"}
            except Exception as e:
                return {"status": "error", "message": "An unexpected error occurred", "details": str(e)}

    async def stream_text(
        self, 
        prompt: str, 
        model: str = "gpt-3.5-turbo"
    ) -> AsyncGenerator[StreamEvent[str], None]:
        """Simulated streaming for text generation."""
        yield StreamEvent(event_type=StreamEventType.START)
        try:
            result = await self.generate_text(prompt, model)
            if result.get("status") == "success":
                yield StreamEvent(event_type=StreamEventType.CONTENT, data=result.get("response"))
            else:
                error_message = result.get("details", "Unknown error")
                yield StreamEvent(event_type=StreamEventType.ERROR, error=Exception(error_message))
        except Exception as e:
            yield StreamEvent(event_type=StreamEventType.ERROR, error=e)
        finally:
            yield StreamEvent(event_type=StreamEventType.DONE)

    async def stream_image(
        self, 
        prompt: str, 
        **kwargs
    ) -> AsyncGenerator[StreamEvent[str], None]:
        """Simulated streaming for image generation."""
        yield StreamEvent(event_type=StreamEventType.START)
        try:
            result = await self.generate_image(prompt, **kwargs)
            if result.get("status") == "success":
                yield StreamEvent(event_type=StreamEventType.CONTENT, data=result.get("url"))
            else:
                error_message = result.get("details", "Unknown error")
                yield StreamEvent(event_type=StreamEventType.ERROR, error=Exception(error_message))
        except Exception as e:
            yield StreamEvent(event_type=StreamEventType.ERROR, error=e)
        finally:
            yield StreamEvent(event_type=StreamEventType.DONE)

    async def stream_audio(
        self, 
        prompt: str, 
        voice: str = "nova"
    ) -> AsyncGenerator[StreamEvent[str], None]:
        """Simulated streaming for audio generation."""
        yield StreamEvent(event_type=StreamEventType.START)
        try:
            result = await self.generate_audio(prompt, voice)
            if result.get("status") == "success":
                yield StreamEvent(event_type=StreamEventType.CONTENT, data=result.get("url"))
            else:
                error_message = result.get("details", "Unknown error")
                yield StreamEvent(event_type=StreamEventType.ERROR, error=Exception(error_message))
        except Exception as e:
            yield StreamEvent(event_type=StreamEventType.ERROR, error=e)
        finally:
            yield StreamEvent(event_type=StreamEventType.DONE)
