import httpx
import asyncio
import urllib.parse
from typing import Optional, Dict, Any

from typing import AsyncGenerator
from base_generator import BaseGenerator
from base_streaming_generator import BaseStreamingGenerator, StreamEvent, StreamEventType

class PollinationsClient(BaseGenerator, BaseStreamingGenerator):
    """Client for interacting with the Pollinations API."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(provider="pollinations", api_key=api_key, config_section="Pollinations", **kwargs)
        self.base_url = "https://gen.pollinations.ai"
        self.max_retries = self.config.get("max_retries", 3)
    def _get_headers(self) -> Dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def generate_image(
        self, 
        prompt: str, 
        model: str = "flux",
        width: int = 1024,
        height: int = 1024,
        seed: int = 42,
        nologo: bool = True,
        private: bool = True,
        enhance: bool = False,
        safe: bool = False
    ) -> Dict[str, Any]:
        # New API: GET https://gen.pollinations.ai/image/{prompt}?model=flux...
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"{self.base_url}/image/{encoded_prompt}"
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
        
        headers = self._get_headers()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, headers=headers, timeout=30.0)
                if response.status_code == 200:
                     return {"status": "success", "url": str(response.url)}
                else:
                     return {"status": "error", "message": f"HTTP Error: {response.status_code}", "details": response.text}
            except httpx.HTTPStatusError as e:
                return {"status": "error", "message": f"HTTP Error: {e.response.status_code}", "details": e.response.text}
            except Exception as e:
                return {"status": "error", "message": "An unexpected error occurred", "details": str(e)}

    async def generate_video(
        self, 
        prompt: str, 
        model: str = "videoalpha",
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Generate video using Pollinations API.
        Assumes /video endpoint or similar structure.
        """
        encoded_prompt = urllib.parse.quote(prompt)
        # Speculative endpoint based on user request and existing pattern
        # /video returned 404. /image on gen.pollinations.ai returned 404 (or error).
        # Trying legacy endpoint: image.pollinations.ai/prompt/...
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        params = {
            "model": model,
            "seed": seed
        }
        
        headers = self._get_headers()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, headers=headers, timeout=60.0)
                if response.status_code == 200:
                     return {"status": "success", "url": str(response.url)}
                else:
                     return {"status": "error", "message": f"HTTP Error: {response.status_code}", "details": response.text}
            except httpx.HTTPStatusError as e:
                return {"status": "error", "message": f"HTTP Error: {e.response.status_code}", "details": e.response.text}
            except Exception as e:
                return {"status": "error", "message": "An unexpected error occurred", "details": str(e)}

    async def generate_text(
        self, 
        prompt: str, 
        model: str = "openai"
    ) -> Dict[str, Any]:
        # New API: POST https://gen.pollinations.ai/v1/chat/completions
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload, headers=headers, timeout=30.0)
                response.raise_for_status()
                # Parse OpenAI-compatible response
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return {"status": "success", "response": content}
            except httpx.HTTPStatusError as e:
                return {"status": "error", "message": f"HTTP Error: {e.response.status_code}", "details": e.response.text}
            except Exception as e:
                return {"status": "error", "message": "An unexpected error occurred", "details": str(e)}

    # Audio endpoint wasn't clearly detailed in the quick summary but sticking to legacy or assumed pattern if needed?
    # Schema doesn't explicitly list /audio endpoint, but /text endpoint mentions openai-audio model.
    # Let's temporarily disable or leave audio as legacy/best-effort
    async def generate_audio(
        self, 
        prompt: str, 
        voice: str = "nova"
    ) -> Dict[str, Any]:
         return {"status": "error", "message": "Audio generation update pending verification"}

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
