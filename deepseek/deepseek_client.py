import httpx
import json
from typing import Any, AsyncGenerator, Dict, Optional

from ..base_generator import BaseGenerator
from ..base_streaming_generator import BaseStreamingGenerator, StreamEvent, StreamEventType

class DeepSeekClient(BaseGenerator, BaseStreamingGenerator):
    """Client for interacting with the DeepSeek API, with async and streaming support."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(provider="deepseek", api_key=api_key, config_section="DeepSeek", **kwargs)
        self.base_url = self.config.get("base_url", "https://api.deepseek.com/v1")
        self.text_model = self.config.get("text_model", "deepseek-chat")

    async def generate_text_async(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate text asynchronously."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.text_model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=30.0)
                response.raise_for_status()
                return {"status": "success", "response": response.json()}
            except httpx.HTTPStatusError as e:
                return {"status": "error", "message": f"HTTP Error: {e.response.status_code}", "details": e.response.text}
            except Exception as e:
                return {"status": "error", "message": "An unexpected error occurred", "details": str(e)}

    async def stream_text(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[StreamEvent[str], None]:
        """Stream text generation using Server-Sent Events."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.text_model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": True
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, headers=headers, json=payload, timeout=30.0) as response:
                    yield StreamEvent(event_type=StreamEventType.START)
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[len('data: '):]
                            if data_str.strip() == '[DONE]':
                                break
                            chunk = json.loads(data_str)
                            content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                yield StreamEvent(event_type=StreamEventType.CONTENT, data=content)
        except Exception as e:
            yield StreamEvent(event_type=StreamEventType.ERROR, error=e)
        finally:
            yield StreamEvent(event_type=StreamEventType.DONE)
