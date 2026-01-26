"""
HeyGen Video Generation Wrapper.
Wraps HeyGen API for avatar video creation.
"""
import os
import sys
import asyncio
import httpx

# Add heygen submodule to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_root = os.path.dirname(current_dir)
heygen_path = os.path.join(ai_root, "heygen")

if heygen_path not in sys.path:
    sys.path.insert(0, heygen_path)

# Import BaseGenerator
sys.path.insert(0, ai_root)
from base_generator import BaseGenerator

class HeyGenGenerator(BaseGenerator):
    def __init__(self, api_key=None):
        super().__init__(provider="heygen", api_key=api_key)
        self.base_url = "https://api.heygen.com"
        
    async def generate_video(
        self, 
        prompt: str,
        avatar_id: str = None,
        voice_id: str = None,
        **kwargs
    ):
        """
        Generates an avatar video using HeyGen API v2.
        
        Args:
            prompt: Script/text for the avatar to speak
            avatar_id: Avatar ID (optional, uses default if not provided)
            voice_id: Voice ID (optional, uses default if not provided)
        """
        try:
            payload = {
                "video_inputs": [{
                    "character": {
                        "type": "avatar",
                        "avatar_id": avatar_id or "default",
                    },
                    "voice": {
                        "type": "text",
                        "input_text": prompt,
                        "voice_id": voice_id or "default"
                    }
                }],
                "dimension": kwargs.get("dimension", {"width": 1920, "height": 1080}),
                "test": kwargs.get("test", False)
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v2/video/generate",
                    json=payload,
                    headers={
                        "X-Api-Key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "status": "success",
                        "video_id": result.get("data", {}).get("video_id"),
                        "message": "Video generation started",
                        "provider": self.provider
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"API error: {response.status_code} - {response.text}",
                        "provider": self.provider
                    }
                    
        except Exception as e:
            return {"status": "error", "message": str(e), "provider": self.provider}
    
    async def check_status(self, video_id: str):
        """Check video generation status."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/v1/video_status.get",
                    params={"video_id": video_id},
                    headers={"X-Api-Key": self.api_key},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "status": "error", 
                        "message": f"Status check failed: {response.status_code}",
                        "provider": self.provider
                    }
                    
        except Exception as e:
            return {"status": "error", "message": str(e), "provider": self.provider}

