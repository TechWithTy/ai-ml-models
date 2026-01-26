
import os
import sys
import asyncio
import json

# Ensure 'kling' submodule is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_root = os.path.dirname(current_dir)
kling_submodule_path = os.path.join(ai_root, "kling")

if kling_submodule_path not in sys.path:
    sys.path.insert(0, kling_submodule_path)

# Import BaseGenerator
sys.path.insert(0, ai_root)
from base_generator import BaseGenerator

# Try imports
try:
    from client import KlingClient
    from api.text_to_video import TextToVideoRequest
    from api.image_to_video import ImageToVideoRequest
    KLING_AVAILABLE = True
except ImportError as e:
    KLING_AVAILABLE = False
    IMPORT_ERROR = str(e)

class KlingGenerator(BaseGenerator):
    def __init__(self, api_key=None):
        super().__init__(provider="kling", api_key=api_key)
        
        if not KLING_AVAILABLE:
            raise ImportError(f"Kling SDK not available: {IMPORT_ERROR}")
            
        self.client = KlingClient(api_key=self.api_key)

    async def generate_video(self, prompt, duration=5.0, resolution="1920x1080", image_url=None):
        """
        Generates a video from text or image.
        """
        try:
            if image_url:
                request = ImageToVideoRequest(
                    prompt=prompt or "Animate this image",
                    image=image_url,
                    duration=duration
                )
                response = await self.client.image_to_video(request)
            else:
                request = TextToVideoRequest(
                    prompt=prompt,
                    duration=duration,
                    resolution=resolution
                )
                response = await self.client.text_to_video(request)
            
            return {
                "status": "success", 
                "task_id": response.task_id, 
                "message": "Video generation started. Check email or implement pooling callback."
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "provider": self.provider}

    async def check_status(self, task_id):
         # Not implemented in snippet, but likely needed.
         pass
