"""
Image handling utilities for the Gemini SDK.
"""
import base64
from pathlib import Path
from typing import Dict, Any

def load_image_as_base64(image_path: Path) -> str:
    """! Loads an image file and returns it as a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_part(mime_type: str, data: str) -> Dict[str, Any]:
    """? Creates a part for an image to be included in a prompt."""
    return {"inline_data": {"mime_type": mime_type, "data": data}}
