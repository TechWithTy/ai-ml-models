"""
Message and prompt handling utilities for the Gemini SDK.
"""
from typing import Dict, Any

def create_text_part(text: str) -> Dict[str, Any]:
    """? Creates a part for text to be included in a prompt."""
    return {"text": text}
