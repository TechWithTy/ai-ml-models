"""
Defines the Gemini models available through the API.
"""
from enum import Enum

class GeminiModel(Enum):
    """! An enumeration of the available Gemini models."""
    GEMINI_2_5_PRO = "gemini-1.5-pro-latest"
    GEMINI_2_5_FLASH = "gemini-1.5-flash-latest"
    GEMINI_2_5_FLASH_LITE = "gemini-1.5-flash-lite-latest" # Not a real model, for testing
