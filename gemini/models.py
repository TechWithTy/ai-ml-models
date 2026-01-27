"""
Defines the Gemini models available through the API.
"""
from enum import Enum

class GeminiModel(Enum):
    """! An enumeration of the available Gemini models."""
    GEMINI_3_0_FLASH = "gemini-3.0-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite-preview-02-05"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_PRO = "gemini-pro"
