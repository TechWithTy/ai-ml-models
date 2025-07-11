"""
Authentication utilities for the Gemini SDK.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_gemini_api_key() -> str | None:
    """! Retrieves the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")
