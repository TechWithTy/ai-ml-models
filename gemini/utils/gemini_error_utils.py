"""
Error handling utilities for the Gemini SDK.
"""
from ..exceptions import APIError, RateLimitError

def handle_api_error(e: Exception) -> None:
    """! Parses an exception and raises a specific Gemini API error."""
    error_message = str(e).lower()
    if "rate limit" in error_message or "429" in error_message:
        raise RateLimitError(f"Rate limit exceeded: {e}")
    raise APIError(f"An unexpected error occurred: {e}")
