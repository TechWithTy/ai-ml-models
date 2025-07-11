"""
Custom exceptions for the Gemini SDK.
"""

class GeminiError(Exception):
    """Base exception for all Gemini SDK errors."""
    pass

class AuthenticationError(GeminiError):
    """Raised when authentication with the Gemini API fails."""
    pass

class RateLimitError(GeminiError):
    """Raised when the rate limit for the Gemini API is exceeded."""
    pass

class APIError(GeminiError):
    """Raised when an unexpected API error occurs."""
    pass
