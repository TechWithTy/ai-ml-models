class DeepSeekError(Exception):
    """Base exception for DeepSeek client errors."""
    pass

class APIError(DeepSeekError):
    """Raised for API-specific errors."""
    pass

class AuthenticationError(DeepSeekError):
    """Raised for authentication errors."""
    pass
