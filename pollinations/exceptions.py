class PollinationsError(Exception):
    """Base exception for Pollinations client errors."""
    pass

class APIError(PollinationsError):
    """Raised for API-specific errors."""
    pass

class AuthenticationError(PollinationsError):
    """Raised for authentication errors."""
    pass
