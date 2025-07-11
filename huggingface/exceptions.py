#!/usr/bin/env python
# src/ml_models/huggingface/exceptions.py

"""Custom exceptions for the Hugging Face client."""

class HuggingFaceError(Exception):
    """Base exception for Hugging Face client errors."""
    pass

class HuggingFaceApiError(HuggingFaceError):
    """Raised for API-related errors."""
    pass

class HuggingFaceAuthError(HuggingFaceError):
    """Raised for authentication errors."""
    pass
