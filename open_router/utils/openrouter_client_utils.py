import os
from openrouter import OpenRouter

def get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")
    return OpenRouter(api_key=api_key)
