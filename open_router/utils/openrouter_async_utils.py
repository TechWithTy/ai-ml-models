import os
from openrouter import AsyncOpenRouter

def get_async_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")
    return AsyncOpenRouter(api_key=api_key)
