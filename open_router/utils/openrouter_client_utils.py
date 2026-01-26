import os
from openai import OpenAI

def get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
