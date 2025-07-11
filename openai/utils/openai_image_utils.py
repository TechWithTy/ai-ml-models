import requests
from typing import Optional
from utils.index import get_env_variable
from utils.config.config_loader import config

OPENAI_API_KEY: Optional[str] = get_env_variable("OPENAI_API_KEY")
openai_cfg = config.get("user_profile", {}).get("llm", {}).get("OpenAI", {})

def generate_openai_image(scoped_prompt: str) -> Optional[str]:
    """
    Generates an image using OpenAI's Image API (DALL·E).
    Configuration is pulled from YAML settings.
    """
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    model = openai_cfg.get("image_model", "dall-e-3")
    size = openai_cfg.get("image_size", "1024x1024")
    payload = {
        "model": model,
        "prompt": scoped_prompt,
        "n": 1,
        "size": size,
        "response_format": "url",
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        image_data = response.json().get("data", [])
        if image_data and "url" in image_data[0]:
            return image_data[0]["url"]
    else:
        print("❌ Failed to generate OpenAI image:", response.json())
    return None
