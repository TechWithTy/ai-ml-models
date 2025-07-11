import requests
from typing import Optional
from utils.index import get_env_variable
from utils.prompt_builder import get_prompt_globals
from utils.config.config_loader import config

OPENAI_API_KEY: Optional[str] = get_env_variable("OPENAI_API_KEY")
openai_config = config.get("user_profile", {}).get("llm", {}).get("OpenAI", {})

def create_openai_assistant() -> str:
    url = "https://api.openai.com/v1/assistants"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    assistant_name = openai_config.get("name", "LinkedIn Content Assistant")
    model = openai_config.get("model", "gpt-4o")
    instructions = openai_config.get(
        "custom_system_instructions", "You are a professional copywriter..."
    )
    temperature = openai_config.get("temperature", 1.0)
    top_p = openai_config.get("top_p", 1.0)
    data = {
        "name": assistant_name,
        "model": model,
        "instructions": instructions,
        "temperature": temperature,
        "top_p": top_p,
        "response_format": "auto",
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        assistant_id = response.json().get("id")
        print(f"✅ OpenAI Assistant created: {assistant_id} ⚠️Save this to .env or github secrets to reuse")
        return assistant_id
    else:
        print("❌ Error creating OpenAI Assistant:", response.json())
        raise RuntimeError("Failed to create OpenAI Assistant.")

def run_openai_assistant(thread_id: str, assistant_id: str) -> Optional[str]:
    url = f"https://api.openai.com/v1/threads/{thread_id}/runs"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    data = {"assistant_id": assistant_id}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("id")
    else:
        print("❌ Error running assistant:", response.json())
        return None

def wait_for_openai_response(thread_id: str, run_id: str) -> Optional[dict]:
    import time
    url = f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    while True:
        response = requests.get(url, headers=headers)
        data = response.json()
        status = data.get("status", "")
        if status == "completed":
            return None
        elif status in ["failed", "cancelled"]:
            print("❌ Assistant failed:", data)
            return data
        time.sleep(5)
