import requests
from typing import Optional, Dict, Any
from utils.index import get_env_variable
from utils.prompt_builder import get_prompt_globals
from utils.config.config_loader import config

OPENAI_API_KEY: Optional[str] = get_env_variable("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID: Optional[str] = get_env_variable("OPENAI_ASSISTANT_ID")

state = get_prompt_globals()
prompt = state["prompt"]
system_instructions = state["system_instructions"]
openai_config = config.get("user_profile", {}).get("llm", {}).get("OpenAI", {})

def create_openai_thread() -> Optional[str]:
    url = "https://api.openai.com/v1/threads"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    response = requests.post(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("id")
    else:
        print("❌ Error creating OpenAI thread:", response.json())
        return None

def send_message_to_openai(thread_id: str, message: str = None) -> bool:
    url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    msg = message or prompt
    data = {
        "role": "user",
        "content": msg,
        "attachments": [],
        "metadata": {"system_instructions": system_instructions},
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("✅ Message sent successfully!")
        return True
    else:
        print("❌ Error sending message:", response.json())
        return False

def get_openai_response(thread_id: str) -> Optional[str]:
    url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        messages = response.json().get("data", [])
        if messages:
            first_message = messages[0]
            content_list = first_message.get("content", [])
            if content_list and isinstance(content_list, list):
                text_content = content_list[0].get("text", {}).get("value")
                return text_content if isinstance(text_content, str) else None
    print("❌ Error fetching response:", response.json())
    return None
