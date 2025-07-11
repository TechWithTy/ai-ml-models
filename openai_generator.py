import sys

import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional
import requests
import time
import os
from dotenv import load_dotenv
from utils.index import get_env_variable
from utils.prompt_builder import get_prompt_globals, init_globals_for_test
from utils.config.config_loader import config

# ✅ Load environment variables
load_dotenv()

env_path = ".env"

OPENAI_API_KEY: Optional[str] = get_env_variable("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID: Optional[str] = get_env_variable("OPENAI_ASSISTANT_ID")


if not OPENAI_API_KEY:
    raise ValueError(
        "❌ OPENAI_API_KEY is missing! Set it in your .env file or GitHub Secrets."
    )
TEST_MODE = get_env_variable("TEST_MODE").lower() == "true"

if TEST_MODE:
    init_globals_for_test()
state = get_prompt_globals()

prompt = state["prompt"]
creative_prompt = state["creative_prompt"]
gif_prompt = state["gif_prompt"]
hashtags = state["hashtags"]
system_instructions = state["system_instructions"]
blog_content = state["blog_content"]
openai_config = config.get("user_profile", {}).get("llm", {}).get("OpenAI", {})


print(prompt, "OpenAI Scoped Prompt")


def create_openai_assistant() -> str:
    """Creates a new OpenAI Assistant using YAML configuration if none exists."""
    url = "https://api.openai.com/v1/assistants"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    # Get the shared global state

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
        print(
            f"✅ OpenAI Assistant created: {assistant_id} ⚠️Save this to .env or github secrets to reuse"
        )
        return assistant_id
    else:
        print("❌ Error creating OpenAI Assistant:", response.json())
        raise RuntimeError("Failed to create OpenAI Assistant.")


# ✅ Ensure an Assistant ID exists
if not OPENAI_ASSISTANT_ID:
    print("🔹 No Assistant ID found. Creating one now...")
    OPENAI_ASSISTANT_ID = create_openai_assistant()


def create_openai_thread() -> Optional[str]:
    """Creates an OpenAI Assistant thread and returns the thread ID."""
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


def send_message_to_openai(
    thread_id: str,
) -> None:
    """Sends a blog post to OpenAI Assistant with instructions, creative cues, and viral preferences from config."""
    url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }

    # 🔧 Config sections

    # 📨 Final Message Content

    print("Message Content:\n\n", prompt)
    data = {
        "role": "user",
        "content": prompt,
        "attachments": [],
        "metadata": {"system_instructions": system_instructions},
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("✅ Message sent successfully!")
    else:
        print("❌ Error sending message:", response.json())


def run_openai_assistant(thread_id: str) -> Optional[str]:
    """Runs the OpenAI Assistant and returns the run ID."""
    url = f"https://api.openai.com/v1/threads/{thread_id}/runs"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    data = {"assistant_id": OPENAI_ASSISTANT_ID}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get("id")
    else:
        print("❌ Error running assistant:", response.json())
        return None


def wait_for_openai_response(thread_id: str, run_id: str) -> Optional[dict]:
    """Waits for OpenAI Assistant to process the request and returns failure details if any."""
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
            return None  # No issues
        elif status in ["failed", "cancelled"]:
            print("❌ Assistant failed:", data)
            return data  # 👈 Return full error response

        time.sleep(5)


def get_openai_response(thread_id: str) -> Optional[str]:
    """Fetches the response from OpenAI Assistant and returns the text content."""
    url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2",
    }
    system_instructions = config.get("ai", {}).get("custom_system_instructions") or (
        "You're a professional copywriter helping turn blog posts into viral LinkedIn content."
    )
    print("system_instructions", system_instructions)
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

    openai_cfg = config.get("user_profile", {}).get("llm", {}).get("OpenAI", {})
    model = openai_cfg.get("image_model", "dall-e-3")
    size = openai_cfg.get(
        "image_size", "1024x1024"
    )  # You can add this to YAML if not present

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


from src.ml_models.openai.openai_generator import OpenAIGenerator

def send_message_to_openai(prompt_text: str = None) -> dict:
    """
    Sends a prompt to OpenAI using the refactored OpenAIGenerator class.
    Returns a structured result dictionary.
    """
    generator = OpenAIGenerator()
    return generator.send_message(prompt_text)

    try:
        thread_id = create_openai_thread()
        if not thread_id:
            return {"status": "error", "response": "❌ Failed to create OpenAI thread."}

        send_message_to_openai(thread_id)
        run_id = run_openai_assistant(thread_id)
        if not run_id:
            return {"status": "error", "response": "❌ Failed to run OpenAI assistant."}

        error_details = wait_for_openai_response(thread_id, run_id)

        if error_details:
            return {
                "status": "failed",
                "response": error_details.get("last_error", {}).get(
                    "message", "Unknown failure."
                ),
                "details": error_details,  # Optional: include full payload
            }

        response_text = get_openai_response(thread_id)

        if response_text:
            return {"status": "success", "response": response_text}
        else:
            return {
                "status": "failed",
                "response": "❌ Assistant did not return a valid response.",
            }

    except Exception as e:
        return {
            "status": "error",
            "response": f"❌ Exception during pipeline: {str(e)}",
        }
