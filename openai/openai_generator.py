from typing import Dict, Any
import requests
from src.ml_models.base_generator import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    """
    OpenAI generator using OpenAI API. Inherits shared logic from BaseGenerator.
    """
    def __init__(self):
        super().__init__(provider="openai", config_section="OpenAI")
        self.model = self.config.get("model", "gpt-4o")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 500)
        self.assistant_id = self.config.get("assistant_id", None)

    def send_message(self, prompt_text: str = None) -> Dict[str, Any]:
        """
        Sends a prompt to OpenAI using the modular utility pipeline (thread, message, assistant, wait, fetch response).
        Returns a structured result dictionary.
        """
        from src.ml_models.openai.utils import (
            create_openai_thread,
            send_message_to_openai,
            get_openai_response,
            create_openai_assistant,
            run_openai_assistant,
            wait_for_openai_response,
        )
        thread_id = create_openai_thread()
        if not thread_id:
            return {"status": "error", "response": "❌ Failed to create OpenAI thread."}
        ok = send_message_to_openai(thread_id, prompt_text)
        if not ok:
            return {"status": "error", "response": "❌ Failed to send message to OpenAI."}
        assistant_id = create_openai_assistant()
        run_id = run_openai_assistant(thread_id, assistant_id)
        if not run_id:
            return {"status": "error", "response": "❌ Failed to run OpenAI assistant."}
        error_details = wait_for_openai_response(thread_id, run_id)
        if error_details:
            return {
                "status": "failed",
                "response": error_details.get("last_error", {}).get(
                    "message", "Unknown failure."
                ),
                "details": error_details,
            }
        response_text = get_openai_response(thread_id)
        if response_text:
            return {"status": "success", "response": response_text}
        else:
            return {
                "status": "failed",
                "response": "❌ Assistant did not return a valid response.",
            }

    def generate_image(self, scoped_prompt: str) -> Dict[str, Any]:
        """
        Generates an image using OpenAI's DALL·E API via the utility function.
        Returns a dictionary with the image URL or error message.
        """
        from src.ml_models.openai.utils import generate_openai_image
        img_url = generate_openai_image(scoped_prompt)
        if img_url:
            return {"status": "success", "image_url": img_url}
        else:
            return {"status": "error", "response": "❌ Failed to generate OpenAI image."}
