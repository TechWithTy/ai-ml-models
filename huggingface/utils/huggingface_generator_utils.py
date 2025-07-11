from typing import Dict, Any
import requests
from src.ml_models.base_generator import BaseGenerator

class HuggingFaceGenerator(BaseGenerator):
    """
    Hugging Face generator using HuggingFace API. Inherits shared logic from BaseGenerator.
    """
    def __init__(self):
        super().__init__(provider="huggingface", config_section="HuggingFace")
        self.text_model = self.config.get("text_model", "mistralai/Mistral-7B-Instruct-v0.1")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 500)
        self.image_model = self.config.get("image_model", "runwayml/stable-diffusion-v1-5")
        self.image_height = self.config.get("image_height", 512)
        self.image_width = self.config.get("image_width", 512)

    def send_message(self, prompt_text: str = None) -> Dict[str, Any]:
        state = self.get_prompt_state()
        prompt = prompt_text or state["prompt"]
        if not prompt:
            return {"status": "error", "message": "No prompt given"}
        url = f"https://api-inference.huggingface.co/models/{self.text_model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
            },
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            # Hugging Face returns a list of dicts with 'generated_text'
            if isinstance(result, list) and result and "generated_text" in result[0]:
                return {"status": "success", "response": result[0]["generated_text"], "raw": result}
            return {"status": "error", "response": "Unexpected HuggingFace response format", "raw": result}
        except Exception as e:
            return {"status": "error", "response": str(e)}

    def generate_image(self, scoped_prompt: str) -> Dict[str, Any]:
        url = f"https://api-inference.huggingface.co/models/{self.image_model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "inputs": scoped_prompt,
            "options": {
                "wait_for_model": True
            },
            "parameters": {
                "height": self.image_height,
                "width": self.image_width
            }
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            # Image is returned as bytes (PNG/JPEG) or a URL, depending on the API
            return {"status": "success", "image": response.content}
        except Exception as e:
            return {"status": "error", "response": str(e)}
