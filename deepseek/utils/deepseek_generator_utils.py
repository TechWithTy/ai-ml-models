from typing import Dict, Any
import requests
from src.ml_models.base_generator import BaseGenerator

class DeepSeekGenerator(BaseGenerator):
    """
    DeepSeek AI generator using the DeepSeek API. Inherits shared logic from BaseGenerator.
    """
    def __init__(self):
        super().__init__(provider="deepseek", config_section="DeepSeek")
        self.deepseek_model = self.config.get("text_model", "deepseek-chat")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 500)
        self.top_p = self.config.get("top_p", 0.95)
        self.frequency_penalty = self.config.get("frequency_penalty", 0.0)
        self.presence_penalty = self.config.get("presence_penalty", 0.0)
        self.response_format = self.config.get("response_format", "json_object")
        self.tool_choice = self.config.get("tool_choice", "auto")
        self.logprobs = self.config.get("logprobs", False)
        self.top_logprobs = self.config.get("top_logprobs", 5)
        self.tools = self.config.get("tools", "function")

    def send_message(self) -> Dict[str, Any]:
        state = self.get_prompt_state()
        prompt = state["prompt"]
        system_instructions = state["system_instructions"]
        if not prompt:
            print("No Prompt Given DeepSeek")
            return {"status": "error", "message": "No prompt given"}
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.deepseek_model,
            "messages": [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "response_format": {"type": self.response_format},
            "tool_choice": self.tool_choice,
            "logprobs": self.logprobs,
        }
        if self.tools == "function":
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "placeholder_function",
                        "description": "A placeholder function as required by the DeepSeek API.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        if self.logprobs:
            payload["top_logprobs"] = self.top_logprobs
        try:
            response = requests.post(url, headers=headers, json=payload)
            try:
                response_data = response.json()
            except Exception as e:
                print("❌ Failed to parse JSON:", e)
                return {
                    "status": "error",
                    "response": "Invalid JSON response",
                    "details": {
                        "status_code": response.status_code,
                        "text": response.text,
                        "error": str(e),
                    },
                }
            if response.status_code == 200:
                completion = (
                    response_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if completion == prompt.strip():
                    return {
                        "status": "failed",
                        "response": "Completion is identical to prompt. Likely an error.",
                        "details": {"status_code": response.status_code, "raw": response_data},
                    }
                return {
                    "status": "success",
                    "response": completion,
                    "details": {"status_code": response.status_code, "raw": response_data},
                }
            return {
                "status": "failed",
                "response": response_data.get("error", "Unknown error from DeepSeek"),
                "details": {"status_code": response.status_code, "raw": response_data},
            }
        except requests.RequestException as req_err:
            print("❌ Request error while sending to DeepSeek:", str(req_err))
            return {
                "status": "error",
                "response": "Request exception occurred",
                "details": {"status_code": None, "raw": str(req_err)},
            }
