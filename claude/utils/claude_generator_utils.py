from typing import Optional, Dict, Any
import requests
from base_generator import BaseGenerator

class ClaudeGenerator(BaseGenerator):
    """
    Claude AI generator using Anthropic API. Inherits shared logic from BaseGenerator.
    """
    def __init__(self):
        super().__init__(provider="anthropic", config_section="Anthropic")
        self.claude_model = self.config.get("text_model", "claude-3-sonnet")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 500)

    def send_message(self, prompt_text: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Sends a blog post to Claude AI and returns a structured result.
        """
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        state = self.get_prompt_state()
        prompt = prompt_text or state.get("prompt")
        # Use provided model or config default (updated default to 3.5 Sonnet)
        target_model = model or self.claude_model or "claude-3-5-sonnet-latest"
        
        # Handle prompt caching
        anthropic_headers = headers.copy()
        messages_payload = []
        
        if kwargs.get("enable_caching"):
             anthropic_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
             
             # Cache the system prompt if provided
             if system_instructions:
                  pass # System caching structure is different, keeping simple for now
             
             # Mark the user content for caching (ephemeral)
             messages_payload.append({
                 "role": "user", 
                 "content": [
                     {
                         "type": "text", 
                         "text": prompt, 
                         "cache_control": {"type": "ephemeral"}
                     }
                 ]
             })
        else:
             messages_payload = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ]
        
        # Adjust for 'system' role separation in modern API if needed, 
        # normally system is a top-level param, but messages list works for simple cases.
        # Actually, modern Anthropic API prefers 'system' as top level param, not in messages.
        # Let's fix that while we are here.
        
        data = {
            "model": target_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_instructions,
            "messages": [
                 {"role": "user", 
                  "content": [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}} if kwargs.get("enable_caching") else prompt}
            ]
        }
        
        try:
            response = requests.post(url, headers=anthropic_headers, json=data)
            try:
                response_json = response.json()
            except Exception as e:
                print("❌ Failed to parse Claude response JSON:", e)
                return {
                    "status": "error",
                    "response": "Invalid JSON response",
                    "status_code": response.status_code,
                    "raw": response.text,
                }
            if response.status_code == 200:
                content_blocks = response_json.get("content", [])
                if content_blocks and isinstance(content_blocks, list):
                    text = content_blocks[0].get("text", "").strip()
                    return {
                        "status": "success",
                        "response": text,
                        "status_code": 200,
                        "raw": response_json,
                    }
                else:
                    return {
                        "status": "error",
                        "response": "Claude returned an unexpected content format.",
                        "status_code": 200,
                        "raw": response_json,
                    }
            else:
                return {
                    "status": "failed",
                    "response": response_json.get("error", "Unknown error from Claude API"),
                    "status_code": response.status_code,
                    "raw": response_json,
                }
        except requests.RequestException as req_err:
            print("❌ Request error while sending to Claude:", str(req_err))
            return {
                "status": "error",
                "response": "Request exception occurred",
                "status_code": None,
                "raw": str(req_err),
            }
