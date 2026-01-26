from typing import Dict, Any, Optional
from openai import OpenAI
from base_generator import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    """
    OpenAI generator using standard Chat Completions API.
    Supports o1 models, reasoning_effort, and structured outputs.
    """
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(provider="openai", api_key=api_key, config_section="OpenAI", **kwargs)
        self.client = OpenAI(api_key=self.api_key)
        self.model = self.config.get("model", "gpt-4o")
        self.max_tokens = self.config.get("max_tokens", 1000)

    def send_message(self, prompt_text: str = None, **kwargs) -> Dict[str, Any]:
        """
        Sends a chat completion request.
        Supports:
        - reasoning_effort (for o1 models)
        - response_format (json_schema)
        - audio (input and output)
        """
        import base64
        
        model = kwargs.get("model", self.model)
        
        # Handle o1 specific parameters
        is_o1 = model.startswith("o1")
        reasoning_effort = kwargs.get("reasoning_effort", "medium") if is_o1 else None
        
        # Construct messages
        messages = []
        
        # Check for audio input
        audio_input_path = kwargs.get("audio_input")
        if audio_input_path:
             try:
                 with open(audio_input_path, "rb") as audio_file:
                     encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
                     
                 # Multimodal content block
                 content_block = [
                     {"type": "text", "text": prompt_text or "Process this audio."},
                     {
                         "type": "input_audio", 
                         "input_audio": {
                             "data": encoded_string, 
                             "format": "wav" # Assuming wav for now, could be passed in
                         }
                     }
                 ]
                 messages.append({"role": "user", "content": content_block})
             except Exception as e:
                 return {"status": "error", "message": f"Failed to read audio file: {str(e)}"}
        else:
             messages.append({"role": "user", "content": prompt_text})

        params = {
            "model": model,
            "messages": messages,
        }
        
        if is_o1:
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            if "max_tokens" in kwargs:
                params["max_completion_tokens"] = kwargs.get("max_tokens")
        else:
            params["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)
            
        # Audio Output request
        if kwargs.get("output_audio"):
             params["modalities"] = ["text", "audio"]
             params["audio"] = {"voice": "alloy", "format": "wav"}

        # Pass through other kwargs
        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]

        try:
            response = self.client.chat.completions.create(**params)
            
            result = {"status": "success"}
            
            # content might be None if purely audio output? 
            # Usually text is present or in audio transcript.
            message = response.choices[0].message
            
            if message.content:
                result["response"] = message.content
                
            if hasattr(message, 'audio') and message.audio:
                 result["audio_transcript"] = message.audio.transcript
                 result["audio_id"] = message.audio.id
                 # We might want to return the base64 audio data if available, 
                 # but standard object might just have ID/transcript.
                 # Wait, 'data' field usually comes in the audio response object (base64).
                 if hasattr(message.audio, 'data'):
                      result["audio_data"] = message.audio.data # Base64 string
            
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_audio(self, text: str, voice: str = "alloy") -> Dict[str, Any]:
        """
        Generates audio from text (TTS) using OpenAI's Audio API.
        """
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            # Response is binary stream
            return {"status": "success", "audio_content": response.content}
        except Exception as e:
             return {"status": "error", "message": str(e)}

    def generate_image(self, scoped_prompt: str) -> Dict[str, Any]:
        """
        Generates an image using OpenAI's DALL·E API via the utility function.
        Returns a dictionary with the image URL or error message.
        """
        from .utils import generate_openai_image
        img_url = generate_openai_image(scoped_prompt)
        if img_url:
            return {"status": "success", "image_url": img_url}
        else:
            return {"status": "error", "response": "❌ Failed to generate OpenAI image."}
