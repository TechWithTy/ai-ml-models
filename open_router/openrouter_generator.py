from src.ml_models.base_generator import BaseGenerator
from src.ml_models.open_router.utils.text.openrouter_message_utils import (
    send_chat_completion,
    send_chat_with_fallbacks,
    send_chat_with_any_route,
)

class OpenRouterGenerator(BaseGenerator):
    """
    OpenRouter generator supporting text, image, PDF, and structured output via modular utils.
    """
    def send_message(
        self,
        prompt_text: str = None,
        images: list = None,
        pdfs: list = None,
        response_schema: dict = None,
        plugins: list = None,
        **kwargs
    ):
        """
        Send a prompt with optional images, PDFs, and structured output schema.
        images: list of image URLs or local paths
        pdfs: list of local PDF paths
        response_schema: JSON schema dict for structured output
        plugins: OpenRouter plugin config (e.g., for PDF engine)
        """
        from src.ml_models.open_router.utils.openrouter_file_utils import build_image_content_entry, build_pdf_content_entry
        from src.ml_models.open_router.utils.openrouter_structured_utils import build_response_format_json_schema
        state = self.get_prompt_state()
        prompt = prompt_text or state["prompt"]
        content = []
        # Always add text first
        if prompt:
            content.append({"type": "text", "text": prompt})
        # Add images
        if images:
            for img in images:
                if img.startswith("http://") or img.startswith("https://"):
                    content.append(build_image_content_entry(image_url=img))
                else:
                    content.append(build_image_content_entry(image_path=img))
        # Add PDFs
        if pdfs:
            for pdf in pdfs:
                content.append(build_pdf_content_entry(pdf))
        messages = [{"role": "user", "content": content}] if content else []
        model = self.config.get("model", "openai/gpt-3.5-turbo")
        fallbacks = self.config.get("fallbacks", [])
        route = self.config.get("route", None)
        # Structured output
        response_format = None
        if response_schema:
            response_format = build_response_format_json_schema(response_schema)
        # Compose payload
        payload = {"messages": messages}
        if plugins:
            payload["plugins"] = plugins
        if response_format:
            payload["response_format"] = response_format
        payload.update(kwargs)  # Allow override of any param
        # Routing logic
        if route == "any":
            models = self.config.get("models", [model] + fallbacks)
            return send_chat_with_any_route(models, messages, **payload)
        elif fallbacks:
            return send_chat_with_fallbacks(model, messages, fallbacks, **payload)
        else:
            return send_chat_completion(model, messages, **payload)
