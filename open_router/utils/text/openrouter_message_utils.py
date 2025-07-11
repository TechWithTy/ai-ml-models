from ..openrouter_client_utils import get_openrouter_client

def send_chat_completion(model: str, messages: list, **kwargs):
    client = get_openrouter_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        return {"error": str(e)}

def send_chat_with_fallbacks(model: str, messages: list, fallbacks: list, **kwargs):
    client = get_openrouter_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            fallbacks=fallbacks,
            **kwargs
        )
        return response
    except Exception as e:
        return {"error": str(e)}

def send_chat_with_any_route(models: list, messages: list, **kwargs):
    client = get_openrouter_client()
    try:
        response = client.chat.completions.create(
            models=models,
            messages=messages,
            route="any",
            **kwargs
        )
        return response
    except Exception as e:
        return {"error": str(e)}
