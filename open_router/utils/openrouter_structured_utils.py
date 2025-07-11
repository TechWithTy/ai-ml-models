def build_response_format_json_schema(schema: dict, name: str = "structured_output", strict: bool = True, description: str = None) -> dict:
    """
    Returns a response_format parameter for OpenRouter structured output.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
            **({"description": description} if description else {})
        }
    }
