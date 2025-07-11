from src.ml_models.claude.utils.claude_generator_utils import ClaudeGenerator

# * This file provides a backward-compatible interface for Claude generation.
# * Use ClaudeGenerator().send_message() to interact with the Claude API.

def send_message_to_claude() -> dict:
    """
    Sends a blog post to Claude AI using the refactored ClaudeGenerator class.
    Returns a structured result dictionary.
    """
    generator = ClaudeGenerator()
    return generator.send_message()
