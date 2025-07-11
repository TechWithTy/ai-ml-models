from ml_models.deepseek.utils.deepseek_generator_utils import DeepSeekGenerator

def send_message_to_deepseek() -> dict:
    """
    Sends a blog post to DeepSeek AI using the refactored DeepSeekGenerator class.
    Returns a structured result dictionary.
    """
    generator = DeepSeekGenerator()
    return generator.send_message()
