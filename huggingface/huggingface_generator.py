from ml_models.huggingface.utils.huggingface_generator_utils import HuggingFaceGenerator

def send_message_to_huggingface(prompt_text: str = None) -> dict:
    """
    Sends a prompt to Hugging Face using the refactored HuggingFaceGenerator class.
    Returns a structured result dictionary.
    """
    generator = HuggingFaceGenerator()
    return generator.send_message(prompt_text)

def generate_image_with_huggingface(scoped_prompt: str) -> dict:
    """
    Generates an image using Hugging Face's image model.
    """
    generator = HuggingFaceGenerator()
    return generator.generate_image(scoped_prompt)
