from pathlib import Path
from utils.helpers.file.image import encode_file_to_base64

def build_image_content_entry(image_url: str = None, image_path: str = None) -> dict:
    """
    Returns a content entry for OpenRouter image input. Accepts either a URL or a local file path.
    """
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}
    elif image_path:
        ext = Path(image_path).suffix.lower().replace('.', '')
        mime = f"image/{ext if ext in ['png', 'jpeg', 'webp'] else 'jpeg'}"
        base64_img = encode_file_to_base64(image_path)
        data_url = f"data:{mime};base64,{base64_img}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    else:
        raise ValueError("Must provide either image_url or image_path.")

def build_pdf_content_entry(pdf_path: str, filename: str = None) -> dict:
    base64_pdf = encode_file_to_base64(pdf_path)
    data_url = f"data:application/pdf;base64,{base64_pdf}"
    return {
        "type": "file",
        "file": {
            "filename": filename or Path(pdf_path).name,
            "file_data": data_url
        }
    }
