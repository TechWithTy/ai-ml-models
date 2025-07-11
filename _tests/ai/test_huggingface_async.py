import sys
import os
import pytest
import warnings
from src.ml_models.huggingface_generator import run_huggingface_pipeline
from utils.prompt_builder import init_globals_for_test

"""
The function `test_run_huggingface_pipeline` is a pytest asynchronous test that checks
the output of a Hugging Face text generation pipeline for specific conditions and prints
the results for debugging and verification.
"""

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize the state
init_globals_for_test()

@pytest.mark.asyncio
async def test_run_huggingface_pipeline():
    # Run the pipeline and get the result
    result = run_huggingface_pipeline()

    # Check for subscription/balance issues
    if "Check Model or Balance / HF Subscription" in result.get("response", ""):
        warnings.warn("HuggingFace API subscription or balance issue detected. Please check your account status.")
        pytest.skip("⚠️ Skipping test: HuggingFace API subscription or balance issue detected")

    # Print debugging information
    print("\n=== Hugging Face Pipeline Test ===")
    print(f"🤗HF Status: {result.get('status')}")
    print(f"🤗HF Status Code: {result.get('status_code')}")
    print(f"🤗HF Response: {result.get('response')}")
    print(f"🤗HF Details: {result.get('details')}")
    
    # Basic validations
    assert isinstance(result, dict), "Expected pipeline to return a dict"
    assert "status_code" in result, "Expected status_code in response"
    assert "details" in result, "Expected details in response"
    assert result.get("status") == "success", f"❌ HuggingFace text generation failed: {result.get('response')}"

    # Validate response format
    assert isinstance(result.get("response"), str), "Expected 'response' to be a string"
    assert len(result.get("response").strip()) > 0, "Text response is empty"

    # Validate details contains the raw API response
    assert isinstance(result.get("details"), (dict, list)), "Expected 'details' to contain the raw API response"

    # If an image URL is returned, validate it
    if "image_url" in result:
        assert result["image_url"].startswith("http"), "Expected a valid image URL"
        print("\n🖼️ Hugging Face image:", result["image_url"])
