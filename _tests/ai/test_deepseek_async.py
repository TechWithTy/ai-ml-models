import sys
import os
import pytest
import warnings

# Add src to PYTHONPATH for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml_models.deepseek_generator import send_message_to_deepseek
from utils.prompt_builder import init_globals_for_test

# Initialize the state
init_globals_for_test()

@pytest.mark.asyncio
async def test_send_message_to_deepseek():
    result = send_message_to_deepseek()

    assert isinstance(result, dict), "❌ Expected result to be a dictionary"

    print("\n=== DeepSeek Pipeline Test ===")
    print("\n🦈 DeepSeek Result:")
    print("🦈 DeepSeek Status:", result.get("status"))
    print("🦈 DeepSeek Response:", result.get("response"))
    print("🦈 DeepSeek Details:", result.get("details"))

    # Check if the request was successful
    if result.get("status") == "success":
        response_text = result.get("response")
        assert isinstance(response_text, str), "❌ 'response' should be a string"
        assert len(response_text.strip()) > 0, "❌ 'response' should not be empty"
        print("\n🧠 DeepSeek AI Response:\n", response_text)
    else:
        # Log and handle failure cases
        print("⚠️ DeepSeek returned a failure")

        # Check if the error is related to insufficient balance
        error_message = result.get("details", {}).get("raw", {}).get("error", {}).get("message")
        if "Insufficient Balance" in error_message:
            warnings.warn("⚠️ Insufficient Balance: Please add credits to your account.")
            pytest.skip("Skipping test due to insufficient balance")
        else:
            # Log other error details
            print("🧾 Error Message:", error_message)
            pytest.fail(f"❌ DeepSeek request did not succeed: {error_message}")
