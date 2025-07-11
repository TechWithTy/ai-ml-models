import sys
import os
import pytest
import warnings  # Import the warnings module

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml_models.claude_generator import send_message_to_claude
from utils.prompt_builder import init_globals_for_test, get_prompt_globals

# Initialize the state
init_globals_for_test()

# Get the shared global state
state = get_prompt_globals()

prompt = state["prompt"]
creative_prompt = state["creative_prompt"]
gif_prompt = state["gif_prompt"]
hashtags = state["hashtags"]
system_instructions = state["system_instructions"]
blog_content = state["blog_content"]

@pytest.mark.asyncio
async def test_send_message_to_claude():
  
    result = send_message_to_claude()

    assert isinstance(result, dict), "❌ Expected result to be a dictionary"
    print("\n===🎅 Claude Pipeline Test ===")
    print("🎅 Claude  Status:", result.get("status"))
    print("🎅 Claude  Response:", result.get("response"))
    print("🎅 Claude 📦 Message:", result.get("response").get("message"))  # Corrected line to extract message
    print("🎅 Claude  Status Code:", result.get("status_code"))

    # Check if the result indicates a successful response
    if result.get("status") == "success":
        assert isinstance(result.get("response"), str), "❌ Claude response should be a string"
        assert len(result.get("response").strip()) > 0, "❌ Claude response should not be empty"
    else:
        # Special case: check if the error is related to billing issues
        if result.get("response").get("message") == "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.":
            warnings.warn("⚠️ Claude test passed, but billing error encountered: Your credit balance is too low.")
            pytest.skip("Skipping test due to billing error")  # Skip the test with a message

        # Log full details on other errors
        print("❌ Claude returned an error.")
        print("🧾 Full Error Info:", result.get("raw"))
        pytest.fail(f"Claude failed: {result.get('response')}")  # Fail for other unexpected errors
