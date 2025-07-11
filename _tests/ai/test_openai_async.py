import sys
import os
import pytest
import warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ml_models.openai_generator import run_openai_pipeline
from utils.prompt_builder import init_globals_for_test

# Initialize the state
init_globals_for_test()

@pytest.mark.asyncio
async def test_run_openai_pipeline(initialized_prompt_state):
    prompt = initialized_prompt_state["prompt"]
    assert prompt is not None, "❌ Prompt is still None"
    
    result = run_openai_pipeline()

    assert isinstance(result, dict), "❌ Expected result to be a dictionary"
    print("\n=== 👁️OpenAi Pipeline Test ===")
    print("👁️OpenAi Status:", result.get("status"))
    print("👁️OpenAi Response:", result.get("response"))
    print("👁️OpenAi Status Code:", result.get("status_code"))
    print(f"👁️OpenAi Details: {result.get('details')}")

    # Check if the pipeline completed successfully
    if result.get("status") == "success":
        assert isinstance(result.get("response"), str), "❌ 'response' should be a string"
        assert len(result.get("response").strip()) > 0, "❌ 'response' should not be empty"
        print("\n✅ Assistant Output:\n", result.get("response"))
    else:
        # Log and handle failure cases
        print("❌ OpenAI Assistant returned an error.")

        # Check if the error is related to exceeding quota
        error_message = result.get("response", "")
        if "You exceeded your current quota" in error_message:
            warnings.warn("⚠️ Exceeded Quota: Please check your plan and billing details.")
            pytest.skip("Skipping test due to quota exceeded")
        else:
            # Log other error details
            print("🧾 Error Message:", error_message)
            pytest.fail(f"❌ OpenAI Assistant request did not succeed: {error_message}")
