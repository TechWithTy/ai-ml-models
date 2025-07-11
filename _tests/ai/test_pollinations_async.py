import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml_models.pollinations_generator import (
    list_image_models,
    generate_image,
    generate_text_advanced,
    generate_text,
    generate_audio,
    list_text_models,
    call_openai_compatible_endpoint
)
from utils.prompt_builder import init_globals_for_test

# Initialize the state
init_globals_for_test()

# Add src path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Constants for tests
TEST_PROMPT = "hello world"
TEST_IMAGE_PROMPT = "a futuristic city skyline at sunset"
TEST_PAYLOAD = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is artificial intelligence?"}
        ],
        "model": "openai",
        "seed": 42,
        "jsonMode": True,  
        "private": True,  
        "reasoning_effort": "high" 
    }



@pytest.mark.asyncio
async def test_generate_image():
    result = await generate_image(TEST_IMAGE_PROMPT)
    
    assert result is not None, "❌ Expected a non-null response"
    assert isinstance(result, dict), "❌ Expected result to be a dictionary"
    assert "status" in result, "❌ Response missing 'status' field"
    assert "status_code" in result, "❌ Response missing 'status_code' field"
    assert "response" in result, "❌ Response missing 'response' field"
    
    print("\n=== 🪷 Pollinations:Generate Image Test ===")
    print("🪷 Pollinations:Generate Image Status:", result.get("status"))
    print("🪷 Pollinations:Generate Image Response:", result.get("response"))
    print("🪷 Pollinations:Generate Image Status Code:", result.get("status_code"))
    print(f"🪷 Pollinations:Generate Image Details: {result.get('details')}")
    
    if result["status"] == "success":
        assert result["status_code"] in [200, 201, 202], "❌ Expected successful status code"
        assert isinstance(result["response"], str), "❌ Expected response to be a string"
        assert result["response"].startswith("http"), "❌ Image URL should start with http"
        print(f"✅ generate_image passed with URL: {result['response']}")
    else:
        print(f"⚠️ generate_image returned non-success status: {result['status']}")
        print(f"⚠️ Details: {result['details']}")


@pytest.mark.asyncio
async def test_list_image_models():
    result = await list_image_models()
    assert result is not None, "u274c No response returned by list_image_models"
    assert isinstance(result, dict), "u274c Expected a dictionary response"
    assert "status" in result, "u274c Response missing 'status' field"
    assert "status_code" in result, "u274c Response missing 'status_code' field"
    assert "response" in result, "u274c Response missing 'response' field"
    
    print("\n=== 🚀 List Image Models Test ===")
    print("🚀 List Image Models Status:", result.get("status"))
    print("🚀 List Image Models Status Code:", result.get("status_code"))
    print(f"🚀 List Image Models Details: {str(result.get('details'))[:100]}...")
    
    # Check if the response was successful
    if result["status"] == "success":
        assert result["status_code"] == 200, "u274c Expected status code 200 for success"
        assert isinstance(result["response"], (list, dict)), "u274c Expected response to be a list or dictionary"
        print(f"✅ list_image_models returned response: {str(result['response'])[:100]}...")
    else:
        print(f"⚠️ list_image_models returned non-success status: {result['status']}")
        print(f"⚠️ Details: {result['details']}")


@pytest.mark.asyncio
async def test_generate_text_advanced():
    result = await generate_text_advanced(TEST_PAYLOAD)

    assert result is not None, "❌ Expected a non-null response"
    assert isinstance(result, dict), "❌ Expected result to be a dictionary"
    assert "status" in result, "❌ Response missing 'status' field"
    assert "status_code" in result, "❌ Response missing 'status_code' field"
    assert "response" in result, "❌ Response missing 'response' field"
    
    print("\n=== 🪷 Pollinations:Generate Text Advanced Test ===")
    print("🪷 Pollinations:Generate Text Advanced Status:", result.get("status"))
    print("🪷 Pollinations:Generate Text Advanced Response:", result.get("response")[:100], "...")
    print("🪷 Pollinations:Generate Text Advanced Status Code:", result.get("status_code"))
    print(f"🪷 Pollinations:Generate Text Advanced Details: {str(result.get('details'))[:100]}...")
    
    if result["status"] == "success":
        assert result["status_code"] == 200, "❌ Expected status code 200 for success"
        assert isinstance(result["response"], str), "❌ Expected response to be a string"
        assert result["response"].strip(), "❌ Response string should not be empty"
        print("✅ generate_text_advanced passed.")
    else:
        print(f"⚠️ generate_text_advanced returned non-success status: {result['status']}")
        print(f"⚠️ Details: {result['details']}")


@pytest.mark.asyncio
async def test_generate_text():
    prompt = "What is the capital of France?"
    result = await generate_text(prompt)

    assert result is not None, "🪷 Pollinations:Expected a non-null response"
    assert isinstance(result, dict), "🪷 Pollinations:Expected result to be a dictionary"
    assert "status" in result, "🪷 Pollinations:Response missing 'status' field"
    assert "status_code" in result, "🪷 Pollinations:Response missing 'status_code' field"
    assert "response" in result, "🪷 Pollinations:Response missing 'response' field"
    
    print("\n=== 🪷 Pollinations:Generate Text Test ===")
    print("🪷 Pollinations:Generate Text Status:", result.get("status"))
    print("🪷 Pollinations:Generate Text Response:", result.get("response")[:100], "...")
    print("🪷 Pollinations:Generate Text Status Code:", result.get("status_code"))
    print(f"🪷 Pollinations:Generate Text Details: {str(result.get('details'))[:100]}...")
    
    if result["status"] == "success":
        assert result["status_code"] == 200, "🪷 Pollinations:Expected status code 200 for success"
        assert isinstance(result["response"], str), "🪷 Pollinations:Expected response to be a string"
        assert result["response"].strip(), "🪷 Pollinations:Response string should not be empty"
        print("✅ generate_text passed.")
    else:
        print(f"⚠️ generate_text returned non-success status: {result['status']}")
        print(f"⚠️ Details: {result['details']}")


@pytest.mark.asyncio
async def test_generate_audio():
    prompt = "Hello world, this is a test of the audio generation API"
    result = await generate_audio(prompt)
    
    assert result is not None, "🪷 Expected a non-null response"
    assert isinstance(result, dict), "🪷 Expected result to be a dictionary"
    assert "status" in result, "🪷 Response missing 'status' field"
    assert "status_code" in result, "🪷 Response missing 'status_code' field"
    assert "response" in result, "🪷 Response missing 'response' field"
    
    print("\n=== 🪷 Generate Audio Test ===")
    print("🪷 Generate Audio Status:", result.get("status"))
    print("🪷 Generate Audio Response:", result.get("response"))
    print("🪷 Generate Audio Status Code:", result.get("status_code"))
    print(f"🪷 Generate Audio Details: {result.get('details')}")
    
    if result["status"] == "success":
        assert result["status_code"] == 200, "🪷 Expected status code 200 for success"
        assert isinstance(result["response"], str), "🪷 Expected response to be a string"
        assert "model=openai-audio" in result["response"], "🪷 Audio URL should contain model=openai-audio parameter"
        assert "voice=" in result["response"], "🪷 Audio URL should contain voice parameter"
        print(f"✅ generate_audio passed with URL: {result['response']}")
    else:
        print(f"⚠️ generate_audio returned non-success status: {result['status']}")
        print(f"⚠️ Details: {result['details']}")


@pytest.mark.asyncio
async def test_list_text_models():
    result = await list_text_models()
    
    assert result is not None, "🪷 Pollinations:Expected a non-null response"
    assert isinstance(result, dict), "🪷 Pollinations:Expected result to be a dictionary"
    assert "status" in result, "🪷 Pollinations:Response missing 'status' field"
    assert "status_code" in result, "🪷 Pollinations:Response missing 'status_code' field"
    assert "response" in result, "🪷 Pollinations:Response missing 'response' field"
    
    print("\n=== 🪷 Pollinations:List Text Models Test ===")
    print("🪷 Pollinations:List Text Models Status:", result.get("status"))
    print("🪷 Pollinations:List Text Models Response:", str(result.get("response"))[:100], "...")
    print("🪷 Pollinations:List Text Models Status Code:", result.get("status_code"))
    print(f"🪷 Pollinations:List Text Models Details: {str(result.get('details'))[:100]}...")
    
    if result["status"] == "success":
        assert result["status_code"] == 200, "🪷 Pollinations:Expected status code 200 for success"
        assert isinstance(result["response"], (list, dict)), "🪷 Pollinations:Expected response to be a list or dictionary"
        print(f"✅ list_text_models returned response: {str(result['response'])[:100]}...")
    else:
        print(f"⚠️ list_text_models returned non-success status: {result['status']}")
        print(f"⚠️ Details: {result['details']}")


@pytest.mark.asyncio
async def test_openai_chat_completions():
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7
    }
    
    result = await call_openai_compatible_endpoint("chat/completions", payload)
    
    assert result is not None, "🪷 Pollinations:Expected a non-null response"
    assert isinstance(result, dict), "🪷 Pollinations:Expected result to be a dictionary"
    assert "status" in result, "🪷 Pollinations:Response missing 'status' field"
    assert "status_code" in result, "🪷 Pollinations:Response missing 'status_code' field"
    assert "response" in result, "🪷 Pollinations:Response missing 'response' field"
    
    print("\n=== 🪷 Pollinations:OpenAI Chat Completions Test ===")
    print("🪷 Pollinations:OpenAI Chat Completions Status:", result.get("status"))
    print("🪷 Pollinations:OpenAI Chat Completions Status Code:", result.get("status_code"))
    print(f"🪷 Pollinations:OpenAI Chat Completions Details: {str(result.get('details'))[:100]}...")
    
    if result["status"] == "success":
        assert result["status_code"] == 200, "🪷 Expected status code 200 for success"
        assert isinstance(result["response"], dict), "🪷 Expected response to be a dictionary"
        assert "choices" in result["response"], "🪷 'choices' key not in response"
        assert isinstance(result["response"]["choices"], list) and result["response"]["choices"], "🪷 'choices' should be a non-empty list"
        assert "message" in result["response"]["choices"][0], "🪷 'message' key missing in first choice"
        print("✅ test_openai_chat_completions passed.")
    else:
        print(f"⚠️ OpenAI chat completions returned non-success status: {result['status']}")
        print(f"⚠️ Details: {result['details']}")


# @pytest.mark.asyncio
# async def test_completions():
#     payload = {
#         "model": "text-davinci-003",
#         "prompt": TEST_PROMPT,
#         "max_tokens": 30
#     }
#     result = await call_openai_compatible_endpoint("/v1/completions", payload=payload)
#     assert result and "choices" in result, "u274c completions failed"

# # Endpoint Functions
# @pytest.mark.asyncio
# async def test_embeddings():
#     payload = {
#         "model": "text-embedding-ada-002",
#         "input": TEST_PROMPT
#     }
#     result = await call_openai_compatible_endpoint("/v1/embeddings", payload=payload)
#     assert result and "data" in result, "u274c embeddings failed"

# @pytest.mark.asyncio
# async def test_moderations():
#     payload = {
#         "input": "You are so stupid!"
#     }
#     result = await call_openai_compatible_endpoint("/v1/moderations", payload=payload)
#     assert result and "results" in result, "u274c moderations failed"

# @pytest.mark.asyncio
# async def test_list_models():
#     result = await call_openai_compatible_endpoint("/v1/models", method="GET")
#     assert result and "data" in result, "u274c models list failed"

# @pytest.mark.asyncio
# async def test_model_info():
#     result = await call_openai_compatible_endpoint("/v1/models/gpt-3.5-turbo", method="GET")
#     assert result and "id" in result, "u274c model info failed"

# @pytest.mark.asyncio
# async def test_image_generation():
#     payload = {
#         "prompt": TEST_IMAGE_PROMPT,
#         "n": 1,
#         "size": "512x512"
#     }
#     result = await call_openai_compatible_endpoint("/v1/images/generations", payload=payload)
#     assert result and "data" in result, "u274c image generation failed"

# @pytest.mark.asyncio
# async def test_audio_transcriptions():
#     pytest.skip("🗣️ Pollinations may not support audio uploads for transcriptions. Manual test or mock required.")

# @pytest.mark.asyncio
# async def test_audio_translations():
#     pytest.skip("🗣️ Pollinations may not support audio translations. Manual test or mock required.")

# @pytest.mark.asyncio
# async def test_file_upload_list():
#     pytest.skip("📁 Pollinations likely doesn’t support file endpoints. Skip or mock for now.")

# @pytest.mark.asyncio
# async def test_fine_tuning_jobs():
#     pytest.skip("🤖 Fine-tuning jobs likely unsupported on Pollinations API.")
# @pytest.mark.asyncio
# async def test_fetch_image_feed():
#     feed = await fetch_image_feed()
#     assert feed is not None and isinstance(feed, dict)
#     print("✅ fetch_image_feed passed.")


# @pytest.mark.asyncio
# async def test_fetch_text_feed():
#     feed = await fetch_text_feed()
#     assert feed is not None and isinstance(feed, dict)
#     print("✅ fetch_text_feed passed.")
