import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import httpx
import asyncio
from typing import Optional, Dict, Any
from utils.prompt_builder import  get_prompt_globals
import urllib.parse


# Constants
BASE_IMAGE_URL = "https://image.pollinations.ai"
BASE_TEXT_URL = "https://text.pollinations.ai"
BASE_AUDIO_URL = "https://text.pollinations.ai"
OPENAI_ENDPOINT = f"{BASE_TEXT_URL}/openai"
DEFAULT_VOICE = "nova"
FALLBACK_VOICE = "echo"

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5


# Get the shared global state
state = get_prompt_globals()

# Access individual variables
prompt = state["prompt"]
creative_prompt = state["creative_prompt"]
gif_prompt = state["gif_prompt"]
hashtags = state["hashtags"]
system_instructions = state["system_instructions"]
blog_content = state["blog_content"]

async def fetch_with_retries(url: str, client: httpx.AsyncClient) -> Optional[httpx.Response]:
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response
            print(f"⚠️ Attempt {attempt + 1} failed ({response.status_code}), retrying...")
        except httpx.RequestError as e:
            print(f"🚨 Request error on attempt {attempt + 1}: {e}")
        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
    print("❌ All retries failed.")
    return None


async def generate_image(prompt: str, model: str = "stable-diffusion") -> Dict[str, Any]:
    # The correct URL format for Pollinations image API is /prompt/ not /i/
    url = f"{BASE_IMAGE_URL}/prompt/{prompt}"
    if model != "stable-diffusion":
        url += f"?model={model}"

    async with httpx.AsyncClient() as client:
        try:
            response = await fetch_with_retries(url, client)
            if response:
                print(f"📸 Image Generation Status Code: {response.status_code}")
                print(f"📸 Image URL: {url}")
                
                return {
                    "status": "success",
                    "status_code": response.status_code,
                    "response": url,
                    "details": url
                }
            else:
                return {
                    "status": "failed",
                    "status_code": 0,
                    "response": "Failed to get response after retries",
                    "details": "All retry attempts failed"
                }
        except Exception as e:
            print(f"Error generating image: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }


async def generate_image_advanced(
    prompt: str,
    model: str = "flux",
    seed: int = 42,
    width: int = 1024,
    height: int = 1024,
    nologo: bool = True,
    private: bool = True,
    enhance: bool = False,
    safe: bool = True
) -> Optional[str]:
    """
    Generates an advanced AI image via Pollinations with extended GET parameters.
    Returns the image URL if successful, otherwise None.
    """

    query = {
        "model": model,
        "seed": seed,
        "width": width,
        "height": height,
        "nologo": str(nologo).lower(),
        "private": str(private).lower(),
        "enhance": str(enhance).lower(),
        "safe": str(safe).lower(),
    }

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"{BASE_IMAGE_URL}/prompt/{encoded_prompt}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=query, timeout=15)
            if response.status_code == 200:
                return str(response.url)
            else:
                print(f"❌ Failed to generate image: {response.status_code}")
                return None
    except Exception as e:
        print(f"🔥 Exception during image generation: {e}")
        return None



async def list_image_models() -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_IMAGE_URL}/models")
            print(f"🖼️ Image Models Status Code: {response.status_code}")
            print(f"🖼️ Image Models Response: {response.text[:100]}...")
            
            if response.status_code == 200:
                try:
                    models_data = response.json()
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "response": models_data,
                        "details": models_data
                    }
                except Exception as e:
                    print(f"Error parsing image models JSON: {e}")
                    return {
                        "status": "error",
                        "status_code": response.status_code,
                        "response": "Error parsing response",
                        "details": str(e)
                    }
            else:
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "response": "Failed to fetch image models",
                    "details": response.text
                }
        except Exception as e:
            print(f"Error fetching image models: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }


async def generate_text(prompt: str) -> Dict[str, Any]:
    url = f"{BASE_TEXT_URL}/{prompt}"
    async with httpx.AsyncClient() as client:
        try:
            response = await fetch_with_retries(url, client)
            if response:
                print(f"📝 Text Generation Status Code: {response.status_code}")
                print(f"📝 Text Generation Response: {response.text[:100]}...")
                
                return {
                    "status": "success",
                    "status_code": response.status_code,
                    "response": response.text,
                    "details": response.text
                }
            else:
                return {
                    "status": "failed",
                    "status_code": 0,
                    "response": "Failed to get response after retries",
                    "details": "All retry attempts failed"
                }
        except Exception as e:
            print(f"Error generating text: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }


async def generate_text_advanced(payload: dict) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:  
        print(f"📥 Advanced Text Generation Payload Model: {payload['model']}") 
        try:
            response = await client.post(BASE_TEXT_URL, json=payload)
            
            print(f"📥 Advanced Text Generation Status Code: {response.status_code}")
            print(f"📦 Advanced Text Generation Response: {response.text[:100]}...")

            if response.status_code == 200:
                try:
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "response": response.text,
                        "details": response.text
                    }
                except Exception as e:
                    print(f"Error parsing advanced text response: {e}")
                    return {
                        "status": "error",
                        "status_code": response.status_code,
                        "response": "Error parsing response",
                        "details": str(e)
                    }
            else:
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "response": "Failed to generate advanced text",
                    "details": response.text
                }
        except Exception as e:
            print(f"Exception during advanced text generation: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }


async def generate_audio(prompt: str, voice: str = DEFAULT_VOICE) -> Dict[str, Any]:
    url = f"{BASE_AUDIO_URL}/{prompt}?model=openai-audio&voice={voice}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await fetch_with_retries(url, client)
            if response and response.status_code == 200:
                # Check if we got an audio file (Content-Type: audio/mpeg)
                content_type = response.headers.get("Content-Type", "")
                if "audio/mpeg" in content_type:
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "response": url,  # Return the URL for playback
                        "details": f"Audio file generated successfully. Content-Type: {content_type}"
                    }
                else:
                    return {
                        "status": "error",
                        "status_code": response.status_code,
                        "response": "Unexpected content type",
                        "details": f"Expected audio/mpeg, got {content_type}"
                    }
            elif response:
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "response": "Failed to generate audio",
                    "details": response.text
                }
            else:
                return {
                    "status": "failed",
                    "status_code": 0,
                    "response": "Failed to get response after retries",
                    "details": "All retry attempts failed"
                }
        except Exception as e:
            print(f"Error generating audio: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }


async def list_text_models() -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_TEXT_URL}/models")
            print(f"📝 Text Models Status Code: {response.status_code}")
            print(f"📝 Text Models Response: {response.text[:100]}...")
            
            if response.status_code == 200:
                try:
                    models_data = response.json()
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "response": models_data,
                        "details": models_data
                    }
                except Exception as e:
                    print(f"Error parsing text models JSON: {e}")
                    return {
                        "status": "error",
                        "status_code": response.status_code,
                        "response": "Error parsing response",
                        "details": str(e)
                    }
            else:
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "response": "Failed to fetch text models",
                    "details": response.text
                }
        except Exception as e:
            print(f"Error fetching text models: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }

async def call_openai_compatible_endpoint(
    endpoint: str,
    payload: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(endpoint, json=payload, headers=headers)
            print(f"🔄 OpenAI Compatible Endpoint Status Code: {response.status_code}")
            print(f"📥 OpenAI Response: {response.text[:100]}...")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "response": response_data,
                        "details": response_data
                    }
                except Exception as e:
                    print(f"Error parsing OpenAI compatible response JSON: {e}")
                    return {
                        "status": "error",
                        "status_code": response.status_code,
                        "response": "Error parsing response",
                        "details": str(e)
                    }
            else:
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "response": "Failed to call OpenAI compatible endpoint",
                    "details": response.text
                }
        except Exception as e:
            print(f"Error calling OpenAI compatible endpoint: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }

async def fetch_image_feed() -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_IMAGE_URL}/feed")
            print(f"📸 Image Feed Status Code: {response.status_code}")
            print(f"📸 Image Feed Response: {response.text[:100]}...")
            
            if response.status_code == 200:
                try:
                    feed_data = response.json()
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "response": feed_data,
                        "details": feed_data
                    }
                except Exception as e:
                    print(f"Error parsing image feed JSON: {e}")
                    return {
                        "status": "error",
                        "status_code": response.status_code,
                        "response": "Error parsing response",
                        "details": str(e)
                    }
            else:
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "response": "Failed to fetch image feed",
                    "details": response.text
                }
        except Exception as e:
            print(f"Error fetching image feed: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }

async def fetch_text_feed() -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_TEXT_URL}/feed")
            print(f"📝 Text Feed Status Code: {response.status_code}")
            print(f"📝 Text Feed Response: {response.text[:100]}...")
            
            if response.status_code == 200:
                try:
                    feed_data = response.json()
                    return {
                        "status": "success",
                        "status_code": response.status_code,
                        "response": feed_data,
                        "details": feed_data
                    }
                except Exception as e:
                    print(f"Error parsing text feed JSON: {e}")
                    return {
                        "status": "error",
                        "status_code": response.status_code,
                        "response": "Error parsing response",
                        "details": str(e)
                    }
            else:
                return {
                    "status": "failed",
                    "status_code": response.status_code,
                    "response": "Failed to fetch text feed",
                    "details": response.text
                }
        except Exception as e:
            print(f"Error fetching text feed: {e}")
            return {
                "status": "error",
                "status_code": 500,
                "response": "Exception during API call",
                "details": str(e)
            }
