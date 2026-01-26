import os
import asyncio
import httpx
from dotenv import load_dotenv

async def test_glif():
    # Load .env from backend root
    current_dir = os.getcwd()
    env_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), '.env')
    print(f"Loading env from: {env_path}")
    load_dotenv(env_path)
    
    api_key = os.getenv("GLIF_API_KEY")
    print(f"GLIF_API_KEY found: {bool(api_key)}")
    
    if not api_key:
        print("Error: GLIF_API_KEY not found")
        return

    url = "https://simple-api.glif.app"
    payload = {
        "id": "test-workflow-id",
        "inputs": ["Test"],
        "visibility": "PRIVATE"
    }
    
    print(f"Sending request to {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    except httpx.TimeoutException:
        print("Request timed out (10s)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_glif())
