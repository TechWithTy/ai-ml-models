
import os
import sys

# Ensure we can import from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from gemini.gemini_client import GeminiClient
from gemini.models import GeminiModel
import google.generativeai as genai

def add_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

def main():
    print("Verifying Gemini Tools Support...")
    
    # Check if API key is present
    if not os.environ.get("GEMINI_API_KEY"):
        print("Skipping execution: GEMINI_API_KEY not found.")
        return

    client = GeminiClient()
    
    # Define tool
    tools_list = [add_numbers]
    
    try:
        # We need to manually construct the request usually, but GeminiClient.generate_content 
        # now accepts 'tools' list
        print("Sending request with tools...")
        response = client.generate_content(
            model=GeminiModel.GEMINI_1_5_FLASH, 
            prompt="What is 57 plus 43?",
            tools=tools_list
        )
        
        # Check if function call was generated
        if response.raw_response.candidates[0].content.parts[0].function_call:
            print("SUCCESS: Function call generated!")
            print(response.raw_response.candidates[0].content.parts[0].function_call)
        else:
             print("Response received (Text):", response.text)
             
    except Exception as e:
        print("Error during tool verification:", e)

if __name__ == "__main__":
    main()
