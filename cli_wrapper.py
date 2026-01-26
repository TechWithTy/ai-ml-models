#!/usr/bin/env python3
"""
CLI Wrapper for AI ML Models submodule.
Allows executing models (OpenAI, Gemini, Pollinations) from the command line.
Returns JSON output.
"""
import sys
import os
import json
import argparse
import asyncio
from typing import Dict, Any

# Ensure we can import from the current directory (package root)
try:
    import huggingface_hub
except ImportError:
    pass
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def load_env():
    """Load environment variables from .env file in parent directories."""
    try:
        from dotenv import load_dotenv
        # Try loading .env from backend root (2 levels up)
        env_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            # Fallback to local .env
            load_dotenv()
    except ImportError:
        pass

async def run_openai(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        # Import lazily to avoid issues if optional deps are missing
        from openai_provider.openai_generator import OpenAIGenerator
        
        client = OpenAIGenerator()
        
        # Override config if provided in args (config parsing override could be added here)
        # For now, we rely on env vars + defaults, but we can update attributes
        if args.model:
            client.model = args.model
            
        if args.image:
            return client.generate_image(args.prompt)
        elif args.tts:
            result = client.generate_audio(args.prompt)
            if result.get("status") == "success":
                try:
                    output_file = "output_tts.mp3"
                    with open(output_file, "wb") as f:
                        f.write(result["audio_content"])
                    result["audio_content"] = "<binary_truncated>"
                    result["file_saved"] = output_file
                except Exception as ex:
                     result["save_error"] = str(ex)
            return result
        else:
            kwargs = {}
            if args.reasoning_effort:
                kwargs["reasoning_effort"] = args.reasoning_effort
            if args.voice_input:
                kwargs["audio_input"] = args.voice_input
            if args.voice_output:
                kwargs["output_audio"] = True
                
            response = client.send_message(args.prompt, **kwargs)
            
            # Handle audio output saving
            if "audio_data" in response:
                try:
                    import base64
                    audio_bytes = base64.b64decode(response["audio_data"])
                    output_filename = "output_audio.wav"
                    with open(output_filename, "wb") as f:
                        f.write(audio_bytes)
                    response["audio_file_saved"] = output_filename
                    # Truncate data for clean JSON output
                    response["audio_data"] = "<base64_audio_data_truncated>"
                except Exception as e:
                    response["audio_save_error"] = str(e)
            
            return response
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "openai"}

async def run_gemini(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from gemini.gemini_client import GeminiClient
        from gemini.models import GeminiModel
        
        # Determine model enum
        model_name = args.model or "gemini-3.0-flash"
        
        # Simple mapping
        if "3.0" in model_name:
            model = GeminiModel.GEMINI_3_0_FLASH
        elif "2.5" in model_name and "pro" in model_name:
            model = GeminiModel.GEMINI_2_5_PRO
        elif "2.5" in model_name:
            model = GeminiModel.GEMINI_2_5_FLASH
        elif "2.0" in model_name:
             model = GeminiModel.GEMINI_2_0_FLASH
        elif "1.5" in model_name and "pro" in model_name:
             model = GeminiModel.GEMINI_1_5_PRO
        elif "1.5" in model_name:
             model = GeminiModel.GEMINI_1_5_FLASH
        else:
             model = GeminiModel.GEMINI_3_0_FLASH # Default to latest
            
        client = GeminiClient()
        
        if args.image:
             # Prefer generic multimodal handler for consistency
             pass
             
        # Prepare media args
        media_path = None
        media_mime = None
        
        # Check for media args
        if args.voice_input:
             media_path = args.voice_input
             media_mime = "audio/mp3" # Default assumption
        elif hasattr(args, 'video_input') and args.video_input:
             media_path = args.video_input
             media_mime = "video/mp4"
        elif hasattr(args, 'image_input') and args.image_input:
             media_path = args.image_input
             # Mime type auto-detected in client if None
        elif args.image and not args.image_input:
             return {"status": "error", "message": "Please use --image-input for Gemini image path"}
             
        # Call synchronous generate_content (since we updated that method, not async yet for media)
        # Note: GeminiClient.generate_content is synchronous in our update.
        # But run_gemini is async. We should wrap it or use to_thread if blocking (upload_file blocks).
        response = await asyncio.to_thread(
            client.generate_content, 
            model, 
            args.prompt, 
            media_path=media_path, 
            media_mime_type=media_mime,
            system_instruction=args.system,
            json_mode=args.json
        )
        return {"status": "success", "response": response.text}
        
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "gemini"}

async def run_pollinations(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from pollinations.pollinations_client import PollinationsClient
        
        client = PollinationsClient()
        
        if args.video:
            return await client.generate_video(
                args.prompt,
                model=args.model or "videoalpha"
            )
        elif args.image:
            return await client.generate_image(
                args.prompt, 
                model=args.model or "flux",
                width=1024,
                height=1024
            )
        else:
            return await client.generate_text(args.prompt, model=args.model or "gpt-4o")
            
    except Exception as e:
         return {"status": "error", "message": str(e), "provider": "pollinations"}

async def run_claude(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from claude.claude_generator import ClaudeGenerator
        client = ClaudeGenerator()
        return client.send_message(prompt_text=args.prompt, model=args.model, enable_caching=args.cache_prompt)
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "claude"}

async def run_openrouter(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from open_router.openrouter_generator import OpenRouterGenerator
        client = OpenRouterGenerator(provider="openrouter")
        
        images = []
        if args.image_input:
             images.append(args.image_input)
             
        # args.video_input for OpenRouter? 
        # Some models support video as sequence of frames, but generator only has 'images' param.
        # Leaving video out for now or mapping it if we implement video processing utils later.
        
        return client.send_message(
            prompt_text=args.prompt, 
            model=args.model,
            images=images,
            json_mode=args.json,
            system_instruction=args.system
        )
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "openrouter"}

async def run_huggingface(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import huggingface_hub
        print(f"DEBUG: huggingface_hub version: {huggingface_hub.__version__}, file: {huggingface_hub.__file__}")
        from huggingface.huggingface_client import HuggingFaceClient
        client = HuggingFaceClient(model=args.model or "gpt2") # Default to something simple
        if args.video:
             return {"status": "error", "message": "HF video not supported in CLI wrapper yet"}
        if args.image:
             return {"status": "error", "message": "HF image not supported in CLI wrapper yet"}
        response = await client.generate_text_async(args.prompt)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "huggingface"}

async def run_deepseek(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from deepseek.deepseek_client import DeepSeekClient
        client = DeepSeekClient()
        response = await client.generate_text_async(args.prompt)
        return response
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "deepseek"}

async def run_reachy(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from reachy.reachy_client import ReachyMiniClient
        client = ReachyMiniClient()
        
        if args.reachy_see:
             return client.see()
             
        # Default to move if any move args are present, or just generic ping
        return client.move(
            head_z=args.reachy_head_z,
            antennas_deg=args.reachy_antennas,
            body_yaw_deg=args.reachy_yaw
        )
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "reachy"}

async def run_kling(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from kling_provider.kling_generator import KlingGenerator
        client = KlingGenerator()
        
        # Determine image url if provided (for Image-to-Video)
        image_url = args.image_input if args.image_input else None
        
        # For now generic args or defaults
        return await client.generate_video(
            prompt=args.prompt,
            image_url=image_url
        )
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "kling"}

async def run_heygen(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from heygen_provider.heygen_generator import HeyGenGenerator
        client = HeyGenGenerator()
        
        return await client.generate_video(
            prompt=args.prompt,
            avatar_id=args.model  # Use --model for avatar selection
        )
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "heygen"}

async def run_glif(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from glif_provider.glif_generator import GlifGenerator
        client = GlifGenerator()
        
        # Workflow ID should be passed via --model flag
        workflow_id = args.model
        if not workflow_id:
            return {
                "status": "error",
                "message": "Glif requires a workflow ID. Pass via --model flag.",
                "provider": "glif"
            }
        
        return await client.run_workflow(
            workflow_id=workflow_id,
            inputs=[args.prompt] if args.prompt else []
        )
    except Exception as e:
        return {"status": "error", "message": str(e), "provider": "glif"}

async def main():
    parser = argparse.ArgumentParser(description="AI ML Models CLI Wrapper")
    parser.add_argument("--provider", required=True, choices=["openai", "gemini", "pollinations", "claude", "openrouter", "huggingface", "deepseek", "reachy", "kling", "heygen", "glif"], help="Model provider")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--model", help="Model name (optional)")
    parser.add_argument("--image", action="store_true", help="Generate image instead of text")
    parser.add_argument("--video", action="store_true", help="Generate video instead of text")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="Reasoning effort for o1 models")
    parser.add_argument("--voice-input", help="Path to audio file for voice input (OpenAI)")
    parser.add_argument("--voice-output", action="store_true", help="Request audio output (OpenAI multimodal)")
    parser.add_argument("--tts", action="store_true", help="Generate text-to-speech audio (OpenAI TTS)")
    parser.add_argument("--cache-prompt", action="store_true", help="Enable prompt caching (Claude)")
    parser.add_argument("--video-input", help="Path to video file for multimodal input")
    parser.add_argument("--image-input", help="Path to image file for multimodal input")
    parser.add_argument("--json", action="store_true", help="Enforce JSON output (schema-based models)")
    parser.add_argument("--system", help="System instructions")
    
    # Reachy Mini Args
    parser.add_argument("--reachy-head-z", type=float, help="Reachy Head Z (mm)")
    parser.add_argument("--reachy-antennas", type=float, nargs=2, help="Reachy Antennas angles (deg)")
    parser.add_argument("--reachy-yaw", type=float, help="Reachy Body Yaw (deg)")
    parser.add_argument("--reachy-see", action="store_true", help="Capture Reachy view")
    
    args = parser.parse_args()
    
    load_env()
    
    result = {}
    
    if args.provider == "openai":
        result = await run_openai(args)
    elif args.provider == "gemini":
        result = await run_gemini(args)
    elif args.provider == "pollinations":
        result = await run_pollinations(args)
    elif args.provider == "claude":
        result = await run_claude(args)
    elif args.provider == "openrouter":
        result = await run_openrouter(args)
    elif args.provider == "huggingface":
        result = await run_huggingface(args)
    elif args.provider == "deepseek":
        result = await run_deepseek(args)
    elif args.provider == "reachy":
        result = await run_reachy(args)
    elif args.provider == "kling":
        result = await run_kling(args)
    elif args.provider == "heygen":
        result = await run_heygen(args)
    elif args.provider == "glif":
        result = await run_glif(args)
        
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "error", "message": "Interrupted"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
