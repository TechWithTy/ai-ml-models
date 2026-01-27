"""
Authentication utilities for the Gemini SDK.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Authentication utilities for the Gemini SDK with Key Rotation.
"""
import os
import time
import logging
import asyncio
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiKeyPool:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiKeyPool, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.keys = []
        # Load numbered keys 1-20 FIRST (Priority)
        for i in range(1, 21):
            key = os.getenv(f"GOOGLE_AI_STUDIO_API_KEY_{i}")
            if key:
                self.keys.append(key)

        # Load MAIN key LAST (Fallback/Final)
        main_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY_MAIN") or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if main_key:
            self.keys.append(main_key)
            
        # Deduplicate while preserving order
        seen = set()
        unique_keys = []
        for k in self.keys:
            if k and k.strip() and k not in seen:
                unique_keys.append(k)
                seen.add(k)
        self.keys = unique_keys
        
        if not self.keys:
            logger.warning("No Gemini API keys found in environment variables!")
        else:
            logger.info(f"Loaded {len(self.keys)} Gemini API keys.")

        # State tracking: 'available', 'rate_limited', 'exhausted'
        self.key_status: Dict[str, Dict] = {
            k: {'status': 'available', 'retry_after_ts': 0} 
            for k in self.keys
        }
        self.current_index = 0
        self._lock = asyncio.Lock()
        self._initialized = True

    async def get_next_available_key(self) -> Optional[str]:
        """Gets the next available API key, rotating through the pool."""
        async with self._lock:
            keys_count = len(self.keys)
            if keys_count == 0:
                return None

            # Try to find an available key
            # We cycle through all keys to find one that is ready
            for _ in range(keys_count):
                key = self.keys[self.current_index]
                status = self.key_status[key]
                
                # Check if rate limit has expired
                if status['status'] == 'rate_limited':
                    if time.time() > status['retry_after_ts']:
                        status['status'] = 'available'
                        status['retry_after_ts'] = 0
                        logger.info(f"Key ...{key[-4:]} rate limit expired, now available.")
                
                if status['status'] == 'available':
                    # Rotate index for next time (round-robin load balancing)
                    self.current_index = (self.current_index + 1) % keys_count
                    return key
                
                # Move to next key to check
                self.current_index = (self.current_index + 1) % keys_count
            
            # If we're here, no keys are immediately available
            # We could return the one with the soonest retry time, or just None.
            return None

    async def mark_key_rate_limited(self, key: str, duration: int = 60):
        """Marks a key as rate limited for a specific duration."""
        async with self._lock:
            if key in self.key_status:
                self.key_status[key]['status'] = 'rate_limited'
                self.key_status[key]['retry_after_ts'] = time.time() + duration
                logger.warning(f"Marking key ...{key[-4:]} as rate limited for {duration}s")

    async def mark_key_exhausted(self, key: str):
        """Marks a key as permanently exhausted (quota limit)."""
        async with self._lock:
            if key in self.key_status:
                self.key_status[key]['status'] = 'exhausted'
                self.key_status[key]['retry_after_ts'] = float('inf')
                logger.error(f"Marking key ...{key[-4:]} as EXHAUSTED (quota limit reached)")

# Global instance
_pool = None

def get_key_pool() -> GeminiKeyPool:
    global _pool
    if _pool is None:
        _pool = GeminiKeyPool()
    return _pool

# Backward compatibility
def get_gemini_api_key() -> str | None:
    """! Retrieves a Gemini API key (non-async wrapper, careful with rotation)."""
    # This is a bit risky if used in sync context expecting rotation, 
    # but for backward compat we return the first designated main key or from env
    return os.getenv("GOOGLE_AI_STUDIO_API_KEY_MAIN") or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
