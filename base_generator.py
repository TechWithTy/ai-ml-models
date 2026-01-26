from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os
from dotenv import load_dotenv

class BaseGenerator(ABC):
    """
    * Abstract base class for all ML model generators.
    * Simplified for CLI wrapper usage.
    """
    def __init__(self, provider: str, config_section: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        load_dotenv()
        self.provider = provider
        self.config_section = config_section or provider
        self.api_key = api_key or self._get_api_key()
        self.config = {} # Config loading removed
        self._validate_api_key()

    def _get_api_key(self) -> Optional[str]:
        key_env = f"{self.provider.upper()}_API_KEY"
        return os.getenv(key_env)

    def _validate_api_key(self):
        # Pollinations doesn't require an API key
        if self.provider == "pollinations":
            return
        if not self.api_key:
            # Don't crash immediately, allow for checking in subclasses or graceful failure
            print(f"⚠️ {self.provider.upper()}_API_KEY is missing. Some features may not work.")

    def get_prompt_state(self) -> Dict[str, Any]:
        return {}

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def send_message(self, *args, **kwargs) -> Dict[str, Any]:
        """Send a message to the provider. Can be overridden."""
        raise NotImplementedError("Method not implemented")

    # * Add any shared utility methods here for all generators (e.g., error formatting, logging, etc.)
