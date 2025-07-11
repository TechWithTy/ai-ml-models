from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from utils.config.config_loader import config
from utils.index import get_env_variable
from utils.prompt_builder import get_prompt_globals, init_globals_for_test
import os
from dotenv import load_dotenv

class BaseGenerator(ABC):
    """
    * Abstract base class for all ML model generators (Claude, DeepSeek, HuggingFace, OpenAI, Pollinations).
    * Handles environment loading, config parsing, prompt state extraction, and provides utility methods.
    * Extend this class and implement the required abstract methods for each provider.
    """
    def __init__(self, provider: str, config_section: Optional[str] = None):
        load_dotenv()
        self.provider = provider
        self.config_section = config_section or provider
        self.api_key = self._get_api_key()
        self.test_mode = get_env_variable("TEST_MODE").lower() == "true"
        if self.test_mode:
            init_globals_for_test()
        self.state = get_prompt_globals()
        self.config = config.get("user_profile", {}).get("llm", {}).get(self.config_section, {})
        self._validate_api_key()

    def _get_api_key(self) -> Optional[str]:
        key_env = f"{self.provider.upper()}_API_KEY"
        return get_env_variable(key_env)

    def _validate_api_key(self):
        if not self.api_key:
            raise ValueError(f"❌ {self.provider.upper()}_API_KEY is missing! Set it in your .env file or GitHub Secrets.")

    def get_prompt_state(self) -> Dict[str, Any]:
        return self.state

    def get_config(self) -> Dict[str, Any]:
        return self.config

    @abstractmethod
    def send_message(self, *args, **kwargs) -> Dict[str, Any]:
        """Send a message to the provider. Must be implemented by subclasses."""
        pass

    # * Add any shared utility methods here for all generators (e.g., error formatting, logging, etc.)
