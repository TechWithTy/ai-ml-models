from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Callable, Dict, Generic, Optional, TypeVar
from enum import Enum
import asyncio

T = TypeVar('T')

class StreamEventType(Enum):
    START = "start"
    CONTENT = "content"
    ERROR = "error"
    DONE = "done"

class StreamEvent(Generic[T]):
    """Represents an event in a streaming response."""
    def __init__(self, event_type: StreamEventType, data: Optional[T] = None, error: Optional[Exception] = None):
        self.event_type = event_type
        self.data = data
        self.error = error

    def __str__(self) -> str:
        return f"StreamEvent(type={self.event_type}, data={self.data}, error={self.error})"

class BaseStreamingGenerator(ABC):
    """Abstract base class for streaming generators."""
    
    @abstractmethod
    async def stream_text(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[StreamEvent[str], None]:
        """Stream text generation."""
        yield
    
    @abstractmethod
    async def stream_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent[bytes], None]:
        """Stream audio generation."""
        yield
    
    @abstractmethod
    async def stream_image(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[StreamEvent[bytes], None]:
        """Stream image generation."""
        yield
