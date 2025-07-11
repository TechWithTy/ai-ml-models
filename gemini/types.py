"""
Defines the data models and types for the Gemini SDK.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Optional, Union, Generator, AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field

# Basic Types
class Part(BaseModel):
    """A part of a prompt, can be text or inline data."""
    text: Optional[str] = None
    inline_data: Optional[Dict[str, Any]] = None

class GenerateContentResponse(BaseModel):
    """Standard response for content generation."""
    text: Optional[str] = None
    raw_response: Optional[Any] = None

# Streaming Types
class StreamingResponseState(str, Enum):
    """The state of a streaming response delta."""
    START = "start"
    CONTENT = "content"
    END = "end"
    ERROR = "error"

class StreamingDelta(BaseModel):
    """A single delta in a streaming response."""
    text: Optional[str] = None
    state: StreamingResponseState
    raw: Dict[str, Any]

class StreamConfig(BaseModel):
    """Configuration for a streaming request."""
    timeout: Optional[int] = Field(None, description="Request timeout in seconds.")

# Callbacks
StreamingCallback = Callable[[StreamingDelta], None]
AsyncStreamingCallback = Callable[[StreamingDelta], Awaitable[None]]

class StreamingResponse(BaseModel):
    """Wrapper for streaming responses with helper methods."""
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    stream: Union[Generator[StreamingDelta, None, None], AsyncGenerator[StreamingDelta, None]]
    is_async: bool = False

    def __iter__(self):
        if self.is_async:
            raise RuntimeError("Cannot iterate over async stream in sync context")
        return self.stream

    def __aiter__(self):
        if not self.is_async:
            raise RuntimeError("Cannot iterate over sync stream in async context")
        return self.stream

    def collect(self) -> str:
        """Collects all text from a synchronous stream."""
        return "".join(delta.text for delta in self if delta.text)

    async def acollect(self) -> str:
        """Collects all text from an asynchronous stream."""
        return "".join([delta.text async for delta in self if delta.text])
