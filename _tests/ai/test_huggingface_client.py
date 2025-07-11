#!/usr/bin/env python
# src/ml_models/_tests/ai/test_huggingface_client.py

import os
import pytest
from typing import AsyncGenerator

from src.ml_models.huggingface import HuggingFaceClient
from src.ml_models.huggingface.exceptions import HuggingFaceAuthError
from src.ml_models.utils.stream_event import StreamEvent

# Constants
TEST_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"

@pytest.fixture
def client():
    """Provides a HuggingFaceClient instance for testing."""
    try:
        return HuggingFaceClient(model=TEST_MODEL)
    except HuggingFaceAuthError:
        pytest.skip("HUGGINGFACE_API_KEY not set, skipping Hugging Face client tests.")

@pytest.fixture
def image_client():
    """Provides a HuggingFaceClient instance for image generation testing."""
    try:
        return HuggingFaceClient(model=IMAGE_MODEL)
    except HuggingFaceAuthError:
        pytest.skip("HUGGINGFACE_API_KEY not set, skipping Hugging Face client tests.")

@pytest.mark.asyncio
async def test_generate_text_async(client: HuggingFaceClient):
    """Tests asynchronous text generation."""
    prompt = "Once upon a time"
    response = await client.generate_text_async(prompt, max_new_tokens=10)
    
    assert isinstance(response, str)
    assert len(response.strip()) > 0
    print(f"\nGenerated Text: {response}")

@pytest.mark.asyncio
async def test_generate_text_stream(client: HuggingFaceClient):
    """Tests streaming text generation."""
    prompt = "The future of AI is"
    stream = client.generate_text_stream(prompt, max_new_tokens=15)
    
    assert isinstance(stream, AsyncGenerator)
    
    events = []
    full_response = ""
    async for event in stream:
        events.append(event.event_type)
        if event.event_type == "CONTENT":
            full_response += event.data
            print(event.data, end="")
    
    print("\n") # for clean output
    assert "START" in events
    assert "CONTENT" in events
    assert "DONE" in events
    assert len(full_response.strip()) > 0

@pytest.mark.asyncio
async def test_generate_image_async(image_client: HuggingFaceClient):
    """Tests asynchronous image generation."""
    prompt = "A photorealistic image of a cat programming a computer"
    image_bytes = await image_client.generate_image_async(prompt)
    
    assert isinstance(image_bytes, bytes)
    assert len(image_bytes) > 0
    print(f"\nGenerated image size: {len(image_bytes)} bytes")
