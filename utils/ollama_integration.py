"""
Ollama integration for local AI model inference.
Provides functions to use Ollama's API for text summarization with streaming support.
"""

import requests
import json
import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")


def check_ollama_available():
    """Check if Ollama service is available."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models():
    """List available models in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []


def summarize_with_ollama(text, model="llama3", max_length=150):
    """Summarize text using Ollama's local API (non-streaming)."""
    if not check_ollama_available():
        logger.warning("Ollama service is not available")
        return None
    
    prompt = f"Summarize the following text in about {max_length} words:\n\n{text}"
    
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": max_length * 2
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Ollama: {e}")
        return None


def stream_summarize_with_ollama(text, model="llama3", max_length=150):
    """
    Summarize text using Ollama with streaming. Yields tokens as they arrive.
    
    Yields:
        str: Individual response tokens
    """
    if not check_ollama_available():
        logger.warning("Ollama service is not available")
        return
    
    prompt = f"Summarize the following text in about {max_length} words:\n\n{text}"
    
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": max_length * 2
                }
            },
            stream=True
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get('response', '')
                    if token:
                        yield token
                    if data.get('done', False):
                        break
        else:
            logger.error(f"Ollama API error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Ollama: {e}")


def chunk_and_summarize(text, model="llama3", chunk_size=4000, max_length=150):
    """Chunk long text and summarize each chunk, then combine."""
    if len(text) <= chunk_size:
        return summarize_with_ollama(text, model, max_length)
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_with_ollama(chunk, model, max_length // len(chunks))
        if summary:
            chunk_summaries.append(summary)
    
    if not chunk_summaries:
        return None
    
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    combined_summary = " ".join(chunk_summaries)
    return summarize_with_ollama(combined_summary, model, max_length)


def stream_chunk_and_summarize(text, model="llama3", chunk_size=4000, max_length=150):
    """
    Chunk and summarize with streaming on the final summary.
    Returns non-streaming chunk summaries, then streams the final combination.
    
    Yields:
        str: Tokens from the final summary
    """
    if len(text) <= chunk_size:
        yield from stream_summarize_with_ollama(text, model, max_length)
        return
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= chunk_size:
            current_chunk.append(word)
            current_length += len(word) + 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_with_ollama(chunk, model, max_length // len(chunks))
        if summary:
            chunk_summaries.append(summary)
    
    if not chunk_summaries:
        return
    
    if len(chunk_summaries) == 1:
        yield chunk_summaries[0]
        return
    
    combined_summary = " ".join(chunk_summaries)
    yield from stream_summarize_with_ollama(combined_summary, model, max_length)