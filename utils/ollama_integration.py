"""
Ollama integration for local AI model inference.
Provides functions to use Ollama's API for text summarization.
"""

import requests
import json
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default Ollama API endpoint - configurable via environment variable
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api")


def check_ollama_available():
    """
    Check if Ollama service is available.
    
    Returns:
        bool: True if Ollama is available, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models():
    """
    List available models in Ollama.
    
    Returns:
        list: List of available model names
    """
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
    """
    Summarize text using Ollama's local API.
    
    Args:
        text (str): Text to summarize
        model (str): Ollama model to use
        max_length (int): Maximum length of the summary
        
    Returns:
        str: Summarized text or None if failed
    """
    if not check_ollama_available():
        logger.warning("Ollama service is not available")
        return None
    
    # Check if the model is available
    available_models = list_available_models()
    if model not in available_models:
        logger.warning(f"Model {model} not available in Ollama. Available models: {available_models}")
        return None
    
    # Prepare the prompt for summarization
    prompt = f"Summarize the following text in about {max_length} words:\n\n{text}"
    
    try:
        # Make the API request
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": max_length * 2  # Approximate token count
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


def chunk_and_summarize(text, model="llama3", chunk_size=4000, max_length=150):
    """
    Chunk long text and summarize each chunk, then combine the summaries.
    
    Args:
        text (str): Text to summarize
        model (str): Ollama model to use
        chunk_size (int): Maximum size of each chunk in characters
        max_length (int): Maximum length of the final summary
        
    Returns:
        str: Combined summary or None if failed
    """
    if len(text) <= chunk_size:
        return summarize_with_ollama(text, model, max_length)
    
    # Split text into chunks
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
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
        summary = summarize_with_ollama(chunk, model, max_length // len(chunks))
        if summary:
            chunk_summaries.append(summary)
    
    if not chunk_summaries:
        return None
    
    # If there's only one chunk summary, return it
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    # Otherwise, combine the summaries and summarize again
    combined_summary = " ".join(chunk_summaries)
    return summarize_with_ollama(combined_summary, model, max_length) 