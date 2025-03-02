"""
Caching utilities for the OBS Recording Transcriber.
Provides functions to cache and retrieve transcription and summarization results.
"""

import json
import hashlib
import os
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default cache directory
CACHE_DIR = Path.home() / ".obs_transcriber_cache"


def get_file_hash(file_path):
    """
    Generate a hash for a file based on its content and modification time.
    
    Args:
        file_path (Path): Path to the file
        
    Returns:
        str: Hash string representing the file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    # Get file stats
    stats = file_path.stat()
    file_size = stats.st_size
    mod_time = stats.st_mtime
    
    # Create a hash based on path, size and modification time
    # This is faster than hashing the entire file content
    hash_input = f"{file_path.absolute()}|{file_size}|{mod_time}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_cache_path(file_path, model=None, operation=None):
    """
    Get the cache file path for a given input file and operation.
    
    Args:
        file_path (Path): Path to the original file
        model (str, optional): Model used for processing
        operation (str, optional): Operation type (e.g., 'transcribe', 'summarize')
        
    Returns:
        Path: Path to the cache file
    """
    file_path = Path(file_path)
    file_hash = get_file_hash(file_path)
    
    if not file_hash:
        return None
    
    # Create cache directory if it doesn't exist
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a cache filename based on the hash and optional parameters
    cache_name = file_hash
    if model:
        cache_name += f"_{model}"
    if operation:
        cache_name += f"_{operation}"
    
    return cache_dir / f"{cache_name}.json"


def save_to_cache(file_path, data, model=None, operation=None):
    """
    Save data to cache.
    
    Args:
        file_path (Path): Path to the original file
        data (dict): Data to cache
        model (str, optional): Model used for processing
        operation (str, optional): Operation type
        
    Returns:
        bool: True if successful, False otherwise
    """
    cache_path = get_cache_path(file_path, model, operation)
    if not cache_path:
        return False
    
    try:
        # Add metadata to the cached data
        cache_data = {
            "original_file": str(Path(file_path).absolute()),
            "timestamp": time.time(),
            "model": model,
            "operation": operation,
            "data": data
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Cached data saved to {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        return False


def load_from_cache(file_path, model=None, operation=None, max_age=None):
    """
    Load data from cache if available and not expired.
    
    Args:
        file_path (Path): Path to the original file
        model (str, optional): Model used for processing
        operation (str, optional): Operation type
        max_age (float, optional): Maximum age of cache in seconds
        
    Returns:
        dict or None: Cached data or None if not available
    """
    cache_path = get_cache_path(file_path, model, operation)
    if not cache_path or not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if cache is expired
        if max_age is not None:
            cache_time = cache_data.get("timestamp", 0)
            if time.time() - cache_time > max_age:
                logger.info(f"Cache expired for {file_path}")
                return None
        
        logger.info(f"Loaded data from cache: {cache_path}")
        return cache_data.get("data")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None


def clear_cache(max_age=None):
    """
    Clear all cache files or only expired ones.
    
    Args:
        max_age (float, optional): Maximum age of cache in seconds
        
    Returns:
        int: Number of files deleted
    """
    if not CACHE_DIR.exists():
        return 0
    
    count = 0
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            if max_age is not None:
                # Check if file is expired
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                cache_time = cache_data.get("timestamp", 0)
                if time.time() - cache_time <= max_age:
                    continue  # Skip non-expired files
            
            # Delete the file
            os.remove(cache_file)
            count += 1
        except Exception as e:
            logger.error(f"Error deleting cache file {cache_file}: {e}")
    
    logger.info(f"Cleared {count} cache files")
    return count


def get_cache_size():
    """
    Get the total size of the cache directory.
    
    Returns:
        tuple: (size_bytes, file_count)
    """
    if not CACHE_DIR.exists():
        return 0, 0
    
    total_size = 0
    file_count = 0
    
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            total_size += cache_file.stat().st_size
            file_count += 1
        except Exception:
            pass
    
    return total_size, file_count 