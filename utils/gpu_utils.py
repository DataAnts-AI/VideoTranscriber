"""
GPU utilities for the OBS Recording Transcriber.
Provides functions to detect and configure GPU acceleration.
"""

import logging
import os
import platform
import subprocess
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gpu_info():
    """
    Get information about available GPUs.
    
    Returns:
        dict: Information about available GPUs
    """
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [],
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    }
    
    # Get CUDA device information
    if gpu_info["cuda_available"]:
        for i in range(gpu_info["cuda_device_count"]):
            device_props = torch.cuda.get_device_properties(i)
            gpu_info["cuda_devices"].append({
                "index": i,
                "name": device_props.name,
                "total_memory": device_props.total_memory,
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })
    
    return gpu_info


def get_optimal_device():
    """
    Get the optimal device for computation.
    
    Returns:
        torch.device: The optimal device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        # If multiple GPUs are available, select the one with the most memory
        if torch.cuda.device_count() > 1:
            max_memory = 0
            best_device = 0
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                if device_props.total_memory > max_memory:
                    max_memory = device_props.total_memory
                    best_device = i
            return torch.device(f"cuda:{best_device}")
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_memory_limits(memory_fraction=0.8):
    global torch
    import torch
    """
    Set memory limits for GPU usage.
    
    Args:
        memory_fraction (float): Fraction of GPU memory to use (0.0 to 1.0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Set memory fraction for each device
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(memory_fraction, i)
        
        return True
    except Exception as e:
        logger.error(f"Error setting memory limits: {e}")
        return False


def optimize_for_inference():
    """
    Apply optimizations for inference.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        
        # Enable cuDNN benchmark mode for optimized performance
        torch.backends.cudnn.benchmark = True
        
        # Disable gradient calculation for inference
        torch.set_grad_enabled(False)
        
        return True
    except Exception as e:
        logger.error(f"Error optimizing for inference: {e}")
        return False


def get_recommended_batch_size(model_size="base"):
    """
    Get recommended batch size based on available GPU memory.
    
    Args:
        model_size (str): Size of the model (tiny, base, small, medium, large)
        
    Returns:
        int: Recommended batch size
    """
    # Default batch sizes for CPU
    default_batch_sizes = {
        "tiny": 16,
        "base": 8,
        "small": 4,
        "medium": 2,
        "large": 1
    }
    
    # If CUDA is not available, return default CPU batch size
    if not torch.cuda.is_available():
        return default_batch_sizes.get(model_size, 1)
    
    # Approximate memory requirements in GB for different model sizes
    memory_requirements = {
        "tiny": 1,
        "base": 2,
        "small": 4,
        "medium": 8,
        "large": 16
    }
    
    # Get available GPU memory
    device = get_optimal_device()
    if device.type == "cuda":
        device_idx = device.index
        device_props = torch.cuda.get_device_properties(device_idx)
        available_memory_gb = device_props.total_memory / (1024 ** 3)
        
        # Calculate batch size based on available memory
        model_memory = memory_requirements.get(model_size, 2)
        max_batch_size = int(available_memory_gb / model_memory)
        
        # Ensure batch size is at least 1
        return max(1, max_batch_size)
    
    # For MPS or other devices, return default
    return default_batch_sizes.get(model_size, 1)


def configure_gpu(model_size="base", memory_fraction=0.8):
    """
    Configure GPU settings for optimal performance.
    
    Args:
        model_size (str): Size of the model (tiny, base, small, medium, large)
        memory_fraction (float): Fraction of GPU memory to use (0.0 to 1.0)
        
    Returns:
        dict: Configuration information
    """
    gpu_info = get_gpu_info()
    device = get_optimal_device()
    
    # Set memory limits if using CUDA
    if device.type == "cuda":
        set_memory_limits(memory_fraction)
    
    # Apply inference optimizations
    optimize_for_inference()
    
    # Get recommended batch size
    batch_size = get_recommended_batch_size(model_size)
    
    config = {
        "device": device,
        "batch_size": batch_size,
        "gpu_info": gpu_info,
        "memory_fraction": memory_fraction if device.type == "cuda" else None
    }
    
    logger.info(f"GPU configuration: Using {device} with batch size {batch_size}")
    return config 