from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)


def validate_environment(obs_path: Path = None):
    """Validate environment and prerequisites."""
    errors = []
    
    if obs_path and not obs_path.exists():
        errors.append(f"Directory not found: {obs_path}")
    
    if not shutil.which("ffmpeg"):
        errors.append("FFmpeg is not installed or not in PATH. Install it from https://ffmpeg.org/download.html")
    
    return errors


def get_system_capabilities():
    """Return a dict of detected system capabilities for display."""
    import torch
    
    caps = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "gpu_name": None,
        "gpu_memory": None,
    }
    
    if caps["cuda"] and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        caps["gpu_name"] = props.name
        caps["gpu_memory"] = props.total_memory
    
    return caps
