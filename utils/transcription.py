import whisper
from pathlib import Path
from utils.audio_processing import extract_audio
import logging
import torch
import streamlit as st

try:
    from utils.gpu_utils import configure_gpu, get_optimal_device
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

try:
    from utils.cache import load_from_cache, save_to_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHISPER_MODEL = "base"


@st.cache_resource
def _load_whisper_model(model_name, device_str):
    """Load and cache a Whisper model. Cached across reruns."""
    logger.info(f"Loading Whisper model: {model_name} on {device_str}")
    device = torch.device(device_str)
    return whisper.load_model(model_name, device=device if device.type != "mps" else "cpu")


def transcribe_audio(audio_path: Path, model=WHISPER_MODEL, use_cache=True, cache_max_age=None, 
                     use_gpu=True, memory_fraction=0.8):
    """
    Transcribe audio using Whisper and return both segments and full transcript.
    
    Args:
        audio_path (Path): Path to the audio or video file
        model (str): Whisper model size to use (tiny, base, small, medium, large)
        use_cache (bool): Whether to use caching
        cache_max_age (float, optional): Maximum age of cache in seconds
        use_gpu (bool): Whether to use GPU acceleration if available
        memory_fraction (float): Fraction of GPU memory to use (0.0 to 1.0)
        
    Returns:
        tuple: (segments, transcript) where segments is a list of dicts with timing info
    """
    audio_path = Path(audio_path)
    
    if use_cache and CACHE_AVAILABLE:
        cached_data = load_from_cache(audio_path, model, "transcribe", cache_max_age)
        if cached_data:
            logger.info(f"Using cached transcription for {audio_path}")
            return cached_data.get("segments", []), cached_data.get("transcript", "")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if audio_path.suffix.lower() in video_extensions:
        audio_path = extract_audio(audio_path)
    
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        gpu_config = configure_gpu(model, memory_fraction)
        device = gpu_config["device"]
        logger.info(f"Using device: {device} for transcription")
    
    whisper_model = _load_whisper_model(model, str(device))
    
    logger.info(f"Transcribing audio: {audio_path}")
    result = whisper_model.transcribe(str(audio_path))
    
    transcript = result["text"]
    segments = result["segments"]
    
    if use_cache and CACHE_AVAILABLE:
        cache_data = {
            "transcript": transcript,
            "segments": segments
        }
        save_to_cache(audio_path, cache_data, model, "transcribe")
    
    return segments, transcript