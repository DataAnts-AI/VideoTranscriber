import whisper
from pathlib import Path
from transformers import pipeline, AutoTokenizer
from utils.audio_processing import extract_audio
from utils.summarization import summarize_text
import logging
import torch

# Try to import GPU utilities, but don't fail if not available
try:
    from utils.gpu_utils import configure_gpu, get_optimal_device
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Try to import caching utilities, but don't fail if not available
try:
    from utils.cache import load_from_cache, save_to_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHISPER_MODEL = "base"

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
    
    # Check cache first if enabled
    if use_cache and CACHE_AVAILABLE:
        cached_data = load_from_cache(audio_path, model, "transcribe", cache_max_age)
        if cached_data:
            logger.info(f"Using cached transcription for {audio_path}")
            return cached_data.get("segments", []), cached_data.get("transcript", "")
    
    # Extract audio if the input is a video file (M4A is already audio)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if audio_path.suffix.lower() in video_extensions:
        audio_path = extract_audio(audio_path)
    
    # Configure GPU if available and requested
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        gpu_config = configure_gpu(model, memory_fraction)
        device = gpu_config["device"]
        logger.info(f"Using device: {device} for transcription")
    
    # Load the specified Whisper model
    logger.info(f"Loading Whisper model: {model}")
    whisper_model = whisper.load_model(model, device=device if device.type != "mps" else "cpu")
    
    # Transcribe the audio
    logger.info(f"Transcribing audio: {audio_path}")
    result = whisper_model.transcribe(str(audio_path))
    
    # Extract the full transcript and segments
    transcript = result["text"]
    segments = result["segments"]
    
    # Cache the results if caching is enabled
    if use_cache and CACHE_AVAILABLE:
        cache_data = {
            "transcript": transcript,
            "segments": segments
        }
        save_to_cache(audio_path, cache_data, model, "transcribe")
    
    return segments, transcript