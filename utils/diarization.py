"""
Speaker diarization utilities for the OBS Recording Transcriber.
Provides functions to identify different speakers in audio recordings.
"""

import logging
import os
import numpy as np
from pathlib import Path
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import GPU utilities, but don't fail if not available
try:
    from utils.gpu_utils import get_optimal_device
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Default HuggingFace auth token environment variable
HF_TOKEN_ENV = "HF_TOKEN"


def get_diarization_pipeline(use_gpu=True, hf_token=None):
    """
    Initialize the speaker diarization pipeline.
    
    Args:
        use_gpu (bool): Whether to use GPU acceleration if available
        hf_token (str, optional): HuggingFace API token for accessing the model
        
    Returns:
        Pipeline or None: Diarization pipeline if successful, None otherwise
    """
    # Check if token is provided or in environment
    if hf_token is None:
        hf_token = os.environ.get(HF_TOKEN_ENV)
        if hf_token is None:
            logger.error(f"HuggingFace token not provided. Set {HF_TOKEN_ENV} environment variable or pass token directly.")
            return None
    
    try:
        # Configure device
        device = torch.device("cpu")
        if use_gpu and GPU_UTILS_AVAILABLE:
            device = get_optimal_device()
            logger.info(f"Using device: {device} for diarization")
        
        # Initialize the pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token
        )
        
        # Move to appropriate device
        if device.type == "cuda":
            pipeline = pipeline.to(torch.device(device))
        
        return pipeline
    except Exception as e:
        logger.error(f"Error initializing diarization pipeline: {e}")
        return None


def diarize_audio(audio_path, pipeline=None, num_speakers=None, use_gpu=True, hf_token=None):
    """
    Perform speaker diarization on an audio file.
    
    Args:
        audio_path (Path): Path to the audio file
        pipeline (Pipeline, optional): Pre-initialized diarization pipeline
        num_speakers (int, optional): Number of speakers (if known)
        use_gpu (bool): Whether to use GPU acceleration if available
        hf_token (str, optional): HuggingFace API token
        
    Returns:
        dict: Dictionary mapping time segments to speaker IDs
    """
    audio_path = Path(audio_path)
    
    # Initialize pipeline if not provided
    if pipeline is None:
        pipeline = get_diarization_pipeline(use_gpu, hf_token)
        if pipeline is None:
            return None
    
    try:
        # Run diarization
        logger.info(f"Running speaker diarization on {audio_path}")
        diarization = pipeline(audio_path, num_speakers=num_speakers)
        
        # Extract speaker segments
        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = (turn.start, turn.end)
            speaker_segments[segment] = speaker
        
        return speaker_segments
    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        return None


def apply_diarization_to_transcript(transcript_segments, speaker_segments):
    """
    Apply speaker diarization results to transcript segments.
    
    Args:
        transcript_segments (list): List of transcript segments with timing info
        speaker_segments (dict): Dictionary mapping time segments to speaker IDs
        
    Returns:
        list: Updated transcript segments with speaker information
    """
    if not speaker_segments:
        return transcript_segments
    
    # Convert speaker segments to a more usable format
    speaker_ranges = [(Segment(start, end), speaker) 
                     for (start, end), speaker in speaker_segments.items()]
    
    # Update transcript segments with speaker information
    for segment in transcript_segments:
        segment_start = segment['start']
        segment_end = segment['end']
        segment_range = Segment(segment_start, segment_end)
        
        # Find overlapping speaker segments
        overlaps = []
        for (spk_range, speaker) in speaker_ranges:
            overlap = segment_range.intersect(spk_range)
            if overlap:
                overlaps.append((overlap.duration, speaker))
        
        # Assign the speaker with the most overlap
        if overlaps:
            overlaps.sort(reverse=True)  # Sort by duration (descending)
            segment['speaker'] = overlaps[0][1]
        else:
            segment['speaker'] = "UNKNOWN"
    
    return transcript_segments


def format_transcript_with_speakers(transcript_segments):
    """
    Format transcript with speaker labels.
    
    Args:
        transcript_segments (list): List of transcript segments with speaker info
        
    Returns:
        str: Formatted transcript with speaker labels
    """
    formatted_lines = []
    current_speaker = None
    
    for segment in transcript_segments:
        speaker = segment.get('speaker', 'UNKNOWN')
        text = segment['text'].strip()
        
        # Add speaker label when speaker changes
        if speaker != current_speaker:
            formatted_lines.append(f"\n[{speaker}]")
            current_speaker = speaker
        
        formatted_lines.append(text)
    
    return " ".join(formatted_lines)


def transcribe_with_diarization(audio_path, whisper_model="base", num_speakers=None, 
                               use_gpu=True, hf_token=None):
    """
    Transcribe audio with speaker diarization.
    
    Args:
        audio_path (Path): Path to the audio file
        whisper_model (str): Whisper model size to use
        num_speakers (int, optional): Number of speakers (if known)
        use_gpu (bool): Whether to use GPU acceleration if available
        hf_token (str, optional): HuggingFace API token
        
    Returns:
        tuple: (diarized_segments, formatted_transcript)
    """
    audio_path = Path(audio_path)
    
    # Configure device
    device = torch.device("cpu")
    if use_gpu and GPU_UTILS_AVAILABLE:
        device = get_optimal_device()
    
    try:
        # Step 1: Transcribe audio with Whisper
        logger.info(f"Transcribing audio with Whisper model: {whisper_model}")
        model = whisper.load_model(whisper_model, device=device if device.type != "mps" else "cpu")
        result = model.transcribe(str(audio_path))
        transcript_segments = result["segments"]
        
        # Step 2: Perform speaker diarization
        logger.info("Performing speaker diarization")
        pipeline = get_diarization_pipeline(use_gpu, hf_token)
        if pipeline is None:
            logger.warning("Diarization pipeline not available, returning transcript without speakers")
            return transcript_segments, result["text"]
        
        speaker_segments = diarize_audio(audio_path, pipeline, num_speakers, use_gpu)
        
        # Step 3: Apply diarization to transcript
        if speaker_segments:
            diarized_segments = apply_diarization_to_transcript(transcript_segments, speaker_segments)
            formatted_transcript = format_transcript_with_speakers(diarized_segments)
            return diarized_segments, formatted_transcript
        else:
            return transcript_segments, result["text"]
    
    except Exception as e:
        logger.error(f"Error in transcribe_with_diarization: {e}")
        return None, None 