"""
Subtitle export utilities for the OBS Recording Transcriber.
Supports exporting transcripts to SRT, ASS, and WebVTT subtitle formats.
"""

from pathlib import Path
import re
from datetime import timedelta
import gzip
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_timestamp_srt(timestamp_ms):
    """
    Format a timestamp in milliseconds to SRT format (HH:MM:SS,mmm).
    
    Args:
        timestamp_ms (int): Timestamp in milliseconds
        
    Returns:
        str: Formatted timestamp string
    """
    hours, remainder = divmod(timestamp_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"


def format_timestamp_ass(timestamp_ms):
    """
    Format a timestamp in milliseconds to ASS format (H:MM:SS.cc).
    
    Args:
        timestamp_ms (int): Timestamp in milliseconds
        
    Returns:
        str: Formatted timestamp string
    """
    hours, remainder = divmod(timestamp_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, remainder = divmod(remainder, 1000)
    centiseconds = remainder // 10
    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}.{int(centiseconds):02d}"


def format_timestamp_vtt(timestamp_ms):
    """
    Format a timestamp in milliseconds to WebVTT format (HH:MM:SS.mmm).
    
    Args:
        timestamp_ms (int): Timestamp in milliseconds
        
    Returns:
        str: Formatted timestamp string
    """
    hours, remainder = divmod(timestamp_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"


def export_to_srt(segments, output_path):
    """
    Export transcript segments to SRT format.
    
    Args:
        segments (list): List of transcript segments with start, end, and text
        output_path (Path): Path to save the SRT file
        
    Returns:
        Path: Path to the saved SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp_srt(int(segment['start'] * 1000))
            end_time = format_timestamp_srt(int(segment['end'] * 1000))
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    
    return output_path


def export_to_ass(segments, output_path, video_width=1920, video_height=1080, style=None):
    """
    Export transcript segments to ASS format with styling.
    
    Args:
        segments (list): List of transcript segments with start, end, and text
        output_path (Path): Path to save the ASS file
        video_width (int): Width of the video in pixels
        video_height (int): Height of the video in pixels
        style (dict, optional): Custom style parameters
        
    Returns:
        Path: Path to the saved ASS file
    """
    # Default style
    default_style = {
        "fontname": "Arial",
        "fontsize": "48",
        "primary_color": "&H00FFFFFF",  # White
        "secondary_color": "&H000000FF",  # Blue
        "outline_color": "&H00000000",  # Black
        "back_color": "&H80000000",  # Semi-transparent black
        "bold": "-1",  # True
        "italic": "0",  # False
        "alignment": "2",  # Bottom center
    }
    
    # Apply custom style if provided
    if style:
        default_style.update(style)
    
    # ASS header template
    ass_header = f"""[Script Info]
Title: Transcription
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{default_style['fontname']},{default_style['fontsize']},{default_style['primary_color']},{default_style['secondary_color']},{default_style['outline_color']},{default_style['back_color']},{default_style['bold']},{default_style['italic']},0,0,100,100,0,0,1,2,2,{default_style['alignment']},10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        
        for segment in segments:
            start_time = format_timestamp_ass(int(segment['start'] * 1000))
            end_time = format_timestamp_ass(int(segment['end'] * 1000))
            text = segment['text'].strip().replace('\n', '\\N')
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
    
    return output_path


def export_to_vtt(segments, output_path):
    """
    Export transcript segments to WebVTT format.
    
    Args:
        segments (list): List of transcript segments with start, end, and text
        output_path (Path): Path to save the WebVTT file
        
    Returns:
        Path: Path to the saved WebVTT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # WebVTT header
        f.write("WEBVTT\n\n")
        
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp_vtt(int(segment['start'] * 1000))
            end_time = format_timestamp_vtt(int(segment['end'] * 1000))
            
            # Optional cue identifier
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    
    return output_path


def transcript_to_segments(transcript, segment_duration=5.0):
    """
    Convert a plain transcript to timed segments for subtitle export.
    Used when the original segments are not available.
    
    Args:
        transcript (str): Full transcript text
        segment_duration (float): Duration of each segment in seconds
        
    Returns:
        list: List of segments with start, end, and text
    """
    # Split transcript into sentences
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    segments = []
    
    current_time = 0.0
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Estimate duration based on word count (approx. 2.5 words per second)
        word_count = len(sentence.split())
        duration = max(2.0, word_count / 2.5)
        
        segments.append({
            'start': current_time,
            'end': current_time + duration,
            'text': sentence
        })
        
        current_time += duration
    
    return segments


def compress_file(input_path, compression_type='gzip'):
    """
    Compress a file using the specified compression method.
    
    Args:
        input_path (Path): Path to the file to compress
        compression_type (str): Type of compression ('gzip' or 'zip')
        
    Returns:
        Path: Path to the compressed file
    """
    input_path = Path(input_path)
    
    if compression_type == 'gzip':
        output_path = input_path.with_suffix(input_path.suffix + '.gz')
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        return output_path
    
    elif compression_type == 'zip':
        output_path = input_path.with_suffix('.zip')
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(input_path, arcname=input_path.name)
        return output_path
    
    else:
        logger.warning(f"Unsupported compression type: {compression_type}")
        return input_path


def export_transcript(transcript, output_path, format_type='srt', segments=None, 
                      compress=False, compression_type='gzip', style=None):
    """
    Export transcript to the specified subtitle format.
    
    Args:
        transcript (str): Full transcript text
        output_path (Path): Base path for the output file (without extension)
        format_type (str): 'srt', 'ass', or 'vtt'
        segments (list, optional): List of transcript segments with timing information
        compress (bool): Whether to compress the output file
        compression_type (str): Type of compression ('gzip' or 'zip')
        style (dict, optional): Custom style parameters for ASS format
        
    Returns:
        Path: Path to the saved subtitle file
    """
    output_path = Path(output_path)
    
    # If segments are not provided, create them from the transcript
    if segments is None:
        segments = transcript_to_segments(transcript)
    
    if format_type.lower() == 'srt':
        output_file = output_path.with_suffix('.srt')
        result_path = export_to_srt(segments, output_file)
    elif format_type.lower() == 'ass':
        output_file = output_path.with_suffix('.ass')
        result_path = export_to_ass(segments, output_file, style=style)
    elif format_type.lower() == 'vtt':
        output_file = output_path.with_suffix('.vtt')
        result_path = export_to_vtt(segments, output_file)
    else:
        raise ValueError(f"Unsupported format type: {format_type}. Use 'srt', 'ass', or 'vtt'.")
    
    # Compress the file if requested
    if compress:
        result_path = compress_file(result_path, compression_type)
    
    return result_path 