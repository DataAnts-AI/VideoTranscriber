from pathlib import Path
import tempfile
import os
import logging

try:
    from moviepy import AudioFileClip
except ImportError:
    from moviepy.editor import AudioFileClip

logger = logging.getLogger(__name__)

_temp_audio_files = []


def extract_audio(video_path: Path):
    """Extract audio from a video file into a temp directory for automatic cleanup."""
    try:
        audio = AudioFileClip(str(video_path))
        temp_dir = tempfile.mkdtemp(prefix="videotranscriber_")
        audio_path = Path(temp_dir) / f"{video_path.stem}_audio.wav"
        audio.write_audiofile(str(audio_path), verbose=False, logger=None)
        audio.close()
        _temp_audio_files.append(str(audio_path))
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def cleanup_temp_audio():
    """Remove all temporary audio files created during processing."""
    cleaned = 0
    for fpath in _temp_audio_files:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                parent = os.path.dirname(fpath)
                if os.path.isdir(parent) and not os.listdir(parent):
                    os.rmdir(parent)
                cleaned += 1
        except Exception as e:
            logger.warning(f"Could not remove temp file {fpath}: {e}")
    _temp_audio_files.clear()
    return cleaned


def get_video_duration(video_path: Path):
    """Get duration of a video/audio file in seconds."""
    try:
        clip = AudioFileClip(str(video_path))
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        return None
