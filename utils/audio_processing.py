from pathlib import Path

# moviepy 2.x removed moviepy.editor; import directly from moviepy
try:
    from moviepy import AudioFileClip
except ImportError:
    # Fallback for moviepy 1.x
    from moviepy.editor import AudioFileClip

def extract_audio(video_path: Path):
    """Extract audio from a video file."""
    try:
        audio = AudioFileClip(str(video_path))
        audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
        audio.write_audiofile(str(audio_path), verbose=False, logger=None)
        audio.close()
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")
