import streamlit as st
from utils.audio_processing import extract_audio
from utils.transcription import transcribe_audio
from utils.summarization import summarize_text
from utils.validation import validate_environment
from utils.export import export_transcript
from pathlib import Path
import os
import logging
import humanize
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Ollama integration, but don't fail if it's not available
try:
    from utils.ollama_integration import check_ollama_available, list_available_models, chunk_and_summarize
    OLLAMA_AVAILABLE = check_ollama_available()
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import GPU utilities, but don't fail if not available
try:
    from utils.gpu_utils import get_gpu_info, configure_gpu
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Try to import caching utilities, but don't fail if not available
try:
    from utils.cache import get_cache_size, clear_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Try to import diarization utilities, but don't fail if not available
try:
    from utils.diarization import transcribe_with_diarization
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

# Try to import translation utilities, but don't fail if not available
try:
    from utils.translation import transcribe_and_translate, get_language_name
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

# Try to import keyword extraction utilities, but don't fail if not available
try:
    from utils.keyword_extraction import extract_keywords_from_transcript, generate_keyword_index, generate_interactive_transcript
    KEYWORD_EXTRACTION_AVAILABLE = True
except ImportError:
    KEYWORD_EXTRACTION_AVAILABLE = False

def main():
    # Set page configuration
    st.set_page_config(
        page_title="OBS Recording Transcriber",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stDownloadButton>button {
        width: 100%;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .speaker {
        font-weight: bold;
        color: #1E88E5;
    }
    .timestamp {
        color: #757575;
        font-size: 0.9em;
        margin-right: 8px;
    }
    .keyword {
        background-color: #FFF9C4;
        padding: 0 2px;
        border-radius: 3px;
    }
    .interactive-transcript p {
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎥 OBS Recording Transcriber")
    st.caption("Process your OBS recordings with AI transcription and summarization")

    # Sidebar configuration
    st.sidebar.header("Settings")
    
    # Allow the user to select a base folder
    base_folder = st.sidebar.text_input(
        "Enter the base folder path:",
        value=str(Path.home())
    )
    
    base_path = Path(base_folder)

    # Model selection
    st.sidebar.subheader("Model Settings")
    
    # Transcription model selection
    transcription_model = st.sidebar.selectbox(
        "Transcription Model",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Select the Whisper model size. Larger models are more accurate but slower."
    )
    
    # Summarization model selection
    summarization_options = ["Hugging Face (Online)", "Ollama (Local)"] if OLLAMA_AVAILABLE else ["Hugging Face (Online)"]
    summarization_method = st.sidebar.selectbox(
        "Summarization Method",
        summarization_options,
        index=0,
        help="Select the summarization method. Ollama runs locally but requires installation."
    )
    
    # If Ollama is selected, show model selection
    ollama_model = None
    if OLLAMA_AVAILABLE and summarization_method == "Ollama (Local)":
        available_models = list_available_models()
        if available_models:
            ollama_model = st.sidebar.selectbox(
                "Ollama Model",
                available_models,
                index=0 if "llama3" in available_models else 0,
                help="Select the Ollama model to use for summarization."
            )
        else:
            st.sidebar.warning("No Ollama models found. Please install models using 'ollama pull model_name'.")
    
    # Advanced features
    st.sidebar.subheader("Advanced Features")
    
    # Speaker diarization
    use_diarization = st.sidebar.checkbox(
        "Speaker Diarization", 
        value=False,
        disabled=not DIARIZATION_AVAILABLE,
        help="Identify different speakers in the recording."
    )
    
    # Show HF token input if diarization is enabled
    hf_token = None
    if use_diarization and DIARIZATION_AVAILABLE:
        hf_token = st.sidebar.text_input(
            "HuggingFace Token",
            type="password",
            help="Required for speaker diarization. Get your token at huggingface.co/settings/tokens"
        )
        
        num_speakers = st.sidebar.number_input(
            "Number of Speakers",
            min_value=1,
            max_value=10,
            value=2,
            help="Specify the number of speakers if known, or leave at default for auto-detection."
        )
    
    # Translation
    use_translation = st.sidebar.checkbox(
        "Translation",
        value=False,
        disabled=not TRANSLATION_AVAILABLE,
        help="Translate the transcript to another language."
    )
    
    # Target language selection if translation is enabled
    target_lang = None
    if use_translation and TRANSLATION_AVAILABLE:
        target_lang = st.sidebar.selectbox(
            "Target Language",
            ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar"],
            format_func=lambda x: f"{get_language_name(x)} ({x})",
            help="Select the language to translate to."
        )
    
    # Keyword extraction
    use_keywords = st.sidebar.checkbox(
        "Keyword Extraction",
        value=False,
        disabled=not KEYWORD_EXTRACTION_AVAILABLE,
        help="Extract keywords and link them to timestamps."
    )
    
    if use_keywords and KEYWORD_EXTRACTION_AVAILABLE:
        max_keywords = st.sidebar.slider(
            "Max Keywords",
            min_value=5,
            max_value=30,
            value=15,
            help="Maximum number of keywords to extract."
        )
    
    # Performance settings
    st.sidebar.subheader("Performance Settings")
    
    # GPU acceleration
    use_gpu = st.sidebar.checkbox(
        "Use GPU Acceleration", 
        value=True if GPU_UTILS_AVAILABLE else False,
        disabled=not GPU_UTILS_AVAILABLE,
        help="Use GPU for faster processing if available."
    )
    
    # Show GPU info if available
    if GPU_UTILS_AVAILABLE and use_gpu:
        gpu_info = get_gpu_info()
        if gpu_info["cuda_available"]:
            gpu_devices = [f"{d['name']} ({humanize.naturalsize(d['total_memory'])})" for d in gpu_info["cuda_devices"]]
            st.sidebar.info(f"GPU(s) available: {', '.join(gpu_devices)}")
        elif gpu_info["mps_available"]:
            st.sidebar.info("Apple Silicon GPU (MPS) available")
        else:
            st.sidebar.warning("No GPU detected. Using CPU.")
    
    # Memory usage
    memory_fraction = st.sidebar.slider(
        "GPU Memory Usage",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.1,
        disabled=not (GPU_UTILS_AVAILABLE and use_gpu),
        help="Fraction of GPU memory to use. Lower if you encounter out-of-memory errors."
    )
    
    # Caching options
    use_cache = st.sidebar.checkbox(
        "Use Caching",
        value=True if CACHE_AVAILABLE else False,
        disabled=not CACHE_AVAILABLE,
        help="Cache transcription results to avoid reprocessing the same files."
    )
    
    # Cache management
    if CACHE_AVAILABLE and use_cache:
        cache_size, cache_files = get_cache_size()
        if cache_size > 0:
            st.sidebar.info(f"Cache: {humanize.naturalsize(cache_size)} ({cache_files} files)")
            if st.sidebar.button("Clear Cache"):
                cleared = clear_cache()
                st.sidebar.success(f"Cleared {cleared} cache files")
    
    # Export options
    st.sidebar.subheader("Export Options")
    export_format = st.sidebar.multiselect(
        "Export Formats",
        ["TXT", "SRT", "VTT", "ASS"],
        default=["TXT"],
        help="Select the formats to export the transcript."
    )
    
    # Compression options
    compress_exports = st.sidebar.checkbox(
        "Compress Exports",
        value=False,
        help="Compress exported files to save space."
    )
    
    if compress_exports:
        compression_type = st.sidebar.radio(
            "Compression Format",
            ["gzip", "zip"],
            index=0,
            help="Select the compression format for exported files."
        )
    else:
        compression_type = None
    
    # ASS subtitle styling
    if "ASS" in export_format:
        st.sidebar.subheader("ASS Subtitle Styling")
        show_style_options = st.sidebar.checkbox("Customize ASS Style", value=False)
        
        if show_style_options:
            ass_style = {}
            ass_style["fontname"] = st.sidebar.selectbox(
                "Font", 
                ["Arial", "Helvetica", "Times New Roman", "Courier New", "Comic Sans MS"],
                index=0
            )
            ass_style["fontsize"] = st.sidebar.slider("Font Size", 12, 72, 48)
            ass_style["alignment"] = st.sidebar.selectbox(
                "Alignment", 
                ["2 (Bottom Center)", "1 (Bottom Left)", "3 (Bottom Right)", "8 (Top Center)"],
                index=0
            ).split()[0]  # Extract just the number
            ass_style["bold"] = "-1" if st.sidebar.checkbox("Bold", value=True) else "0"
            ass_style["italic"] = "-1" if st.sidebar.checkbox("Italic", value=False) else "0"
        else:
            ass_style = None

    # Validate environment
    env_errors = validate_environment(base_path)
    if env_errors:
        st.error("## Environment Issues")
        for error in env_errors:
            st.markdown(f"- {error}")
        return

    # File selection - support multiple video and audio formats
    supported_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4a"]
    recordings = []
    for extension in supported_extensions:
        recordings.extend(base_path.glob(extension))
    
    if not recordings:
        st.warning(f"📂 No recordings found in the folder: {base_folder}!")
        st.info("💡 Supported formats: MP4, AVI, MOV, MKV, M4A")
        return

    selected_file = st.selectbox("Choose a recording", recordings)

    # Process button with spinner
    if st.button("🚀 Start Processing"):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Update progress
            status_text.text("Extracting audio...")
            progress_bar.progress(10)
            
            # Process based on selected features
            if use_diarization and DIARIZATION_AVAILABLE and hf_token:
                # Transcribe with speaker diarization
                status_text.text("Transcribing with speaker diarization...")
                num_speakers_arg = int(num_speakers) if num_speakers > 0 else None
                diarized_segments, diarized_transcript = transcribe_with_diarization(
                    selected_file,
                    whisper_model=transcription_model,
                    num_speakers=num_speakers_arg,
                    use_gpu=use_gpu,
                    hf_token=hf_token
                )
                segments = diarized_segments
                transcript = diarized_transcript
            elif use_translation and TRANSLATION_AVAILABLE:
                # Transcribe and translate
                status_text.text("Transcribing and translating...")
                original_segments, translated_segments, original_transcript, translated_transcript = transcribe_and_translate(
                    selected_file,
                    whisper_model=transcription_model,
                    target_lang=target_lang,
                    use_gpu=use_gpu
                )
                segments = translated_segments
                transcript = translated_transcript
                # Store original for display
                original_text = original_transcript
            else:
                # Standard transcription
                status_text.text("Transcribing audio...")
                segments, transcript = transcribe_audio(
                    selected_file, 
                    model=transcription_model,
                    use_cache=use_cache,
                    use_gpu=use_gpu,
                    memory_fraction=memory_fraction
                )
            
            progress_bar.progress(50)
            
            if transcript:
                # Extract keywords if requested
                keyword_timestamps = None
                entity_timestamps = None
                if use_keywords and KEYWORD_EXTRACTION_AVAILABLE:
                    status_text.text("Extracting keywords...")
                    keyword_timestamps, entity_timestamps = extract_keywords_from_transcript(
                        transcript, 
                        segments, 
                        max_keywords=max_keywords,
                        use_gpu=use_gpu
                    )
                    
                    # Generate keyword index
                    keyword_index = generate_keyword_index(keyword_timestamps, entity_timestamps)
                    
                    # Generate interactive transcript
                    interactive_transcript = generate_interactive_transcript(
                        segments, 
                        keyword_timestamps, 
                        entity_timestamps
                    )
                
                # Generate summary based on selected method
                status_text.text("Generating summary...")
                if OLLAMA_AVAILABLE and summarization_method == "Ollama (Local)" and ollama_model:
                    summary = chunk_and_summarize(transcript, model=ollama_model)
                    if not summary:
                        st.warning("Ollama summarization failed. Falling back to Hugging Face.")
                        summary = summarize_text(
                            transcript,
                            use_gpu=use_gpu,
                            memory_fraction=memory_fraction
                        )
                else:
                    summary = summarize_text(
                        transcript,
                        use_gpu=use_gpu,
                        memory_fraction=memory_fraction
                    )
                
                progress_bar.progress(80)
                status_text.text("Preparing results...")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["Summary", "Transcript", "Advanced"])
                
                with tab1:
                    st.subheader("🖍 Summary")
                    st.write(summary)
                    
                    # If translation was used, show original language
                    if use_translation and TRANSLATION_AVAILABLE and 'original_text' in locals():
                        with st.expander("Original Language Summary"):
                            original_summary = summarize_text(
                                original_text,
                                use_gpu=use_gpu,
                                memory_fraction=memory_fraction
                            )
                            st.write(original_summary)
                
                with tab2:
                    st.subheader("📜 Full Transcript")
                    
                    # Show interactive transcript if keywords were extracted
                    if use_keywords and KEYWORD_EXTRACTION_AVAILABLE and 'interactive_transcript' in locals():
                        st.markdown(interactive_transcript, unsafe_allow_html=True)
                    else:
                        st.text(transcript)
                    
                    # If translation was used, show original language
                    if use_translation and TRANSLATION_AVAILABLE and 'original_text' in locals():
                        with st.expander("Original Language Transcript"):
                            st.text(original_text)
                
                with tab3:
                    # Show keyword index if available
                    if use_keywords and KEYWORD_EXTRACTION_AVAILABLE and 'keyword_index' in locals():
                        st.subheader("🔑 Keyword Index")
                        st.markdown(keyword_index)
                    
                    # Show speaker information if available
                    if use_diarization and DIARIZATION_AVAILABLE:
                        st.subheader("🎙️ Speaker Information")
                        speakers = set(segment.get('speaker', 'UNKNOWN') for segment in segments)
                        st.write(f"Detected {len(speakers)} speakers: {', '.join(speakers)}")
                        
                        # Count words per speaker
                        speaker_words = {}
                        for segment in segments:
                            speaker = segment.get('speaker', 'UNKNOWN')
                            words = len(segment['text'].split())
                            if speaker in speaker_words:
                                speaker_words[speaker] += words
                            else:
                                speaker_words[speaker] = words
                        
                        # Display speaker statistics
                        st.write("### Speaker Statistics")
                        for speaker, words in speaker_words.items():
                            st.write(f"- **{speaker}**: {words} words")
                
                # Export options
                st.subheader("💾 Export Options")
                export_cols = st.columns(len(export_format))
                
                output_base = Path(selected_file).stem
                
                for i, format_type in enumerate(export_format):
                    with export_cols[i]:
                        if format_type == "TXT":
                            st.download_button(
                                label=f"Download {format_type}",
                                data=transcript,
                                file_name=f"{output_base}_transcript.txt",
                                mime="text/plain"
                            )
                        elif format_type in ["SRT", "VTT", "ASS"]:
                            # Export to subtitle format
                            output_path = export_transcript(
                                transcript, 
                                output_base, 
                                format_type.lower(),
                                segments=segments,
                                compress=compress_exports,
                                compression_type=compression_type,
                                style=ass_style if format_type == "ASS" and ass_style else None
                            )
                            
                            # Read the exported file for download
                            with open(output_path, 'rb') as f:
                                subtitle_content = f.read()
                            
                            # Determine file extension
                            file_ext = f".{format_type.lower()}"
                            if compress_exports:
                                file_ext += ".gz" if compression_type == "gzip" else ".zip"
                            
                            st.download_button(
                                label=f"Download {format_type}",
                                data=subtitle_content,
                                file_name=f"{output_base}{file_ext}",
                                mime="application/octet-stream"
                            )
                            
                            # Clean up the temporary file
                            os.remove(output_path)
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("Processing complete!")
            else:
                st.error("❌ Failed to process recording")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write(e)  # This will show the traceback in the Streamlit app

if __name__ == "__main__":
    main()
