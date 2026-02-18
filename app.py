import streamlit as st
from utils.audio_processing import extract_audio, cleanup_temp_audio, get_video_duration
from utils.transcription import transcribe_audio
from utils.summarization import summarize_text
from utils.validation import validate_environment, get_system_capabilities
from utils.export import export_transcript
from pathlib import Path
import os
import logging
import humanize
import time
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from utils.ollama_integration import (
        check_ollama_available, list_available_models,
        chunk_and_summarize, stream_chunk_and_summarize
    )
    OLLAMA_AVAILABLE = check_ollama_available()
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from utils.gpu_utils import get_gpu_info, configure_gpu, optimize_for_inference
    GPU_UTILS_AVAILABLE = True
    optimize_for_inference()
except ImportError:
    GPU_UTILS_AVAILABLE = False

try:
    from utils.cache import get_cache_size, clear_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from utils.diarization import transcribe_with_diarization
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

try:
    from utils.translation import transcribe_and_translate, get_language_name
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

try:
    from utils.keyword_extraction import (
        extract_keywords_from_transcript, generate_keyword_index,
        generate_interactive_transcript
    )
    KEYWORD_EXTRACTION_AVAILABLE = True
except ImportError:
    KEYWORD_EXTRACTION_AVAILABLE = False


def init_session_state():
    """Initialize session state with defaults for persistence across reruns."""
    defaults = {
        "transcription_model": "base",
        "summarization_method": "Hugging Face (Online)",
        "use_diarization": False,
        "use_translation": False,
        "use_keywords": False,
        "use_gpu": GPU_UTILS_AVAILABLE,
        "use_cache": CACHE_AVAILABLE,
        "memory_fraction": 0.8,
        "export_formats": ["TXT"],
        "compress_exports": False,
        "base_folder": str(Path.home()),
        "recursive_search": False,
        "results": None,
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def format_duration(seconds):
    """Format seconds into MM:SS or HH:MM:SS."""
    if seconds is None:
        return "Unknown"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temp directory and return its path."""
    temp_dir = tempfile.mkdtemp(prefix="vt_upload_")
    file_path = Path(temp_dir) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def render_sidebar():
    """Render the sidebar with collapsible settings groups."""
    st.sidebar.markdown("### Settings")

    # -- Model Settings (expanded by default) --
    with st.sidebar.expander("Model Settings", expanded=True):
        st.session_state.transcription_model = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=["tiny", "base", "small", "medium", "large"].index(
                st.session_state.transcription_model
            ),
            help="Larger models are more accurate but slower.",
            key="sb_whisper_model",
        )

        summarization_options = (
            ["Hugging Face (Online)", "Ollama (Local)"]
            if OLLAMA_AVAILABLE
            else ["Hugging Face (Online)"]
        )
        st.session_state.summarization_method = st.selectbox(
            "Summarization",
            summarization_options,
            index=0,
            help="Ollama runs locally but requires installation.",
            key="sb_summarization",
        )

        ollama_model = None
        if OLLAMA_AVAILABLE and st.session_state.summarization_method == "Ollama (Local)":
            available_models = list_available_models()
            if available_models:
                ollama_model = st.selectbox(
                    "Ollama Model",
                    available_models,
                    index=0,
                    key="sb_ollama_model",
                )
            else:
                st.warning("No Ollama models found. Run `ollama pull <model>`.")

    # -- Advanced Features (collapsed) --
    with st.sidebar.expander("Advanced Features"):
        st.session_state.use_diarization = st.checkbox(
            "Speaker Diarization",
            value=st.session_state.use_diarization,
            disabled=not DIARIZATION_AVAILABLE,
            help="Identify different speakers in the recording.",
            key="sb_diarization",
        )

        hf_token = None
        num_speakers = 2
        if st.session_state.use_diarization and DIARIZATION_AVAILABLE:
            hf_token = st.text_input(
                "HuggingFace Token",
                type="password",
                help="Required for diarization. Get token at huggingface.co/settings/tokens",
                key="sb_hf_token",
            )
            num_speakers = st.number_input(
                "Number of Speakers", min_value=1, max_value=10, value=2,
                key="sb_num_speakers",
            )

        st.session_state.use_translation = st.checkbox(
            "Translation",
            value=st.session_state.use_translation,
            disabled=not TRANSLATION_AVAILABLE,
            help="Translate the transcript to another language.",
            key="sb_translation",
        )

        target_lang = None
        if st.session_state.use_translation and TRANSLATION_AVAILABLE:
            target_lang = st.selectbox(
                "Target Language",
                ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar"],
                format_func=lambda x: f"{get_language_name(x)} ({x})",
                key="sb_target_lang",
            )

        st.session_state.use_keywords = st.checkbox(
            "Keyword Extraction",
            value=st.session_state.use_keywords,
            disabled=not KEYWORD_EXTRACTION_AVAILABLE,
            help="Extract keywords and link them to timestamps.",
            key="sb_keywords",
        )

        max_keywords = 15
        if st.session_state.use_keywords and KEYWORD_EXTRACTION_AVAILABLE:
            max_keywords = st.slider(
                "Max Keywords", min_value=5, max_value=30, value=15,
                key="sb_max_keywords",
            )

    # -- Performance (collapsed) --
    with st.sidebar.expander("Performance"):
        st.session_state.use_gpu = st.checkbox(
            "GPU Acceleration",
            value=st.session_state.use_gpu,
            disabled=not GPU_UTILS_AVAILABLE,
            help="Use GPU for faster processing if available.",
            key="sb_gpu",
        )

        if GPU_UTILS_AVAILABLE and st.session_state.use_gpu:
            gpu_info = get_gpu_info()
            if gpu_info["cuda_available"]:
                gpu_devices = [
                    f"{d['name']} ({humanize.naturalsize(d['total_memory'])})"
                    for d in gpu_info["cuda_devices"]
                ]
                st.info(f"GPU: {', '.join(gpu_devices)}")
            elif gpu_info["mps_available"]:
                st.info("Apple Silicon GPU (MPS)")
            else:
                st.warning("No GPU detected. Using CPU.")

        st.session_state.memory_fraction = st.slider(
            "GPU Memory %",
            min_value=0.1, max_value=1.0,
            value=st.session_state.memory_fraction, step=0.1,
            disabled=not (GPU_UTILS_AVAILABLE and st.session_state.use_gpu),
            key="sb_memory",
        )

        st.session_state.use_cache = st.checkbox(
            "Cache Results",
            value=st.session_state.use_cache,
            disabled=not CACHE_AVAILABLE,
            help="Cache transcriptions to avoid reprocessing.",
            key="sb_cache",
        )

        if CACHE_AVAILABLE and st.session_state.use_cache:
            cache_size, cache_files = get_cache_size()
            if cache_size > 0:
                st.caption(f"Cache: {humanize.naturalsize(cache_size)} ({cache_files} files)")
                if st.button("Clear Cache", key="sb_clear_cache"):
                    cleared = clear_cache()
                    st.success(f"Cleared {cleared} files")

    # -- Export (collapsed) --
    with st.sidebar.expander("Export Options"):
        st.session_state.export_formats = st.multiselect(
            "Formats",
            ["TXT", "SRT", "VTT", "ASS"],
            default=st.session_state.export_formats,
            key="sb_export_formats",
        )

        st.session_state.compress_exports = st.checkbox(
            "Compress Exports",
            value=st.session_state.compress_exports,
            key="sb_compress",
        )

        compression_type = None
        if st.session_state.compress_exports:
            compression_type = st.radio(
                "Compression", ["gzip", "zip"], index=0,
                key="sb_compression_type",
            )

        ass_style = None
        if "ASS" in st.session_state.export_formats:
            if st.checkbox("Customize ASS Style", value=False, key="sb_ass_custom"):
                ass_style = {
                    "fontname": st.selectbox(
                        "Font",
                        ["Arial", "Helvetica", "Times New Roman", "Courier New"],
                        key="sb_ass_font",
                    ),
                    "fontsize": str(st.slider("Font Size", 12, 72, 48, key="sb_ass_size")),
                    "alignment": st.selectbox(
                        "Alignment",
                        ["2 (Bottom Center)", "1 (Bottom Left)", "3 (Bottom Right)", "8 (Top Center)"],
                        key="sb_ass_align",
                    ).split()[0],
                    "bold": "-1" if st.checkbox("Bold", value=True, key="sb_ass_bold") else "0",
                    "italic": "-1" if st.checkbox("Italic", value=False, key="sb_ass_italic") else "0",
                }

    # -- System Info (collapsed) --
    with st.sidebar.expander("System Info"):
        caps = get_system_capabilities()
        st.markdown(f"- **FFmpeg:** {'Installed' if caps['ffmpeg'] else 'Not found'}")
        st.markdown(f"- **CUDA:** {'Available' if caps['cuda'] else 'Not available'}")
        st.markdown(f"- **MPS:** {'Available' if caps['mps'] else 'Not available'}")
        if caps["gpu_name"]:
            st.markdown(f"- **GPU:** {caps['gpu_name']} ({humanize.naturalsize(caps['gpu_memory'])})")
        st.markdown(f"- **Ollama:** {'Connected' if OLLAMA_AVAILABLE else 'Not available'}")
        st.markdown(f"- **Diarization:** {'Ready' if DIARIZATION_AVAILABLE else 'Not available'}")

    return {
        "ollama_model": ollama_model,
        "hf_token": hf_token,
        "num_speakers": num_speakers,
        "target_lang": target_lang,
        "max_keywords": max_keywords,
        "compression_type": compression_type,
        "ass_style": ass_style,
    }


def render_file_input():
    """Render the file input section with upload + folder browse tabs."""
    upload_tab, browse_tab = st.tabs(["Upload Files", "Browse Folder"])

    selected_file = None

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Drag and drop your recordings here",
            type=["mp4", "avi", "mov", "mkv", "m4a"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded_files:
            if len(uploaded_files) == 1:
                selected_file = ("upload", uploaded_files[0])
            else:
                file_names = [f.name for f in uploaded_files]
                chosen = st.selectbox("Choose a recording", file_names, key="upload_select")
                idx = file_names.index(chosen)
                selected_file = ("upload", uploaded_files[idx])

    with browse_tab:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state.base_folder = st.text_input(
                "Folder path",
                value=st.session_state.base_folder,
                key="folder_input",
            )
        with col2:
            st.session_state.recursive_search = st.checkbox(
                "Recursive", value=st.session_state.recursive_search,
                key="recursive_check",
            )

        base_path = Path(st.session_state.base_folder)
        env_errors = validate_environment(base_path)
        if env_errors:
            for error in env_errors:
                st.warning(error)
        else:
            extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4a"]
            recordings = []
            glob_fn = base_path.rglob if st.session_state.recursive_search else base_path.glob
            for ext in extensions:
                recordings.extend(glob_fn(ext))

            if recordings:
                chosen = st.selectbox(
                    "Choose a recording",
                    recordings,
                    format_func=lambda p: str(p.relative_to(base_path)) if str(p).startswith(str(base_path)) else str(p),
                    key="folder_select",
                )
                selected_file = ("path", chosen)
            else:
                st.info("No recordings found. Supported formats: MP4, AVI, MOV, MKV, M4A")

    return selected_file


def render_file_preview(selected_file):
    """Show file metadata before processing."""
    if selected_file is None:
        return

    source_type, file_ref = selected_file

    if source_type == "upload":
        file_size = file_ref.size
        file_name = file_ref.name
        duration = None
    else:
        file_size = file_ref.stat().st_size
        file_name = file_ref.name
        duration = get_video_duration(file_ref)

    cols = st.columns(4)
    cols[0].metric("File", file_name)
    cols[1].metric("Size", humanize.naturalsize(file_size))
    cols[2].metric("Format", Path(file_name).suffix.upper().lstrip("."))
    cols[3].metric("Duration", format_duration(duration))


def resolve_file_path(selected_file):
    """Convert the selected file reference to an actual file path."""
    source_type, file_ref = selected_file
    if source_type == "upload":
        return save_uploaded_file(file_ref)
    return file_ref


def process_recording(file_path, sidebar_opts):
    """Run the full processing pipeline with granular status updates."""
    results = {}
    start_time = time.time()

    with st.status("Processing recording...", expanded=True) as status:

        # Step 1: Transcription
        st.write(f"Transcribing with Whisper ({st.session_state.transcription_model} model)...")
        t0 = time.time()

        if st.session_state.use_diarization and DIARIZATION_AVAILABLE and sidebar_opts["hf_token"]:
            num_spk = int(sidebar_opts["num_speakers"]) if sidebar_opts["num_speakers"] > 0 else None
            segments, transcript = transcribe_with_diarization(
                file_path,
                whisper_model=st.session_state.transcription_model,
                num_speakers=num_spk,
                use_gpu=st.session_state.use_gpu,
                hf_token=sidebar_opts["hf_token"],
            )
            results["diarized"] = True
        elif st.session_state.use_translation and TRANSLATION_AVAILABLE:
            st.write("Transcribing and translating...")
            orig_seg, trans_seg, orig_text, trans_text = transcribe_and_translate(
                file_path,
                whisper_model=st.session_state.transcription_model,
                target_lang=sidebar_opts["target_lang"],
                use_gpu=st.session_state.use_gpu,
            )
            segments = trans_seg
            transcript = trans_text
            results["original_text"] = orig_text
            results["original_segments"] = orig_seg
            results["translated"] = True
        else:
            segments, transcript = transcribe_audio(
                file_path,
                model=st.session_state.transcription_model,
                use_cache=st.session_state.use_cache,
                use_gpu=st.session_state.use_gpu,
                memory_fraction=st.session_state.memory_fraction,
            )

        transcription_time = time.time() - t0
        st.write(f"Transcription complete ({transcription_time:.1f}s)")

        if not transcript:
            status.update(label="Processing failed", state="error")
            return None

        results["segments"] = segments
        results["transcript"] = transcript

        # Step 2: Keyword extraction
        if st.session_state.use_keywords and KEYWORD_EXTRACTION_AVAILABLE:
            st.write("Extracting keywords...")
            t0 = time.time()
            kw_ts, ent_ts = extract_keywords_from_transcript(
                transcript, segments,
                max_keywords=sidebar_opts["max_keywords"],
                use_gpu=st.session_state.use_gpu,
            )
            results["keyword_timestamps"] = kw_ts
            results["entity_timestamps"] = ent_ts
            results["keyword_index"] = generate_keyword_index(kw_ts, ent_ts)
            results["interactive_transcript"] = generate_interactive_transcript(segments, kw_ts, ent_ts)
            st.write(f"Keywords extracted ({time.time() - t0:.1f}s)")

        # Step 3: Summarization
        st.write("Generating summary...")
        t0 = time.time()

        use_ollama = (
            OLLAMA_AVAILABLE
            and st.session_state.summarization_method == "Ollama (Local)"
            and sidebar_opts["ollama_model"]
        )

        if use_ollama:
            summary = chunk_and_summarize(transcript, model=sidebar_opts["ollama_model"])
            if not summary:
                st.write("Ollama failed, falling back to Hugging Face...")
                summary = summarize_text(
                    transcript,
                    use_gpu=st.session_state.use_gpu,
                    memory_fraction=st.session_state.memory_fraction,
                )
            results["ollama_streaming"] = True
        else:
            summary = summarize_text(
                transcript,
                use_gpu=st.session_state.use_gpu,
                memory_fraction=st.session_state.memory_fraction,
            )

        results["summary"] = summary
        st.write(f"Summary generated ({time.time() - t0:.1f}s)")

        # Cleanup temp audio files
        cleanup_temp_audio()

        total_time = time.time() - start_time
        results["processing_time"] = total_time
        results["word_count"] = len(transcript.split())

        status.update(label=f"Complete in {total_time:.1f}s", state="complete")

    return results


def render_results(results, sidebar_opts):
    """Display processing results with metrics, tabs, and export options."""
    if results is None:
        st.error("Processing failed. Check logs for details.")
        return

    # Metric cards
    st.markdown("---")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Words", f"{results['word_count']:,}")
    metric_cols[1].metric("Segments", str(len(results.get("segments", []))))
    metric_cols[2].metric("Processing Time", f"{results['processing_time']:.1f}s")

    if results.get("diarized"):
        speakers = set(seg.get("speaker", "UNKNOWN") for seg in results["segments"])
        metric_cols[3].metric("Speakers", str(len(speakers)))
    elif results.get("translated"):
        metric_cols[3].metric("Translated", "Yes")
    else:
        metric_cols[3].metric("Model", st.session_state.transcription_model.capitalize())

    # Results tabs
    tab_names = ["Summary", "Transcript", "Advanced"]
    tab1, tab2, tab3 = st.tabs(tab_names)

    with tab1:
        st.subheader("Summary")
        if results.get("ollama_streaming") and OLLAMA_AVAILABLE and sidebar_opts["ollama_model"]:
            st.write(results["summary"])
            with st.expander("Re-generate with streaming"):
                if st.button("Stream Summary", key="stream_btn"):
                    st.write_stream(
                        stream_chunk_and_summarize(
                            results["transcript"],
                            model=sidebar_opts["ollama_model"],
                        )
                    )
        else:
            st.write(results["summary"])

        if results.get("original_text"):
            with st.expander("Original Language Summary"):
                original_summary = summarize_text(
                    results["original_text"],
                    use_gpu=st.session_state.use_gpu,
                    memory_fraction=st.session_state.memory_fraction,
                )
                st.write(original_summary)

    with tab2:
        st.subheader("Full Transcript")

        if results.get("interactive_transcript"):
            st.markdown(results["interactive_transcript"], unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='max-height:500px; overflow-y:auto; padding:1rem; "
                f"border:1px solid #333; border-radius:8px; font-family:monospace; "
                f"font-size:0.9em; line-height:1.6;'>{_format_segments_html(results['segments'])}</div>",
                unsafe_allow_html=True,
            )

        st.download_button(
            "Copy Transcript (Download TXT)",
            data=results["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
            key="copy_transcript",
        )

        if results.get("original_text"):
            with st.expander("Original Language Transcript"):
                st.text(results["original_text"])

    with tab3:
        if results.get("keyword_index"):
            st.subheader("Keyword Index")
            st.markdown(results["keyword_index"])

        if results.get("diarized"):
            st.subheader("Speaker Information")
            speakers = set(seg.get("speaker", "UNKNOWN") for seg in results["segments"])
            st.write(f"Detected {len(speakers)} speakers: {', '.join(speakers)}")

            speaker_words = {}
            for seg in results["segments"]:
                spk = seg.get("speaker", "UNKNOWN")
                speaker_words[spk] = speaker_words.get(spk, 0) + len(seg["text"].split())

            for spk, words in speaker_words.items():
                st.write(f"- **{spk}**: {words} words")

    # Export section
    export_formats = st.session_state.export_formats
    if export_formats:
        st.markdown("---")
        st.subheader("Export")
        export_cols = st.columns(len(export_formats))

        output_base = Path(results.get("file_name", "transcript")).stem

        for i, fmt in enumerate(export_formats):
            with export_cols[i]:
                if fmt == "TXT":
                    st.download_button(
                        label=f"Download {fmt}",
                        data=results["transcript"],
                        file_name=f"{output_base}_transcript.txt",
                        mime="text/plain",
                        key=f"dl_{fmt}",
                    )
                elif fmt in ["SRT", "VTT", "ASS"]:
                    output_path = export_transcript(
                        results["transcript"],
                        output_base,
                        fmt.lower(),
                        segments=results["segments"],
                        compress=st.session_state.compress_exports,
                        compression_type=sidebar_opts["compression_type"],
                        style=sidebar_opts["ass_style"] if fmt == "ASS" else None,
                    )

                    with open(output_path, "rb") as f:
                        content = f.read()

                    file_ext = f".{fmt.lower()}"
                    if st.session_state.compress_exports:
                        file_ext += ".gz" if sidebar_opts["compression_type"] == "gzip" else ".zip"

                    st.download_button(
                        label=f"Download {fmt}",
                        data=content,
                        file_name=f"{output_base}{file_ext}",
                        mime="application/octet-stream",
                        key=f"dl_{fmt}",
                    )

                    try:
                        os.remove(output_path)
                    except OSError:
                        pass


def _format_segments_html(segments):
    """Format transcript segments as HTML with timestamps."""
    if not segments:
        return "<p>No segments available.</p>"

    lines = []
    for seg in segments:
        start = seg.get("start", 0)
        ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
        speaker = seg.get("speaker", "")
        speaker_html = f"<strong style='color:#4FC3F7;'>[{speaker}]</strong> " if speaker else ""
        text = seg.get("text", "").strip()
        lines.append(
            f"<p style='margin:4px 0;'>"
            f"<span style='color:#888; font-size:0.85em; margin-right:8px;'>{ts}</span>"
            f"{speaker_html}{text}</p>"
        )
    return "\n".join(lines)


def main():
    st.set_page_config(
        page_title="Video Transcriber",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .stDownloadButton>button { width: 100%; border-radius: 8px; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .speaker { font-weight: bold; color: #4FC3F7; }
    .timestamp { color: #888; font-size: 0.9em; margin-right: 8px; }
    .keyword { background-color: rgba(255,235,59,0.2); padding: 0 3px; border-radius: 3px; }
    .interactive-transcript p { margin-bottom: 6px; }
    div[data-testid="stExpander"] { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    st.title("Video Transcriber")
    st.caption("AI-powered transcription, summarization, and analysis for video and audio recordings")

    sidebar_opts = render_sidebar()

    # FFmpeg check
    ffmpeg_errors = validate_environment()
    if ffmpeg_errors:
        for err in ffmpeg_errors:
            st.warning(err)

    selected_file = render_file_input()

    if selected_file:
        render_file_preview(selected_file)

        st.markdown("")
        if st.button("Start Processing", type="primary", use_container_width=True):
            file_path = resolve_file_path(selected_file)

            results = process_recording(file_path, sidebar_opts)

            if results:
                source_type, file_ref = selected_file
                results["file_name"] = file_ref.name if source_type == "upload" else file_ref.name
                st.session_state.results = results
                st.toast("Processing complete!", icon="✅")

            # Clean up uploaded temp files
            if selected_file[0] == "upload":
                try:
                    os.remove(file_path)
                    os.rmdir(file_path.parent)
                except OSError:
                    pass

        # Show persisted results from session state
        if st.session_state.results:
            render_results(st.session_state.results, sidebar_opts)


if __name__ == "__main__":
    main()
