# Installation Guide for OBS Recording Transcriber

This guide will help you install all the necessary dependencies for the OBS Recording Transcriber application, including the advanced features from Phase 3.

## Prerequisites

Before installing the Python packages, you need to set up some prerequisites:

### 1. Python 3.8 or higher

Make sure you have Python 3.8 or higher installed. You can download it from [python.org](https://www.python.org/downloads/).

### 2. FFmpeg

FFmpeg is required for audio processing:

- **Windows**: 
  - Download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)
  - Extract the ZIP file
  - Add the `bin` folder to your system PATH

- **macOS**:
  ```bash
  brew install ffmpeg
  ```

- **Linux**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

### 3. Visual C++ Build Tools (Windows only)

Some packages like `tokenizers` require C++ build tools:

1. Download and install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. During installation, select "Desktop development with C++"

## Installation Steps

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install PyTorch

For better performance, install PyTorch with CUDA support if you have an NVIDIA GPU:

```bash
# Windows/Linux with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# macOS or CPU-only
pip install torch torchvision torchaudio
```

### 3. Install Dependencies

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 4. Troubleshooting Common Issues

#### Tokenizers Installation Issues

If you encounter issues with `tokenizers` installation:

1. Make sure you have Visual C++ Build Tools installed (Windows)
2. Try installing Rust: [rustup.rs](https://rustup.rs/)
3. Install tokenizers separately:
   ```bash
   pip install tokenizers --no-binary tokenizers
   ```

#### PyAnnote.Audio Access

To use speaker diarization, you need a HuggingFace token with access to the pyannote models:

1. Create an account on [HuggingFace](https://huggingface.co/)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Request access to [pyannote/speaker-diarization-3.0](https://huggingface.co/pyannote/speaker-diarization-3.0)
4. Set the token in the application when prompted or as an environment variable:
   ```bash
   # Windows
   set HF_TOKEN=your_token_here
   # macOS/Linux
   export HF_TOKEN=your_token_here
   ```

#### Memory Issues with Large Files

If you encounter memory issues with large files:

1. Use a smaller Whisper model (e.g., "base" instead of "large")
2. Reduce the GPU memory fraction in the application settings
3. Increase your system's swap space/virtual memory

## Running the Application

After installation, run the application with:

```bash
streamlit run app.py
```

## Optional: Ollama Setup for Local Summarization

To use Ollama for local summarization:

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model:
   ```bash
   ollama pull llama3
   ```
3. Uncomment the Ollama line in requirements.txt and install:
   ```bash
   pip install ollama
   ```

## Verifying Installation

To verify that all components are working correctly:

1. Run the application
2. Check that GPU acceleration is available (if applicable)
3. Test a small video file with basic transcription
4. Gradually enable advanced features like diarization and translation

If you encounter any issues, check the application logs for specific error messages. 