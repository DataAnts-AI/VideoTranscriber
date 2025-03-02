# OBS Recording Transcriber

Process OBS recordings or any video/audio files with AI-based transcription and summarization locally on your machine.


## Features
- AI transcription using Whisper.
- Summarization using Hugging Face Transformers.
- File selection, resource validation, and error handling.
- Speaker diarization to identify different speakers in recordings.
- Language detection and translation capabilities.
- Keyword extraction with timestamp linking.
- Interactive transcript with keyword highlighting.
- Export to TXT, SRT, VTT, and ASS subtitle formats with compression options.
- GPU acceleration for faster processing.
- Caching system for previously processed files.

## Installation

### Easy Installation (Recommended)

#### Windows
1. Download or clone the repository
2. Run `install.bat` by double-clicking it
3. Follow the on-screen instructions

#### Linux/macOS
1. Download or clone the repository
2. Open a terminal in the project directory
3. Make the install script executable: `chmod +x install.sh`
4. Run the script: `./install.sh`
5. Follow the on-screen instructions

### Manual Installation
1. Clone the repo.
```
git clone https://github.com/DataAnts-AI/VideoTranscriber.git
cd VideoTranscriber
```

2. Install dependencies:
```
pip install -r requirements.txt
```

Notes:
- Ensure that the versions align with the features you use and your system compatibility.
- torch version should match the capabilities of your hardware (e.g., CUDA support for GPUs).
- For advanced features like speaker diarization, you'll need a HuggingFace token.
- See `INSTALLATION.md` for detailed instructions and troubleshooting.

3. Run the application:
```
streamlit run app.py
```

## Usage
1. Set your base folder where OBS recordings are stored
2. Select a recording from the dropdown
3. Choose transcription and summarization models
4. Configure performance settings (GPU acceleration, caching)
5. Select export formats and compression options
6. Click "Process Recording" to start

## Advanced Features
- **Speaker Diarization**: Identify and label different speakers in your recordings
- **Translation**: Automatically detect language and translate to multiple languages
- **Keyword Extraction**: Extract important keywords with timestamp links
- **Interactive Transcript**: Navigate through the transcript with keyword highlighting
- **GPU Acceleration**: Utilize your GPU for faster processing
- **Caching**: Save processing time by caching results

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
