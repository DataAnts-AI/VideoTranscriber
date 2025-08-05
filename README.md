# Video Transcriber

## Project Overview
The Video Recording Transcriber is a Python application built with Streamlit that processes video and audio recordings to generate transcripts and summaries using AI models. The application uses Whisper for transcription and Hugging Face Transformers for summarization.

**Supported Formats**: MP4, AVI, MOV, MKV (video) and M4A (audio)


![SuiteQL_query_UI-1-Thumbnail](https://github.com/user-attachments/assets/72aaf238-6615-4739-b77f-c4eb9ff96996)

Demo here

https://github.com/user-attachments/assets/990e63fc-232e-46a0-afdf-ca8836d46a13


## Installation

### 🐳 Docker Installation (Recommended)

**Benefits**: Isolated environment, no dependency conflicts, easy deployment

#### Option A: Prebuilt Images (Fastest & Most Reliable)
```bash
# 1. Clone repository for config files
git clone https://github.com/DataAnts-AI/VideoTranscriber.git
cd VideoTranscriber

# 2. Setup environment
cp docker.env.example .env
# Edit .env with your video directory paths

# 3. Ensure Ollama is running on host
ollama serve  # In separate terminal
ollama pull llama3

# 4. Start with prebuilt image
docker-compose -f docker-compose.prebuilt.yml up -d

# 5. Access application
# Open browser to: http://localhost:8501
```

#### Option B: Build from Source (Development)
```bash
# Use the local build approach
docker-compose up -d
```

See [DOCKER.md](DOCKER.md) for complete Docker setup guide.

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
1. Set your base folder where video/audio recordings are stored
2. Select a recording from the dropdown (supports MP4, AVI, MOV, MKV, M4A)
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



## Key Improvement Areas

### 1. UI Enhancements
- **Implemented:**
  - Responsive layout with columns for better organization
  - Expanded sidebar with categorized settings
  - Custom CSS for improved button styling
  - Spinner for long-running operations
  - Expanded transcript view by default

- **Additional Recommendations:**
  - Add a dark mode toggle
  - Implement progress bars for each processing step
  - Add tooltips for complex options
  - Create a dashboard view for batch processing results
  - Add visualization of transcript segments with timestamps

### 2. Ollama Local API Integration
- **Implemented:**
  - Local API integration for offline summarization
  - Model selection from available Ollama models
  - Chunking for long texts
  - Fallback to online models when Ollama fails

- **Additional Recommendations:**
  - Add temperature and other generation parameters as advanced options
  - Implement streaming responses for real-time feedback
  - Cache results to avoid reprocessing
  - Add support for custom Ollama model creation with specific instructions
  - Implement parallel processing for multiple chunks

### 3. Subtitle Export Formats
- **Implemented:**
  - SRT export with proper formatting
  - ASS export with basic styling
  - Multi-format export options
  - Automatic segment creation from plain text

- **Additional Recommendations:**
  - Add customizable styling options for ASS subtitles
  - Implement subtitle editing before export
  - Add support for VTT format for web videos
  - Implement subtitle timing adjustment
  - Add batch export for multiple files

### 4. Architecture and Code Quality
- **Recommendations:**
  - Implement proper error handling and logging throughout
  - Add unit tests for critical components
  - Create a configuration file for default settings
  - Implement caching for processed files
  - Add type hints throughout the codebase
  - Document API endpoints for potential future web service

### 5. Performance Optimizations
- **Recommendations:**
  - Implement parallel processing for batch operations
  - Add GPU acceleration configuration options
  - Optimize memory usage for large files
  - Implement incremental processing for very long recordings
  - Add compression options for exported files

### 6. Additional Features
- **Recommendations:**
  - Speaker diarization (identifying different speakers)
  - Language detection and translation
  - Keyword extraction and timestamp linking
  - Integration with video editing software
  - Batch processing queue with email notifications
  - Custom vocabulary for domain-specific terminology

## Implementation Roadmap
1. **Phase 1 (Completed):** Basic UI improvements, Ollama integration, and subtitle export
2. **Phase 2 (Completed):** Performance optimizations and additional export formats
   - Added WebVTT export format for web videos
   - Implemented GPU acceleration with automatic device selection
   - Added caching system for faster processing of previously transcribed files
   - Optimized memory usage with configurable memory limits
   - Added compression options for exported files
   - Enhanced ASS subtitle styling options
   - Added progress indicators for better user feedback
3. **Phase 3 (Completed):** Advanced features like speaker diarization and translation
   - Implemented speaker diarization to identify different speakers in recordings
   - Added language detection and translation capabilities
   - Integrated keyword extraction with timestamp linking
   - Created interactive transcript with keyword highlighting
   - Added named entity recognition for better content analysis
   - Generated keyword index with timestamp references
   - Provided speaker statistics and word count analysis
4. **Phase 4:** Integration with other tools and services (In progess)


Reach out to support@dataants.org if you need assistance with any AI solutions - we offer support for n8n workflows, local RAG chatbots, and ERP and Financial reporting.
