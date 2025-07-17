# Docker Deployment Guide for VideoTranscriber

This guide explains how to run VideoTranscriber in a Docker container while using Ollama models on your host system.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│ Host System                             │
│ ┌─────────────────┐  ┌──────────────────│
│ │ Ollama Service  │  │ Video Files      │
│ │ (port 11434)    │  │ Directory        │
│ └─────────────────┘  └──────────────────│
│         ▲                     ▲         │
│         │                     │         │
│ ┌───────┼─────────────────────┼─────────│
│ │ Docker Container            │         │
│ │ ┌─────▼─────────┐          │         │
│ │ │ VideoTranscriber         │         │
│ │ │ - Streamlit App          │         │
│ │ │ - Whisper Models         │         │
│ │ │ - ML Dependencies        │         │
│ │ └───────────────┘          │         │
│ └────────────────────────────┼─────────│
│                              │         │
│         Mounted Volumes ─────┘         │
└─────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

1. **Docker & Docker Compose** installed
2. **Ollama running on host**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama service
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull llama3
   ```

### 1. Setup Environment

```bash
# Copy environment template
cp docker.env.example .env

# Edit .env file with your paths
# Key settings to update:
VIDEO_PATH=/path/to/your/videos
OUTPUT_PATH=/path/to/save/outputs
HF_TOKEN=your_huggingface_token_if_needed
```

### 2. Create Required Directories

```bash
# Create directories for mounting
mkdir -p videos outputs cache config
```

### 3. Build and Run

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Access the application
# Open browser to: http://localhost:8501
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VIDEO_PATH` | Host directory containing video files | `./videos` | Yes |
| `OUTPUT_PATH` | Host directory for outputs | `./outputs` | Yes |
| `CACHE_PATH` | Host directory for model cache | `./cache` | No |
| `OLLAMA_API_URL` | Ollama API endpoint | `http://host.docker.internal:11434/api` | No |
| `HF_TOKEN` | HuggingFace token for advanced features | - | No |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | - | No |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `${VIDEO_PATH}` | `/app/data/videos` | Input video files |
| `${OUTPUT_PATH}` | `/app/data/outputs` | Generated transcripts/summaries |
| `${CACHE_PATH}` | `/app/data/cache` | Model and processing cache |
| `${CONFIG_PATH}` | `/app/config` | Configuration files |

## Platform-Specific Setup

### Windows (Docker Desktop)

```yaml
# In docker-compose.yml - use bridge networking
networks:
  - videotranscriber-network

environment:
  - OLLAMA_API_URL=http://host.docker.internal:11434/api
```

### macOS (Docker Desktop)

Same as Windows - uses `host.docker.internal` to access host services.

### Linux

Option 1 - Host Networking (Recommended):
```yaml
# In docker-compose.yml
network_mode: host

environment:
  - OLLAMA_API_URL=http://localhost:11434/api
```

Option 2 - Bridge Networking:
```yaml
environment:
  - OLLAMA_API_URL=http://172.17.0.1:11434/api  # Docker bridge IP
```

## GPU Support

### NVIDIA GPU Setup

1. **Install NVIDIA Container Toolkit**:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Enable in docker-compose.yml**:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

## Usage in Container

### Application Settings

When running in Docker, update these settings in the VideoTranscriber UI:

1. **Base Folder**: Set to `/app/data/videos`
2. **Ollama Models**: Should auto-detect from host
3. **GPU Settings**: Will use container GPU if configured

### File Access

- **Input Videos**: Place in your `${VIDEO_PATH}` directory on host
- **Outputs**: Generated files appear in `${OUTPUT_PATH}` on host
- **Cache**: Models cached in `${CACHE_PATH}` for faster subsequent runs

## Troubleshooting

### Common Issues

#### 1. Can't Connect to Ollama

**Symptoms**: "Ollama service is not available" message

**Solutions**:
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check firewall settings
- For Linux, try host networking mode
- Verify OLLAMA_API_URL in environment

#### 2. No Video Files Detected

**Symptoms**: "No recordings found" message

**Solutions**:
- Check VIDEO_PATH points to correct directory
- Ensure directory contains supported formats (.mp4, .avi, .mov, .mkv)
- Check file permissions

#### 3. GPU Not Detected

**Symptoms**: Processing is slow, no GPU utilization

**Solutions**:
- Install NVIDIA Container Toolkit
- Uncomment GPU section in docker-compose.yml
- Verify: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

#### 4. Permission Issues

**Symptoms**: Cannot write to output directory

**Solutions**:
```bash
# Fix permissions
sudo chown -R $(id -u):$(id -g) outputs cache config
chmod -R 755 outputs cache config
```

### Debugging

```bash
# View container logs
docker-compose logs -f videotranscriber

# Execute shell in container
docker-compose exec videotranscriber bash

# Check Ollama connectivity from container
docker-compose exec videotranscriber curl -f $OLLAMA_API_URL/tags

# Monitor resource usage
docker stats videotranscriber
```

## Advanced Configuration

### Custom Dockerfile

For specialized requirements, modify the Dockerfile:

```dockerfile
# Add custom dependencies
RUN pip install your-custom-package

# Set custom environment variables
ENV YOUR_CUSTOM_VAR=value

# Copy custom configuration
COPY custom-config.yaml /app/config/
```

### Multi-Instance Deployment

Run multiple instances for different use cases:

```bash
# Copy docker-compose.yml to docker-compose.prod.yml
# Modify ports and paths
docker-compose -f docker-compose.prod.yml up -d
```

### CI/CD Integration

```yaml
# .github/workflows/docker.yml
name: Build and Deploy
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t videotranscriber .
```

## Performance Optimization

### Memory Management

```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
    reservations:
      memory: 4G
```

### Model Caching

- Use persistent volumes for `/app/data/cache`
- Pre-download models to reduce startup time
- Configure appropriate cache size limits

### Network Optimization

- Use host networking on Linux for better performance
- Consider running Ollama and VideoTranscriber on same machine
- Use SSD storage for cache directories 