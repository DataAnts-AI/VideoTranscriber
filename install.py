#!/usr/bin/env python
"""
Installation script for OBS Recording Transcriber.
This script helps install all required dependencies and checks for common issues.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_step(text):
    """Print a step in the installation process."""
    print(f"\n>> {text}")

def run_command(command, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    print_step("Checking Python version")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Python 3.8 or higher is required. You have {sys.version}")
        print("Please upgrade your Python installation.")
        return False
    print(f"Python version: {sys.version}")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    print_step("Checking FFmpeg installation")
    result = shutil.which("ffmpeg")
    if result is None:
        print("FFmpeg not found in PATH.")
        print("Please install FFmpeg:")
        if platform.system() == "Windows":
            print("  - Download from: https://www.gyan.dev/ffmpeg/builds/")
            print("  - Extract and add the bin folder to your PATH")
        elif platform.system() == "Darwin":  # macOS
            print("  - Install with Homebrew: brew install ffmpeg")
        else:  # Linux
            print("  - Install with apt: sudo apt update && sudo apt install ffmpeg")
        return False
    
    # Check FFmpeg version
    version_result = run_command("ffmpeg -version")
    if version_result:
        print(f"FFmpeg is installed: {version_result.stdout.splitlines()[0]}")
        return True
    return False

def check_gpu():
    """Check for GPU availability."""
    print_step("Checking GPU availability")
    
    # Check for NVIDIA GPU
    if platform.system() == "Windows":
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            result = run_command("nvidia-smi", check=False)
            if result and result.returncode == 0:
                print("NVIDIA GPU detected:")
                for line in result.stdout.splitlines()[:10]:
                    print(f"  {line}")
                return "nvidia"
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("Apple Silicon (M1/M2) detected")
        return "apple"
    
    print("No GPU detected or GPU drivers not installed. CPU will be used for processing.")
    return "cpu"

def setup_virtual_env():
    """Set up a virtual environment."""
    print_step("Setting up virtual environment")
    
    # Check if venv module is available
    try:
        import venv
        print("Python venv module is available")
    except ImportError:
        print("Python venv module is not available. Please install it.")
        return False
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        activate_venv()
        return True
    
    print(f"Creating virtual environment at {venv_path}")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully")
        activate_venv()
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def activate_venv():
    """Activate the virtual environment."""
    print_step("Activating virtual environment")
    
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Virtual environment not found")
        return False
    
    # Get the path to the activate script
    if platform.system() == "Windows":
        activate_script = venv_path / "Scripts" / "activate.bat"
        activate_cmd = f"call {activate_script}"
    else:
        activate_script = venv_path / "bin" / "activate"
        activate_cmd = f"source {activate_script}"
    
    print(f"To activate the virtual environment, run:")
    print(f"  {activate_cmd}")
    
    # We can't actually activate the venv in this script because it would only
    # affect the subprocess, not the parent process. We just provide instructions.
    return True

def install_pytorch(gpu_type):
    """Install PyTorch with appropriate GPU support."""
    print_step("Installing PyTorch")
    
    if gpu_type == "nvidia":
        print("Installing PyTorch with CUDA support")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif gpu_type == "apple":
        print("Installing PyTorch with MPS support")
        cmd = "pip install torch torchvision torchaudio"
    else:
        print("Installing PyTorch (CPU version)")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    result = run_command(cmd)
    if result and result.returncode == 0:
        print("PyTorch installed successfully")
        return True
    else:
        print("Failed to install PyTorch")
        return False

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print_step("Installing dependencies from requirements.txt")
    
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("requirements.txt not found")
        return False
    
    result = run_command("pip install -r requirements.txt")
    if result and result.returncode == 0:
        print("Dependencies installed successfully")
        return True
    else:
        print("Some dependencies failed to install. See error messages above.")
        return False

def install_tokenizers():
    """Install tokenizers package separately."""
    print_step("Installing tokenizers package")
    
    # First try the normal installation
    result = run_command("pip install tokenizers", check=False)
    if result and result.returncode == 0:
        print("Tokenizers installed successfully")
        return True
    
    # If that fails, try the no-binary option
    print("Standard installation failed, trying alternative method...")
    result = run_command("pip install tokenizers --no-binary tokenizers", check=False)
    if result and result.returncode == 0:
        print("Tokenizers installed successfully with alternative method")
        return True
    
    print("Failed to install tokenizers. You may need to install Rust or Visual C++ Build Tools.")
    if platform.system() == "Windows":
        print("Download Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("Install Rust: https://rustup.rs/")
    return False

def check_installation():
    """Verify the installation by importing key packages."""
    print_step("Verifying installation")
    
    packages_to_check = [
        "streamlit",
        "torch",
        "transformers",
        "whisper",
        "numpy",
        "sklearn"
    ]
    
    all_successful = True
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError:
            print(f"✗ Failed to import {package}")
            all_successful = False
    
    # Check optional packages
    optional_packages = [
        "pyannote.audio",
        "iso639"
    ]
    
    print("\nChecking optional packages:")
    for package in optional_packages:
        try:
            if package == "pyannote.audio":
                # Just try to import pyannote
                __import__("pyannote")
            else:
                __import__(package)
            print(f"✓ {package} imported successfully")
        except ImportError:
            print(f"⚠ {package} not available (required for some advanced features)")
    
    return all_successful

def main():
    """Main installation function."""
    print_header("OBS Recording Transcriber - Installation Script")
    
    # Check prerequisites
    if not check_python_version():
        return
    
    ffmpeg_available = check_ffmpeg()
    gpu_type = check_gpu()
    
    # Setup environment
    if not setup_virtual_env():
        print("Failed to set up virtual environment. Continuing with system Python...")
    
    # Install packages
    print("\nReady to install packages. Make sure your virtual environment is activated.")
    input("Press Enter to continue...")
    
    install_pytorch(gpu_type)
    install_dependencies()
    install_tokenizers()
    
    # Verify installation
    success = check_installation()
    
    print_header("Installation Summary")
    print(f"Python: {'✓ OK' if check_python_version() else '✗ Needs upgrade'}")
    print(f"FFmpeg: {'✓ Installed' if ffmpeg_available else '✗ Not found'}")
    print(f"GPU Support: {gpu_type.upper()}")
    print(f"Dependencies: {'✓ Installed' if success else '⚠ Some issues'}")
    
    print("\nNext steps:")
    if not ffmpeg_available:
        print("1. Install FFmpeg (required for audio processing)")
    
    print("1. Activate your virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("2. Run the application:")
    print("   streamlit run app.py")
    
    print("\nFor advanced features like speaker diarization:")
    print("1. Get a HuggingFace token: https://huggingface.co/settings/tokens")
    print("2. Request access to pyannote models: https://huggingface.co/pyannote/speaker-diarization-3.0")
    
    print("\nSee INSTALLATION.md for more details and troubleshooting.")

if __name__ == "__main__":
    main() 