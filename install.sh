#!/bin/bash

echo "==================================================="
echo " OBS Recording Transcriber - Unix Installation"
echo "==================================================="
echo

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found! Please install Python 3.8 or higher."
    echo "For Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "For macOS: brew install python3"
    exit 1
fi

# Make the script executable
chmod +x install.py

# Run the installation script
echo "Running installation script..."
python3 ./install.py

echo
echo "If the installation was successful, you can run the application with:"
echo "streamlit run app.py"
echo 