@echo off
echo ===================================================
echo OBS Recording Transcriber - Windows Installation
echo ===================================================
echo.

:: Check for Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Run the installation script
echo Running installation script...
python install.py

echo.
echo If the installation was successful, you can run the application with:
echo streamlit run app.py
echo.
pause 