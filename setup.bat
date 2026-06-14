@echo off
:: =============================================================
::  AMBIENT NOTES - First-Time Setup Script
::  Run this ONCE in Command Prompt as Administrator
::  Usage: setup.bat
:: =============================================================

echo.
echo ============================================================
echo   AMBIENT NOTES SETUP  -  Canary-Qwen 2.5B
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10 from:
    echo         https://www.python.org/downloads/release/python-3100/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

:: Create virtual environment
echo [1/6] Creating virtual environment...
python -m venv canary_env
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo       Done.
echo.

:: Activate venv
call canary_env\Scripts\activate.bat

:: Upgrade pip
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo       Done.
echo.

:: Install PyTorch CPU
echo [3/6] Installing PyTorch (CPU version)...
echo       This may take 5-10 minutes...
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu --quiet
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch install failed. Check your internet connection.
    pause
    exit /b 1
)
echo       Done.
echo.

:: Install NeMo
echo [4/6] Installing NVIDIA NeMo toolkit...
echo       This may take 10-15 minutes (~3 GB download)...
pip install "nemo_toolkit[asr]" --extra-index-url https://pypi.nvidia.com --quiet
if %errorlevel% neq 0 (
    echo [WARN] NeMo install had issues. Trying without extra index...
    pip install "nemo_toolkit[asr]" --quiet
)
echo       Done.
echo.

:: Install pyaudio (Windows-friendly way)
echo [5/6] Installing PyAudio and audio tools...
pip install pipwin --quiet
pipwin install pyaudio
if %errorlevel% neq 0 (
    echo [WARN] pipwin failed. Trying direct wheel install...
    pip install pyaudio --quiet
)
pip install soundfile numpy huggingface_hub --quiet
echo       Done.
echo.

:: Final check
echo [6/6] Verifying installation...
python -c "import torch; import soundfile; print('  torch:', torch.__version__)"
python -c "from nemo.collections.speechlm2.models import SALM; print('  nemo : OK')"
echo.

echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo   To run the app:
echo   1. Open Command Prompt in this folder
echo   2. Run:  canary_env\Scripts\activate
echo   3. Run:  python ambient_notes.py
echo.
echo   NOTE: First launch will download the model (~5 GB).
echo         This only happens once.
echo ============================================================
echo.
pause
