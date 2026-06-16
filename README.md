# Speech-to-Text (STT) PoC

This directory contains standalone and comparative Speech-to-Text implementations using AssemblyAI, Faster-Whisper, OpenAI Whisper, and Qwen3-ASR.

## Files in this Directory

* **Main Comparison App**:
  * [comparison.py](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/comparison.py) - Flask web app to compare all models side-by-side.
  * [comparison.jpeg](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/comparison.jpeg) - UI screenshot.

* **AssemblyAI (Cloud)**:
  * [assembleai.py](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/assembleai.py) - Standalone Flask web app.
  * [assembleai.jpeg](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/assembleai.jpeg) - UI screenshot.
  * [assemblai_notes.txt](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/assemblai_notes.txt) - Sample JSON output.

* **Faster-Whisper (Local CPU)**:
  * [fastwhisper_app.py](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/fastwhisper_app.py) - Standalone Flask web app.
  * [fastwhisper.py](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/fastwhisper.py) - CLI recorder/transcriber.
  * [fastwhisper_notes.txt](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/fastwhisper_notes.txt) - Console logs.

* **OpenAI Whisper (Local CPU)**:
  * [openwhisper.py](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/openwhisper.py) - Standalone Flask web app.
  * [openwhisper.jpeg](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/openwhisper.jpeg) - UI screenshot.
  * [openwhisper_notes.txt](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/openwhisper_notes.txt) - Sample JSON output.

* **Qwen3-ASR (Local CPU)**:
  * [qwen_app.py](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/qwen_app.py) - Standalone Flask web app.
  * [qwen.py](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/qwen.py) - CLI recorder/transcriber.
  * [qwen_app.jpeg](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/qwen_app.jpeg) - UI screenshot.
  * [qwen_notes.txt](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/qwen_notes.txt) - Console logs.

* **Documentation & Setup**:
  * [STT_PoC_Comparison_Report.docx](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/STT_PoC_Comparison_Report.docx) - PoC comparison report.
  * [steps.txt](file:///c:/Users/osivankutty/OneDrive%20-%20ChiSquare%20Labs%20Pvt.%20Ltd/Documents/aether/STT/steps.txt) - Environment setup commands.

---

## Quick Setup

Run these commands to set up the environment:

```bash
# Setup virtual environment
python -m venv canary_env
canary_env\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install soundfile numpy huggingface_hub flask assemblyai faster-whisper openai-whisper qwen-asr
```

## Running the Apps

* **Start the Comparison Web App**: `python comparison.py` (open [http://localhost:5000](http://localhost:5000))
* **Start a standalone Web App**: `python <app_filename.py>` (e.g., `qwen_app.py`)
* **Start a CLI recorder**: `python <cli_filename.py>` (e.g., `fastwhisper.py`)
