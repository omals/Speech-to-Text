"""
=============================================================
  AMBIENT NOTES - Speech to Text App
  Model  : Alibaba Qwen3-ASR 1.7B (CPU mode)
  Usage  : python ambient_notes.py
  Tested : Windows 10/11 Command Prompt
=============================================================
"""

import os
import sys
import wave
import time
import torch
import threading
import numpy as np
from datetime import datetime

# ── Banner ────────────────────────────────────────────────
print("=" * 60)
print("   AMBIENT NOTES  |  Qwen3-ASR 1.7B  |  CPU Mode")
print("=" * 60)
print()

# ── Config (edit these if needed) ─────────────────────────
CHUNK_SECONDS  = 10          # Seconds to record per chunk
SAMPLE_RATE    = 16000       # Hz (Qwen3-ASR expects 16kHz)
CHANNELS       = 1           # Mono
NOTES_FILE     = "notes.txt" # Output file (same folder as script)
TEMP_WAV       = "~temp_chunk.wav"
# ──────────────────────────────────────────────────────────


def check_imports():
    """Check all dependencies before loading the heavy model."""
    missing = []
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio  -> pip install pipwin && pipwin install pyaudio")
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile -> pip install soundfile")
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        missing.append("qwen-asr -> pip install -U qwen-asr")
        
    if missing:
        print("[ERROR] Missing packages. Install them first:\n")
        for m in missing:
            print("   pip install", m)
        print()
        sys.exit(1)


def load_model():
    print("[1/3] Loading Qwen3-ASR 1.7B (first run downloads weights)...")
    print("      This takes a few minutes on CPU. Please wait.\n")
    from qwen_asr import Qwen3ASRModel
    
    # Using bfloat16 for optimized memory/speed on modern CPUs; fallback to float32 if unsupported.
    try:
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", 
            dtype=torch.bfloat16, 
            device_map="cpu",
            max_new_tokens=256
        )
    except Exception:
        print("      [INFO] bfloat16 fallback. Trying float32...")
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", 
            dtype=torch.float32, 
            device_map="cpu",
            max_new_tokens=256
        )
        
    print("      Model loaded successfully!\n")
    return model


def record_chunk(filename, seconds, sample_rate, channels):
    """Record audio from microphone and save as WAV."""
    import pyaudio
    p = pyaudio.PyAudio()

    # List available input devices
    default_device = p.get_default_input_device_info()
    print(f"[MIC]  Using device: {default_device['name']}")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024
    )

    frames = []
    total_frames = int(sample_rate / 1024 * seconds)

    print(f"[REC]  Recording {seconds}s", end="", flush=True)
    for i in range(total_frames):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
        # Progress dots every ~2 seconds
        if i % int(total_frames / 5) == 0:
            print(".", end="", flush=True)
    print(" Done.")

    stream.stop_stream()
    stream.close()

    # Save WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    p.terminate()


def transcribe(model, wav_path):
    """Run Qwen3-ASR inference on a WAV file."""
    try:
        # Setting language=None activates automatic language identification (supports 52 languages)
        results = model.transcribe(audio=wav_path, language=None)
        if results and len(results) > 0:
            return results[0].text.strip()
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
    return ""


def save_note(text, notes_file):
    """Append transcribed text with timestamp to notes file."""
    if not text:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(notes_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n{text}\n\n")


def cleanup(temp_file):
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception:
        pass


def print_separator():
    print("-" * 60)


# ── Main ──────────────────────────────────────────────────
def main():
    check_imports()
    model = load_model()

    notes_path = os.path.abspath(NOTES_FILE)
    print("[2/3] Notes will be saved to:")
    print(f"      {notes_path}\n")
    print("[3/3] Starting ambient recording loop.")
    print("      Speak naturally. Press Ctrl+C to stop.\n")
    print_separator()

    session_count = 0
    session_start = datetime.now()

    try:
        while True:
            session_count += 1
            print(f"\n[CHUNK #{session_count}]")

            # Step 1: Record
            record_chunk(TEMP_WAV, CHUNK_SECONDS, SAMPLE_RATE, CHANNELS)

            # Step 2: Transcribe
            print("[STT]  Transcribing with Qwen3-ASR... (Processing on CPU)")
            t_start = time.time()
            text = transcribe(model, TEMP_WAV)
            elapsed = time.time() - t_start

            # Step 3: Show result
            if text:
                print(f"[TEXT] {text}")
                print(f"[TIME] Transcribed in {elapsed:.1f}s")
                save_note(text, NOTES_FILE)
                print(f"[SAVE] Appended to {NOTES_FILE}")
            else:
                print("[TEXT] (silence or error detected, nothing saved)")

            print_separator()
            cleanup(TEMP_WAV)

    except KeyboardInterrupt:
        print("\n\n[STOP] Recording stopped by user.")
        duration = datetime.now() - session_start
        print(f"       Session duration : {str(duration).split('.')[0]}")
        print(f"       Chunks recorded  : {session_count}")
        print(f"       Notes saved to   : {os.path.abspath(NOTES_FILE)}")
        cleanup(TEMP_WAV)
        print("\nGoodbye!")


if __name__ == "__main__":
    main()