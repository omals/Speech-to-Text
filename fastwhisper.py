"""
=============================================================
  AMBIENT NOTES - Speech to Text App
  Model  : Faster-Whisper (base) - runs on CPU
  Audio  : PyAudio (no C++ compiler needed on most systems)
  Speed  : ~3-5 seconds per 10s chunk
  Usage  : python ambient_notes_pyaudio.py
=============================================================

SETUP (run once):
  pip install faster-whisper pyaudio wave

  If PyAudio fails to install on Windows:
    pip install pipwin
    pipwin install pyaudio
  OR download the wheel from:
    https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
    pip install PyAudio‑0.2.14‑cp310‑cp310‑win_amd64.whl
=============================================================
"""

import os
import sys
import time
import wave
import struct
import numpy as np
from datetime import datetime

# ── Banner ────────────────────────────────────────────────
print("=" * 60)
print("   AMBIENT NOTES  |  Faster-Whisper  |  PyAudio Mode")
print("=" * 60)
print()

# ── Config ────────────────────────────────────────────────
CHUNK_SECONDS  = 10          # Seconds per recording chunk
SAMPLE_RATE    = 16000       # Hz (Whisper expects 16 kHz)
CHANNELS       = 1           # Mono
CHUNK_SIZE     = 1024        # PyAudio buffer frames per read
FORMAT         = None        # Set after PyAudio import (paInt16)
NOTES_FILE     = "notes.txt" # Output notes file
TEMP_WAV       = "~temp_chunk.wav"
WHISPER_MODEL  = "base"      # tiny | base | small | medium | large-v3
#   tiny   = fastest (~1-2s),  lower accuracy
#   base   = fast    (~3-5s),  good accuracy  ← recommended
#   small  = slower  (~8-12s), better accuracy
# ──────────────────────────────────────────────────────────


def check_imports():
    """Verify all required packages are installed."""
    missing = []
    try:
        import pyaudio
    except ImportError:
        missing.append(
            "pyaudio  -> pip install pyaudio\n"
            "           (Windows fallback: pip install pipwin && pipwin install pyaudio)"
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        missing.append("faster-whisper -> pip install faster-whisper")

    if missing:
        print("[ERROR] Missing packages:\n")
        for m in missing:
            print("       ", m)
            print()
        sys.exit(1)


def load_model():
    """Load Faster-Whisper model (downloads on first run)."""
    from faster_whisper import WhisperModel
    print(f"[1/3] Loading Faster-Whisper '{WHISPER_MODEL}' model...")
    print("      First run downloads model weights. Please wait.\n")
    model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8"
    )
    print("      Model loaded successfully!\n")
    return model


def list_input_devices():
    """Print all available audio input devices."""
    import pyaudio
    pa = pyaudio.PyAudio()
    print("[MIC]  Available input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"         [{i}] {info['name']}")
    pa.terminate()
    print()


def record_chunk(filename, seconds, sample_rate, chunk_size=1024):
    """
    Record audio using PyAudio and save to a WAV file.

    Parameters
    ----------
    filename    : output .wav path
    seconds     : how many seconds to record
    sample_rate : samples per second (16000 for Whisper)
    chunk_size  : number of frames per PyAudio buffer read
    """
    import pyaudio

    pa     = pyaudio.PyAudio()
    fmt    = pyaudio.paInt16
    frames = []

    # Open the default input stream
    try:
        stream = pa.open(
            format=fmt,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
    except OSError as e:
        print(f"\n[ERROR] Could not open microphone: {e}")
        print("        Make sure a microphone is connected and not in use.")
        pa.terminate()
        return False

    total_frames = int(sample_rate / chunk_size * seconds)
    dot_every    = max(1, total_frames // 5)   # 5 progress dots

    print(f"[REC]  Recording {seconds}s", end="", flush=True)

    for i in range(total_frames):
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        except Exception as e:
            print(f"\n[WARN] Audio read error (skipped): {e}")
            continue

        if (i + 1) % dot_every == 0:
            print(".", end="", flush=True)

    print(" Done.")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # ── Write WAV file ──────────────────────────────────────
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    return True


def get_rms(filename):
    """Return the RMS amplitude of the WAV — used for silence detection."""
    with wave.open(filename, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        n   = len(raw) // 2
        if n == 0:
            return 0.0
        samples = struct.unpack(f"{n}h", raw)
        rms = (sum(s * s for s in samples) / n) ** 0.5
    return rms


def transcribe(model, wav_path):
    """Run Faster-Whisper on the WAV file and return (text, lang, confidence)."""
    segments, info = model.transcribe(
        wav_path,
        beam_size=5,
        language=None,          # auto-detect language
        vad_filter=True,        # skip silent sections
        vad_parameters=dict(
            min_silence_duration_ms=500
        )
    )
    text       = " ".join(seg.text.strip() for seg in segments)
    lang       = info.language
    confidence = round(info.language_probability * 100, 1)
    return text.strip(), lang, confidence


def save_note(text, notes_file):
    """Append transcribed text with timestamp to notes file."""
    if not text:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(notes_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n{text}\n\n")


def cleanup(temp_file):
    """Delete the temporary WAV file."""
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

    # Show available mics before starting
    list_input_devices()

    model = load_model()

    notes_path = os.path.abspath(NOTES_FILE)
    print("[2/3] Notes will be saved to:")
    print(f"      {notes_path}\n")
    print("[3/3] Starting ambient recording loop.")
    print("      Speak naturally. Press Ctrl+C to stop.\n")
    print_separator()

    session_count  = 0
    session_start  = datetime.now()
    silence_streak = 0              # consecutive silent chunks

    try:
        while True:
            session_count += 1
            print(f"\n[CHUNK #{session_count}]")

            # ── Record ─────────────────────────────────────
            ok = record_chunk(TEMP_WAV, CHUNK_SECONDS, SAMPLE_RATE, CHUNK_SIZE)
            if not ok:
                print("[SKIP] Recording failed, retrying in 2s...")
                time.sleep(2)
                session_count -= 1
                continue

            # ── Silence gate (optional) ────────────────────
            rms = get_rms(TEMP_WAV)
            if rms < 50:                        # tweak threshold if needed
                silence_streak += 1
                print(f"[TEXT] (silence detected, RMS={rms:.0f} — nothing saved)")
                if silence_streak >= 3:
                    print(f"[INFO] {silence_streak} consecutive silent chunks.")
                cleanup(TEMP_WAV)
                print_separator()
                continue
            else:
                silence_streak = 0

            # ── Transcribe ─────────────────────────────────
            print("[STT]  Transcribing...", end="", flush=True)
            t_start = time.time()
            text, lang, confidence = transcribe(model, TEMP_WAV)
            elapsed = time.time() - t_start
            print(f" Done in {elapsed:.1f}s")

            # ── Output ─────────────────────────────────────
            if text:
                print(f"[LANG] {lang.upper()} ({confidence}% confidence)")
                print(f"[TEXT] {text}")
                save_note(text, NOTES_FILE)
                print(f"[SAVE] Appended to {NOTES_FILE}")
            else:
                print("[TEXT] (no speech detected after VAD filter)")

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