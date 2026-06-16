"""
=============================================================
  AMBIENT NOTES - Unified STT Benchmark Web App
  Backend : Flask
  Engines : AssemblyAI (cloud) · Faster-Whisper (local CPU)
            OpenAI Whisper (local CPU) · Qwen3-ASR (local CPU)
  Frontend: HTML/JS — one click records once, runs through
            every available engine, shows timing/accuracy
            comparison side-by-side. White theme.
  Usage   : python app.py → open http://localhost:5000
=============================================================

SETUP:
  pip install flask
  Optional per-engine deps (only needed for engines you enable):
    AssemblyAI       : pip install assemblyai      + set API key below
    Faster-Whisper   : pip install faster-whisper
    OpenAI Whisper   : pip install openai-whisper   + ffmpeg installed
    Qwen3-ASR        : pip install torch qwen-asr

  Engines whose dependencies are not installed are automatically
  disabled (greyed out in the UI) — the app still runs fine with
  only a subset of engines available.
=============================================================
"""

from flask import Flask, request, jsonify, render_template_string
import tempfile
import time
import os
import datetime
import traceback

app = Flask(__name__)

# ── AssemblyAI API key (only needed if assemblyai engine is enabled) ──
ASSEMBLYAI_API_KEY = "2fda9e8909d44266bd3cab7463198d7e"

# ── Local model config ─────────────────────────────────────
FASTER_WHISPER_MODEL = "base"      # tiny | base | small | medium | large-v3
OPENAI_WHISPER_MODEL = "base"      # tiny | base | small | medium | large
VAD_MIN_SILENCE_MS = 500
QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# =============================================================
#  Lazy / guarded engine loading
#  Each engine is independently optional. If its package isn't
#  installed, it's marked unavailable and skipped at runtime.
# =============================================================

ENGINES = {}  # name -> {"available": bool, "reason": str, "label": str, "color": str}

# ---- AssemblyAI ----
try:
    import assemblyai as aai
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    ENGINES["assemblyai"] = {"available": True, "reason": "", "label": "AssemblyAI", "color": "#2563eb"}
except Exception as e:
    ENGINES["assemblyai"] = {"available": False, "reason": str(e), "label": "AssemblyAI", "color": "#2563eb"}

# ---- Faster-Whisper ----
fw_model = None
try:
    from faster_whisper import WhisperModel
    print("Loading Faster-Whisper model...")
    fw_model = WhisperModel(FASTER_WHISPER_MODEL, device="cpu", compute_type="int8")
    ENGINES["faster_whisper"] = {"available": True, "reason": "", "label": f"Faster-Whisper ({FASTER_WHISPER_MODEL})", "color": "#ea580c"}
    print("Faster-Whisper loaded.")
except Exception as e:
    ENGINES["faster_whisper"] = {"available": False, "reason": str(e), "label": f"Faster-Whisper ({FASTER_WHISPER_MODEL})", "color": "#ea580c"}

# ---- OpenAI Whisper ----
ow_model = None
try:
    import whisper
    print("Loading OpenAI Whisper model...")
    ow_model = whisper.load_model(OPENAI_WHISPER_MODEL)
    ENGINES["openai_whisper"] = {"available": True, "reason": "", "label": f"OpenAI Whisper ({OPENAI_WHISPER_MODEL})", "color": "#0891b2"}
    print("OpenAI Whisper loaded.")
except Exception as e:
    ENGINES["openai_whisper"] = {"available": False, "reason": str(e), "label": f"OpenAI Whisper ({OPENAI_WHISPER_MODEL})", "color": "#0891b2"}

# ---- Qwen3-ASR ----
qwen_model = None
try:
    import torch
    from qwen_asr import Qwen3ASRModel
    print("Loading Qwen3-ASR model...")
    try:
        qwen_model = Qwen3ASRModel.from_pretrained(QWEN_MODEL_ID, dtype=torch.bfloat16, device_map="cpu", max_new_tokens=256)
    except Exception:
        qwen_model = Qwen3ASRModel.from_pretrained(QWEN_MODEL_ID, dtype=torch.float32, device_map="cpu", max_new_tokens=256)
    ENGINES["qwen_asr"] = {"available": True, "reason": "", "label": "Qwen3-ASR 1.7B", "color": "#7c3aed"}
    print("Qwen3-ASR loaded.")
except Exception as e:
    ENGINES["qwen_asr"] = {"available": False, "reason": str(e), "label": "Qwen3-ASR 1.7B", "color": "#7c3aed"}


# =============================================================
#  Per-engine transcription functions
#  Each returns a dict with a consistent shape:
#  { text, language, confidence, audio_duration, timing:{inference_ms, rtf}, extra:{...} }
# =============================================================

def run_assemblyai(path):
    t0 = time.time()
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(path)
    ms = round((time.time() - t0) * 1000)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(transcript.error)

    rtf = round(transcript.audio_duration / (ms / 1000), 2) if transcript.audio_duration and ms > 0 else None

    return {
        "text": transcript.text or "",
        "language": transcript.language_code,
        "confidence": round(transcript.confidence * 100, 1) if transcript.confidence else None,
        "audio_duration": transcript.audio_duration,
        "timing": {"inference_ms": ms, "inference_sec": round(ms / 1000, 2), "rtf": rtf},
        "extra": {
            "words": [{"text": w.text, "start_ms": w.start, "end_ms": w.end, "confidence": w.confidence}
                      for w in (transcript.words or [])][:50]
        }
    }


def run_faster_whisper(path):
    t0 = time.time()
    segments, info = fw_model.transcribe(
        path, beam_size=5, language=None,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
    )
    segments = list(segments)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    ms = round((time.time() - t0) * 1000)

    audio_duration = info.duration
    rtf = round(audio_duration / (ms / 1000), 2) if audio_duration and ms > 0 else None

    return {
        "text": text,
        "language": info.language,
        "confidence": round(info.language_probability * 100, 1),
        "audio_duration": audio_duration,
        "timing": {"inference_ms": ms, "inference_sec": round(ms / 1000, 2), "rtf": rtf},
        "extra": {
            "segments": [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
        }
    }


def run_openai_whisper(path):
    t0 = time.time()
    result = ow_model.transcribe(path, language=None, fp16=False, verbose=False)
    ms = round((time.time() - t0) * 1000)

    text = (result.get("text") or "").strip()
    language = result.get("language", "")
    segments = result.get("segments", [])
    audio_duration = segments[-1]["end"] if segments else None
    rtf = round(audio_duration / (ms / 1000), 2) if audio_duration and ms > 0 else None

    return {
        "text": text,
        "language": language,
        "confidence": None,
        "audio_duration": audio_duration,
        "timing": {"inference_ms": ms, "inference_sec": round(ms / 1000, 2), "rtf": rtf},
        "extra": {
            "segments": [{"id": s["id"], "start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()} for s in segments]
        }
    }


def run_qwen_asr(path):
    t0 = time.time()
    results = qwen_model.transcribe(audio=path, language=None)
    text = results[0].text.strip() if results and len(results) > 0 else ""
    ms = round((time.time() - t0) * 1000)

    audio_duration = getattr(results[0], "duration", None) if results else None
    language = getattr(results[0], "language", None) if results else None
    rtf = round(audio_duration / (ms / 1000), 2) if audio_duration and ms > 0 else None

    return {
        "text": text,
        "language": language,
        "confidence": None,
        "audio_duration": audio_duration,
        "timing": {"inference_ms": ms, "inference_sec": round(ms / 1000, 2), "rtf": rtf},
        "extra": {}
    }


RUNNERS = {
    "assemblyai": run_assemblyai,
    "faster_whisper": run_faster_whisper,
    "openai_whisper": run_openai_whisper,
    "qwen_asr": run_qwen_asr,
}


# =============================================================
#  Frontend
# =============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Ambient Notes | STT Engine Benchmark</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #ffffff;
    --surface:   #f8fafc;
    --surface-2: #f1f5f9;
    --border:    #e2e8f0;
    --border-hi: #cbd5e1;
    --text:      #0f172a;
    --muted:     #64748b;
    --muted-2:   #94a3b8;
    --accent:    #2563eb;
    --accent-lo: #eff6ff;
    --green:     #16a34a;
    --green-lo:  #f0fdf4;
    --amber:     #d97706;
    --amber-lo:  #fffbeb;
    --red:       #dc2626;
    --red-lo:    #fef2f2;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
    --shadow:    0 1px 2px rgba(15,23,42,0.04), 0 1px 8px rgba(15,23,42,0.04);
  }

  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 20px 80px;
  }

  .header { width: 100%; max-width: 1080px; margin-bottom: 32px; }

  .header-top { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }

  .mic-icon {
    width: 38px; height: 38px;
    background: var(--accent-lo);
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 19px;
  }

  h1 { font-size: 1.65rem; font-weight: 800; letter-spacing: -0.5px; color: var(--text); }

  .subtitle { font-size: 0.85rem; color: var(--muted); letter-spacing: 0.2px; margin-left: 50px; }

  .card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    width: 100%;
    max-width: 1080px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
  }

  .engine-toggles {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 22px;
  }

  .engine-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 999px;
    border: 1.5px solid var(--border);
    background: var(--surface);
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
    transition: all 0.15s;
  }

  .engine-chip .swatch { width: 9px; height: 9px; border-radius: 50%; }

  .engine-chip.on {
    background: var(--bg);
    color: var(--text);
    border-color: var(--border-hi);
    box-shadow: var(--shadow);
  }

  .engine-chip.disabled {
    opacity: 0.45;
    cursor: not-allowed;
    text-decoration: line-through;
  }

  .record-btn {
    width: 100%;
    padding: 18px;
    border-radius: 12px;
    border: none;
    font-size: 1.02rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
    background: var(--accent);
    color: #fff;
    letter-spacing: 0.2px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }

  .record-btn:hover:not(:disabled) { background: #1d4ed8; transform: translateY(-1px); }
  .record-btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

  .record-btn.recording { background: var(--red); animation: pulse 1.3s infinite; }

  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.78; } }

  .waveform { display: flex; align-items: center; justify-content: center; gap: 5px; height: 36px; margin: 18px 0 6px; }

  .bar { width: 3px; border-radius: 3px; background: var(--accent); opacity: 0.15; height: 6px; transition: opacity 0.3s; }
  .waveform.active .bar { opacity: 1; animation: wave 1s ease-in-out infinite; }

  .bar:nth-child(1) { animation-delay: 0.00s; } .bar:nth-child(2) { animation-delay: 0.08s; }
  .bar:nth-child(3) { animation-delay: 0.16s; } .bar:nth-child(4) { animation-delay: 0.24s; }
  .bar:nth-child(5) { animation-delay: 0.32s; } .bar:nth-child(6) { animation-delay: 0.40s; }
  .bar:nth-child(7) { animation-delay: 0.32s; } .bar:nth-child(8) { animation-delay: 0.24s; }
  .bar:nth-child(9) { animation-delay: 0.16s; } .bar:nth-child(10) { animation-delay: 0.08s; }
  .bar:nth-child(11) { animation-delay: 0.00s; }

  @keyframes wave { 0%, 100% { height: 4px; } 50% { height: 28px; } }

  .status { text-align: center; font-size: 0.85rem; color: var(--muted); min-height: 20px; margin-top: 6px; }
  .status.active     { color: var(--green); }
  .status.processing { color: var(--amber); }
  .status.error       { color: var(--red); }

  .section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* ── Summary comparison table ── */
  .summary-table { width: 100%; border-collapse: collapse; margin-bottom: 4px; }
  .summary-table th, .summary-table td {
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
  }
  .summary-table th {
    color: var(--muted);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
  }
  .summary-table td.engine-name { font-weight: 700; display: flex; align-items: center; gap: 8px; }
  .summary-table .swatch { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .summary-table td.mono, .summary-table th.mono { font-family: var(--mono); }
  .summary-table tr.fastest td { background: var(--green-lo); }
  .summary-table tr:last-child td { border-bottom: none; }
  .pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    background: var(--green-lo);
    color: var(--green);
    border: 1px solid #bbf7d0;
  }
  .pill.err { background: var(--red-lo); color: var(--red); border-color: #fecaca; }

  /* ── Per-engine result cards ── */
  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 16px;
  }

  .result-card {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    background: var(--surface);
  }

  .result-card.fastest { border-color: #bbf7d0; background: var(--green-lo); }
  .result-card.errored { border-color: #fecaca; background: var(--red-lo); }

  .result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .result-title { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 0.92rem; }
  .result-title .swatch { width: 10px; height: 10px; border-radius: 50%; }

  .timing-row { display: flex; gap: 16px; margin-bottom: 14px; flex-wrap: wrap; }
  .timing-stat { font-family: var(--mono); font-size: 0.78rem; color: var(--muted); }
  .timing-stat b { color: var(--text); font-size: 0.95rem; }

  .transcript-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    min-height: 56px;
    font-size: 0.92rem;
    line-height: 1.6;
    color: var(--text);
    margin-bottom: 10px;
  }
  .transcript-box.empty { color: var(--muted-2); font-style: italic; }

  details.json-details { margin-top: 8px; }
  details.json-details summary {
    cursor: pointer;
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
  }
  .json-box {
    background: #0f172a;
    border-radius: 8px;
    padding: 14px;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: #93c5fd;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 260px;
    overflow-y: auto;
    margin-top: 8px;
    line-height: 1.55;
  }

  .copy-btn {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  .copy-btn:hover { color: var(--text); border-color: var(--border-hi); }

  .empty-hint {
    text-align: center;
    color: var(--muted-2);
    font-size: 0.88rem;
    padding: 30px 0;
  }

  .history { width: 100%; max-width: 1080px; }
  .history-heading {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--muted); margin-bottom: 12px; padding-left: 2px;
  }
  .history-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    box-shadow: var(--shadow);
  }
  .history-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .history-time { font-size: 0.72rem; color: var(--muted); }
  .history-best { font-size: 0.72rem; color: var(--muted); margin-top: 6px; font-family: var(--mono); }

  ::-webkit-scrollbar { width: 7px; height: 7px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }
</style>
</head>
<body>

  <div class="header">
    <div class="header-top">
      <div class="mic-icon">🎙️</div>
      <h1>Ambient Notes — STT Benchmark</h1>
    </div>
    <div class="subtitle">Record once, transcribe with every engine, compare speed &amp; accuracy side-by-side</div>
  </div>

  <div class="card">
    <div class="section-label">Engines</div>
    <div class="engine-toggles" id="engineToggles"></div>

    <button class="record-btn" id="recordBtn" onclick="toggleRecording()">
      <span id="btnIcon">●</span>
      <span id="btnLabel">Start Recording</span>
    </button>

    <div class="waveform" id="waveform">
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
      <div class="bar"></div><div class="bar"></div>
    </div>

    <div class="status" id="status">Click to start recording</div>
  </div>

  <div class="card" id="summaryCard" style="display:none">
    <div class="section-label">Comparison Summary</div>
    <table class="summary-table" id="summaryTable">
      <thead>
        <tr>
          <th>Engine</th>
          <th class="mono">Inference</th>
          <th class="mono">RTF</th>
          <th class="mono">Confidence</th>
          <th>Language</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody id="summaryBody"></tbody>
    </table>
  </div>

  <div class="card">
    <div class="section-label">Engine Results</div>
    <div id="resultsGrid" class="results-grid">
      <div class="empty-hint">Run a recording to see transcripts and timing from each engine.</div>
    </div>
  </div>

  <div class="history" id="historyWrap" style="display:none">
    <div class="history-heading">Session History</div>
    <div id="history"></div>
  </div>

<script>
  const ENGINES = __ENGINES_JSON__;

  let mediaRecorder = null;
  let audioChunks   = [];
  let isRecording   = false;
  let enabledEngines = new Set(Object.keys(ENGINES).filter(k => ENGINES[k].available));

  function renderToggles() {
    const wrap = document.getElementById('engineToggles');
    wrap.innerHTML = '';
    Object.keys(ENGINES).forEach(key => {
      const eng = ENGINES[key];
      const chip = document.createElement('div');
      chip.className = 'engine-chip' + (enabledEngines.has(key) ? ' on' : '') + (eng.available ? '' : ' disabled');
      chip.innerHTML = `<span class="swatch" style="background:${eng.color}"></span> ${eng.label}${eng.available ? '' : ' (unavailable)'}`;
      if (eng.available) {
        chip.onclick = () => {
          if (enabledEngines.has(key)) enabledEngines.delete(key); else enabledEngines.add(key);
          renderToggles();
        };
      } else {
        chip.title = eng.reason || 'Dependency not installed';
      }
      wrap.appendChild(chip);
    });
  }
  renderToggles();

  async function toggleRecording() {
    isRecording ? stopRecording() : await startRecording();
  }

  async function startRecording() {
    if (enabledEngines.size === 0) {
      setStatus('Select at least one engine first.', 'error');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks   = [];

      mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = processAudio;
      mediaRecorder.start();

      isRecording = true;

      document.getElementById('recordBtn').classList.add('recording');
      document.getElementById('btnIcon').textContent  = '■';
      document.getElementById('btnLabel').textContent = 'Stop Recording';
      document.getElementById('waveform').classList.add('active');
      setStatus('Recording… speak now', 'active');
    } catch {
      setStatus('Microphone access denied — please allow mic access.', 'error');
    }
  }

  function stopRecording() {
    if (!mediaRecorder || !isRecording) return;
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    isRecording = false;

    document.getElementById('recordBtn').classList.remove('recording');
    document.getElementById('recordBtn').disabled = true;
    document.getElementById('btnIcon').textContent  = '●';
    document.getElementById('btnLabel').textContent = 'Start Recording';
    document.getElementById('waveform').classList.remove('active');
    setStatus('Transcribing with ' + enabledEngines.size + ' engine(s)…', 'processing');
  }

  async function processAudio() {
    const blob     = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');
    formData.append('engines', JSON.stringify(Array.from(enabledEngines)));

    try {
      const res  = await fetch('/transcribe', { method: 'POST', body: formData });
      const data = await res.json();
      displayResults(data);
    } catch (err) {
      setStatus('Request failed: ' + err.message, 'error');
    } finally {
      document.getElementById('recordBtn').disabled = false;
    }
  }

  function fmtMs(ms) {
    if (ms === null || ms === undefined) return '--';
    return ms >= 1000 ? (ms / 1000).toFixed(2) + 's' : ms + 'ms';
  }

  function displayResults(data) {
    const results = data.results || {};
    const keys = Object.keys(results);

    if (keys.length === 0) {
      setStatus('No results returned.', 'error');
      return;
    }

    // find fastest successful engine
    let fastestKey = null, fastestMs = Infinity;
    keys.forEach(k => {
      const r = results[k];
      if (r.status === 'success' && r.timing && r.timing.inference_ms < fastestMs) {
        fastestMs = r.timing.inference_ms;
        fastestKey = k;
      }
    });

    // ── Summary table ──
    document.getElementById('summaryCard').style.display = '';
    const body = document.getElementById('summaryBody');
    body.innerHTML = '';
    keys.forEach(k => {
      const eng = ENGINES[k];
      const r = results[k];
      const tr = document.createElement('tr');
      if (k === fastestKey) tr.className = 'fastest';
      const statusPill = r.status === 'success'
        ? '<span class="pill">success</span>'
        : `<span class="pill err">${r.status || 'error'}</span>`;
      tr.innerHTML = `
        <td class="engine-name"><span class="swatch" style="background:${eng.color}"></span>${eng.label}</td>
        <td class="mono">${r.timing ? fmtMs(r.timing.inference_ms) : '--'}</td>
        <td class="mono">${r.timing && r.timing.rtf ? r.timing.rtf + '×' : '--'}</td>
        <td class="mono">${r.confidence !== null && r.confidence !== undefined ? r.confidence + '%' : '--'}</td>
        <td>${r.language || '--'}</td>
        <td>${statusPill}</td>
      `;
      body.appendChild(tr);
    });

    // ── Result cards ──
    const grid = document.getElementById('resultsGrid');
    grid.innerHTML = '';
    keys.forEach(k => {
      const eng = ENGINES[k];
      const r = results[k];
      const card = document.createElement('div');
      card.className = 'result-card' + (k === fastestKey ? ' fastest' : '') + (r.status !== 'success' ? ' errored' : '');

      const transcriptText = r.status === 'success' ? (r.text || '(empty transcript)') : (r.error || 'Failed');
      const isEmpty = !(r.status === 'success' && r.text && r.text.trim());

      card.innerHTML = `
        <div class="result-header">
          <div class="result-title"><span class="swatch" style="background:${eng.color}"></span>${eng.label}</div>
          <button class="copy-btn" onclick="copyText(this, ${JSON.stringify(r.text || '')})">Copy</button>
        </div>
        <div class="timing-row">
          <div class="timing-stat">Total <b>${r.timing ? fmtMs(r.timing.inference_ms) : '--'}</b></div>
          <div class="timing-stat">RTF <b>${r.timing && r.timing.rtf ? r.timing.rtf + '×' : '--'}</b></div>
          <div class="timing-stat">Audio <b>${r.audio_duration ? r.audio_duration.toFixed(1) + 's' : '--'}</b></div>
          <div class="timing-stat">Confidence <b>${r.confidence !== null && r.confidence !== undefined ? r.confidence + '%' : '--'}</b></div>
        </div>
        <div class="transcript-box ${isEmpty ? 'empty' : ''}">${transcriptText}</div>
        <details class="json-details">
          <summary>Raw JSON</summary>
          <div class="json-box">${JSON.stringify(r, null, 2)}</div>
        </details>
      `;
      grid.appendChild(card);
    });

    setStatus('Done — click to record again.', '');
    addToHistory(data, fastestKey);
  }

  function addToHistory(data, fastestKey) {
    document.getElementById('historyWrap').style.display = '';
    const history = document.getElementById('history');
    const item = document.createElement('div');
    item.className = 'history-item';
    const fastestLabel = fastestKey ? ENGINES[fastestKey].label : '--';
    const engineCount = Object.keys(data.results || {}).length;
    item.innerHTML = `
      <div class="history-top">
        <div class="history-time">${data.timestamp}</div>
      </div>
      <div class="history-best">⚡ Fastest: ${fastestLabel} &nbsp;·&nbsp; Engines run: ${engineCount}</div>
    `;
    history.insertBefore(item, history.firstChild);
  }

  function copyText(btn, text) {
    navigator.clipboard.writeText(text);
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1500);
  }

  function setStatus(msg, cls) {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className   = 'status ' + (cls || '');
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    import json as _json
    engines_meta = {k: {"available": v["available"], "label": v["label"], "color": v["color"]} for k, v in ENGINES.items()}
    return render_template_string(HTML.replace("__ENGINES_JSON__", _json.dumps(engines_meta)))


@app.route("/transcribe", methods=["POST"])
def transcribe():
    import json as _json

    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    requested = _json.loads(request.form.get("engines", "[]"))
    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    results = {}
    try:
        for engine_key in requested:
            if engine_key not in RUNNERS:
                continue
            if not ENGINES.get(engine_key, {}).get("available"):
                results[engine_key] = {"status": "unavailable", "error": ENGINES.get(engine_key, {}).get("reason", "Engine not installed")}
                continue
            try:
                out = RUNNERS[engine_key](tmp_path)
                out["status"] = "success" if out.get("text", "").strip() else "no_speech"
                results[engine_key] = out
            except Exception as e:
                traceback.print_exc()
                results[engine_key] = {"status": "error", "error": str(e), "text": None}

        return jsonify({
            "results": results,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("  AMBIENT NOTES — UNIFIED STT BENCHMARK")
    print("=" * 60)
    for k, v in ENGINES.items():
        status = "AVAILABLE" if v["available"] else f"DISABLED ({v['reason'][:60]})"
        print(f"  {v['label']:<28} {status}")
    print("=" * 60)
    print("  Open browser at: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    app.run(debug=False, port=5000)