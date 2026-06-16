"""
=============================================================
  AMBIENT NOTES - Web App
  Backend : Flask + Faster-Whisper (base, CPU, local)
  Frontend: HTML/JS (record button → JSON output)
  Usage   : python app.py → open http://localhost:5000

SETUP:
  pip install flask faster-whisper
  No API key needed — runs fully locally on CPU.
  First run downloads the 'base' model weights (~140MB).
=============================================================
"""

from flask import Flask, request, jsonify, render_template_string
from faster_whisper import WhisperModel
import tempfile
import time
import os
import datetime

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────
WHISPER_MODEL = "base"      # tiny | base | small | medium | large-v3
#   tiny   = fastest (~1-2s),  lower accuracy
#   base   = fast    (~3-5s),  good accuracy  ← recommended
#   small  = slower  (~8-12s), better accuracy
VAD_MIN_SILENCE_MS = 500    # consecutive silence (ms) before VAD cuts a segment
# ──────────────────────────────────────────────────────────────

# ── Load Faster-Whisper model once at startup ────────────────
print("=" * 55)
print(f"  AMBIENT NOTES  |  Faster-Whisper ({WHISPER_MODEL})  |  CPU Mode")
print("=" * 55)
print("  Loading model (first run downloads weights)...\n")

model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print("  Model loaded successfully!\n")
# ──────────────────────────────────────────────────────────────


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Ambient Notes | Faster-Whisper Local STT</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0c0e14;
      --surface:   #13161f;
      --border:    #1f2537;
      --border-hi: #2e3650;
      --text:      #dde3f0;
      --muted:     #515c7a;
      --accent:    #fb923c;
      --accent-lo: #2e1f12;
      --green:     #34d399;
      --amber:     #fbbf24;
      --red:       #f87171;
      --mono:      'JetBrains Mono', monospace;
      --sans:      'Inter', sans-serif;
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

    .header { width: 100%; max-width: 700px; margin-bottom: 40px; }

    .header-top {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 6px;
    }

    .mic-icon {
      width: 36px; height: 36px;
      background: var(--accent-lo);
      border: 1px solid var(--accent);
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px;
    }

    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      color: var(--text);
    }

    .subtitle {
      font-size: 0.82rem;
      color: var(--muted);
      letter-spacing: 0.3px;
      margin-left: 48px;
    }

    .model-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--accent-lo);
      border: 1px solid var(--accent);
      border-radius: 20px;
      padding: 3px 10px;
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--accent);
      margin-left: 48px;
      margin-top: 8px;
    }

    .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 28px;
      width: 100%;
      max-width: 700px;
      margin-bottom: 16px;
    }

    .record-btn {
      width: 100%;
      padding: 16px;
      border-radius: 12px;
      border: none;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      background: var(--accent);
      color: #1c1206;
      letter-spacing: 0.3px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }

    .record-btn:hover:not(:disabled) { opacity: 0.88; transform: translateY(-1px); }
    .record-btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

    .record-btn.recording {
      background: var(--red);
      color: #fff;
      animation: pulse 1.3s infinite;
    }

    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }

    .waveform {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      height: 36px;
      margin: 18px 0 6px;
    }

    .bar {
      width: 3px;
      border-radius: 3px;
      background: var(--accent);
      opacity: 0.15;
      height: 6px;
      transition: opacity 0.3s;
    }

    .waveform.active .bar { opacity: 1; animation: wave 1s ease-in-out infinite; }

    .bar:nth-child(1)  { animation-delay: 0.00s; }
    .bar:nth-child(2)  { animation-delay: 0.08s; }
    .bar:nth-child(3)  { animation-delay: 0.16s; }
    .bar:nth-child(4)  { animation-delay: 0.24s; }
    .bar:nth-child(5)  { animation-delay: 0.32s; }
    .bar:nth-child(6)  { animation-delay: 0.40s; }
    .bar:nth-child(7)  { animation-delay: 0.32s; }
    .bar:nth-child(8)  { animation-delay: 0.24s; }
    .bar:nth-child(9)  { animation-delay: 0.16s; }
    .bar:nth-child(10) { animation-delay: 0.08s; }
    .bar:nth-child(11) { animation-delay: 0.00s; }

    @keyframes wave { 0%, 100% { height: 4px; } 50% { height: 28px; } }

    .status { text-align: center; font-size: 0.82rem; color: var(--muted); min-height: 20px; }
    .status.active     { color: var(--green); }
    .status.processing { color: var(--amber); }
    .status.error      { color: var(--red); }
    .status.silence    { color: var(--muted); }

    .timing-bar {
      display: none;
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      margin-top: 20px;
    }
    .timing-bar.visible { display: flex; }

    .timing-item {
      flex: 1;
      padding: 12px 10px;
      text-align: center;
      border-right: 1px solid var(--border);
    }
    .timing-item:last-child { border-right: none; }

    .timing-value {
      font-family: var(--mono);
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--green);
      letter-spacing: -0.5px;
    }

    .timing-label {
      font-size: 0.65rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-top: 3px;
    }

    .section-label {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--muted);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .badge {
      background: var(--accent-lo);
      color: var(--accent);
      border: 1px solid var(--accent);
      border-radius: 20px;
      padding: 2px 8px;
      font-size: 0.68rem;
      font-weight: 600;
      letter-spacing: 0.5px;
    }

    .badge.vad {
      background: #1a2e26;
      color: var(--green);
      border-color: var(--green);
    }

    .transcript-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
      min-height: 72px;
      font-size: 1rem;
      line-height: 1.65;
      color: var(--text);
      margin-bottom: 24px;
    }
    .transcript-box.empty { color: var(--muted); font-style: italic; }

    .json-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px;
      font-family: var(--mono);
      font-size: 0.78rem;
      color: #fdba74;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 340px;
      overflow-y: auto;
      min-height: 72px;
      line-height: 1.6;
    }

    .history { width: 100%; max-width: 700px; }

    .history-heading {
      font-size: 0.68rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--muted);
      margin-bottom: 12px;
      padding-left: 2px;
    }

    .history-item {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 10px;
      font-size: 0.9rem;
      transition: border-color 0.2s;
    }
    .history-item:hover { border-color: var(--border-hi); }

    .history-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .history-time { font-size: 0.72rem; color: var(--muted); }

    .copy-btn {
      background: none;
      border: 1px solid var(--border);
      color: var(--muted);
      border-radius: 6px;
      padding: 2px 10px;
      font-size: 0.72rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .copy-btn:hover { color: var(--text); border-color: var(--border-hi); }

    .history-text { line-height: 1.55; margin-bottom: 6px; }

    .history-meta { font-size: 0.7rem; color: var(--muted); font-family: var(--mono); }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
  </style>
</head>
<body>

  <div class="header">
    <div class="header-top">
      <div class="mic-icon">🎙️</div>
      <h1>Ambient Notes</h1>
    </div>
    <div class="subtitle">Local speech-to-text — no API key, no cloud, fully private</div>
    <div class="model-pill"><span class="dot"></span> Faster-Whisper base · running locally on CPU</div>
  </div>

  <div class="card">
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

    <div class="timing-bar" id="timingBar">
      <div class="timing-item">
        <div class="timing-value" id="tTotal">--</div>
        <div class="timing-label">Total Time</div>
      </div>
      <div class="timing-item">
        <div class="timing-value" id="tInfer">--</div>
        <div class="timing-label">Inference</div>
      </div>
      <div class="timing-item">
        <div class="timing-value" id="tAudio">--</div>
        <div class="timing-label">Audio Length</div>
      </div>
      <div class="timing-item">
        <div class="timing-value" id="tRTF">--</div>
        <div class="timing-label">RTF×</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="section-label">
      Transcript
      <span class="badge">Faster-Whisper base</span>
      <span class="badge vad">VAD filter on</span>
    </div>
    <div class="transcript-box empty" id="transcriptBox">Your transcribed text will appear here…</div>

    <div class="section-label">JSON Response</div>
    <div class="json-box" id="jsonBox">{
  "status": "waiting for audio…"
}</div>
  </div>

  <div class="history" id="historyWrap" style="display:none">
    <div class="history-heading">Session history</div>
    <div id="history"></div>
  </div>

<script>
  let mediaRecorder = null;
  let audioChunks   = [];
  let isRecording   = false;

  async function toggleRecording() {
    isRecording ? stopRecording() : await startRecording();
  }

  async function startRecording() {
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
      document.getElementById('timingBar').classList.remove('visible');
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
    setStatus('Transcribing with Faster-Whisper…', 'processing');
  }

  async function processAudio() {
    const clientStart = Date.now();
    const blob     = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');

    try {
      const res  = await fetch('/transcribe', { method: 'POST', body: formData });
      const data = await res.json();
      const totalMs = Date.now() - clientStart;
      showTiming(data, totalMs);
      displayResult(data);
    } catch (err) {
      displayError(err.message);
    } finally {
      document.getElementById('recordBtn').disabled = false;
    }
  }

  function showTiming(data, totalMs) {
    const bar = document.getElementById('timingBar');
    bar.classList.add('visible');

    const fmt = ms => ms >= 1000 ? (ms / 1000).toFixed(1) + 's' : ms + 'ms';

    document.getElementById('tTotal').textContent = fmt(totalMs);
    document.getElementById('tInfer').textContent = data.timing ? fmt(data.timing.inference_ms) : '--';
    document.getElementById('tAudio').textContent = data.audio_duration ? data.audio_duration.toFixed(1) + 's' : '--';
    document.getElementById('tRTF').textContent = data.timing && data.timing.rtf ? data.timing.rtf + '×' : '--';
  }

  function displayResult(data) {
    const tBox = document.getElementById('transcriptBox');
    const jBox = document.getElementById('jsonBox');

    if (data.status === 'no_speech') {
      tBox.textContent = 'No speech detected (filtered out by VAD).';
      tBox.classList.add('empty');
      setStatus('Silence detected — nothing to transcribe. Click to record again.', 'silence');
    } else if (data.text && data.text.trim()) {
      tBox.textContent = data.text;
      tBox.classList.remove('empty');
      addToHistory(data);
      setStatus('Done — click to record again.', '');
    } else {
      tBox.textContent = data.error || 'No speech detected.';
      tBox.classList.add('empty');
      setStatus('Done — click to record again.', '');
    }

    jBox.textContent = JSON.stringify(data, null, 2);
  }

  function displayError(msg) {
    document.getElementById('transcriptBox').textContent = 'Error: ' + msg;
    document.getElementById('transcriptBox').classList.add('empty');
    document.getElementById('jsonBox').textContent = JSON.stringify({ error: msg }, null, 2);
    setStatus('Something went wrong — try again.', 'error');
  }

  function addToHistory(data) {
    document.getElementById('historyWrap').style.display = '';
    const history = document.getElementById('history');
    const safe    = (data.text || '').replace(/'/g, "\\'");
    const item    = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
      <div class="history-top">
        <div class="history-time">${data.timestamp}</div>
        <button class="copy-btn" onclick="copyText(this, '${safe}')">Copy</button>
      </div>
      <div class="history-text">${data.text}</div>
      <div class="history-meta">
        lang: ${data.language ? data.language.toUpperCase() : '--'} &nbsp;·&nbsp;
        confidence: ${data.confidence ? data.confidence + '%' : '--'} &nbsp;·&nbsp;
        inference: ${data.timing ? data.timing.inference_ms + 'ms' : '--'}
      </div>`;
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
    el.className   = 'status ' + cls;
  }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ── Faster-Whisper inference (with VAD filter) ────
        infer_start = time.time()

        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language=None,            # auto-detect language
            vad_filter=True,          # skip silent sections
            vad_parameters=dict(
                min_silence_duration_ms=VAD_MIN_SILENCE_MS
            )
        )

        segments = list(segments)     # materialize generator
        text = " ".join(seg.text.strip() for seg in segments).strip()

        infer_ms = round((time.time() - infer_start) * 1000)
        # ──────────────────────────────────────────────────

        language       = info.language
        confidence     = round(info.language_probability * 100, 1)
        audio_duration = info.duration

        rtf = round(audio_duration / (infer_ms / 1000), 2) \
              if audio_duration and infer_ms > 0 else None

        if not text:
            # VAD filtered out all audio as silence — mirrors the CLI's silence gate
            return jsonify({
                "status":         "no_speech",
                "text":           "",
                "language":       language,
                "confidence":     confidence,
                "audio_duration": audio_duration,
                "timing": {
                    "inference_ms":  infer_ms,
                    "inference_sec": round(infer_ms / 1000, 2),
                    "rtf":           rtf
                },
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model":     f"faster-whisper-{WHISPER_MODEL} (local CPU)"
            })

        response = {
            "status":         "success",
            "text":            text,
            "language":        language,
            "confidence":      confidence,
            "audio_duration":  audio_duration,
            "timing": {
                "inference_ms":  infer_ms,
                "inference_sec": round(infer_ms / 1000, 2),
                "rtf":           rtf
            },
            "segments": [
                {
                    "start": round(seg.start, 2),
                    "end":   round(seg.end, 2),
                    "text":  seg.text.strip()
                }
                for seg in segments
            ],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":     f"faster-whisper-{WHISPER_MODEL} (local CPU)"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "text": None})

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("  Open browser at: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000)