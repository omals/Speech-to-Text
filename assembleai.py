"""
=============================================================
  AMBIENT NOTES - Web App
  Backend : Flask + AssemblyAI Python SDK
  Frontend: HTML/JS (record button → JSON output)
  Usage   : python app.py → open http://localhost:5000
=============================================================

SETUP:
  pip install flask assemblyai
  Get free API key: https://www.assemblyai.com/dashboard
=============================================================
"""

from flask import Flask, request, jsonify, render_template_string
import assemblyai as aai
import tempfile
import time
import os

app = Flask(__name__)

# ── Put your AssemblyAI API key here ──────────────────────
aai.settings.api_key = "2fda9e8909d44266bd3cab7463198d7e"
# ──────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Ambient Notes | Speech to Text</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 20px;
    }

    h1 {
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 6px;
      background: linear-gradient(90deg, #6ee7b7, #3b82f6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      color: #64748b;
      font-size: 0.9rem;
      margin-bottom: 40px;
    }

    .card {
      background: #1e2330;
      border: 1px solid #2d3748;
      border-radius: 16px;
      padding: 32px;
      width: 100%;
      max-width: 680px;
      margin-bottom: 24px;
    }

    .record-btn {
      width: 100%;
      padding: 18px;
      border-radius: 12px;
      border: none;
      font-size: 1.1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      background: linear-gradient(135deg, #10b981, #3b82f6);
      color: white;
      letter-spacing: 0.5px;
    }

    .record-btn:hover { opacity: 0.9; transform: translateY(-1px); }
    .record-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    .record-btn.recording {
      background: linear-gradient(135deg, #ef4444, #f97316);
      animation: pulse 1.2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.75; }
    }

    .status {
      text-align: center;
      margin-top: 16px;
      font-size: 0.9rem;
      color: #64748b;
      min-height: 22px;
    }

    .status.active { color: #10b981; }
    .status.processing { color: #f59e0b; }
    .status.error { color: #ef4444; }

    /* ── Timing Bar ── */
    .timing-bar {
      display: none;
      background: #0f1117;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 14px 20px;
      margin-top: 16px;
      display: none;
      gap: 0;
      flex-wrap: wrap;
    }

    .timing-bar.visible { display: flex; }

    .timing-item {
      flex: 1;
      min-width: 120px;
      text-align: center;
      padding: 8px 12px;
      border-right: 1px solid #2d3748;
    }

    .timing-item:last-child { border-right: none; }

    .timing-value {
      font-size: 1.4rem;
      font-weight: 700;
      color: #6ee7b7;
    }

    .timing-label {
      font-size: 0.7rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin-top: 2px;
    }

    .section-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #64748b;
      margin-bottom: 12px;
    }

    .transcript-box {
      background: #0f1117;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 20px;
      min-height: 80px;
      font-size: 1.05rem;
      line-height: 1.6;
      color: #e2e8f0;
      margin-bottom: 24px;
    }

    .transcript-box.empty { color: #4a5568; font-style: italic; }

    .json-box {
      background: #0f1117;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 20px;
      font-family: 'Courier New', monospace;
      font-size: 0.82rem;
      color: #6ee7b7;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 380px;
      overflow-y: auto;
      min-height: 80px;
    }

    .badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 600;
      background: #1a2740;
      color: #3b82f6;
      border: 1px solid #2d4a70;
      margin-left: 8px;
    }

    .history { width: 100%; max-width: 680px; }

    .history-item {
      background: #1e2330;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 10px;
      font-size: 0.9rem;
    }

    .history-time { font-size: 0.75rem; color: #64748b; margin-bottom: 4px; }
    .history-meta { font-size: 0.72rem; color: #3b82f6; margin-top: 6px; }

    .copy-btn {
      float: right;
      background: none;
      border: 1px solid #2d3748;
      color: #64748b;
      border-radius: 6px;
      padding: 2px 10px;
      font-size: 0.75rem;
      cursor: pointer;
    }

    .copy-btn:hover { color: #e2e8f0; border-color: #4a5568; }

    .waveform {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
      height: 40px;
      margin: 16px 0;
    }

    .bar {
      width: 4px;
      background: #10b981;
      border-radius: 2px;
      animation: wave 1s ease-in-out infinite;
      opacity: 0;
    }

    .bar:nth-child(1) { animation-delay: 0s; }
    .bar:nth-child(2) { animation-delay: 0.1s; }
    .bar:nth-child(3) { animation-delay: 0.2s; }
    .bar:nth-child(4) { animation-delay: 0.3s; }
    .bar:nth-child(5) { animation-delay: 0.4s; }
    .bar:nth-child(6) { animation-delay: 0.3s; }
    .bar:nth-child(7) { animation-delay: 0.2s; }
    .bar:nth-child(8) { animation-delay: 0.1s; }
    .bar:nth-child(9) { animation-delay: 0s; }

    @keyframes wave {
      0%, 100% { height: 6px; }
      50% { height: 32px; }
    }

    .waveform.active .bar { opacity: 1; }
  </style>
</head>
<body>

  <h1>🎙️ Ambient Notes</h1>
  <p class="subtitle">AssemblyAI · Real-time Speech to Text · JSON Output</p>

  <div class="card">
    <button class="record-btn" id="recordBtn" onclick="toggleRecording()">
      ● Start Recording
    </button>

    <div class="waveform" id="waveform">
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
    </div>

    <div class="status" id="status">Click to start recording</div>

    <!-- Timing Bar -->
    <div class="timing-bar" id="timingBar">
      <div class="timing-item">
        <div class="timing-value" id="tTotal">--</div>
        <div class="timing-label">Total Time</div>
      </div>
      <div class="timing-item">
        <div class="timing-value" id="tApi">--</div>
        <div class="timing-label">API Time</div>
      </div>
      <div class="timing-item">
        <div class="timing-value" id="tAudio">--</div>
        <div class="timing-label">Audio Duration</div>
      </div>
      <div class="timing-item">
        <div class="timing-value" id="tFactor">--</div>
        <div class="timing-label">Realtime Factor</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="section-label">Transcript <span class="badge">AssemblyAI</span></div>
    <div class="transcript-box empty" id="transcriptBox">
      Your transcribed text will appear here...
    </div>

    <div class="section-label">JSON Response</div>
    <div class="json-box" id="jsonBox">{
  "status": "waiting for audio..."
}</div>
  </div>

  <div class="history" id="history"></div>

<script>
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let recordingStartTime = null;

  async function toggleRecording() {
    if (!isRecording) {
      await startRecording();
    } else {
      stopRecording();
    }
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        await processAudio();
      };

      mediaRecorder.start();
      isRecording = true;
      recordingStartTime = Date.now();

      document.getElementById('recordBtn').textContent = '■ Stop Recording';
      document.getElementById('recordBtn').classList.add('recording');
      document.getElementById('waveform').classList.add('active');
      document.getElementById('timingBar').classList.remove('visible');
      setStatus('Recording... speak now', 'active');
    } catch (err) {
      setStatus('Microphone access denied. Please allow mic access.', 'error');
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
      isRecording = false;

      document.getElementById('recordBtn').textContent = '● Start Recording';
      document.getElementById('recordBtn').classList.remove('recording');
      document.getElementById('recordBtn').disabled = true;
      document.getElementById('waveform').classList.remove('active');
      setStatus('Processing audio...', 'processing');
    }
  }

  async function processAudio() {
    const clientStart = Date.now();
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');

    try {
      const res = await fetch('/transcribe', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      const totalClientMs = Date.now() - clientStart;

      // Show timing bar
      showTimingBar(data, totalClientMs);
      displayResult(data);
    } catch (err) {
      displayError(err.message);
    } finally {
      document.getElementById('recordBtn').disabled = false;
      setStatus('Done! Click to record again.', '');
    }
  }

  function showTimingBar(data, totalClientMs) {
    const bar = document.getElementById('timingBar');
    bar.classList.add('visible');

    const apiMs = data.timing ? data.timing.transcription_time_ms : null;
    const audioDur = data.audio_duration ? data.audio_duration.toFixed(1) + 's' : '--';
    const factor = data.timing && data.timing.realtime_factor ? data.timing.realtime_factor + 'x' : '--';

    document.getElementById('tTotal').textContent  = totalClientMs > 1000
      ? (totalClientMs / 1000).toFixed(1) + 's'
      : totalClientMs + 'ms';

    document.getElementById('tApi').textContent = apiMs
      ? (apiMs > 1000 ? (apiMs / 1000).toFixed(1) + 's' : apiMs + 'ms')
      : '--';

    document.getElementById('tAudio').textContent = audioDur;
    document.getElementById('tFactor').textContent = factor;
  }

  function displayResult(data) {
    const transcriptBox = document.getElementById('transcriptBox');
    const jsonBox = document.getElementById('jsonBox');

    if (data.text) {
      transcriptBox.textContent = data.text;
      transcriptBox.classList.remove('empty');
    } else {
      transcriptBox.textContent = data.error || 'No speech detected.';
      transcriptBox.classList.add('empty');
    }

    jsonBox.textContent = JSON.stringify(data, null, 2);

    if (data.text) {
      addToHistory(data);
    }
  }

  function displayError(msg) {
    document.getElementById('transcriptBox').textContent = 'Error: ' + msg;
    document.getElementById('transcriptBox').classList.add('empty');
    document.getElementById('jsonBox').textContent = JSON.stringify({ error: msg }, null, 2);
    setStatus('Error occurred. Try again.', 'error');
  }

  function addToHistory(data) {
    const history = document.getElementById('history');
    const item = document.createElement('div');
    item.className = 'history-item';
    const apiTime = data.timing ? data.timing.transcription_time_ms + 'ms' : '--';
    const safeText = data.text.replace(/'/g, "\\'");
    item.innerHTML = `
      <button class="copy-btn" onclick="copyText(this, '${safeText}')">Copy</button>
      <div class="history-time">${data.timestamp}</div>
      <div>${data.text}</div>
      <div class="history-meta">⏱ API: ${apiTime} · 🌐 Lang: ${data.language || '--'} · 🎯 Confidence: ${data.confidence ? Math.round(data.confidence * 100) + '%' : '--'}</div>
    `;
    history.insertBefore(item, history.firstChild);
  }

  function copyText(btn, text) {
    navigator.clipboard.writeText(text);
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1500);
  }

  function setStatus(msg, type) {
    const el = document.getElementById('status');
    el.textContent = msg;
    el.className = 'status ' + type;
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
        import datetime

        # ── Start timing ──────────────────────────────────
        transcription_start = time.time()

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path)

        transcription_end = time.time()
        transcription_time_ms = round((transcription_end - transcription_start) * 1000)
        # ──────────────────────────────────────────────────

        if transcript.status == aai.TranscriptStatus.error:
            return jsonify({
                "status": "error",
                "error": transcript.error,
                "text": None
            })

        result = {
            "status": "success",
            "text": transcript.text,
            "confidence": transcript.confidence,
            "audio_duration": transcript.audio_duration,
            "language": transcript.language_code,
            "timing": {
                "transcription_time_ms": transcription_time_ms,
                "transcription_time_sec": round(transcription_time_ms / 1000, 2),
                "realtime_factor": round(
                    transcript.audio_duration / (transcription_time_ms / 1000), 2
                ) if transcript.audio_duration and transcription_time_ms > 0 else None
            },
            "words": [
                {
                    "text": w.text,
                    "start_ms": w.start,
                    "end_ms": w.end,
                    "confidence": w.confidence
                }
                for w in (transcript.words or [])
            ],
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "AssemblyAI Universal"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "text": None})

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 55)
    print("  AMBIENT NOTES WEB APP  |  AssemblyAI")
    print("=" * 55)
    print("  Open browser at: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 55)
    app.run(debug=True, port=5000)