import { buildWav, formatMs, friendlyError, micErrorMessage } from "./lib.js";

const $ = (id) => document.getElementById(id);
const el = {
  mic: $("mic"), cancel: $("cancel"), stateText: $("state-text"), stateDetail: $("state-detail"),
  meter: $("meter"), meterBar: $("meter-bar"), messages: $("messages"), live: $("live-transcript"),
  banner: $("banner"), bannerText: $("banner-text"), connPill: $("conn-pill"), connText: $("conn-text"),
  modelsPill: $("models-pill"), modelsText: $("models-text"), diagBody: $("diag-body"),
  clientLatency: $("client-latency"), turnCounts: $("turn-counts"), transport: $("transport"),
  transportHelp: $("transport-help"), reset: $("reset"), announcer: $("announcer"),
};
const announce = (text) => { el.announcer.textContent = text; };

// Rows of the diagnostics table: [label, key in the server's metrics, emphasise]
const DIAG_ROWS = [
  ["Upload (HTTP only)", "upload_ms"], ["Decode (HTTP only)", "decode_ms"],
  ["STT queue", "stt_queue_ms"], ["STT", "stt_ms"],
  ["LLM queue", "llm_queue_ms"], ["LLM first token", "llm_ttft_ms"], ["LLM total", "llm_ms"],
  ["TTS queue", "tts_queue_ms"], ["TTS first audio", "tts_first_chunk_ms"], ["TTS total", "tts_ms"],
  ["First audio ready", "first_audio_ms", true], ["Turn total", "total_ms", true],
];
const PHASE_TEXT = {
  idle: "Ready", "requesting-mic": "Waiting for microphone permission…", recording: "Recording",
  transcribing: "Transcribing…", thinking: "Thinking…", speaking: "Speaking…",
};
const TRANSPORT_HELP = {
  stream: "Audio is sent while you speak, and the reply is spoken sentence by sentence as it is generated.",
  upload: "Audio is recorded, uploaded in one request, and the whole reply is returned at once. Slower to first sound.",
};

const state = {
  transport: "stream", phase: "idle", ws: null, wsOpen: false, reconnectAttempts: 0, reconnectTimer: null,
  maxSeconds: 30, modelsReady: false, serverReachable: true, turnActive: false, turn: null,
  pendingAudio: null, capture: null, timer: null, history: [], abort: null, replayAudio: null,
  lastLlmWarning: "", summary: null, lastMetrics: null,
};

try {
  const saved = localStorage.getItem("transport");
  if (saved === "stream" || saved === "upload") state.transport = saved;
} catch { /* storage can be blocked; the default is fine */ }

// ---------------------------------------------------------------- rendering

function setPhase(phase, detail = "") {
  state.phase = phase;
  el.stateDetail.textContent = detail;
  render();
}

function render() {
  const { phase } = state;
  const idleReady = phase === "idle" && state.modelsReady && (state.transport === "upload" ? state.serverReachable : state.wsOpen);
  el.mic.disabled = !(idleReady || phase === "recording");
  el.mic.dataset.phase = phase;
  el.mic.setAttribute("aria-pressed", String(phase === "recording"));
  el.mic.setAttribute("aria-label", phase === "recording" ? "Stop recording and send" : "Start recording");
  el.cancel.hidden = phase === "idle";
  el.cancel.textContent = phase === "recording" ? "Discard" : phase === "speaking" ? "Stop" : "Cancel";
  el.meter.hidden = phase !== "recording";
  if (phase === "idle" && !el.mic.disabled) el.stateText.textContent = "Ready";
  else if (phase === "idle") el.stateText.textContent = state.modelsReady ? "Not connected" : "Waiting for the server";
  else el.stateText.textContent = PHASE_TEXT[phase];
  el.transportHelp.textContent = TRANSPORT_HELP[state.transport];
}

function showBanner(text, kind = "error") {
  el.bannerText.textContent = text;
  el.banner.dataset.kind = kind;
  el.banner.hidden = false;
}
const clearBanner = () => { el.banner.hidden = true; };
$("banner-dismiss").addEventListener("click", clearBanner);

function addMessage(role, text, pending = false) {
  el.messages.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  li.className = `msg ${role}${pending ? " pending" : ""}`;
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = role === "user" ? "You" : "Assistant";
  const p = document.createElement("p");
  p.textContent = text; // textContent, never innerHTML: model output is untrusted
  li.append(label, p);
  el.messages.append(li);
  li.scrollIntoView({ block: "end" });
  return { li, p };
}

function renderDiagnostics() {
  const last = state.lastMetrics;
  const stages = state.summary?.stages_ms ?? {};
  el.diagBody.replaceChildren(...DIAG_ROWS.map(([label, key, emphasise]) => {
    const row = document.createElement("tr");
    if (emphasise) row.className = "key";
    const median = stages[key] ? `${formatMs(stages[key].median)} (n=${stages[key].n})` : "—";
    for (const text of [label, formatMs(last?.[key]), median]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    return row;
  }));
  const counts = state.summary?.turns_since_start;
  el.turnCounts.textContent = counts && Object.keys(counts).length
    ? `Turns handled by this server: ${Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(", ")}`
    : "";
}

async function refreshSummary() {
  try {
    state.summary = (await (await fetch("/api/metrics", { cache: "no-store" })).json()).summary;
  } catch { /* diagnostics are optional */ }
  renderDiagnostics();
}

// ---------------------------------------------------------------- health

async function pollHealth() {
  try {
    const health = await (await fetch("/health", { cache: "no-store" })).json();
    const c = health.components;
    state.serverReachable = true;
    state.modelsReady = c.stt?.state === "ready" && c.tts?.state === "ready";
    const failed = Object.entries(c).find(([, v]) => v.state === "error");
    if (health.status === "ok") setPill(el.modelsPill, el.modelsText, "ok", "Models ready", "");
    else if (failed) setPill(el.modelsPill, el.modelsText, "error", `${failed[0].toUpperCase()} unavailable`, failed[1].detail);
    else setPill(el.modelsPill, el.modelsText, "warn", "Loading models…", "The first start downloads and loads the models.");
    const llmDetail = c.llm?.state === "error" ? c.llm.detail : "";
    if (llmDetail && llmDetail !== state.lastLlmWarning) showBanner(llmDetail, "warn");
    state.lastLlmWarning = llmDetail;
  } catch {
    state.serverReachable = false;
    state.modelsReady = false;
    setPill(el.modelsPill, el.modelsText, "error", "Server unreachable", "Could not reach /health");
  }
  if (state.transport === "upload") {
    setPill(el.connPill, el.connText, state.serverReachable ? "ok" : "error", state.serverReachable ? "HTTP upload mode" : "Server unreachable", "");
  }
  render();
}

function setPill(pill, textEl, kind, text, title) {
  pill.dataset.state = kind;
  textEl.textContent = text;
  pill.title = title;
}

// ---------------------------------------------------------------- audio playback (streaming mode)

class Player {
  constructor(onIdle) { this.ctx = null; this.nextTime = 0; this.active = new Set(); this.onIdle = onIdle; }
  get busy() { return this.active.size > 0; }
  unlock() { // must run inside a user gesture, or the browser keeps the AudioContext suspended
    this.ctx ??= new AudioContext();
    if (this.ctx.state === "suspended") this.ctx.resume();
  }
  enqueue(int16, sampleRate) {
    this.unlock();
    const buffer = this.ctx.createBuffer(1, int16.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < int16.length; i++) channel[i] = int16[i] / 32768;
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    // Chain each sentence directly after the previous one so playback has no gaps.
    const startAt = Math.max(this.ctx.currentTime + 0.05, this.nextTime);
    source.start(startAt);
    this.nextTime = startAt + buffer.duration;
    this.active.add(source);
    source.onended = () => { this.active.delete(source); if (!this.busy) this.onIdle(); };
  }
  stop() {
    for (const source of this.active) { source.onended = null; try { source.stop(); } catch { /* already ended */ } }
    this.active.clear();
    this.nextTime = 0;
  }
}
const player = new Player(() => { if (!state.turnActive) setPhase("idle"); });

// ---------------------------------------------------------------- WebSocket

function connectSocket() {
  if (state.transport !== "stream" || state.ws) return;
  setPill(el.connPill, el.connText, "warn", state.reconnectAttempts ? "Reconnecting…" : "Connecting…", "");
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/voice`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onmessage = onSocketMessage;
  ws.onclose = () => {
    if (state.ws !== ws) return; // an old socket we replaced on purpose
    state.ws = null;
    state.wsOpen = false;
    onConnectionLost();
    if (state.transport === "stream") {
      setPill(el.connPill, el.connText, "error", "Disconnected", "");
      const delay = Math.min(10000, 1000 * 2 ** state.reconnectAttempts++);
      state.reconnectTimer = setTimeout(connectSocket, delay);
    }
    render();
  };
}

function closeSocket() {
  clearTimeout(state.reconnectTimer);
  const ws = state.ws;
  state.ws = null;
  state.wsOpen = false;
  ws?.close();
}

function onConnectionLost() {
  if (state.phase === "recording") {
    discardCapture();
    showBanner("The connection to the server was lost while recording. Your audio was discarded.");
  } else if (state.turnActive) {
    finishTurn();
    showBanner("The connection to the server was lost during the request.");
  }
  player.stop();
  setPhase("idle");
}

function sendControl(type) { state.ws?.send(JSON.stringify({ type })); }

function onSocketMessage(event) {
  if (typeof event.data !== "string") return onAudioFrame(event.data);
  const m = JSON.parse(event.data);
  switch (m.type) {
    case "ready":
      state.wsOpen = true;
      state.reconnectAttempts = 0;
      state.maxSeconds = m.max_audio_seconds;
      setPill(el.connPill, el.connText, "ok", "Connected", "WebSocket /ws/voice");
      render();
      break;
    case "status": onServerStatus(m.state); break;
    case "transcript": onTranscript(m); break;
    case "reply_delta": if (state.turn?.assistant) state.turn.assistant.p.textContent += m.text; break;
    case "audio": state.pendingAudio = m; break;
    case "done": onDone(m); break;
    case "cancelled": finishTurn(); player.stop(); setPhase("idle"); break;
    case "error": onServerError(m); break;
  }
}

function onServerStatus(status) {
  if (!state.turnActive) return;
  if (status === "transcribing" || status === "thinking" || status === "speaking") setPhase(status);
}

function onTranscript(m) {
  if (!m.final) { el.live.textContent = m.text; return; }
  el.live.textContent = "";
  if (!m.text) {
    showBanner("No speech was detected. Try again, a little closer to the microphone.", "warn");
    return;
  }
  addMessage("user", m.text);
  announce(`You said: ${m.text}`);
  state.turn.assistant = addMessage("assistant", "", true);
}

function onAudioFrame(buffer) {
  const meta = state.pendingAudio;
  state.pendingAudio = null;
  if (!meta || !state.turn) return;
  const samples = new Int16Array(buffer);
  state.turn.chunks.push(samples);
  state.turn.sampleRate = meta.sample_rate;
  if (state.turn.firstAudioAt === null) {
    state.turn.firstAudioAt = performance.now();
    el.clientLatency.textContent = `Stop → first audio (measured in this browser): ${formatMs(state.turn.firstAudioAt - state.turn.stopAt)}`;
  }
  player.enqueue(samples, meta.sample_rate);
}

function onDone(m) {
  const turn = state.turn;
  state.lastMetrics = m.metrics;
  if (turn?.assistant) {
    turn.assistant.li.classList.remove("pending");
    if (turn.chunks.length) attachReplay(turn.assistant.li, buildWav(turn.chunks, turn.sampleRate));
    announce(`Assistant: ${turn.assistant.p.textContent}`);
  }
  finishTurn();
  renderDiagnostics();
  refreshSummary();
  if (!player.busy) setPhase("idle");
}

function onServerError(m) {
  showBanner(friendlyError(m.code, m.message));
  if (state.phase === "recording") discardCapture();
  const hadTurn = state.turnActive;
  finishTurn();
  if (hadTurn) refreshSummary();
  setPhase(player.busy ? "speaking" : "idle");
}

// ---------------------------------------------------------------- one turn

function startTurn() {
  state.turn = { stopAt: performance.now(), firstAudioAt: null, assistant: null, chunks: [], sampleRate: 22050 };
  state.turnActive = true;
  el.clientLatency.textContent = "";
}

function finishTurn() {
  const assistant = state.turn?.assistant;
  if (assistant) {
    assistant.li.classList.remove("pending");
    if (!assistant.p.textContent) assistant.li.remove();
  }
  state.turn = null;
  state.turnActive = false;
}

function attachReplay(li, wavBytes) {
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "none";
  audio.setAttribute("aria-label", "Replay this response");
  audio.src = URL.createObjectURL(new Blob([wavBytes], { type: "audio/wav" }));
  li.append(audio);
}

// ---------------------------------------------------------------- microphone capture

async function openStreamCapture() {
  if (!window.AudioWorkletNode) {
    throw Object.assign(new Error("this browser has no AudioWorklet; use Upload mode in Settings"), { name: "Unsupported" });
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext(); // device rate; the worklet downsamples to 16 kHz
  await ctx.audioWorklet.addModule("pcm-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  const silent = ctx.createGain(); // keeps the node in the graph without playing the mic back
  silent.gain.value = 0;
  source.connect(node).connect(silent).connect(ctx.destination);

  let sending = false;
  let onFlushed = () => {};
  node.port.onmessage = (e) => {
    if (e.data.flushed) return onFlushed();
    if (e.data.pcm && sending) state.ws?.send(e.data.pcm);
    if (e.data.level !== undefined) el.meterBar.style.width = `${Math.min(100, Math.sqrt(e.data.level) * 140)}%`;
  };
  const release = () => { source.disconnect(); node.disconnect(); stream.getTracks().forEach((t) => t.stop()); ctx.close(); };
  return {
    begin() { sending = true; },
    async stop() { // resolves once the last audio chunk has been handed to the socket
      const flushed = new Promise((resolve) => { onFlushed = resolve; setTimeout(resolve, 500); });
      node.port.postMessage("flush");
      await flushed;
      sending = false;
      release();
    },
    abort() { sending = false; release(); },
  };
}

async function openRecorderCapture() {
  if (!window.MediaRecorder) throw Object.assign(new Error("MediaRecorder is not supported"), { name: "Unsupported" });
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const recorder = new MediaRecorder(stream); // the browser picks WebM/Opus or MP4; the server decodes either
  const parts = [];
  recorder.ondataavailable = (e) => { if (e.data.size) parts.push(e.data); };
  const ctx = new AudioContext();
  const analyser = ctx.createAnalyser();
  ctx.createMediaStreamSource(stream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  let frame = 0;
  const drawMeter = () => {
    analyser.getFloatTimeDomainData(samples);
    el.meterBar.style.width = `${Math.min(100, Math.sqrt(Math.max(...samples.map(Math.abs))) * 140)}%`;
    frame = requestAnimationFrame(drawMeter);
  };
  const release = () => { cancelAnimationFrame(frame); stream.getTracks().forEach((t) => t.stop()); ctx.close(); };
  return {
    begin() { recorder.start(); drawMeter(); },
    async stop() {
      const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
      recorder.stop();
      await stopped;
      release();
      return new Blob(parts, { type: recorder.mimeType });
    },
    abort() { recorder.onstop = null; if (recorder.state !== "inactive") recorder.stop(); release(); },
  };
}

function discardCapture() {
  clearInterval(state.timer);
  state.capture?.abort();
  state.capture = null;
  el.live.textContent = "";
}

function startTimer() {
  const started = performance.now();
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  state.timer = setInterval(() => {
    const elapsed = (performance.now() - started) / 1000;
    el.stateDetail.textContent = `${fmt(elapsed)} of ${fmt(state.maxSeconds)} maximum`;
    if (elapsed >= state.maxSeconds - 0.5) stopRecording(); // stop cleanly before the server's limit
  }, 200);
}

// ---------------------------------------------------------------- user actions

async function startRecording() {
  clearBanner();
  player.stop();
  setPhase("requesting-mic");
  state.micRequestCancelled = false;
  let capture;
  try {
    capture = state.transport === "stream" ? await openStreamCapture() : await openRecorderCapture();
  } catch (error) {
    showBanner(error.name === "Unsupported" ? `Cannot record here: ${error.message}.` : micErrorMessage(error));
    setPhase("idle");
    return;
  }
  if (state.micRequestCancelled) { // cancelled while the permission prompt was open
    capture.abort();
    setPhase("idle");
    return;
  }
  if (state.transport === "stream" && !state.wsOpen) { // the socket dropped while the permission prompt was open
    capture.abort();
    showBanner("The connection to the server was lost. Try again once it reconnects.");
    setPhase("idle");
    return;
  }
  state.capture = capture;
  if (state.transport === "stream") sendControl("start");
  capture.begin();
  startTimer();
  setPhase("recording");
}

async function stopRecording() {
  if (state.phase !== "recording" || !state.capture) return;
  clearInterval(state.timer);
  const capture = state.capture;
  state.capture = null;
  startTurn();
  if (state.transport === "stream") {
    setPhase("transcribing");
    await capture.stop();
    sendControl("stop");
  } else {
    setPhase("thinking", "Uploading and processing…");
    await sendUpload(await capture.stop());
  }
}

async function sendUpload(blob) {
  const form = new FormData();
  form.append("audio", blob, "speech");
  form.append("history", JSON.stringify(state.history.slice(-12)));
  state.abort = new AbortController();
  try {
    const response = await fetch("/api/turn", { method: "POST", body: form, signal: state.abort.signal });
    const body = await response.json();
    if (!response.ok) {
      showBanner(friendlyError(body.error.code, body.error.message));
      return finishUpload();
    }
    state.lastMetrics = body.metrics;
    el.clientLatency.textContent = `Stop → response received (measured in this browser): ${formatMs(performance.now() - state.turn.stopAt)}`;
    if (body.status === "no_speech") {
      showBanner("No speech was detected. Try again, a little closer to the microphone.", "warn");
      return finishUpload();
    }
    addMessage("user", body.transcript);
    const reply = addMessage("assistant", body.reply);
    announce(`You said: ${body.transcript}. Assistant: ${body.reply}`);
    state.history.push({ role: "user", content: body.transcript }, { role: "assistant", content: body.reply });
    finishUpload();
    if (body.audio) playUploaded(reply.li, body.audio.data_base64);
  } catch (error) {
    if (error.name !== "AbortError") showBanner(`Network error: ${error.message}. Is the server running?`);
    finishUpload();
  }
}

function finishUpload() {
  state.turn = null;
  state.turnActive = false;
  state.abort = null;
  renderDiagnostics();
  refreshSummary();
  if (state.phase !== "speaking") setPhase("idle");
}

function playUploaded(li, base64) {
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.setAttribute("aria-label", "Replay this response");
  audio.src = `data:audio/wav;base64,${base64}`;
  li.append(audio);
  state.replayAudio = audio;
  audio.onended = audio.onpause = () => { if (state.phase === "speaking") setPhase("idle"); };
  audio.play().then(() => setPhase("speaking")).catch(() => { /* autoplay blocked: the controls are there */ });
}

function cancelCurrent() {
  switch (state.phase) {
    case "requesting-mic":
      state.micRequestCancelled = true; // startRecording() notices when the permission prompt resolves
      setPhase("idle");
      break;
    case "recording":
      discardCapture();
      if (state.transport === "stream") sendControl("cancel");
      setPhase("idle");
      break;
    case "transcribing":
    case "thinking":
    case "speaking":
      player.stop();
      state.replayAudio?.pause();
      if (state.turnActive && state.transport === "stream") sendControl("cancel"); // server replies "cancelled"
      else if (state.abort) state.abort.abort();
      else setPhase("idle");
      break;
  }
}

el.mic.addEventListener("click", () => {
  player.unlock(); // inside the click handler, so playback is allowed later
  if (state.phase === "recording") stopRecording();
  else if (state.phase === "idle") startRecording();
});
el.cancel.addEventListener("click", cancelCurrent);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.phase !== "idle") cancelCurrent(); });

el.reset.addEventListener("click", () => {
  if (state.phase !== "idle") cancelCurrent();
  sendControl("reset");
  state.history = [];
  el.messages.replaceChildren();
  const empty = document.createElement("li");
  empty.className = "empty";
  empty.textContent = "New conversation. Press the microphone to start.";
  el.messages.append(empty);
});

el.transport.value = state.transport;
el.transport.addEventListener("change", () => {
  if (state.phase !== "idle") cancelCurrent();
  state.transport = el.transport.value;
  try { localStorage.setItem("transport", state.transport); } catch { /* optional */ }
  if (state.transport === "stream") connectSocket();
  else closeSocket();
  pollHealth();
  render();
});

// ---------------------------------------------------------------- start

renderDiagnostics();
render();
connectSocket();
pollHealth();
setInterval(pollHealth, 3000);
refreshSummary();
