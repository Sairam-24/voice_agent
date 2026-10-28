// Everything with side effects: the WebSocket, health polling, microphone capture, audio playback,
// cancellation. It reads and writes the pure store in lib/state.ts. This is a port of the logic in
// web/app.js (the classic UI), so both clients speak the same protocol the same way.
import { type RecorderCapture, type StreamCapture, openRecorderCapture, openStreamCapture } from "./audio/capture.ts";
import { Player, SPECTRUM_BINS } from "./audio/player.ts";
import { UnsupportedError, friendlyError, micErrorMessage } from "./lib/errors.ts";
import { formatClock } from "./lib/format.ts";
import {
  parseServerMessage,
  type DocumentSummary,
  type Health,
  type MetricsSummary,
  type ServerInfo,
  type ServerMessage,
  type TurnMetrics,
} from "./lib/protocol.ts";
import { SpeechTimeline } from "./lib/speech.ts";
import { type Action, type Phase, type Transport, attachedIds, createStore, initialState, reduce, turnLanguage } from "./lib/state.ts";
import { buildWav } from "./lib/wav.ts";

/** What the visualiser reads each frame: the real microphone while recording, the real reply audio while speaking. */
export interface AudioTap {
  /** Fills `bins` (length SPECTRUM_BINS) with 0-255 magnitudes and returns an overall level from 0 to 1. */
  read(bins: Uint8Array<ArrayBuffer>): number;
}

interface TurnTrace {
  stopAt: number;
  firstAudioAt: number | null;
  chunks: Int16Array[];
  sampleRate: number;
}

interface UploadResponse {
  status: string;
  transcript: string;
  reply: string;
  metrics: TurnMetrics;
  language?: string | null;
  audio?: { data_base64: string; duration_s: number };
}

const TRANSPORT_KEY = "transport";
const LANGUAGE_KEY = "language";
const GEMINI_KEY = "gemini_key";
const DEFAULT_MAX_FILE_BYTES = 10_000_000;
const DEFAULT_MAX_FILES = 8;

function loadSetting(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null; // storage can be blocked; the defaults are fine
  }
}

function saveSetting(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* optional */
  }
}

function loadTransport(): Transport {
  const saved = loadSetting(TRANSPORT_KEY);
  return saved === "upload" ? "upload" : "stream";
}

/** The reply being spoken, and the clock its text follows (the player's, or an <audio> element's). */
interface SpeechFollow {
  timeline: SpeechTimeline;
  clock: () => number;
  shown: number;
  frame: number;
}

export class VoiceController {
  /** The user's own Gemini key, if they saved one. Kept here and in localStorage, never in the store. */
  private geminiKey = loadSetting(GEMINI_KEY);
  readonly store = createStore(reduce, initialState(loadTransport(), loadSetting(LANGUAGE_KEY) ?? "auto", this.geminiKey !== null));
  readonly player = new Player(() => {
    if (this.store.getState().turnActive) return; // more audio may still come
    this.endSpeech();
    this.setPhase("idle");
  });

  readonly tap: AudioTap = {
    read: (bins) => {
      const node = this.capture?.analyser ?? this.player.analyser;
      if (!node || node.frequencyBinCount !== SPECTRUM_BINS) {
        bins.fill(0);
        return 0;
      }
      node.getByteFrequencyData(bins);
      let sum = 0;
      for (let i = 2; i < 50; i++) sum += bins[i] ?? 0;
      return Math.min(1, Math.sqrt(sum / 48 / 255) * 1.35);
    },
  };

  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | undefined;
  private healthTimer: ReturnType<typeof setInterval> | undefined;
  private recordTimer: ReturnType<typeof setInterval> | undefined;
  private capture: StreamCapture | RecorderCapture | null = null;
  private turn: TurnTrace | null = null;
  private pendingAudio: { sample_rate: number; index: number; sentence?: string; segment?: number } | null = null;
  private abort: AbortController | null = null;
  private replayAudio: HTMLAudioElement | null = null;
  private history: { role: string; content: string }[] = [];
  private micRequestCancelled = false;
  private lastLlmWarning = "";
  private blobUrls: string[] = [];
  private running = false;
  private speech: SpeechFollow | null = null;
  private fileCounter = 0;

  // ------------------------------------------------------------------------ lifecycle

  start(): void {
    if (this.running) return;
    this.running = true;
    this.connectSocket();
    void this.pollHealth();
    this.healthTimer = setInterval(() => void this.pollHealth(), 3000);
    void this.refreshSummary();
  }

  dispose(): void {
    this.running = false;
    clearInterval(this.healthTimer);
    this.closeSocket();
    this.discardCapture();
    this.player.close();
    this.replayAudio?.pause();
    this.abort?.abort();
    this.blobUrls.forEach((url) => URL.revokeObjectURL(url));
    this.blobUrls = [];
  }

  // ------------------------------------------------------------------------ helpers

  private dispatch(action: Action): void {
    this.store.dispatch(action);
  }

  private get state() {
    return this.store.getState();
  }

  private setPhase(phase: Phase, detail = ""): void {
    this.dispatch({ type: "phase", phase, detail });
  }

  private showBanner(text: string, kind: "error" | "warn" = "error"): void {
    this.dispatch({ type: "banner", banner: { text, kind } });
  }

  private sendControl(type: string, fields: Record<string, unknown> = {}): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type, ...fields }));
  }

  // ------------------------------------------------------------------------ text that follows the voice

  /** From now on the reply's text is revealed as its audio plays (see lib/speech.ts). */
  private followSpeech(clock: () => number): SpeechTimeline {
    // Not endSpeech(): that would reveal the new reply, which the reducer has just hidden.
    if (this.speech) cancelAnimationFrame(this.speech.frame);
    const timeline = new SpeechTimeline();
    this.speech = { timeline, clock, shown: 0, frame: requestAnimationFrame(this.tickSpeech) };
    return timeline;
  }

  private readonly tickSpeech = (): void => {
    const speech = this.speech;
    if (!speech) return;
    if (this.player.suspended && speech.clock === this.playerClock) return this.endSpeech(); // nothing will be heard
    const heard = speech.timeline.heard(speech.clock());
    if (heard.length > speech.shown) {
      // Only ever grows: a piece's estimated length can be corrected upwards once all its audio is in.
      speech.shown = heard.length;
      this.dispatch({ type: "spoken", text: heard });
    }
    speech.frame = requestAnimationFrame(this.tickSpeech);
  };

  /** Shows the whole reply: it has been heard, or there is nothing left to follow (stopped, failed, text only). */
  private endSpeech(): void {
    if (this.speech) cancelAnimationFrame(this.speech.frame);
    this.speech = null;
    this.dispatch({ type: "spoken", text: null });
  }

  private readonly playerClock = (): number => this.player.audibleTime;

  // ------------------------------------------------------------------------ server state

  private async pollHealth(): Promise<void> {
    try {
      const health = (await (await fetch("/health", { cache: "no-store" })).json()) as Health;
      this.dispatch({ type: "health", health });
      const llm = health.components.llm;
      // The server's default model; questions asked with the user's own key do not use it.
      const warning = llm?.state === "error" && !this.geminiKey ? llm.detail : "";
      if (warning && warning !== this.lastLlmWarning) this.showBanner(warning, "warn");
      this.lastLlmWarning = warning;
      if (!this.state.info) void this.loadInfo();
    } catch {
      this.dispatch({ type: "unreachable" });
    }
    if (this.state.transport === "upload") {
      const reachable = this.state.serverReachable;
      this.dispatch({
        type: "connection",
        connection: { kind: reachable ? "ok" : "error", text: reachable ? "HTTP upload mode" : "Server unreachable", title: "" },
      });
    }
  }

  private async loadInfo(): Promise<void> {
    try {
      this.dispatch({ type: "info", info: (await (await fetch("/api/info", { cache: "no-store" })).json()) as ServerInfo });
    } catch {
      /* the HUD works without it */
    }
  }

  private async refreshSummary(): Promise<void> {
    try {
      const body = (await (await fetch("/api/metrics", { cache: "no-store" })).json()) as { summary: MetricsSummary };
      this.dispatch({ type: "summary", summary: body.summary });
    } catch {
      /* diagnostics are optional */
    }
  }

  // ------------------------------------------------------------------------ WebSocket

  private connectSocket(): void {
    if (this.state.transport !== "stream" || this.ws || !this.running) return;
    this.dispatch({
      type: "connection",
      connection: { kind: "warn", text: this.reconnectAttempts ? "Reconnecting…" : "Connecting…", title: "" },
    });
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/voice`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
      if (typeof event.data === "string") this.onSocketMessage(parseServerMessage(event.data));
      else this.onAudioFrame(event.data);
    };
    ws.onclose = () => {
      if (this.ws !== ws) return; // an old socket that was replaced on purpose
      this.ws = null;
      this.dispatch({ type: "connection", connection: { kind: "error", text: "Disconnected", title: "" }, wsOpen: false });
      this.onConnectionLost();
      if (this.state.transport === "stream" && this.running) {
        const delay = Math.min(10_000, 1000 * 2 ** this.reconnectAttempts++);
        this.reconnectTimer = setTimeout(() => this.connectSocket(), delay);
      }
    };
  }

  private closeSocket(): void {
    clearTimeout(this.reconnectTimer);
    const ws = this.ws;
    this.ws = null;
    this.dispatch({ type: "connection", connection: this.state.connection, wsOpen: false });
    ws?.close();
  }

  private onConnectionLost(): void {
    if (this.state.phase === "recording") {
      this.discardCapture();
      this.showBanner("The connection to the server was lost while recording. Your audio was discarded.");
    } else if (this.state.turnActive) {
      this.finishTurn();
      this.showBanner("The connection to the server was lost during the request.");
    }
    this.player.stop();
    this.endSpeech();
    this.setPhase("idle");
  }

  private onSocketMessage(message: ServerMessage | null): void {
    if (!message) return;
    switch (message.type) {
      case "ready":
        this.reconnectAttempts = 0;
        this.dispatch({
          type: "connection",
          connection: { kind: "ok", text: "Connected", title: "WebSocket /ws/voice" },
          wsOpen: true,
          maxSeconds: message.max_audio_seconds,
        });
        if (this.geminiKey) this.sendControl("configure", { gemini_key: this.geminiKey }); // a new connection starts on the default
        break;
      case "configured":
        this.dispatch({ type: "configured", cloud: message.cloud });
        break;
      case "status":
        if (this.state.turnActive && (message.state === "transcribing" || message.state === "thinking" || message.state === "speaking")) {
          this.setPhase(message.state);
        }
        break;
      case "transcript":
        if (!message.final) {
          this.dispatch({ type: "live-transcript", text: message.text });
        } else {
          this.dispatch({ type: "final-transcript", text: message.text, ...(message.language ? { language: message.language } : {}) });
          if (message.text) this.followSpeech(this.playerClock);
          else this.showBanner("No speech was detected. Try again, a little closer to the microphone.", "warn");
        }
        break;
      case "reply_delta":
        this.dispatch({ type: "reply-delta", text: message.text });
        break;
      case "audio":
        this.pendingAudio = message;
        break;
      case "done":
        this.onDone(message.metrics);
        break;
      case "cancelled":
        this.finishTurn();
        this.player.stop();
        this.endSpeech();
        this.setPhase("idle");
        break;
      case "error":
        this.onServerError(message.code, message.message, message.ids);
        break;
    }
  }

  private onAudioFrame(buffer: ArrayBuffer): void {
    const meta = this.pendingAudio;
    this.pendingAudio = null;
    if (!meta || !this.turn) return;
    const samples = new Int16Array(buffer);
    this.turn.chunks.push(samples);
    this.turn.sampleRate = meta.sample_rate;
    if (this.turn.firstAudioAt === null) {
      this.turn.firstAudioAt = performance.now();
      this.dispatch({ type: "client-latency", latency: { kind: "first-audio", ms: this.turn.firstAudioAt - this.turn.stopAt } });
    }
    const { start, end } = this.player.enqueue(samples, meta.sample_rate);
    this.speech?.timeline.add(meta.segment ?? meta.index, meta.sentence ?? "", start, end);
  }

  /** Cloud models are rate-limited on the free tier; say so when a local model stood in, or there was no voice. */
  private noteVoiceFallback(metrics: TurnMetrics): void {
    const notes: string[] = [];
    if (metrics.stt_fallback) notes.push("Cloud transcription was unavailable, so the local recogniser transcribed this turn.");
    if (metrics.llm_fallback) {
      notes.push(`The cloud model was unavailable, so the local model answered${this.geminiKey ? " (check your key on the System page)" : ""}.`);
    }
    const voice = metrics.tts_fallback;
    if (voice?.startsWith("Text only: ")) notes.push(`No voice for this reply. ${voice.slice("Text only: ".length)}`);
    else if (voice) notes.push(`The natural voice was unavailable, so a local voice spoke this reply. ${voice}`);
    if (notes.length) this.showBanner(notes.join(" "), "warn");
  }

  private onDone(metrics: TurnMetrics): void {
    const turn = this.turn;
    this.dispatch({ type: "done", metrics });
    this.noteVoiceFallback(metrics);
    if (turn?.chunks.length) {
      const blob = new Blob([buildWav(turn.chunks, turn.sampleRate)], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      this.blobUrls.push(url);
      this.dispatch({ type: "replay", src: url });
    }
    this.turn = null;
    void this.refreshSummary();
    this.speech?.timeline.complete();
    if (!this.player.busy) {
      this.endSpeech(); // text only, or the audio has already finished
      this.setPhase("idle");
    }
  }

  private onServerError(code: string, serverMessage: string, ids?: string[]): void {
    if (code === "document_missing" && ids?.length) {
      const names = this.state.files.filter((f) => f.id && ids.includes(f.id)).map((f) => `“${f.name}”`);
      this.dispatch({ type: "file-remove", ids });
      this.showBanner(`${names.join(", ") || "A file"} is no longer on the server (it was restarted). Attach it again.`);
    } else {
      this.showBanner(friendlyError(code, serverMessage));
    }
    this.endSpeech();
    if (this.state.phase === "recording") this.discardCapture();
    const hadTurn = this.state.turnActive;
    this.finishTurn();
    if (hadTurn) void this.refreshSummary();
    this.setPhase(this.player.busy ? "speaking" : "idle");
  }

  // ------------------------------------------------------------------------ one turn

  private startTurn(): void {
    this.turn = { stopAt: performance.now(), firstAudioAt: null, chunks: [], sampleRate: 22050 };
    this.dispatch({ type: "turn-start" });
  }

  private finishTurn(): void {
    this.turn = null;
    this.dispatch({ type: "turn-end" });
  }

  // ------------------------------------------------------------------------ recording

  private discardCapture(): void {
    clearInterval(this.recordTimer);
    this.capture?.abort();
    this.capture = null;
    this.dispatch({ type: "live-transcript", text: "" });
  }

  private startTimer(): void {
    const started = performance.now();
    this.recordTimer = setInterval(() => {
      const elapsed = (performance.now() - started) / 1000;
      const { maxSeconds } = this.state;
      this.dispatch({ type: "detail", detail: `${formatClock(elapsed)} of ${formatClock(maxSeconds)} maximum` });
      if (elapsed >= maxSeconds - 0.5) void this.stopRecording(); // stop cleanly before the server's limit
    }, 200);
  }

  private async startRecording(): Promise<void> {
    this.dispatch({ type: "banner-clear" });
    this.player.stop();
    this.endSpeech();
    this.setPhase("requesting-mic");
    this.micRequestCancelled = false;
    const transport = this.state.transport;
    let capture: StreamCapture | RecorderCapture;
    try {
      capture =
        transport === "stream"
          ? await openStreamCapture((pcm) => {
              if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(pcm);
            })
          : await openRecorderCapture();
    } catch (error) {
      this.showBanner(error instanceof UnsupportedError ? `Cannot record here: ${error.message}.` : micErrorMessage(error));
      this.setPhase("idle");
      return;
    }
    if (this.micRequestCancelled) {
      capture.abort(); // cancelled while the permission prompt was open
      this.setPhase("idle");
      return;
    }
    if (transport === "stream" && !this.state.wsOpen) {
      capture.abort(); // the socket dropped while the permission prompt was open
      this.showBanner("The connection to the server was lost. Try again once it reconnects.");
      this.setPhase("idle");
      return;
    }
    this.capture = capture;
    if (transport === "stream") this.sendControl("start", { language: turnLanguage(this.state) ?? "auto", documents: attachedIds(this.state) });
    capture.begin();
    this.startTimer();
    this.setPhase("recording");
  }

  private async stopRecording(): Promise<void> {
    const capture = this.capture;
    if (this.state.phase !== "recording" || !capture) return;
    clearInterval(this.recordTimer);
    this.capture = null;
    this.startTurn();
    if (this.state.transport === "stream") {
      this.setPhase("transcribing");
      await (capture as StreamCapture).stop();
      this.sendControl("stop");
    } else {
      this.setPhase("thinking", "Uploading and processing…");
      await this.sendUpload(await (capture as RecorderCapture).stop());
    }
  }

  // ------------------------------------------------------------------------ upload transport

  private async sendUpload(blob: Blob): Promise<void> {
    const form = new FormData();
    form.append("audio", blob, "speech");
    form.append("history", JSON.stringify(this.history.slice(-12)));
    form.append("language", turnLanguage(this.state) ?? "auto");
    const documents = attachedIds(this.state);
    if (documents.length) form.append("documents", JSON.stringify(documents));
    this.abort = new AbortController();
    try {
      const headers: Record<string, string> = this.geminiKey ? { "X-Gemini-Key": this.geminiKey } : {};
      const response = await fetch("/api/turn", { method: "POST", body: form, headers, signal: this.abort.signal });
      const body = (await response.json()) as UploadResponse & { error?: { code: string; message: string; ids?: string[] } };
      if (!response.ok) {
        this.onServerError(body.error?.code ?? "unknown", body.error?.message ?? "", body.error?.ids);
        return this.finishUpload();
      }
      if (this.turn) {
        this.dispatch({ type: "client-latency", latency: { kind: "response", ms: performance.now() - this.turn.stopAt } });
      }
      if (body.status === "no_speech") {
        this.showBanner("No speech was detected. Try again, a little closer to the microphone.", "warn");
        return this.finishUpload();
      }
      this.dispatch({
        type: "upload-turn",
        transcript: body.transcript,
        reply: body.reply,
        metrics: body.metrics,
        ...(body.language ? { language: body.language } : {}),
        ...(body.audio ? { spoken: "" } : {}), // shown as it is heard, like a streamed reply
      });
      this.noteVoiceFallback(body.metrics);
      this.history.push({ role: "user", content: body.transcript }, { role: "assistant", content: body.reply });
      this.finishUpload();
      if (body.audio) this.playUploaded(`data:audio/wav;base64,${body.audio.data_base64}`, body.reply, body.audio.duration_s);
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        this.showBanner(`Network error: ${(error as Error).message}. Is the server running?`);
      }
      this.finishUpload();
    }
  }

  private finishUpload(): void {
    this.turn = null;
    this.abort = null;
    this.dispatch({ type: "turn-end" });
    void this.refreshSummary();
    if (this.state.phase !== "speaking") this.setPhase("idle");
  }

  private playUploaded(src: string, reply: string, seconds: number): void {
    this.dispatch({ type: "replay", src });
    const audio = new Audio(src);
    this.replayAudio = audio;
    const settle = () => {
      this.endSpeech();
      if (this.state.phase === "speaking") this.setPhase("idle");
    };
    audio.onended = settle;
    audio.onpause = settle;
    audio.play().then(
      () => {
        const timeline = this.followSpeech(() => audio.currentTime);
        timeline.add(0, reply, 0, seconds);
        timeline.complete();
        this.setPhase("speaking");
      },
      () => this.endSpeech(), // autoplay blocked: show the text; the replay controls are there
    );
  }

  // ------------------------------------------------------------------------ user actions

  toggleRecording(): void {
    this.player.unlock(); // inside the click handler, so playback is allowed later
    const { phase } = this.state;
    if (phase === "recording") void this.stopRecording();
    else if (phase === "idle") void this.startRecording();
  }

  cancel(): void {
    switch (this.state.phase) {
      case "requesting-mic":
        this.micRequestCancelled = true; // startRecording() notices when the permission prompt resolves
        this.setPhase("idle");
        break;
      case "recording":
        this.discardCapture();
        if (this.state.transport === "stream") this.sendControl("cancel");
        this.setPhase("idle");
        break;
      case "transcribing":
      case "thinking":
      case "speaking":
        this.player.stop();
        this.endSpeech();
        this.replayAudio?.pause();
        if (this.state.turnActive && this.state.transport === "stream") this.sendControl("cancel"); // server replies "cancelled"
        else if (this.abort) this.abort.abort();
        else this.setPhase("idle");
        break;
      case "idle":
        break;
    }
  }

  dismissBanner(): void {
    this.dispatch({ type: "banner-clear" });
  }

  resetConversation(): void {
    if (this.state.phase !== "idle") this.cancel();
    this.sendControl("reset");
    this.endSpeech();
    this.history = [];
    this.dispatch({ type: "new-conversation" });
    this.blobUrls.forEach((url) => URL.revokeObjectURL(url));
    this.blobUrls = [];
  }

  /** Checks a Gemini key with the server and, if it works, uses it for this user's questions from now on. */
  async saveGeminiKey(raw: string): Promise<boolean> {
    const key = raw.trim();
    if (!key) return false;
    this.dispatch({ type: "key", key: { checking: true, message: "", ok: null } });
    try {
      const response = await fetch("/api/gemini-key/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      const body = (await response.json()) as { valid?: boolean; message?: string; error?: { message: string } };
      if (!response.ok || !body.valid) {
        const message = body.message ?? body.error?.message ?? "The key could not be checked.";
        this.dispatch({ type: "key", key: { checking: false, ok: false, message } });
        return false;
      }
      this.geminiKey = key;
      saveSetting(GEMINI_KEY, key);
      this.sendControl("configure", { gemini_key: key });
      this.dispatch({ type: "key", key: { saved: true, checking: false, ok: true, message: body.message ?? "The key works." } });
      return true;
    } catch (error) {
      this.dispatch({ type: "key", key: { checking: false, ok: false, message: `Could not check the key: ${(error as Error).message}` } });
      return false;
    }
  }

  forgetGeminiKey(): void {
    this.geminiKey = null;
    saveSetting(GEMINI_KEY, null);
    this.sendControl("configure", { gemini_key: null });
    this.dispatch({ type: "key", key: { saved: false, ok: null, message: "Your key was removed from this browser." } });
  }

  setLanguage(language: string): void {
    this.dispatch({ type: "language", language });
    saveSetting(LANGUAGE_KEY, language);
  }

  /** Uploads files to ask questions about. Their ids go with every question until they are removed. */
  attachFiles(files: File[]): void {
    const limits = this.state.info?.documents;
    const maxBytes = limits?.max_bytes ?? DEFAULT_MAX_FILE_BYTES;
    const maxFiles = limits?.max_per_turn ?? DEFAULT_MAX_FILES;
    const room = Math.max(0, maxFiles - this.state.files.length);
    if (files.length > room) this.showBanner(`At most ${maxFiles} files can be attached at once.`, "warn");
    for (const file of files.slice(0, room)) {
      if (file.size > maxBytes) this.showBanner(`“${file.name}” is larger than ${Math.round(maxBytes / 1_000_000)} MB.`);
      else void this.uploadFile(file);
    }
  }

  private async uploadFile(file: File): Promise<void> {
    const key = `file-${++this.fileCounter}`;
    this.dispatch({ type: "file-add", file: { key, name: file.name, status: "uploading" } });
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const response = await fetch("/api/documents", { method: "POST", body: form });
      const body = (await response.json()) as DocumentSummary & { error?: { message: string } };
      if (!this.state.files.some((f) => f.key === key)) {
        if (response.ok) void this.deleteOnServer(body.id); // removed while it was uploading
        return;
      }
      if (!response.ok) {
        this.dispatch({ type: "file-remove", key });
        this.showBanner(`Could not attach “${file.name}”: ${body.error?.message ?? response.statusText}`);
        return;
      }
      this.dispatch({ type: "file-ready", key, summary: body });
    } catch (error) {
      this.dispatch({ type: "file-remove", key });
      this.showBanner(`Could not attach “${file.name}”: ${(error as Error).message}. Is the server running?`);
    }
  }

  removeFile(key: string): void {
    const file = this.state.files.find((f) => f.key === key);
    this.dispatch({ type: "file-remove", key });
    if (file?.id) void this.deleteOnServer(file.id);
  }

  private async deleteOnServer(id: string): Promise<void> {
    try {
      await fetch(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
    } catch {
      /* the server drops old files by itself */
    }
  }

  setTransport(transport: Transport): void {
    if (transport === this.state.transport) return;
    if (this.state.phase !== "idle") this.cancel();
    this.dispatch({ type: "transport", transport });
    saveSetting(TRANSPORT_KEY, transport);
    if (transport === "stream") this.connectSocket();
    else this.closeSocket();
    void this.pollHealth();
  }
}
