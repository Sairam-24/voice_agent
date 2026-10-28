// The UI's view state and the pure function that updates it. Everything with side effects
// (WebSocket, microphone, audio playback) lives in controller.ts; this file has none, so it can
// be tested with Node alone.
import type { DocumentSummary, Health, MetricsSummary, ServerInfo, TurnMetrics } from "./protocol.ts";

export type Phase = "idle" | "requesting-mic" | "recording" | "transcribing" | "thinking" | "speaking";
export type Transport = "stream" | "upload";
export type PillKind = "ok" | "warn" | "error";

export interface Pill {
  kind: PillKind;
  text: string;
  title: string;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  pending: boolean;
  /** Language code of a user message, when the server reported one (e.g. "en", "te"). */
  language?: string;
  /** The model behind the message: the recogniser for the user's words, the voice for a reply. */
  engine?: string;
  /** Object or data URL of the spoken reply, once there is one. */
  replaySrc: string | null;
  /** While a reply is being spoken: the part already heard, which is what is shown. Absent = show all of `text`. */
  spoken?: string;
}

/** A file attached to the conversation, to ask questions about. */
export interface AttachedFile {
  key: string;
  name: string;
  status: "uploading" | "ready";
  /** The server's id for it, once uploaded; sent with every question. */
  id?: string;
  chars?: number;
  pages?: number | null;
  truncated?: boolean;
}

/** The user's own Gemini key. The key itself is never in the store, only whether there is one. */
export interface KeyState {
  /** Saved in this browser, and used for this user's questions. */
  saved: boolean;
  checking: boolean;
  /** The outcome of the last check or change, shown next to the field. */
  message: string;
  ok: boolean | null;
}

export interface Banner {
  text: string;
  kind: "error" | "warn";
}

/** Latency as seen by this browser: from the press of "stop" to the first audio (or the full response). */
export interface ClientLatency {
  kind: "first-audio" | "response";
  ms: number;
}

export interface AgentState {
  transport: Transport;
  phase: Phase;
  detail: string;
  connection: Pill;
  models: Pill;
  wsOpen: boolean;
  serverReachable: boolean;
  modelsReady: boolean;
  maxSeconds: number;
  turnActive: boolean;
  messages: Message[];
  liveTranscript: string;
  banner: Banner | null;
  lastMetrics: TurnMetrics | null;
  summary: MetricsSummary | null;
  health: Health | null;
  info: ServerInfo | null;
  clientLatency: ClientLatency | null;
  announcement: string;
  nextMessageId: number;
  /** The language picked for the conversation: "auto" or a code such as "te". */
  language: string;
  files: AttachedFile[];
  key: KeyState;
  /** The stages on Gemini for this connection, as the server confirmed; null until it has. */
  sessionCloud: string[] | null;
}

export function initialState(transport: Transport = "stream", language = "auto", keySaved = false): AgentState {
  return {
    transport,
    phase: "idle",
    detail: "",
    connection: { kind: "warn", text: "Connecting…", title: "" },
    models: { kind: "warn", text: "Checking server…", title: "" },
    wsOpen: false,
    serverReachable: true,
    modelsReady: false,
    maxSeconds: 30,
    turnActive: false,
    messages: [],
    liveTranscript: "",
    banner: null,
    lastMetrics: null,
    summary: null,
    health: null,
    info: null,
    clientLatency: null,
    announcement: "",
    nextMessageId: 1,
    language,
    files: [],
    key: { saved: keySaved, checking: false, message: "", ok: null },
    sessionCloud: null,
  };
}

export type Action =
  | { type: "transport"; transport: Transport }
  | { type: "phase"; phase: Phase; detail?: string }
  | { type: "detail"; detail: string }
  | { type: "connection"; connection: Pill; wsOpen?: boolean; maxSeconds?: number }
  | { type: "health"; health: Health }
  | { type: "unreachable" }
  | { type: "info"; info: ServerInfo }
  | { type: "summary"; summary: MetricsSummary }
  | { type: "banner"; banner: Banner }
  | { type: "banner-clear" }
  | { type: "turn-start" }
  | { type: "turn-end" }
  | { type: "live-transcript"; text: string }
  | { type: "final-transcript"; text: string; language?: string }
  | { type: "reply-delta"; text: string }
  | { type: "replay"; src: string }
  | { type: "done"; metrics: TurnMetrics }
  | { type: "client-latency"; latency: ClientLatency }
  | { type: "upload-turn"; transcript: string; reply: string; metrics: TurnMetrics; language?: string; spoken?: string }
  | { type: "new-conversation" }
  | { type: "spoken"; text: string | null }
  | { type: "language"; language: string }
  | { type: "file-add"; file: AttachedFile }
  | { type: "file-ready"; key: string; summary: DocumentSummary }
  | { type: "file-remove"; key?: string; ids?: string[] }
  | { type: "key"; key: Partial<KeyState> }
  | { type: "configured"; cloud: string[] };

function lastPendingAssistantIndex(messages: Message[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message?.role === "assistant" && message.pending) return i;
  }
  return -1;
}

function replaceAt(messages: Message[], index: number, change: Partial<Message>): Message[] {
  return messages.map((message, i) => (i === index ? { ...message, ...change } : message));
}

/** Cloud (Gemini) models are not named in the UI; local ones are, since they show when a fallback stood in. */
export function shownEngine(name: string | null | undefined): string | undefined {
  return name && !name.startsWith("gemini") ? name : undefined;
}

/** The voice that spoke a reply, as the server reported it; "text only" when there was none. */
export function voiceUsed(metrics: TurnMetrics): string | undefined {
  if (metrics.tts_engine) return shownEngine(metrics.tts_engine);
  return metrics.tts_fallback?.startsWith("Text only") ? "text only" : undefined;
}

/** Records which models handled the last exchange on its two messages. */
function annotateLastExchange(messages: Message[], metrics: TurnMetrics): Message[] {
  const voice = voiceUsed(metrics);
  const lastUser = messages.findLastIndex((m) => m.role === "user");
  const lastAssistant = messages.findLastIndex((m) => m.role === "assistant");
  const recogniser = shownEngine(metrics.stt_engine);
  return messages.map((m, i) => {
    if (i === lastUser && recogniser) return { ...m, engine: recogniser };
    if (i === lastAssistant && i > lastUser && voice) return { ...m, engine: voice };
    return m;
  });
}

/** Drops a reply bubble that never received any text and stops showing the others as in progress. */
function settleMessages(messages: Message[]): Message[] {
  return messages
    .filter((m) => !(m.role === "assistant" && m.pending && m.text === ""))
    .map((m) => (m.pending ? { ...m, pending: false } : m));
}

export function healthToModelsPill(health: Health): Pill {
  const failed = Object.entries(health.components).find(([, component]) => component?.state === "error");
  if (health.status === "ok") return { kind: "ok", text: "Models ready", title: "" };
  if (failed) return { kind: "error", text: `${failed[0].toUpperCase()} unavailable`, title: failed[1]?.detail ?? "" };
  return { kind: "warn", text: "Loading models…", title: "The first start downloads and loads the models." };
}

export function reduce(state: AgentState, action: Action): AgentState {
  switch (action.type) {
    case "transport":
      return { ...state, transport: action.transport };
    case "phase":
      return { ...state, phase: action.phase, detail: action.detail ?? "" };
    case "detail":
      return action.detail === state.detail ? state : { ...state, detail: action.detail };
    case "connection":
      return {
        ...state,
        connection: action.connection,
        wsOpen: action.wsOpen ?? state.wsOpen,
        maxSeconds: action.maxSeconds ?? state.maxSeconds,
      };
    case "health": {
      const { components } = action.health;
      return {
        ...state,
        health: action.health,
        serverReachable: true,
        modelsReady: components.stt?.state === "ready" && components.tts?.state === "ready",
        models: healthToModelsPill(action.health),
      };
    }
    case "unreachable":
      return {
        ...state,
        health: null,
        serverReachable: false,
        modelsReady: false,
        models: { kind: "error", text: "Server unreachable", title: "Could not reach /health" },
      };
    case "info":
      return { ...state, info: action.info };
    case "summary":
      return { ...state, summary: action.summary };
    case "banner":
      return { ...state, banner: action.banner };
    case "banner-clear":
      return state.banner ? { ...state, banner: null } : state;
    case "turn-start":
      return { ...state, turnActive: true, clientLatency: null };
    case "turn-end":
      return { ...state, turnActive: false, messages: settleMessages(state.messages) };
    case "live-transcript":
      return { ...state, liveTranscript: action.text };
    case "final-transcript": {
      if (!action.text) return { ...state, liveTranscript: "" };
      const user: Message = {
        id: state.nextMessageId,
        role: "user",
        text: action.text,
        pending: false,
        replaySrc: null,
        ...(action.language ? { language: action.language } : {}),
      };
      // Its text stays hidden until it is heard (see lib/speech.ts); the controller reveals it.
      const assistant: Message = { id: state.nextMessageId + 1, role: "assistant", text: "", pending: true, replaySrc: null, spoken: "" };
      return {
        ...state,
        liveTranscript: "",
        messages: [...state.messages, user, assistant],
        nextMessageId: state.nextMessageId + 2,
        announcement: `You said: ${action.text}`,
      };
    }
    case "reply-delta": {
      const index = lastPendingAssistantIndex(state.messages);
      const current = state.messages[index];
      if (!current) return state;
      return { ...state, messages: replaceAt(state.messages, index, { text: current.text + action.text }) };
    }
    case "replay": {
      // The reply may already be settled (its audio can finish after "done"), so take the last assistant message.
      let index = -1;
      state.messages.forEach((m, i) => {
        if (m.role === "assistant") index = i;
      });
      return index < 0 ? state : { ...state, messages: replaceAt(state.messages, index, { replaySrc: action.src }) };
    }
    case "done": {
      const reply = [...state.messages].reverse().find((m) => m.role === "assistant" && m.pending);
      return {
        ...state,
        lastMetrics: action.metrics,
        turnActive: false,
        messages: annotateLastExchange(settleMessages(state.messages), action.metrics),
        announcement: reply?.text ? `Assistant: ${reply.text}` : state.announcement,
      };
    }
    case "client-latency":
      return { ...state, clientLatency: action.latency };
    case "upload-turn": {
      const id = state.nextMessageId;
      return {
        ...state,
        lastMetrics: action.metrics,
        nextMessageId: id + 2,
        messages: annotateLastExchange([
          ...state.messages,
          {
            id,
            role: "user",
            text: action.transcript,
            pending: false,
            replaySrc: null,
            ...(action.language ? { language: action.language } : {}),
          },
          {
            id: id + 1,
            role: "assistant",
            text: action.reply,
            pending: false,
            replaySrc: null,
            ...(action.spoken !== undefined ? { spoken: action.spoken } : {}),
          },
        ], action.metrics),
        announcement: `You said: ${action.transcript}. Assistant: ${action.reply}`,
      };
    }
    case "new-conversation":
      return { ...state, messages: [], liveTranscript: "", announcement: "" };
    case "spoken": {
      const index = state.messages.findLastIndex((m) => m.role === "assistant");
      const current = state.messages[index];
      if (!current || (action.text === null && current.spoken === undefined) || current.spoken === action.text) return state;
      const { spoken: _, ...rest } = current;
      const next = action.text === null ? rest : { ...rest, spoken: action.text };
      return { ...state, messages: state.messages.map((m, i) => (i === index ? next : m)) };
    }
    case "language":
      return { ...state, language: action.language };
    case "file-add":
      return { ...state, files: [...state.files, action.file] };
    case "file-ready": {
      const { id, chars, pages, truncated } = action.summary;
      const files = state.files.map((f) => (f.key === action.key ? { ...f, status: "ready" as const, id, chars, pages, truncated } : f));
      return { ...state, files };
    }
    case "key":
      return { ...state, key: { ...state.key, ...action.key } };
    case "configured":
      return { ...state, sessionCloud: action.cloud };
    case "file-remove": {
      const files = state.files.filter((f) => f.key !== action.key && !(f.id && action.ids?.includes(f.id)));
      return files.length === state.files.length ? state : { ...state, files };
    }
  }
}

// ---------------------------------------------------------------------------- derived values

export type Stage = "stt" | "llm" | "tts";

export function canRecord(state: AgentState): boolean {
  if (state.phase === "recording") return true;
  const linked = state.transport === "upload" ? state.serverReachable : state.wsOpen;
  return state.phase === "idle" && state.modelsReady && linked && !readingFiles(state);
}

export type Mode = "local" | "server-key" | "own-key";

/** Where this browser's questions are answered: local models, Gemini with the server's key, or with the user's. */
export function mode(state: AgentState): Mode {
  if (state.key.saved) return "own-key";
  return state.info?.cloud?.server_key ? "server-key" : "local";
}

export function modeLabel(state: AgentState): string {
  return { local: "Local models", "server-key": "Cloud", "own-key": "Cloud · your key" }[mode(state)];
}

/** The stages that use Gemini for this browser's questions. */
export function cloudStages(state: AgentState): string[] {
  if (state.sessionCloud) return state.sessionCloud;
  return mode(state) === "local" ? [] : (state.info?.cloud?.stages ?? []);
}

/** A question asked while a file is still uploading would be answered without it. */
export function readingFiles(state: AgentState): boolean {
  return state.files.some((f) => f.status === "uploading");
}

/** The ids of the attached files, to send with a question. */
export function attachedIds(state: AgentState): string[] {
  return state.files.flatMap((f) => (f.status === "ready" && f.id ? [f.id] : []));
}

/** The language choices to offer: "auto" first when the server can detect them. Fewer than two = no choice. */
export function languageOptions(info: ServerInfo | null): string[] {
  const choices = info?.languages?.choices ?? [];
  const options = info?.languages?.auto ? ["auto", ...choices] : choices;
  return options.length > 1 ? options : [];
}

/** The language to send with a turn: null = let the server detect it (or use its fixed language). */
export function turnLanguage(state: AgentState): string | null {
  return state.language !== "auto" && languageOptions(state.info).includes(state.language) ? state.language : null;
}

const PHASE_TEXT: Record<Exclude<Phase, "idle">, string> = {
  "requesting-mic": "Waiting for microphone permission…",
  recording: "Recording",
  transcribing: "Transcribing…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

export function stateLabel(state: AgentState): string {
  if (state.phase !== "idle") return PHASE_TEXT[state.phase];
  if (readingFiles(state)) return "Reading the file…";
  if (canRecord(state)) return "Ready";
  return state.modelsReady ? "Not connected" : "Waiting for the server";
}

/** The one-word status shown inside the core. It only restates the real state in the HUD's vocabulary. */
export function coreWord(state: AgentState): string {
  switch (state.phase) {
    case "requesting-mic":
      return "ACCESS";
    case "recording":
      return "LISTENING";
    case "transcribing":
      return "DECODING";
    case "thinking":
      return "THINKING";
    case "speaking":
      return "SPEAKING";
    case "idle":
      if (canRecord(state)) return "READY";
      if (!state.serverReachable) return "OFFLINE";
      return state.modelsReady ? "LINKING" : "BOOTING";
  }
}

export function cancelLabel(phase: Phase): string {
  return phase === "recording" ? "Discard" : phase === "speaking" ? "Stop" : "Cancel";
}

/** Which pipeline stage the server is working on, judged from the phase. */
export function activeStage(phase: Phase): Stage | null {
  switch (phase) {
    case "transcribing":
      return "stt";
    case "thinking":
      return "llm";
    case "speaking":
      return "tts";
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------- tiny store

export interface Store<S, A> {
  getState(): S;
  dispatch(action: A): void;
  subscribe(listener: () => void): () => void;
}

/** A synchronous store: dispatch updates the state immediately, so event handlers never read stale state. */
export function createStore<S, A>(reducer: (state: S, action: A) => S, initial: S): Store<S, A> {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    getState: () => state,
    dispatch(action) {
      const next = reducer(state, action);
      if (next === state) return;
      state = next;
      listeners.forEach((listener) => listener());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
  };
}
