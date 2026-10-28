// Types for what the server sends. The protocol is documented at the top of app/voice_socket.py.

/** Per-turn timings from the server, in milliseconds. A stage that did not run is null. */
export interface TurnMetrics {
  request_id?: string;
  transport?: string;
  status?: string;
  audio_seconds?: number | null;
  upload_ms?: number | null;
  decode_ms?: number | null;
  stt_queue_ms?: number | null;
  stt_ms?: number | null;
  llm_queue_ms?: number | null;
  llm_ttft_ms?: number | null;
  llm_ms?: number | null;
  tts_queue_ms?: number | null;
  tts_first_chunk_ms?: number | null;
  tts_ms?: number | null;
  first_audio_ms?: number | null;
  total_ms?: number | null;
  /** Language of the user's speech (detected by STT, or the configured one). */
  language?: string | null;
  /** The voice that spoke the reply, and why a fallback voice was used if it was. */
  tts_engine?: string | null;
  tts_fallback?: string | null;
  /** Why a fallback recogniser transcribed this turn, if one did, and which recogniser did. */
  stt_fallback?: string | null;
  stt_engine?: string | null;
  /** Why the local model answered instead of the cloud one, if it did. */
  llm_fallback?: string | null;
}

/** The timing fields (milliseconds). */
export type StageKey = Extract<keyof TurnMetrics, `${string}_ms`>;

export interface StageSummary {
  n: number;
  median: number;
  p95: number;
}

export interface MetricsSummary {
  stages_ms: Partial<Record<StageKey, StageSummary>>;
  turns_since_start: Record<string, number>;
}

export type ComponentState = "ready" | "loading" | "error" | string;

export interface Health {
  status: "ok" | "starting" | "degraded" | string;
  components: Partial<Record<"stt" | "tts" | "llm", { state: ComponentState; detail: string }>>;
}

/** A voice, or a set of local voices keyed by script ("latin", "telugu"), possibly with a fallback. */
export interface TtsInfo {
  engine?: string;
  model?: string;
  voice?: string;
  sample_rate?: number;
  voices?: Record<string, TtsInfo>;
  fallback?: TtsInfo;
}

export interface SttInfo {
  engine?: string;
  model?: string;
  device?: string;
  compute_type?: string;
  language?: string;
  fallback?: SttInfo;
}

/** The parts of /api/info the HUD displays. */
export interface ServerInfo {
  components?: {
    stt?: SttInfo;
    llm?: { provider?: string; model?: string; parameter_size?: string; quantization?: string; thinking_level?: string | null };
    tts?: TtsInfo;
  };
  /** Languages the user may pick; `auto` = the server can also detect them. */
  languages?: { choices: string[]; auto: boolean };
  documents?: { max_bytes: number; max_per_turn: number; context_chars: number };
  /** Whether the server has its own Gemini key, and which stages use Gemini when a key is available. */
  cloud?: { server_key: boolean; stages: string[] };
}

/** What POST /api/documents returns for an attached file. */
export interface DocumentSummary {
  id: string;
  name: string;
  chars: number;
  pages: number | null;
  truncated: boolean;
}

export type ServerStatus = "idle" | "listening" | "transcribing" | "thinking" | "speaking";

export type ServerMessage =
  | { type: "ready"; max_audio_seconds: number; partial_transcripts?: boolean; models_ready?: boolean }
  /** After "configure": the stages that run on Gemini for this connection. */
  | { type: "configured"; cloud: string[] }
  | { type: "status"; state: ServerStatus }
  | { type: "transcript"; text: string; final: boolean; language?: string }
  | { type: "reply_delta"; text: string }
  /** `sentence`: the words this chunk speaks; consecutive chunks of one piece of text share `segment`. */
  | { type: "audio"; index: number; sample_rate: number; sentence?: string; segment?: number; engine?: string }
  | { type: "done"; status: string; metrics: TurnMetrics }
  | { type: "cancelled" }
  | { type: "error"; code: string; message: string; ids?: string[] };

const KNOWN_TYPES = new Set(["ready", "configured", "status", "transcript", "reply_delta", "audio", "done", "cancelled", "error"]);

/** Parses one JSON text frame. Anything malformed or of an unknown type becomes null and is ignored. */
export function parseServerMessage(raw: string): ServerMessage | null {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof value !== "object" || value === null) return null;
  const { type } = value as { type?: unknown };
  if (typeof type !== "string" || !KNOWN_TYPES.has(type)) return null;
  return value as ServerMessage;
}
