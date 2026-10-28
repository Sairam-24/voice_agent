// Turns server error codes and browser microphone failures into text a person can act on.

const ERROR_TEXT: Record<string, string> = {
  unsupported_audio: "The server could not decode that audio.",
  empty_audio: "No audio was recorded. Speak after the recording indicator appears, then press stop.",
  audio_too_long: "That recording was too long. Try a shorter message.",
  upload_too_large: "That recording was too large to upload.",
  server_busy: "The server is busy with other requests. Try again in a moment.",
  shutting_down: "The server is shutting down.",
  models_not_ready: "The speech models are still loading. Wait for the status to read “Models ready”.",
  stt_timeout: "Transcription took too long and was stopped.",
  llm_timeout: "The language model took too long and was stopped.",
  tts_timeout: "Speech synthesis took too long and was stopped.",
  turn_timeout: "The whole request took too long and was stopped.",
};

/**
 * `*_error` codes carry a specific, already-readable message from the server (for example how to
 * start Ollama), so that is shown. Known codes get friendlier text; anything else falls back to
 * the server's message.
 */
export function friendlyError(code: string, serverMessage?: string): string {
  return ERROR_TEXT[code] ?? (serverMessage || `Something went wrong (${code}).`);
}

export function micErrorMessage(error: unknown): string {
  const { name, message } = (error ?? {}) as { name?: string; message?: string };
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Microphone permission was denied. Allow microphone access for this site in your browser’s address-bar settings, then try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No microphone was found. Connect one and try again.";
    case "NotReadableError":
    case "TrackStartError":
      return "The microphone is in use by another application or could not be started.";
    case "SecurityError":
      return "Microphone access requires a secure page (https:// or http://localhost).";
    default:
      return `Could not start the microphone${message ? `: ${message}` : "."}`;
  }
}

/** Thrown when the browser lacks a needed API; the UI says so instead of showing a raw error. */
export class UnsupportedError extends Error {
  override name = "Unsupported";
}
