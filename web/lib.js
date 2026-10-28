// Small pure helpers used by app.js (kept separate so they can be tested with Node).

// 12.3 -> "12 ms", 1234 -> "1.23 s", null -> "—". Never invents a number.
// The space is a non-breaking one so a number never wraps away from its unit.
export function formatMs(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${Math.round(value)} ms`;
}

// Wraps 16-bit mono PCM chunks (Int16Array[]) in a WAV container so an <audio> element can replay it.
export function buildWav(chunks, sampleRate) {
  const samples = chunks.reduce((n, c) => n + c.length, 0);
  const bytes = new Uint8Array(44 + samples * 2);
  const view = new DataView(bytes.buffer);
  const text = (offset, s) => [...s].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  text(36, "data");
  view.setUint32(40, samples * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++, offset += 2) view.setInt16(offset, chunk[i], true);
  }
  return bytes;
}

const ERROR_TEXT = {
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

// *_error codes carry a specific, already-readable message from the server (e.g. how to start
// Ollama), so show that. Known codes get friendlier text; anything else falls back to the message.
export function friendlyError(code, serverMessage) {
  if (ERROR_TEXT[code]) return ERROR_TEXT[code];
  return serverMessage || `Something went wrong (${code}).`;
}

// Maps getUserMedia failures to something a person can act on.
export function micErrorMessage(error) {
  switch (error && error.name) {
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
      return `Could not start the microphone${error && error.message ? `: ${error.message}` : "."}`;
  }
}
