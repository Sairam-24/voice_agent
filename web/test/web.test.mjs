// Run with:  node --test web/test
import assert from "node:assert/strict";
import test from "node:test";

import { Downsampler } from "../downsampler.js";
import { buildWav, formatMs, friendlyError, micErrorMessage } from "../lib.js";

function sine(rate, seconds, freq = 440, amplitude = 0.5) {
  return Float32Array.from({ length: Math.round(rate * seconds) }, (_, i) => amplitude * Math.sin((2 * Math.PI * freq * i) / rate));
}

function collect(downsampler, samples, blockSize = 128) {
  const chunks = [];
  for (let i = 0; i < samples.length; i += blockSize) chunks.push(...downsampler.push(samples.subarray(i, i + blockSize)));
  chunks.push(downsampler.flush());
  return chunks;
}

test("48 kHz -> 16 kHz keeps the duration (within one output sample)", () => {
  const total = collect(new Downsampler(48000), sine(48000, 2)).reduce((n, c) => n + c.length, 0);
  assert.ok(Math.abs(total - 32000) <= 1, `got ${total} samples`);
});

test("44.1 kHz (fractional ratio) -> 16 kHz keeps the duration", () => {
  const total = collect(new Downsampler(44100), sine(44100, 2)).reduce((n, c) => n + c.length, 0);
  assert.ok(Math.abs(total - 32000) <= 2, `got ${total} samples`);
});

test("full chunks are 100 ms (1600 samples)", () => {
  const d = new Downsampler(48000);
  const chunks = d.push(sine(48000, 0.35));
  assert.equal(chunks.length, 3);
  assert.ok(chunks.every((c) => c.length === 1600));
  assert.ok(d.flush().length > 0 && d.flush().length === 0, "flush returns the remainder once");
});

test("amplitude survives resampling and is scaled to int16", () => {
  const peak = Math.max(...collect(new Downsampler(48000), sine(48000, 0.5, 440, 0.5)).flatMap((c) => [...c].map(Math.abs)));
  assert.ok(peak > 0.45 * 32767 && peak <= 0.5 * 32767 + 50, `peak ${peak}`);
});

test("out-of-range input is clipped, not wrapped", () => {
  const loud = new Float32Array(480).fill(1.7);
  const out = collect(new Downsampler(48000), loud).flatMap((c) => [...c]);
  assert.ok(out.every((v) => v === 32767));
});

test("buildWav writes a valid 44-byte PCM header and the samples", () => {
  const wav = buildWav([Int16Array.of(1, -2, 3), Int16Array.of(4)], 22050);
  const view = new DataView(wav.buffer);
  assert.equal(new TextDecoder().decode(wav.slice(0, 4)), "RIFF");
  assert.equal(new TextDecoder().decode(wav.slice(8, 12)), "WAVE");
  assert.equal(view.getUint32(24, true), 22050); // sample rate
  assert.equal(view.getUint16(22, true), 1); // mono
  assert.equal(view.getUint16(34, true), 16); // bits
  assert.equal(view.getUint32(40, true), 8); // 4 samples * 2 bytes
  assert.equal(view.getUint32(4, true), wav.length - 8); // RIFF size
  assert.deepEqual([1, 2, 3, 4].map((i) => view.getInt16(44 + (i - 1) * 2, true)), [1, -2, 3, 4]);
});

test("formatMs never fabricates a value", () => {
  assert.equal(formatMs(null), "—");
  assert.equal(formatMs(undefined), "—");
  assert.equal(formatMs(NaN), "—");
  assert.equal(formatMs(12.4), "12 ms"); // non-breaking space keeps number and unit together
  assert.equal(formatMs(1234), "1.23 s");
  assert.equal(formatMs(0), "0 ms");
});

test("known error codes get friendly text; *_error codes pass the server message through", () => {
  assert.match(friendlyError("server_busy", "x"), /busy/i);
  assert.equal(friendlyError("llm_error", "Ollama is not reachable at http://localhost:11434."), "Ollama is not reachable at http://localhost:11434.");
  assert.match(friendlyError("mystery", ""), /mystery/);
});

test("microphone errors explain what to do", () => {
  assert.match(micErrorMessage({ name: "NotAllowedError" }), /permission was denied/i);
  assert.match(micErrorMessage({ name: "NotFoundError" }), /No microphone/i);
  assert.match(micErrorMessage({ name: "NotReadableError" }), /in use/i);
  assert.match(micErrorMessage({ name: "Weird", message: "boom" }), /boom/);
});
