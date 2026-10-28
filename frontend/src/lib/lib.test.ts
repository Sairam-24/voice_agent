import assert from "node:assert/strict";
import { test } from "node:test";
import { friendlyError, micErrorMessage } from "./errors.ts";
import { formatClock, formatMs } from "./format.ts";
import { parseServerMessage } from "./protocol.ts";
import { buildWav } from "./wav.ts";

test("formatMs never fabricates a value", () => {
  assert.equal(formatMs(null), "—");
  assert.equal(formatMs(undefined), "—");
  assert.equal(formatMs(Number.NaN), "—");
  assert.equal(formatMs(12.4), "12 ms");
  assert.equal(formatMs(1234), "1.23 s");
  assert.equal(formatMs(0), "0 ms");
});

test("formatClock renders minutes and zero-padded seconds", () => {
  assert.equal(formatClock(0), "0:00");
  assert.equal(formatClock(75.9), "1:15");
  assert.equal(formatClock(-3), "0:00");
});

test("parseServerMessage accepts the protocol's message types", () => {
  assert.deepEqual(parseServerMessage('{"type":"transcript","text":"hi","final":true}'), { type: "transcript", text: "hi", final: true });
  assert.equal(parseServerMessage('{"type":"cancelled"}')?.type, "cancelled");
});

test("parseServerMessage ignores anything it does not understand", () => {
  assert.equal(parseServerMessage("not json"), null);
  assert.equal(parseServerMessage("[1,2]"), null);
  assert.equal(parseServerMessage("null"), null);
  assert.equal(parseServerMessage('{"type":"surprise"}'), null);
  assert.equal(parseServerMessage('{"text":"no type"}'), null);
});

test("buildWav writes a valid 44-byte PCM header followed by the samples", () => {
  const wav = buildWav([Int16Array.of(1, -2), Int16Array.of(300)], 22050);
  const view = new DataView(wav.buffer);
  const ascii = (offset: number, length: number) => String.fromCharCode(...wav.slice(offset, offset + length));
  assert.equal(wav.length, 44 + 3 * 2);
  assert.equal(ascii(0, 4), "RIFF");
  assert.equal(ascii(8, 4), "WAVE");
  assert.equal(view.getUint32(24, true), 22050);
  assert.equal(view.getUint32(40, true), 6);
  assert.deepEqual([view.getInt16(44, true), view.getInt16(46, true), view.getInt16(48, true)], [1, -2, 300]);
});

test("known error codes get friendly text; other codes pass the server's message through", () => {
  assert.match(friendlyError("server_busy"), /busy/);
  assert.equal(friendlyError("llm_error", "Ollama is not reachable. Start it with `ollama serve`."), "Ollama is not reachable. Start it with `ollama serve`.");
  assert.equal(friendlyError("weird"), "Something went wrong (weird).");
});

test("microphone errors say what to do", () => {
  assert.match(micErrorMessage({ name: "NotAllowedError" }), /permission was denied/);
  assert.match(micErrorMessage({ name: "NotFoundError" }), /No microphone/);
  assert.match(micErrorMessage({ name: "NotReadableError" }), /in use/);
  assert.match(micErrorMessage({ name: "SecurityError" }), /secure page/);
  assert.match(micErrorMessage(new Error("boom")), /boom/);
  assert.match(micErrorMessage(undefined), /Could not start the microphone/);
});
