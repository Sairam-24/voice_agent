import assert from "node:assert/strict";
import { test } from "node:test";
import type { Health } from "./protocol.ts";
import {
  activeStage,
  attachedIds,
  type Action,
  type AgentState,
  canRecord,
  cancelLabel,
  cloudStages,
  coreWord,
  createStore,
  initialState,
  languageOptions,
  mode,
  modeLabel,
  reduce,
  stateLabel,
  turnLanguage,
} from "./state.ts";

const run = (state: AgentState, ...actions: Action[]) => actions.reduce(reduce, state);
const READY: Health = {
  status: "ok",
  components: { stt: { state: "ready", detail: "" }, tts: { state: "ready", detail: "" }, llm: { state: "ready", detail: "" } },
};

test("a final transcript adds the user's message and an empty pending reply", () => {
  const state = run(initialState(), { type: "final-transcript", text: "What is the capital of France?" });
  assert.deepEqual(
    state.messages.map((m) => [m.role, m.text, m.pending]),
    [
      ["user", "What is the capital of France?", false],
      ["assistant", "", true],
    ],
  );
  assert.equal(state.announcement, "You said: What is the capital of France?");
});

test("the detected language is kept on the user's message, and left off when unknown", () => {
  const withLanguage = run(initialState(), { type: "final-transcript", text: "नमस्ते", language: "hi" });
  assert.equal(withLanguage.messages[0]?.language, "hi");
  const upload = run(initialState("upload"), { type: "upload-turn", transcript: "hola", reply: "¡Hola!", metrics: {}, language: "es" });
  assert.equal(upload.messages[0]?.language, "es");
  const without = run(initialState(), { type: "final-transcript", text: "hi" });
  assert.equal("language" in (without.messages[0] ?? {}), false);
});

test("the models that handled a turn are recorded on its two messages", () => {
  const state = run(
    initialState(),
    { type: "final-transcript", text: "నమస్కారం", language: "te" },
    { type: "reply-delta", text: "నమస్కారం!" },
    { type: "done", metrics: { stt_engine: "whisper base", tts_engine: "piper te_IN-padmavathi-medium" } },
  );
  assert.equal(state.messages[0]?.engine, "whisper base");
  assert.equal(state.messages[1]?.engine, "piper te_IN-padmavathi-medium");

  const cloud = run(
    initialState(),
    { type: "final-transcript", text: "hi" },
    { type: "reply-delta", text: "Hello." },
    { type: "done", metrics: { stt_engine: "gemini gemini-3.5-flash-lite", tts_engine: "gemini gemini-3.8-flash-lite-tts (Kore)" } },
  );
  assert.equal(cloud.messages[0]?.engine, undefined, "cloud models are not named in the UI");
  assert.equal(cloud.messages[1]?.engine, undefined);

  const textOnly = run(
    initialState(),
    { type: "final-transcript", text: "hi" },
    { type: "reply-delta", text: "Hello." },
    { type: "done", metrics: { tts_fallback: "Text only: quota used up" } },
  );
  assert.equal(textOnly.messages[1]?.engine, "text only");
});

test("an empty final transcript adds nothing and clears the live caption", () => {
  const state = run(initialState(), { type: "live-transcript", text: "uh" }, { type: "final-transcript", text: "" });
  assert.equal(state.messages.length, 0);
  assert.equal(state.liveTranscript, "");
});

test("reply text accumulates in the pending reply only", () => {
  const state = run(
    initialState(),
    { type: "final-transcript", text: "first" },
    { type: "reply-delta", text: "Hello" },
    { type: "reply-delta", text: " there." },
  );
  assert.equal(state.messages[0]?.text, "first");
  assert.equal(state.messages[1]?.text, "Hello there.");
});

test("a reply delta with no pending reply is ignored", () => {
  const before = initialState();
  assert.equal(reduce(before, { type: "reply-delta", text: "stray" }), before);
});

test("done settles the reply, stores the metrics and announces the answer", () => {
  const state = run(
    initialState(),
    { type: "turn-start" },
    { type: "final-transcript", text: "hi" },
    { type: "reply-delta", text: "Hello." },
    { type: "done", metrics: { first_audio_ms: 4900, total_ms: 4912 } },
  );
  assert.equal(state.turnActive, false);
  assert.equal(state.messages[1]?.pending, false);
  assert.equal(state.lastMetrics?.first_audio_ms, 4900);
  assert.equal(state.announcement, "Assistant: Hello.");
});

test("ending a turn that produced no reply text removes the empty bubble", () => {
  const state = run(initialState(), { type: "final-transcript", text: "hi" }, { type: "turn-end" });
  assert.deepEqual(
    state.messages.map((m) => m.role),
    ["user"],
  );
});

test("the replay audio attaches to the latest reply even after the turn has settled", () => {
  const state = run(
    initialState(),
    { type: "final-transcript", text: "hi" },
    { type: "reply-delta", text: "Hello." },
    { type: "done", metrics: {} },
    { type: "replay", src: "blob:reply" },
  );
  assert.equal(state.messages[1]?.replaySrc, "blob:reply");
  assert.equal(state.messages[0]?.replaySrc, null);
});

test("an upload turn adds both messages at once", () => {
  const state = run(initialState("upload"), { type: "upload-turn", transcript: "hi", reply: "Hello.", metrics: { total_ms: 3000 } });
  assert.deepEqual(
    state.messages.map((m) => [m.role, m.text, m.pending]),
    [
      ["user", "hi", false],
      ["assistant", "Hello.", false],
    ],
  );
  assert.equal(state.lastMetrics?.total_ms, 3000);
});

test("starting a turn clears the previous client latency", () => {
  const state = run(
    initialState(),
    { type: "client-latency", latency: { kind: "first-audio", ms: 4200 } },
    { type: "turn-start" },
  );
  assert.equal(state.clientLatency, null);
});

test("a new conversation clears messages and the live caption", () => {
  const state = run(initialState(), { type: "final-transcript", text: "hi" }, { type: "new-conversation" });
  assert.deepEqual(state.messages, []);
});

test("health: ok, loading and a failed component each produce the right pill", () => {
  const ok = run(initialState(), { type: "health", health: READY });
  assert.deepEqual([ok.models.kind, ok.models.text, ok.modelsReady], ["ok", "Models ready", true]);

  const loading = run(initialState(), {
    type: "health",
    health: { status: "starting", components: { stt: { state: "loading", detail: "" }, tts: { state: "loading", detail: "" } } },
  });
  assert.deepEqual([loading.models.kind, loading.modelsReady], ["warn", false]);

  const failed = run(initialState(), {
    type: "health",
    health: { status: "degraded", components: { stt: { state: "ready", detail: "" }, tts: { state: "error", detail: "voice file missing" } } },
  });
  assert.equal(failed.models.kind, "error");
  assert.equal(failed.models.text, "TTS unavailable");
  assert.equal(failed.models.title, "voice file missing");
});

test("an unreachable server makes recording impossible", () => {
  const state = run(initialState(), { type: "health", health: READY }, { type: "connection", connection: { kind: "ok", text: "Connected", title: "" }, wsOpen: true }, { type: "unreachable" });
  assert.equal(state.modelsReady, false);
  assert.equal(canRecord(state), false);
});

test("recording needs ready models and a live link for the chosen transport", () => {
  const ready = run(initialState(), { type: "health", health: READY });
  assert.equal(canRecord(ready), false, "stream mode needs the socket");

  const connected = run(ready, { type: "connection", connection: { kind: "ok", text: "Connected", title: "" }, wsOpen: true });
  assert.equal(canRecord(connected), true);
  assert.equal(canRecord(run(connected, { type: "phase", phase: "thinking" })), false, "busy");
  assert.equal(canRecord(run(connected, { type: "phase", phase: "recording" })), true, "can always stop");

  const upload = run(ready, { type: "transport", transport: "upload" });
  assert.equal(canRecord(upload), true, "upload mode only needs the server to be reachable");
  assert.equal(canRecord(run(initialState(), { type: "transport", transport: "upload" })), false, "models not ready yet");
});

test("the state label explains why nothing can be recorded", () => {
  assert.equal(stateLabel(initialState()), "Waiting for the server");
  const modelsOnly = run(initialState(), { type: "health", health: READY });
  assert.equal(stateLabel(modelsOnly), "Not connected");
  const connected = run(modelsOnly, { type: "connection", connection: { kind: "ok", text: "Connected", title: "" }, wsOpen: true });
  assert.equal(stateLabel(connected), "Ready");
  assert.equal(stateLabel(run(connected, { type: "phase", phase: "thinking" })), "Thinking…");
});

test("the core word restates the real state", () => {
  const word = (state: AgentState) => coreWord(state);
  assert.equal(word(initialState()), "BOOTING");
  const ready = run(initialState(), { type: "health", health: READY });
  assert.equal(word(ready), "LINKING");
  assert.equal(word(run(ready, { type: "connection", connection: { kind: "ok", text: "Connected", title: "" }, wsOpen: true })), "READY");
  assert.equal(word(run(ready, { type: "unreachable" })), "OFFLINE");
  assert.equal(word(run(ready, { type: "phase", phase: "recording" })), "LISTENING");
  assert.equal(word(run(ready, { type: "phase", phase: "thinking" })), "THINKING");
});

test("the cancel button is labelled for what it will do", () => {
  assert.equal(cancelLabel("recording"), "Discard");
  assert.equal(cancelLabel("speaking"), "Stop");
  assert.equal(cancelLabel("thinking"), "Cancel");
});

test("pipeline stage follows the phase", () => {
  assert.deepEqual(
    (["idle", "recording", "transcribing", "thinking", "speaking"] as const).map(activeStage),
    [null, null, "stt", "llm", "tts"],
  );
});

test("the store notifies on change, stays quiet when nothing changed, and can be unsubscribed", () => {
  const store = createStore(reduce, initialState());
  let calls = 0;
  const unsubscribe = store.subscribe(() => calls++);
  store.dispatch({ type: "banner", banner: { text: "x", kind: "error" } });
  assert.equal(calls, 1);
  store.dispatch({ type: "banner-clear" });
  store.dispatch({ type: "banner-clear" }); // already clear: same state object, no notification
  assert.equal(calls, 2);
  unsubscribe();
  store.dispatch({ type: "banner", banner: { text: "y", kind: "warn" } });
  assert.equal(calls, 2);
  assert.equal(store.getState().banner?.text, "y");
});

test("a streamed reply is hidden until it is heard, then shown in full", () => {
  let state = run(initialState(), { type: "final-transcript", text: "Capital of India?" }, { type: "reply-delta", text: "New Delhi is the capital." });
  assert.equal(state.messages[1]?.spoken, "", "generated but not yet heard");
  state = run(state, { type: "spoken", text: "New Delhi" });
  assert.equal(state.messages[1]?.spoken, "New Delhi");
  state = run(state, { type: "done", metrics: {} }, { type: "spoken", text: null });
  assert.equal(state.messages[1]?.text, "New Delhi is the capital.");
  assert.equal("spoken" in (state.messages[1] ?? {}), false);
  assert.equal(run(state, { type: "spoken", text: null }), state, "nothing to change: same state");
});

test("attached files: uploading blocks recording, ready ones are sent, missing ones are dropped", () => {
  const ready = run(initialState(), { type: "health", health: READY }, { type: "connection", connection: { kind: "ok", text: "", title: "" }, wsOpen: true });
  const uploading = run(ready, { type: "file-add", file: { key: "f1", name: "plan.pdf", status: "uploading" } });
  assert.equal(canRecord(uploading), false);
  assert.equal(stateLabel(uploading), "Reading the file…");
  const done = run(uploading, { type: "file-ready", key: "f1", summary: { id: "abc", name: "plan.pdf", chars: 900, pages: 3, truncated: false } });
  assert.equal(canRecord(done), true);
  assert.deepEqual(attachedIds(done), ["abc"]);
  assert.deepEqual(run(done, { type: "file-remove", ids: ["abc"] }).files, []);
});

test("the language toggle offers Auto plus the server's languages, and only sends a valid choice", () => {
  const info = { languages: { choices: ["te", "en"], auto: true } };
  assert.deepEqual(languageOptions(info), ["auto", "te", "en"]);
  assert.deepEqual(languageOptions({ languages: { choices: ["en"], auto: false } }), [], "one fixed language: nothing to choose");
  const state = run(initialState(), { type: "info", info }, { type: "language", language: "te" });
  assert.equal(turnLanguage(state), "te");
  assert.equal(turnLanguage(run(state, { type: "language", language: "fr" })), null, "not offered: detect instead");
});

test("the mode says where questions are answered: local, the server's key, or the user's own key", () => {
  const local = run(initialState(), { type: "info", info: { cloud: { server_key: false, stages: ["llm", "stt", "tts"] } } });
  assert.equal(mode(local), "local");
  assert.equal(modeLabel(local), "Local models");
  assert.deepEqual(cloudStages(local), []);

  const server = run(initialState(), { type: "info", info: { cloud: { server_key: true, stages: ["llm", "stt", "tts"] } } });
  assert.equal(mode(server), "server-key");
  assert.deepEqual(cloudStages(server), ["llm", "stt", "tts"]);

  const own = run(local, { type: "key", key: { saved: true, ok: true, message: "The key works." } });
  assert.equal(mode(own), "own-key");
  assert.deepEqual(cloudStages(own), ["llm", "stt", "tts"]);
  assert.deepEqual(cloudStages(run(own, { type: "configured", cloud: ["llm"] })), ["llm"], "the server's answer wins");
});
