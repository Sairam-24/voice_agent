import assert from "node:assert/strict";
import { test } from "node:test";
import { activity, buildGraph, compactModelName, MAX_MESSAGE_NODES, sessionModels, stageModels } from "./graph.ts";
import type { ServerInfo } from "./protocol.ts";
import { initialState, type Message } from "./state.ts";

const GEMINI_WITH_LOCAL_FALLBACKS: ServerInfo = {
  components: {
    stt: { engine: "gemini", model: "gemini-3.5-flash-lite", fallback: { engine: "faster-whisper", model: "base" } },
    llm: { provider: "gemini", model: "gemini-3.5-flash-lite" },
    tts: {
      engine: "gemini",
      model: "gemini-3.8-flash-lite-tts",
      voice: "Kore",
      fallback: { engine: "local", voices: { latin: { engine: "kokoro", voice: "af_heart" }, telugu: { engine: "piper", voice: "te_IN-padmavathi-medium" } } },
    },
  },
};

const message = (id: number, role: "user" | "assistant", text: string, language?: string): Message => ({
  id,
  role,
  text,
  pending: false,
  replaySrc: null,
  ...(language ? { language } : {}),
});

test("every stage lists its models: cloud ones as 'cloud', local ones by name, fallbacks marked", () => {
  assert.deepEqual(stageModels(GEMINI_WITH_LOCAL_FALLBACKS), {
    stt: ["cloud", "faster-whisper base (fallback)"],
    llm: ["cloud"],
    tts: ["cloud", "kokoro af_heart (fallback)", "piper te_IN-padmavathi-medium (fallback)"],
  });
  assert.ok(!JSON.stringify(buildGraph({ info: GEMINI_WITH_LOCAL_FALLBACKS, messages: [] })).includes("gemini"));
});

test("local voices are listed directly when they are the main voice", () => {
  const info: ServerInfo = { components: { tts: { engine: "local", voices: { latin: { engine: "piper", voice: "en_US-ljspeech-medium" } } } } };
  assert.deepEqual(stageModels(info).tts, ["piper en_US-ljspeech-medium"]);
});

test("model names are shortened for the graph without losing the model", () => {
  assert.equal(compactModelName("faster-whisper base (fallback)"), "whisper base");
  assert.equal(compactModelName("piper te_IN-padmavathi-medium (fallback)"), "piper te_IN-padmavathi-medium");
});

test("before /api/info arrives the graph has only the pipeline", () => {
  const graph = buildGraph({ info: null, messages: [] });
  assert.deepEqual(
    graph.nodes.map((n) => n.id),
    ["mic", "stt", "llm", "tts", "out"],
  );
  assert.deepEqual(graph.links, [
    ["mic", "stt"],
    ["stt", "llm"],
    ["llm", "tts"],
    ["tts", "out"],
  ]);
});

test("languages and messages become nodes linked into the pipeline", () => {
  const graph = buildGraph({
    info: GEMINI_WITH_LOCAL_FALLBACKS,
    messages: [message(1, "user", "భారతదేశ రాజధాని ఏమిటి?", "te"), message(2, "assistant", "న్యూఢిల్లీ."), message(3, "user", "Thanks", "en")],
  });
  const ids = graph.nodes.map((n) => n.id);
  assert.ok(ids.includes("lang:te") && ids.includes("lang:en"));
  assert.equal(graph.nodes.filter((n) => n.kind === "model").length, 6);
  assert.ok(graph.links.some(([a, b]) => a === "msg:1" && b === "lang:te"));
  assert.ok(graph.links.some(([a, b]) => a === "msg:1" && b === "msg:2"), "a reply is linked to the question it answers");
  assert.ok(graph.links.some(([a, b]) => a === "msg:2" && b === "tts"));
  const ids2 = new Set(ids);
  assert.ok(graph.links.every(([a, b]) => ids2.has(a) && ids2.has(b)), "no link points at a missing node");
});

test("a reply being spoken is labelled with what has been heard so far", () => {
  const speaking = { ...message(2, "assistant", "Hyderabad is the capital."), spoken: "Hyderabad" };
  const graph = buildGraph({ info: null, messages: [message(1, "user", "Capital?"), speaking] });
  assert.equal(graph.nodes.find((n) => n.id === "msg:2")?.label, "Hyderabad");
});

test("only the most recent messages are drawn", () => {
  const messages = Array.from({ length: 25 }, (_, i) => message(i + 1, i % 2 ? "assistant" : "user", `m${i}`, "en"));
  const shown = buildGraph({ info: null, messages }).nodes.filter((n) => n.kind === "user" || n.kind === "assistant");
  assert.equal(shown.length, MAX_MESSAGE_NODES);
  assert.equal(shown.at(-1)?.id, "msg:25");
});

test("data flows along the links of the stage that is working", () => {
  assert.deepEqual(activity("idle"), { links: [], focus: null });
  assert.deepEqual(activity("thinking"), { links: [["stt", "llm"]], focus: "llm" });
  assert.deepEqual(activity("speaking").links, [
    ["llm", "tts"],
    ["tts", "out"],
  ]);
});

test("with the user's own key, the stages it moves to the cloud show the local models as fallbacks", () => {
  const info: ServerInfo = {
    components: { stt: { engine: "faster-whisper", model: "base" }, llm: { provider: "ollama", model: "llama3.2:1b" }, tts: { engine: "local", voices: { latin: { engine: "kokoro", voice: "af_heart" } } } },
    cloud: { server_key: false, stages: ["llm", "stt", "tts"] },
  };
  const state = { ...initialState(), info };
  assert.deepEqual(sessionModels(state).llm, ["ollama llama3.2:1b"], "no key: the local model");
  const own = { ...state, key: { saved: true, checking: false, message: "", ok: true } };
  assert.deepEqual(sessionModels(own), {
    stt: ["cloud", "faster-whisper base (fallback)"],
    llm: ["cloud", "ollama llama3.2:1b (fallback)"],
    tts: ["cloud", "kokoro af_heart (fallback)"],
  });
});
