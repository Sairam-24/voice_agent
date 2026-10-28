// The graph view's model: which nodes exist and how they connect, built only from real state
// (the server's /api/info, the conversation, the current phase). Pure, so Node can test it.
import type { ServerInfo, TtsInfo } from "./protocol.ts";
import { type AgentState, cloudStages, type Phase } from "./state.ts";

export type NodeKind = "io" | "stage" | "model" | "language" | "user" | "assistant";

export interface GraphNodeSpec {
  id: string;
  kind: NodeKind;
  label: string;
  sub?: string;
  /** Fixed position in world space; nodes without one are placed by the force simulation. */
  anchor?: readonly [number, number, number];
}

export interface GraphSpec {
  nodes: GraphNodeSpec[];
  links: [string, string][];
}

export const RING_RADIUS = 230;
export const BACKBONE = ["mic", "stt", "llm", "tts", "out"] as const;
export type BackboneId = (typeof BACKBONE)[number];
const BACKBONE_LABEL: Record<BackboneId, string> = { mic: "MIC", stt: "STT", llm: "LLM", tts: "TTS", out: "SPEAKER" };
export const MAX_MESSAGE_NODES = 10;

export function backboneAngle(id: BackboneId): number {
  return (BACKBONE.indexOf(id) / BACKBONE.length) * Math.PI * 2;
}

/** Cloud (Gemini) models are shown as "cloud", not by name; local models are named. */
export const CLOUD = "cloud";

function modelName(engine: string | undefined, ...details: (string | undefined)[]): string {
  return engine === "gemini" ? CLOUD : [engine, ...details].filter(Boolean).join(" ");
}

function voiceName(tts: TtsInfo): string {
  return modelName(tts.engine, tts.model, tts.voice);
}

/** Names of the models behind each stage, primary first, fallbacks marked. */
export function stageModels(info: ServerInfo | null): Record<"stt" | "llm" | "tts", string[]> {
  const c = info?.components;
  const stt = c?.stt;
  const llm = c?.llm;
  const tts = c?.tts;
  const localVoices = (t: TtsInfo | undefined) => Object.values(t?.voices ?? {}).map(voiceName);
  const sttNames = stt ? [modelName(stt.engine, stt.model)] : [];
  if (stt?.fallback) sttNames.push(`${modelName(stt.fallback.engine, stt.fallback.model)} (fallback)`);
  let ttsNames: string[] = [];
  if (tts?.engine === "local") ttsNames = localVoices(tts);
  else if (tts) ttsNames = [voiceName(tts), ...localVoices(tts.fallback).map((name) => `${name} (fallback)`)];
  return {
    stt: sttNames,
    llm: llm ? [modelName(llm.provider, llm.model)] : [],
    tts: ttsNames,
  };
}

/** The models behind each stage for this browser's questions. With the user's own key, the stages it
 * moves to the cloud say "cloud", and the server's local models become their fallbacks. */
export function sessionModels(state: AgentState): Record<"stt" | "llm" | "tts", string[]> {
  const models = stageModels(state.info);
  if (!state.key.saved) return models;
  for (const stage of cloudStages(state)) {
    if (stage !== "stt" && stage !== "llm" && stage !== "tts") continue;
    const current = models[stage];
    if (current[0] !== CLOUD) models[stage] = [CLOUD, ...current.map((name) => `${name} (fallback)`)];
  }
  return models;
}

/** "faster-whisper base (fallback)" -> "whisper base": the graph has less room than the panels. */
export function compactModelName(name: string): string {
  return name.replace(" (fallback)", "").replace(/^faster-whisper /, "whisper ");
}

function shorten(text: string, max = 22): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max - 1)}…` : clean;
}

export function buildGraph(
  state: Pick<AgentState, "info" | "messages">,
  models: Record<"stt" | "llm" | "tts", string[]> = stageModels(state.info),
): GraphSpec {
  const nodes: GraphNodeSpec[] = [];
  const links: [string, string][] = [];

  BACKBONE.forEach((id, index) => {
    const angle = backboneAngle(id);
    nodes.push({
      id,
      kind: id === "mic" || id === "out" ? "io" : "stage",
      label: BACKBONE_LABEL[id],
      anchor: [Math.cos(angle) * RING_RADIUS, 0, Math.sin(angle) * RING_RADIUS],
    });
    if (index > 0) links.push([BACKBONE[index - 1] as BackboneId, id]);
  });

  for (const stage of ["stt", "llm", "tts"] as const) {
    const angle = backboneAngle(stage);
    models[stage].forEach((name, i, all) => {
      const spread = (i - (all.length - 1) / 2) * 0.32;
      const radius = RING_RADIUS * 1.08;
      const id = `model:${stage}:${i}`;
      nodes.push({
        id,
        kind: "model",
        label: compactModelName(name),
        ...(name.endsWith("(fallback)") ? { sub: "fallback" } : {}),
        anchor: [Math.cos(angle + spread) * radius, -100 - i * 14, Math.sin(angle + spread) * radius],
      });
      links.push([stage, id]);
    });
  }

  const languages = [...new Set(state.messages.map((m) => m.language).filter((l): l is string => Boolean(l)))];
  for (const code of languages) {
    nodes.push({ id: `lang:${code}`, kind: "language", label: code.toUpperCase() });
    links.push([`lang:${code}`, "stt"], [`lang:${code}`, "llm"]);
  }

  const recent = state.messages.slice(-MAX_MESSAGE_NODES);
  recent.forEach((message, i) => {
    const id = `msg:${message.id}`;
    // A reply being spoken shows only what has been heard, as in the transcript.
    nodes.push({ id, kind: message.role, label: shorten((message.spoken ?? message.text) || "…") });
    if (message.role === "user") links.push([id, message.language ? `lang:${message.language}` : "stt"]);
    else {
      links.push([id, "tts"]);
      const previous = recent[i - 1];
      if (previous) links.push([`msg:${previous.id}`, id]);
    }
  });

  return { nodes, links };
}

/** The pipeline links carrying data in this phase, and the node the camera should travel to. */
export function activity(phase: Phase): { links: [BackboneId, BackboneId][]; focus: BackboneId | null } {
  switch (phase) {
    case "recording":
      return { links: [["mic", "stt"]], focus: "mic" };
    case "transcribing":
      return { links: [["mic", "stt"]], focus: "stt" };
    case "thinking":
      return { links: [["stt", "llm"]], focus: "llm" };
    case "speaking":
      return { links: [["llm", "tts"], ["tts", "out"]], focus: "tts" };
    default:
      return { links: [], focus: null };
  }
}
