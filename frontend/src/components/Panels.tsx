import { AnimatePresence, motion } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { formatMs } from "../lib/format.ts";
import type { StageKey, TurnMetrics } from "../lib/protocol.ts";
import { sessionModels } from "../lib/graph.ts";
import { type AgentState, type Stage, activeStage, cloudStages } from "../lib/state.ts";

export function Panel({ title, tag, className, children }: { title: string; tag?: string; className?: string; children: ReactNode }) {
  return (
    <motion.section
      className={className ? `panel ${className}` : "panel"}
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div className="panel-head">
        <h2>{title}</h2>
        {tag && <span>{tag}</span>}
      </div>
      {children}
    </motion.section>
  );
}

// ---------------------------------------------------------------------------- system

const COMPONENTS = [
  { key: "stt", label: "STT" },
  { key: "llm", label: "LLM" },
  { key: "tts", label: "TTS" },
] as const;


const STATE_KIND: Record<string, "ok" | "warn" | "error"> = { ready: "ok", loading: "warn", error: "error" };

export function SystemPanel({ state }: { state: AgentState }) {
  const models = sessionModels(state);
  // With the user's own key a cloud stage does not depend on the server's default model, and the key
  // was checked when it was saved.
  const ownCloud = state.key.saved ? cloudStages(state) : [];
  return (
    <Panel title="System" tag="/health">
      {COMPONENTS.map(({ key, label }) => {
        const health = state.health?.components[key];
        const component = ownCloud.includes(key) && health?.state === "error" ? { state: "ready", detail: "" } : health;
        const kind = component ? STATE_KIND[component.state] : undefined;
        return (
          <div className="sys-row" key={key}>
            <span className="label">{label}</span>
            <span className="state" data-state={kind}>
              <span className="dot" data-state={kind} />
              {component ? component.state : state.serverReachable ? "…" : "offline"}
            </span>
            <span className="model">
              {(models[key].length ? models[key] : ["—"]).map((name) => (
                <span key={name} className={name.endsWith("(fallback)") ? "fallback" : undefined}>
                  {name}
                </span>
              ))}
            </span>
            {component?.state === "error" && <span className="detail">{component.detail}</span>}
          </div>
        );
      })}
    </Panel>
  );
}

// ---------------------------------------------------------------------------- pipeline

const STAGE_ROWS: { stage: Stage; label: string; key: StageKey }[] = [
  { stage: "stt", label: "STT", key: "stt_ms" },
  { stage: "llm", label: "LLM", key: "llm_ms" },
  { stage: "tts", label: "TTS", key: "tts_ms" },
];

function latencyText(state: AgentState): string {
  const latency = state.clientLatency;
  if (!latency) return "";
  const label = latency.kind === "first-audio" ? "Stop → first audio" : "Stop → response received";
  return `${label} (measured in this browser): ${formatMs(latency.ms)}`;
}

export function PipelinePanel({ state }: { state: AgentState }) {
  const metrics: TurnMetrics | null = state.lastMetrics;
  const total = metrics?.total_ms ?? null;
  const active = activeStage(state.phase);
  return (
    <Panel title="Pipeline" tag="last turn">
      <div className="hero">
        <span className="caption">First audio</span>
        <span className="value">{formatMs(metrics?.first_audio_ms)}</span>
      </div>
      {STAGE_ROWS.map(({ stage, label, key }) => {
        const ms = metrics?.[key] ?? null;
        const share = ms !== null && total ? Math.max(0, Math.min(100, (ms / total) * 100)) : 0;
        const isActive = active === stage;
        return (
          <div className="stage-row" key={stage} data-active={isActive}>
            <span className="name">{label}</span>
            <span className="bar" data-active={isActive}>
              <motion.i initial={false} animate={{ width: `${share}%` }} transition={{ type: "spring", stiffness: 120, damping: 20 }} />
            </span>
            <span className="value">{formatMs(ms)}</span>
          </div>
        );
      })}
      <div className="stage-row">
        <span className="name">ALL</span>
        <span className="bar">
          <motion.i initial={false} animate={{ width: total ? "100%" : "0%" }} transition={{ type: "spring", stiffness: 120, damping: 20 }} />
        </span>
        <span className="value">{formatMs(total)}</span>
      </div>
      <p className="note">Bars show each stage as a share of the turn total. They overlap when streaming, so they can add up to more than 100%.</p>
      <p id="client-latency" className="client-latency">
        {latencyText(state)}
      </p>
    </Panel>
  );
}

// ---------------------------------------------------------------------------- diagnostics

const DIAG_ROWS: [label: string, key: StageKey, emphasise?: boolean][] = [
  ["Upload (HTTP only)", "upload_ms"],
  ["Decode (HTTP only)", "decode_ms"],
  ["STT queue", "stt_queue_ms"],
  ["STT", "stt_ms"],
  ["LLM queue", "llm_queue_ms"],
  ["LLM first token", "llm_ttft_ms"],
  ["LLM total", "llm_ms"],
  ["TTS queue", "tts_queue_ms"],
  ["TTS first audio", "tts_first_chunk_ms"],
  ["TTS total", "tts_ms"],
  ["First audio ready", "first_audio_ms", true],
  ["Turn total", "total_ms", true],
];

export function DiagnosticsPanel({ state }: { state: AgentState }) {
  const [open, setOpen] = useState(true);
  const stages = state.summary?.stages_ms ?? {};
  const counts = Object.entries(state.summary?.turns_since_start ?? {});
  return (
    <motion.section className="panel" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}>
      <button type="button" className="diag-toggle" aria-expanded={open} aria-controls="diag" onClick={() => setOpen((v) => !v)}>
        <span>Diagnostics</span>
        <span className="mono dim">{open ? "−" : "+"}</span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div id="diag" initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} style={{ overflow: "hidden" }}>
            <table className="diag-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Last</th>
                  <th>Median</th>
                </tr>
              </thead>
              <tbody id="diag-body">
                {DIAG_ROWS.map(([label, key, emphasise]) => {
                  const summary = stages[key];
                  return (
                    <tr key={key} className={emphasise ? "key" : undefined}>
                      <td>{label}</td>
                      <td>{formatMs(state.lastMetrics?.[key])}</td>
                      <td>{summary ? `${formatMs(summary.median)} (n=${summary.n})` : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {counts.length > 0 && (
              <p className="note mono">Turns handled by this server: {counts.map(([k, v]) => `${k} ${v}`).join(", ")}</p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  );
}

// ---------------------------------------------------------------------------- conversation

export function ConversationPanel({ state }: { state: AgentState }) {
  const list = useRef<HTMLOListElement>(null);
  // Keep the newest text in view while the reply streams in.
  useEffect(() => {
    list.current?.scrollTo({ top: list.current.scrollHeight });
  }, [state.messages]);

  return (
    <Panel title="Conversation" tag={`${state.messages.length} msg`} className="chat-panel">
      <ol id="messages" className="messages" ref={list} aria-label="Conversation">
        {state.messages.length === 0 && <li className="empty">Press the microphone, or hit Space, and ask something.</li>}
        <AnimatePresence initial={false}>
          {state.messages.map((message) => (
            <motion.li
              key={message.id}
              className={`msg ${message.role}${message.pending || message.spoken !== undefined ? " pending" : ""}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25 }}
            >
              <span className="role">
                {message.role === "user" ? "You" : "Assistant"}
                {message.language && <span className="lang">{message.language}</span>}
              </span>
              {/* Plain text on purpose: model output is untrusted and must never be rendered as HTML. */}
              {/* While a reply is spoken, only the part already heard is shown (lib/speech.ts). */}
              <p>{message.spoken ?? message.text}</p>
              {message.engine && (
                <span className="meta">{message.role === "user" ? `heard by ${message.engine}` : `voice: ${message.engine}`}</span>
              )}
              {message.replaySrc && message.spoken === undefined && (
                <audio controls preload="none" src={message.replaySrc} aria-label="Replay this response" />
              )}
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
    </Panel>
  );
}
