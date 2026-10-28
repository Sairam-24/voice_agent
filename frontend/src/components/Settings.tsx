import { useState } from "react";
import type { VoiceController } from "../controller.ts";
import { type AgentState, type Transport, mode } from "../lib/state.ts";
import { Panel } from "./Panels.tsx";

const TRANSPORT_HELP: Record<Transport, string> = {
  stream: "Audio is sent while you speak, and the reply is spoken sentence by sentence as it is generated.",
  upload: "Audio is recorded, uploaded in one request, and the whole reply is returned at once. Slower to first sound.",
};

const MODE_TAG = { local: "not set", "server-key": "server key", "own-key": "your key" } as const;

export function KeyPanel({ state, controller }: { state: AgentState; controller: VoiceController }) {
  const [draft, setDraft] = useState("");
  const [visible, setVisible] = useState(false);
  const { key } = state;
  const serverKey = state.info?.cloud?.server_key ?? false;
  return (
    <Panel title="Gemini API key" tag={MODE_TAG[mode(state)]} className="key-panel">
      <p className="note">
        {serverKey
          ? "This server has its own key, so questions already use Gemini. Add your own key to use your quota instead."
          : "Without a key, everything runs on this server's local models. With a free Gemini key, transcription, answers and the voice use Gemini: better in many languages, with a natural voice."}
      </p>
      {key.saved ? (
        <div className="key-saved">
          <span className="dot" data-state="ok" />
          <span>Your key is saved in this browser and used for your questions.</span>
          <button id="key-remove" type="button" className="ghost" onClick={() => controller.forgetGeminiKey()}>
            Remove key
          </button>
        </div>
      ) : (
        <form
          className="key-form"
          onSubmit={(event) => {
            event.preventDefault();
            void controller.saveGeminiKey(draft).then((saved) => saved && setDraft(""));
          }}
        >
          <label htmlFor="gemini-key">Your Gemini API key</label>
          <div className="key-row">
            <input
              id="gemini-key"
              type={visible ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              placeholder="AIza…"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
            <button type="button" className="ghost" aria-pressed={visible} onClick={() => setVisible((v) => !v)}>
              {visible ? "Hide" : "Show"}
            </button>
            <button id="key-save" type="submit" className="ghost" disabled={!draft.trim() || key.checking}>
              {key.checking ? "Checking…" : "Save"}
            </button>
          </div>
        </form>
      )}
      {key.message && (
        <p className="key-message" data-ok={key.ok === null ? undefined : String(key.ok)} role="status">
          {key.message}
        </p>
      )}
      <p className="note">
        Get a free key at{" "}
        <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">
          aistudio.google.com/apikey
        </a>
        . It stays in this browser and is sent to this server only to call Gemini for you; the server does not store or log it.
        With a key, your voice, the conversation and attached files go to Google, and on the free tier Google may use them to
        improve its products.
      </p>
    </Panel>
  );
}

export function ConnectionPanel({ state, controller }: { state: AgentState; controller: VoiceController }) {
  return (
    <Panel title="Connection" tag={state.transport === "stream" ? "WebSocket" : "HTTP"}>
      <div className="segmented" role="radiogroup" aria-label="Audio transport">
        {(["stream", "upload"] as const).map((transport) => (
          <button
            key={transport}
            id={`transport-${transport}`}
            type="button"
            role="radio"
            aria-checked={state.transport === transport}
            onClick={() => controller.setTransport(transport)}
          >
            {transport === "stream" ? "Stream" : "Upload"}
          </button>
        ))}
      </div>
      <p className="note">{TRANSPORT_HELP[state.transport]}</p>
    </Panel>
  );
}
