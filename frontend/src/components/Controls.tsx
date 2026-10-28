import { AnimatePresence, motion } from "motion/react";
import { useRef } from "react";
import type { VoiceController } from "../controller.ts";
import { describeFile, languageLabel } from "../lib/format.ts";
import { type AgentState, canRecord, cancelLabel, languageOptions, stateLabel } from "../lib/state.ts";

/** What the file picker offers; the server decides what it can actually read. */
const ACCEPTED_FILES = ".pdf,.docx,.txt,.md,.markdown,.csv,.tsv,.json,.html,.htm,.xml,.yaml,.yml,.log";

function MicIcon({ recording }: { recording: boolean }) {
  return recording ? (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M8.5 21h7" />
    </svg>
  );
}

function PaperclipIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 11.5 12.6 19.9a5.5 5.5 0 0 1-7.8-7.8l8.5-8.5a3.7 3.7 0 0 1 5.2 5.2l-8.5 8.5a1.8 1.8 0 0 1-2.6-2.6l7.8-7.8" />
    </svg>
  );
}

export function Readout({ state }: { state: AgentState }) {
  return (
    <div className="readout">
      <h2 id="state-text" className="state-text">
        {stateLabel(state)}
      </h2>
      <p className="live" id="live-transcript">
        {state.liveTranscript}
      </p>
    </div>
  );
}

function LanguageSwitch({ state, controller }: { state: AgentState; controller: VoiceController }) {
  const options = languageOptions(state.info);
  if (!options.length) return null;
  const current = options.includes(state.language) ? state.language : options[0];
  return (
    <div className="segmented lang-switch" role="radiogroup" aria-label="Language you speak and want replies in">
      {options.map((code) => (
        <button
          key={code}
          id={`language-${code}`}
          type="button"
          role="radio"
          lang={code === "auto" ? undefined : code}
          aria-checked={current === code}
          title={code === "auto" ? "Detect the language of each question" : `Speak and get replies in ${languageLabel(code)}`}
          onClick={() => controller.setLanguage(code)}
        >
          {languageLabel(code)}
        </button>
      ))}
    </div>
  );
}

function Files({ state, controller }: { state: AgentState; controller: VoiceController }) {
  const input = useRef<HTMLInputElement>(null);
  const max = state.info?.documents?.max_per_turn ?? 8;
  return (
    <div className="files">
      <input
        ref={input}
        id="file-input"
        type="file"
        multiple
        hidden
        accept={ACCEPTED_FILES}
        onChange={(event) => {
          if (event.target.files) controller.attachFiles([...event.target.files]);
          event.target.value = ""; // so the same file can be picked again after removing it
        }}
      />
      <button
        id="attach"
        type="button"
        className="ghost attach"
        disabled={state.files.length >= max}
        title="PDF, Word or text files. Ask questions about them by voice. You can also drop files anywhere on the page."
        onClick={() => input.current?.click()}
      >
        <PaperclipIcon />
        Attach file
      </button>
      <AnimatePresence initial={false}>
        {state.files.map((file) => (
          <motion.span
            key={file.key}
            className="file-chip"
            data-status={file.status}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.18 }}
          >
            <span className="file-name" title={file.name}>
              {file.name}
            </span>
            <span className="file-size">{file.status === "uploading" ? "reading…" : describeFile(file)}</span>
            <button type="button" aria-label={`Remove ${file.name}`} onClick={() => controller.removeFile(file.key)}>
              ×
            </button>
          </motion.span>
        ))}
      </AnimatePresence>
    </div>
  );
}

export function Controls({ state, controller }: { state: AgentState; controller: VoiceController }) {
  const recording = state.phase === "recording";
  const idle = state.phase === "idle";
  return (
    <>
      <div className="controls">
        <button
          id="cancel"
          type="button"
          className="ghost"
          onClick={() => controller.cancel()}
          hidden={idle}
        >
          {cancelLabel(state.phase)}
        </button>

        <div className="mic-wrap">
          {recording && (
            <motion.span
              className="mic-ring"
              initial={{ scale: 1, opacity: 0.6 }}
              animate={{ scale: 1.38, opacity: 0 }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
            />
          )}
          <motion.button
            id="mic"
            type="button"
            className="mic"
            data-phase={state.phase}
            disabled={!canRecord(state)}
            aria-pressed={recording}
            aria-label={recording ? "Stop recording and send" : "Start recording"}
            onClick={() => controller.toggleRecording()}
            whileTap={{ scale: 0.94 }}
          >
            <MicIcon recording={recording} />
          </motion.button>
        </div>

        <button id="reset" type="button" className="ghost" onClick={() => controller.resetConversation()}>
          New chat
        </button>
      </div>

      <LanguageSwitch state={state} controller={controller} />

      <Files state={state} controller={controller} />

      <p className="hint">
        {state.files.length > 0 ? "Questions are answered from the attached files." : "Ask anything, or attach a file and ask about it."}
        <span className="kbd-hint">
          {" "}
          <kbd>Space</kbd> starts and sends, <kbd>Esc</kbd> cancels.
        </span>
      </p>
    </>
  );
}
