import { AnimatePresence, motion } from "motion/react";
import { type AgentState, mode, modeLabel } from "../lib/state.ts";

export type Page = "home" | "system";

export function Backdrop() {
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="grid-lines" />
      <div className="sweep" />
      <div className="scan" />
      <div className="vignette" />
    </div>
  );
}

export function TopBar({ state, page }: { state: AgentState; page: Page }) {
  const { connection, models } = state;
  return (
    <header className="topbar">
      <div className="brand">
        <svg className="brand-mark" viewBox="0 0 34 34" fill="none" stroke="currentColor" aria-hidden="true">
          <circle cx="17" cy="17" r="15" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.7" />
          <circle cx="17" cy="17" r="10" strokeWidth="2" />
          <circle cx="17" cy="17" r="3.5" fill="currentColor" stroke="none" />
        </svg>
        <div>
          <h1>Voice Agent</h1>
          <p>Speech in · answer · voice out</p>
        </div>
      </div>
      <nav className="nav" aria-label="Pages">
        <a href="#/" id="nav-home" aria-current={page === "home" ? "page" : undefined}>
          Assistant
        </a>
        <a href="#/system" id="nav-system" aria-current={page === "system" ? "page" : undefined}>
          System
        </a>
      </nav>
      <div className="pills">
        <a
          id="mode-pill"
          className="pill"
          href="#/system"
          data-state={mode(state) === "local" ? "warn" : "ok"}
          title={mode(state) === "local" ? "Running on local models. Add a Gemini key on the System page for the cloud models." : "Using Gemini in the cloud. Change it on the System page."}
        >
          <span className="dot" />
          <span>{modeLabel(state)}</span>
        </a>
        <span id="conn-pill" className="pill" data-state={connection.kind} title={connection.title}>
          <span className="dot" />
          <span id="conn-text">{connection.text}</span>
        </span>
        <span id="models-pill" className="pill" data-state={models.kind} title={models.title}>
          <span className="dot" />
          <span id="models-text">{models.text}</span>
        </span>
      </div>
    </header>
  );
}

export function Banner({ state, onDismiss }: { state: AgentState; onDismiss: () => void }) {
  const banner = state.banner;
  return (
    <AnimatePresence>
      {banner && (
        <motion.div
          id="banner"
          className="banner"
          role="alert"
          data-kind={banner.kind}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 24 }}
          transition={{ duration: 0.22 }}
        >
          <p id="banner-text">{banner.text}</p>
          <button id="banner-dismiss" type="button" onClick={onDismiss} aria-label="Dismiss message">
            Dismiss
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
