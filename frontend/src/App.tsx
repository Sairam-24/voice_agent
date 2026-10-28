import { MotionConfig } from "motion/react";
import { type DragEvent, useEffect, useState } from "react";
import { Banner, Backdrop, type Page, TopBar } from "./components/Chrome.tsx";
import { Controls, Readout } from "./components/Controls.tsx";
import { Core } from "./components/Core.tsx";
import { GraphView } from "./components/GraphView.tsx";
import { ConversationPanel, DiagnosticsPanel, PipelinePanel, SystemPanel } from "./components/Panels.tsx";
import { ConnectionPanel, KeyPanel } from "./components/Settings.tsx";
import { useAgent } from "./hooks/useAgent.ts";
import { canRecord, coreWord } from "./lib/state.ts";

/** Space and Escape should not hijack a focused control: a focused button already reacts to Space itself. */
function isInteractive(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && target.closest("button, a, input, select, textarea, audio, [role='radio']") !== null;
}

type View = "core" | "graph";

const carriesFiles = (event: DragEvent) => event.dataTransfer.types.includes("Files");

/** The page comes from the URL hash (#/ or #/system), so links, reloads and the back button all work. */
const pageFromHash = (): Page => (location.hash === "#/system" ? "system" : "home");

function savedView(): View {
  try {
    return localStorage.getItem("view") === "graph" ? "graph" : "core";
  } catch {
    return "core"; // storage can be blocked; the default is fine
  }
}

export function App() {
  const { controller, state } = useAgent();
  const [view, setView] = useState<View>(savedView);
  const [page, setPage] = useState<Page>(pageFromHash);
  const [dropping, setDropping] = useState(false);
  const chooseView = (next: View) => {
    setView(next);
    try {
      localStorage.setItem("view", next);
    } catch {
      /* optional */
    }
  };

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (page !== "home") return; // the talk shortcuts belong to the assistant page
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") controller.cancel();
      else if (event.code === "Space" && !event.repeat && !isInteractive(event.target)) {
        event.preventDefault();
        controller.toggleRecording();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [controller, page]);

  const alert = state.banner?.kind === "error" && state.phase === "idle";
  const touch = window.matchMedia("(pointer: coarse)").matches; // phones and tablets have no space bar
  const sub = state.detail || (state.phase === "idle" && canRecord(state) ? (touch ? "TAP THE MIC TO TALK" : "PRESS SPACE TO TALK") : "");

  return (
    <MotionConfig reducedMotion="user">
      <div
        className="hud"
        data-phase={state.phase}
        data-alert={alert ? "error" : undefined}
        onDragOver={(event) => {
          if (!carriesFiles(event)) return;
          event.preventDefault(); // allows the drop
          setDropping(true);
        }}
        onDragLeave={(event) => {
          if (event.relatedTarget === null) setDropping(false); // left the window
        }}
        onDrop={(event) => {
          if (!carriesFiles(event)) return;
          event.preventDefault(); // otherwise the browser opens the file
          setDropping(false);
          controller.attachFiles([...event.dataTransfer.files]);
        }}
      >
        <Backdrop />
        <TopBar state={state} page={page} />
        {page === "home" ? (
          <main className="home">
            <section className="stage" aria-label="Voice control">
              <div className="segmented view-switch" role="radiogroup" aria-label="Visual">
                {(["core", "graph"] as const).map((option) => (
                  <button key={option} id={`view-${option}`} type="button" role="radio" aria-checked={view === option} onClick={() => chooseView(option)}>
                    {option === "core" ? "Core" : "Graph"}
                  </button>
                ))}
              </div>
              {view === "core" ? (
                <Core phase={state.phase} word={coreWord(state)} sub={sub} tap={controller.tap} />
              ) : (
                <GraphView state={state} />
              )}
              <Readout state={state} />
              <Controls state={state} controller={controller} />
            </section>
            <section className="chat" aria-label="Conversation">
              <ConversationPanel state={state} />
            </section>
          </main>
        ) : (
          <main className="system-page">
            <div className="system-col">
              <KeyPanel state={state} controller={controller} />
              <ConnectionPanel state={state} controller={controller} />
            </div>
            <div className="system-col">
              <SystemPanel state={state} />
              <PipelinePanel state={state} />
            </div>
            <div className="system-col">
              <DiagnosticsPanel state={state} />
            </div>
          </main>
        )}
        <Banner state={state} onDismiss={() => controller.dismissBanner()} />
        {dropping && (
          <div className="drop-overlay" aria-hidden="true">
            <span>Drop to attach: ask about it by voice</span>
          </div>
        )}
        <div id="announcer" className="sr-only" aria-live="polite">
          {state.announcement}
        </div>
      </div>
    </MotionConfig>
  );
}
