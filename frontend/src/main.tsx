import "@fontsource/orbitron/latin-500.css";
import "@fontsource/orbitron/latin-600.css";
import "@fontsource/rajdhani/latin-500.css";
import "@fontsource/rajdhani/latin-600.css";
import "@fontsource/share-tech-mono/latin-400.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.tsx";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
