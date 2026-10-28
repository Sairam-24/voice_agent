import { useAnimationFrame, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef } from "react";
import { activity, BACKBONE, type BackboneId, backboneAngle, buildGraph, type GraphNodeSpec, type NodeKind, sessionModels } from "../lib/graph.ts";
import type { AgentState } from "../lib/state.ts";

// A pseudo-3D graph drawn on a canvas: the pipeline as a ring, the models above each stage, the
// languages and messages of this conversation floating below. The camera travels to whichever
// stage is working and data visibly flows along the links between them. Drag to look around.

interface SimNode extends GraphNodeSpec {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}

type Tone = "accent" | "text" | "blue" | "dim";
const STYLE: Record<NodeKind, { radius: number; tone: Tone }> = {
  io: { radius: 10, tone: "text" },
  stage: { radius: 16, tone: "accent" },
  model: { radius: 6, tone: "dim" },
  language: { radius: 12, tone: "blue" },
  user: { radius: 6, tone: "blue" },
  assistant: { radius: 6, tone: "accent" },
};
const DISTANCE = 820; // camera distance: perspective strength
const TAU = Math.PI * 2;

// Fixed "stars" so the space has depth; seeded so they do not jump between renders.
const STARS = (() => {
  let seed = 7;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  return Array.from({ length: 160 }, () => [random() * 900, random() * 600, random() * 900] as const);
})();

const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

export function GraphView({ state }: { state: AgentState }) {
  const reduced = useReducedMotion() ?? false;
  const models = useMemo(() => sessionModels(state), [state.info, state.key.saved, state.sessionCloud]);
  const spec = useMemo(() => buildGraph({ info: state.info, messages: state.messages }, models), [state.info, state.messages, models]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodes = useRef(new Map<string, SimNode>());
  const camera = useRef({ yaw: 0.9, pitch: -0.42, zoom: 1, fx: 0, fy: 0, fz: 0, userUntil: 0 });
  const colors = useRef({ accent: "#38e8ff", text: "#d8f7ff", blue: "#6aa6ff", dim: "#8bbccb", frame: 0 });
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);
  const live = useRef({ phase: state.phase, spec });
  live.current = { phase: state.phase, spec };

  // Keep simulated nodes in step with the graph; new ones appear next to what they link to.
  useEffect(() => {
    const map = nodes.current;
    const wanted = new Set(spec.nodes.map((n) => n.id));
    for (const id of [...map.keys()]) if (!wanted.has(id)) map.delete(id);
    spec.nodes.forEach((node, i) => {
      const existing = map.get(node.id);
      if (existing) {
        Object.assign(existing, node);
        return;
      }
      const link = spec.links.find(([a, b]) => a === node.id || b === node.id);
      const other = link ? map.get(link[0] === node.id ? link[1] : link[0]) : undefined;
      const jitter = (k: number) => Math.sin(i * 12.9898 + k * 78.233) * 40;
      const [x, y, z] = node.anchor ?? [(other?.x ?? 0) + jitter(1), (other?.y ?? 0) + 70 + jitter(2), (other?.z ?? 0) + jitter(3)];
      map.set(node.id, { ...node, x, y, z, vx: 0, vy: 0, vz: 0 });
    });
  }, [spec]);

  // Match the canvas backing store to its size on screen.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
      const width = Math.round(canvas.clientWidth * ratio);
      const height = Math.round(canvas.clientHeight * ratio);
      if (width && height && (canvas.width !== width || canvas.height !== height)) {
        canvas.width = width;
        canvas.height = height;
      }
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useAnimationFrame((timeMs, deltaMs) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !canvas.width) return;
    const dt = Math.min(deltaMs, 50) / 1000;
    const t = timeMs / 1000;
    const { phase, spec: graph } = live.current;
    const map = nodes.current;
    const cam = camera.current;
    const palette = colors.current;
    if (palette.frame++ % 10 === 0) {
      const style = getComputedStyle(canvas);
      for (const key of ["accent", "text", "blue", "dim"] as const) {
        const value = style.getPropertyValue(`--${key}`).trim();
        if (value) palette[key] = value;
      }
    }

    // 1. Forces for the free-floating nodes: repel each other, springs along links, settle in a band below the ring.
    const free = [...map.values()].filter((n) => !n.anchor);
    for (const a of free) {
      for (const b of map.values()) {
        if (a === b) continue;
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        const d2 = Math.max(dx * dx + dy * dy + dz * dz, 400);
        const push = 2400 / d2;
        const d = Math.sqrt(d2);
        a.vx += (dx / d) * push;
        a.vy += (dy / d) * push;
        a.vz += (dz / d) * push;
      }
      const band = a.kind === "language" ? 95 : 175;
      a.vy += (band - a.y) * 0.004;
      a.vx -= a.x * 0.0015;
      a.vz -= a.z * 0.0015;
    }
    for (const [from, to] of graph.links) {
      const a = map.get(from), b = map.get(to);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      const pull = (d - 120) * 0.006;
      for (const [node, sign] of [[a, 1], [b, -1]] as const) {
        if (node.anchor) continue;
        node.vx += (dx / d) * pull * sign;
        node.vy += (dy / d) * pull * sign;
        node.vz += (dz / d) * pull * sign;
      }
    }
    for (const n of map.values()) {
      if (n.anchor) {
        [n.x, n.y, n.z] = n.anchor;
        continue;
      }
      n.vx *= 0.82; n.vy *= 0.82; n.vz *= 0.82;
      n.x += Math.max(-6, Math.min(6, n.vx));
      n.y += Math.max(-6, Math.min(6, n.vy));
      n.z += Math.max(-6, Math.min(6, n.vz));
    }

    // 2. Camera: travel to the stage that is working, otherwise drift slowly around the graph.
    const { links: activeLinks, focus } = activity(phase);
    const userControl = performance.now() < cam.userUntil;
    const ease = 1 - Math.exp(-dt * (reduced ? 50 : 1.8));
    const focused = focus ? map.get(focus) : undefined;
    if (!userControl && !drag.current) {
      if (focus) cam.yaw += wrapAngle(-Math.PI / 2 - backboneAngle(focus) - cam.yaw) * ease;
      else if (!reduced) cam.yaw += dt * 0.07;
    }
    cam.fx += ((focused ? focused.x * 0.45 : 0) - cam.fx) * ease;
    cam.fy += ((focused ? focused.y * 0.45 : 0) - cam.fy) * ease;
    cam.fz += ((focused ? focused.z * 0.45 : 0) - cam.fz) * ease;
    cam.zoom += ((focused ? 1.22 : 1) - cam.zoom) * ease;

    const width = canvas.width, height = canvas.height;
    const base = Math.min(width, height) / 640;
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const project = (x: number, y: number, z: number) => {
      const px = x - cam.fx, py = y - cam.fy, pz = z - cam.fz;
      const rx = px * cy - pz * sy;
      const rz = px * sy + pz * cy;
      const ry = py * cp - rz * sp;
      const depth = py * sp + rz * cp + DISTANCE;
      const scale = (DISTANCE / Math.max(depth, 60)) * cam.zoom * base;
      return { x: width / 2 + rx * scale, y: height * 0.52 + ry * scale, scale, depth };
    };

    // 3. Draw: space, floor rings, links (with data in flight), nodes and labels.
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = palette.text;
    for (const [x, y, z] of STARS) {
      const p = project(x, y, z);
      ctx.globalAlpha = Math.max(0, Math.min(0.5, 1.1 - p.depth / 1700));
      ctx.fillRect(p.x, p.y, 1.4 * base, 1.4 * base);
    }
    ctx.strokeStyle = palette.accent;
    ctx.lineWidth = 1;
    for (const radius of [140, 260, 380]) {
      ctx.globalAlpha = 0.1;
      ctx.beginPath();
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * TAU;
        const p = project(Math.cos(a) * radius, 240, Math.sin(a) * radius);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }

    const active = new Set(activeLinks.map(([a, b]) => `${a}>${b}`));
    for (const [from, to] of graph.links) {
      const a = map.get(from), b = map.get(to);
      if (!a || !b) continue;
      const pa = project(a.x, a.y, a.z), pb = project(b.x, b.y, b.z);
      const hot = active.has(`${from}>${to}`);
      ctx.globalAlpha = hot ? 0.95 : Math.max(0.1, 0.5 - (pa.depth + pb.depth) / 2 / 3000);
      ctx.strokeStyle = hot ? palette.accent : palette.dim;
      ctx.lineWidth = (hot ? 2.4 : 1) * base;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      if (hot && !reduced) {
        ctx.fillStyle = palette.text;
        for (let k = 0; k < 4; k++) {
          const f = (t * 0.7 + k / 4) % 1;
          ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.arc(pa.x + (pb.x - pa.x) * f, pa.y + (pb.y - pa.y) * f, 3 * base, 0, TAU);
          ctx.fill();
        }
      }
    }

    const working = new Set<string>(activeLinks.flat());
    const recentMessages = new Set(graph.nodes.filter((n) => n.kind === "user" || n.kind === "assistant").slice(-4).map((n) => n.id));
    const ordered = [...map.values()].map((n) => ({ n, p: project(n.x, n.y, n.z) })).sort((a, b) => b.p.depth - a.p.depth);
    ctx.textAlign = "center";
    for (const { n, p } of ordered) {
      const style = STYLE[n.kind];
      const color = palette[style.tone];
      const pulse = working.has(n.id) && !reduced ? 1 + 0.18 * Math.sin(t * 6) : 1;
      const radius = style.radius * p.scale * pulse;
      const fade = Math.max(0.25, Math.min(1, 1.35 - p.depth / 1400));
      const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 2.6);
      glow.addColorStop(0, color);
      glow.addColorStop(1, "transparent");
      ctx.globalAlpha = fade * (working.has(n.id) ? 0.55 : 0.25);
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius * 2.6, 0, TAU);
      ctx.fill();
      ctx.globalAlpha = fade;
      ctx.fillStyle = n.kind === "stage" || n.kind === "io" ? "#031019" : color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6 * base;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, TAU);
      ctx.fill();
      ctx.stroke();

      // Model labels: the primary model of every stage when nothing is happening, all of a stage's
      // models (fallbacks too) while the camera is on it. Messages: only the latest few.
      const modelStage = n.kind === "model" ? n.id.split(":")[1] : undefined;
      const labelled =
        n.kind === "model"
          ? focus ? modelStage === focus : !n.sub
          : n.kind !== "user" && n.kind !== "assistant" || recentMessages.has(n.id);
      if (!labelled) continue;
      const big = n.kind === "stage" || n.kind === "io" || n.kind === "language";
      ctx.font = `${big ? 600 : 500} ${Math.round((big ? 12 : 10.5) * Math.max(0.8, p.scale))}px ${big ? "Orbitron, sans-serif" : "'Share Tech Mono', monospace"}`;
      ctx.fillStyle = big ? palette.text : palette.dim;
      ctx.globalAlpha = fade;
      const offset = radius + (big ? 14 : 11) * p.scale;
      ctx.fillText(n.sub ? `${n.label} · ${n.sub}` : n.label, p.x, n.kind === "model" ? p.y - offset : p.y + offset + 4);
    }
    ctx.globalAlpha = 1;
  });

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { x: event.clientX, y: event.clientY, yaw: camera.current.yaw, pitch: camera.current.pitch };
  };
  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const start = drag.current;
    if (!start) return;
    camera.current.yaw = start.yaw + (event.clientX - start.x) * 0.008;
    camera.current.pitch = Math.max(-1.2, Math.min(0.25, start.pitch - (event.clientY - start.y) * 0.006));
  };
  const onPointerUp = () => {
    drag.current = null;
    camera.current.userUntil = performance.now() + 5000; // leave the view where the user put it for a while
  };

  const { focus } = activity(state.phase);
  const focusLabel = focus ? `${focus.toUpperCase()}${focus in models ? ` · ${models[focus as "stt" | "llm" | "tts"][0] ?? ""}` : ""}` : "";
  const summary =
    `Pipeline graph: ${BACKBONE.map((id: BackboneId) => id.toUpperCase()).join(" to ")}. ` +
    `Models: ${[...models.stt, ...models.llm, ...models.tts].join(", ") || "loading"}. ` +
    `${spec.nodes.filter((n) => n.kind === "language").length} languages, ${state.messages.length} messages.`;

  return (
    <div className="graph">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={summary}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      />
      <div className="graph-focus mono" aria-hidden="true">
        {focusLabel ? `FOCUS ▸ ${focusLabel}` : "DRAG TO LOOK AROUND"}
      </div>
      <ul className="graph-legend" aria-hidden="true">
        <li data-tone="accent">Stage</li>
        <li data-tone="dim">Model</li>
        <li data-tone="blue">Language</li>
        <li data-tone="blue">You</li>
        <li data-tone="accent">Assistant</li>
      </ul>
    </div>
  );
}
