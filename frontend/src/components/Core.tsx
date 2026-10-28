import { AnimatePresence, motion, useAnimationFrame, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef } from "react";
import { SPECTRUM_BINS } from "../audio/player.ts";
import type { AudioTap } from "../controller.ts";
import { activeStage, type Phase, type Stage } from "../lib/state.ts";

// All geometry is in a 600 x 600 box centred on (300, 300) and scaled by CSS.
const C = 300;
const TAU = Math.PI * 2;
const BAR_COUNT = 96;
const BAR_INNER = 154;

const point = (radius: number, angle: number) => [C + Math.cos(angle) * radius, C + Math.sin(angle) * radius] as const;

/** Radial tick marks as one path. `longEvery` makes every n-th tick longer. */
function ticks(count: number, outer: number, shortLen: number, longLen: number, longEvery: number): string {
  let d = "";
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * TAU;
    const [x1, y1] = point(outer, angle);
    const [x2, y2] = point(outer - (i % longEvery === 0 ? longLen : shortLen), angle);
    d += `M${x1.toFixed(1)} ${y1.toFixed(1)}L${x2.toFixed(1)} ${y2.toFixed(1)}`;
  }
  return d;
}

function arc(radius: number, startDeg: number, sweepDeg: number): string {
  const a0 = (startDeg * Math.PI) / 180;
  const a1 = ((startDeg + sweepDeg) * Math.PI) / 180;
  const [x0, y0] = point(radius, a0);
  const [x1, y1] = point(radius, a1);
  return `M${x0.toFixed(1)} ${y0.toFixed(1)}A${radius} ${radius} 0 ${sweepDeg > 180 ? 1 : 0} 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
}

// Degrees per second of the three rotating ring groups, per phase. Purely decorative: it only
// signals that the system is busier, it does not measure anything.
const SPEEDS: Record<Phase, readonly [number, number, number]> = {
  idle: [3, -5, 12],
  "requesting-mic": [4, -8, 26],
  recording: [5, -9, 34],
  transcribing: [8, -16, 95],
  thinking: [10, -22, 170],
  speaking: [6, -12, 28],
};

const STAGE_NODES: { stage: Stage; label: string; angle: number }[] = [
  { stage: "stt", label: "STT", angle: -90 },
  { stage: "llm", label: "LLM", angle: 30 },
  { stage: "tts", label: "TTS", angle: 150 },
];

interface CoreProps {
  phase: Phase;
  word: string;
  sub: string;
  tap: AudioTap;
}

export function Core({ phase, word, sub, tap }: CoreProps) {
  const reduced = useReducedMotion() ?? false;
  const stage = activeStage(phase);

  const outerTicks = useMemo(() => ticks(180, 288, 7, 15, 5), []);
  const arcs = useMemo(() => [0, 95, 190, 285].map((start) => arc(232, start, 52)), []);

  const groupRefs = [useRef<SVGGElement>(null), useRef<SVGGElement>(null), useRef<SVGGElement>(null)];
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);
  const live = useRef({
    angles: [0, 0, 0],
    speeds: [0, 0, 0],
    level: 0,
    bars: new Float32Array(BAR_COUNT),
    bins: new Uint8Array(SPECTRUM_BINS),
    color: "#38e8ff",
    frame: 0,
  });
  const phaseRef = useRef(phase);
  phaseRef.current = phase;

  // Keep the canvas backing store matched to its on-screen size (and the device pixel ratio).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const size = Math.round(canvas.clientWidth * Math.min(window.devicePixelRatio || 1, 2));
      if (size > 0 && canvas.width !== size) {
        canvas.width = size;
        canvas.height = size;
      }
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useAnimationFrame((timeMs, deltaMs) => {
    const s = live.current;
    const now = phaseRef.current;
    const dt = Math.min(deltaMs, 60) / 1000;
    const t = timeMs / 1000;

    // 1. rotating rings, with speed eased so a phase change never makes them jump
    const targets = reduced ? [0, 0, 0] : SPEEDS[now];
    const ease = 1 - Math.exp(-dt * 2.5);
    for (let i = 0; i < 3; i++) {
      s.speeds[i] = (s.speeds[i] ?? 0) + ((targets[i] ?? 0) - (s.speeds[i] ?? 0)) * ease;
      s.angles[i] = ((s.angles[i] ?? 0) + (s.speeds[i] ?? 0) * dt) % 360;
      groupRefs[i]?.current?.setAttribute("transform", `rotate(${(s.angles[i] ?? 0).toFixed(2)} ${C} ${C})`);
    }

    // 2. the real audio level: microphone while recording, the reply while it is spoken
    const listening = now === "recording" || now === "speaking";
    const raw = listening ? tap.read(s.bins) : 0;
    s.level += (raw - s.level) * (1 - Math.exp(-dt * (raw > s.level ? 20 : 6)));

    // 3. bar targets
    const half = BAR_COUNT / 2;
    const busy = now === "transcribing" || now === "thinking";
    const sweep = (t * (now === "thinking" ? 3.2 : 2.2)) % TAU;
    for (let i = 0; i < BAR_COUNT; i++) {
      const angle = (i / BAR_COUNT) * TAU;
      let target: number;
      if (listening && raw > 0) {
        const mirrored = i < half ? i : BAR_COUNT - 1 - i;
        const bin = 1 + Math.round(Math.pow(mirrored / (half - 1), 1.6) * 58);
        target = Math.min(1, Math.pow((s.bins[bin] ?? 0) / 255, 0.95) * 1.35);
      } else if (busy && !reduced) {
        const d = Math.abs(((angle - sweep + Math.PI * 3) % TAU) - Math.PI);
        const d2 = Math.abs(((angle - sweep + TAU * 2) % TAU) - Math.PI);
        target = 0.08 + 0.72 * Math.exp(-(d * d) / 0.2) + 0.35 * Math.exp(-(d2 * d2) / 0.2);
      } else {
        target = reduced ? 0.07 : 0.06 + 0.05 * (0.5 + 0.5 * Math.sin(t * 1.3 + i * 0.45));
      }
      const current = s.bars[i] ?? 0;
      s.bars[i] = current + (target - current) * (target > current ? 0.55 : 0.16);
    }

    // 4. canvas
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx && canvas.width > 0) {
      if (s.frame++ % 5 === 0) {
        const color = getComputedStyle(canvas).getPropertyValue("--accent").trim();
        if (color) s.color = color;
      }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const scale = canvas.width / 600;
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.strokeStyle = s.color;
      ctx.lineCap = "round";
      ctx.lineWidth = 3.4;
      for (let i = 0; i < BAR_COUNT; i++) {
        const v = s.bars[i] ?? 0;
        const angle = (i / BAR_COUNT) * TAU - Math.PI / 2;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const length = BAR_INNER + 3 + v * 58;
        ctx.globalAlpha = 0.3 + v * 0.7;
        ctx.beginPath();
        ctx.moveTo(C + cos * BAR_INNER, C + sin * BAR_INNER);
        ctx.lineTo(C + cos * length, C + sin * length);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // 5. glow behind the orb follows the level
    const glow = glowRef.current;
    if (glow) {
      glow.style.opacity = String(0.32 + Math.min(1, s.level) * 0.6);
      glow.style.transform = `scale(${(1 + s.level * 0.22).toFixed(3)})`;
    }
  });

  return (
    <div className="core" role="img" aria-label={`Voice core, ${word.toLowerCase()}`}>
      <div className="core-glow" ref={glowRef} />
      <motion.svg
        viewBox="0 0 600 600"
        initial={reduced ? false : { opacity: 0, scale: 0.92 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.1, ease: "easeOut" }}
        aria-hidden="true"
      >
        <defs>
          <radialGradient id="orb-fill" cx="50%" cy="42%" r="62%">
            <stop offset="0%" style={{ stopColor: "var(--accent)", stopOpacity: 0.34 }} />
            <stop offset="55%" style={{ stopColor: "var(--accent)", stopOpacity: 0.08 }} />
            <stop offset="100%" style={{ stopColor: "#02060c", stopOpacity: 0.85 }} />
          </radialGradient>
        </defs>

        {/* outer ruler, slow */}
        <g ref={groupRefs[0]}>
          <path className="ring thin" d={outerTicks} />
        </g>

        {/* dashed rings, counter-rotating */}
        <g ref={groupRefs[1]}>
          <circle className="ring faint" cx={C} cy={C} r={262} strokeDasharray="2 9" />
          <circle className="ring thin" cx={C} cy={C} r={254} strokeDasharray="70 16 22 16" />
        </g>

        {/* bright arcs, speed follows the phase */}
        <g ref={groupRefs[2]}>
          {arcs.map((d) => (
            <path key={d} className="ring bold" d={d} />
          ))}
        </g>

        {/* rings that draw themselves in at start-up */}
        <motion.circle
          className="ring faint"
          cx={C}
          cy={C}
          r={232}
          initial={reduced ? false : { pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.8, ease: "easeInOut" }}
        />
        <motion.circle
          className="ring thin"
          cx={C}
          cy={C}
          r={222}
          initial={reduced ? false : { pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.5, ease: "easeInOut", delay: 0.2 }}
        />
        <circle className="ring faint" cx={C} cy={C} r={BAR_INNER - 6} />

        {/* compass markers */}
        {[0, 90, 180, 270].map((deg) => {
          const a = (deg * Math.PI) / 180;
          const [x, y] = point(216, a);
          const [x1, y1] = point(204, a + 0.05);
          const [x2, y2] = point(204, a - 0.05);
          return <path key={deg} className="mark" d={`M${x.toFixed(1)} ${y.toFixed(1)}L${x1.toFixed(1)} ${y1.toFixed(1)}L${x2.toFixed(1)} ${y2.toFixed(1)}Z`} />;
        })}

        {/* thinking ripples */}
        <AnimatePresence>
          {(phase === "thinking" || phase === "transcribing") && !reduced &&
            [0, 0.9].map((delay) => (
              <motion.circle
                key={delay}
                className="ring bold"
                cx={C}
                cy={C}
                initial={{ r: 128, opacity: 0.55 }}
                animate={{ r: 205, opacity: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1.8, ease: "easeOut", repeat: Infinity, delay }}
              />
            ))}
        </AnimatePresence>

        {/* the orb */}
        <circle cx={C} cy={C} r={138} fill="url(#orb-fill)" className="orb-edge" />
        <circle className="ring faint" cx={C} cy={C} r={122} strokeDasharray="1 5" />

        {/* pipeline nodes: they light up with the stage the server is working on */}
        {STAGE_NODES.map(({ stage: nodeStage, label, angle }) => {
          const [x, y] = point(232, (angle * Math.PI) / 180);
          const active = stage === nodeStage;
          return (
            <g key={nodeStage} className="node" data-active={active} transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`}>
              <rect x={-23} y={-10} width={46} height={20} />
              <text y={4} textAnchor="middle">
                {label}
              </text>
            </g>
          );
        })}
      </motion.svg>
      <canvas ref={canvasRef} aria-hidden="true" />
      <div className="core-label">
        <AnimatePresence mode="wait">
          <motion.div
            key={word}
            className="core-word"
            initial={reduced ? false : { opacity: 0, y: 6, filter: "blur(4px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={reduced ? undefined : { opacity: 0, y: -6, filter: "blur(4px)" }}
            transition={{ duration: 0.22 }}
          >
            {word}
          </motion.div>
        </AnimatePresence>
        <div className="core-sub">{sub}</div>
      </div>
    </div>
  );
}
