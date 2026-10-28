// Real-browser end-to-end check of the web UI, with no npm dependencies.
//
//   node scripts/browser_e2e.mjs [--url http://127.0.0.1:8000] [--audio benchmarks/inputs/short.wav]
//        [--transport stream|upload] [--expect "capital"] [--browser <path to chrome/edge>]
//        [--screenshots <dir>] [--viewport 1440x900] [--scenario turn|deny-mic] [--reduced-motion true]
//        [--speak-seconds 2.6] [--gif <out.gif>] [--view core|graph]
//        [--language te] [--attach <file to ask about>]
//
// With E2E_GEMINI_KEY set in the environment, the key is typed into the System page's key form first
// (React UI only), as a user would; it is never printed.
//
// It starts Chrome or Edge headless with a WAV file standing in for the microphone, opens the
// page, clicks the record button, waits, clicks stop, and then reads what the UI actually shows.
// The server (with real models) must already be running. Exit code 0 = the whole loop worked.
// It works against both UIs (the React HUD at / and the classic page at /classic/) because they
// share element ids and classes.
//
// With --gif the turn is also recorded (real frames, real speed, no audio) and encoded with ffmpeg.
// With --screenshots the page is captured at each stage of the turn. Those images show the real UI
// driven by the real models, but the "microphone" is the WAV file, not a person.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
const url = args.url ?? "http://127.0.0.1:8000";
const audio = resolve(args.audio ?? "benchmarks/inputs/short.wav");
const transport = args.transport ?? "stream";
const expected = (args.expect ?? "capital").toLowerCase();
const scenario = args.scenario ?? "turn"; // "turn" = a full voice turn; "deny-mic" = the permission-denied error path
const shotDir = args.screenshots ? resolve(args.screenshots) : null;
const [width, height] = (args.viewport ?? "1440x900").split("x").map(Number);
const speakSeconds = Number(args["speak-seconds"] ?? 2.6); // how long to talk: the fake microphone plays the WAV once
const gifPath = args.gif ? resolve(args.gif) : null;
const frames = []; // screencast frames for --gif: { time (s), data (base64 jpeg) }
const port = 9333;

const candidates = [
  args.browser,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);
const browserPath = candidates.find(existsSync);
if (!browserPath) throw new Error("No Chrome/Edge found. Pass --browser <path>.");
if (!existsSync(audio)) throw new Error(`Audio file not found: ${audio} (run scripts/make_test_audio.py)`);
if (shotDir) mkdirSync(shotDir, { recursive: true });

const profile = mkdtempSync(join(tmpdir(), "voice-agent-e2e-"));
const browser = spawn(browserPath, [
  "--headless=new", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  "--use-fake-device-for-media-stream",
  // The fake permission UI auto-grants the microphone, which would hide the denied-permission path.
  ...(scenario === "deny-mic" ? [] : ["--use-fake-ui-for-media-stream"]),
  `--use-file-for-fake-audio-capture=${audio}%noloop`,
  "--autoplay-policy=no-user-gesture-required", "--no-first-run", "--no-default-browser-check", "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const problems = [];

async function main() {
  let targets;
  for (let i = 0; i < 50 && !targets; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); } catch { await sleep(200); }
  }
  const page = targets?.find((t) => t.type === "page");
  if (!page) throw new Error("browser did not start");

  const socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let nextId = 1;
  const waiting = new Map();
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.id && waiting.has(message.id)) return waiting.get(message.id)(message);
    if (message.method === "Page.screencastFrame") {
      frames.push({ time: message.params.metadata.timestamp, data: message.params.data });
      socket.send(JSON.stringify({ id: nextId++, method: "Page.screencastFrameAck", params: { sessionId: message.params.sessionId } }));
      return;
    }
    if (message.method === "Runtime.exceptionThrown") problems.push(`exception: ${message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text}`);
    if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
      problems.push(`console.error: ${message.params.args.map((a) => a.value ?? a.description).join(" ")}`);
    }
  };
  const send = (method, params = {}) => new Promise((resolve) => {
    const id = nextId++;
    waiting.set(id, resolve);
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const { result } = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true, userGesture: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };
  const waitFor = async (expression, what, timeoutMs = 60000) => {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      if (await evaluate(expression)) return;
      await sleep(250);
    }
    throw new Error(`timed out waiting for: ${what}`);
  };
  const shot = async (name) => {
    if (!shotDir) return;
    const { result } = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(join(shotDir, `${name}.png`), Buffer.from(result.data, "base64"));
    console.log(`screenshot: ${name}.png`);
  };

  await send("Runtime.enable");
  await send("Page.enable");
  if (args["reduced-motion"]) {
    await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
  }
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 600 });
  await send("Page.navigate", { url });
  // Scripts run after the DOM is parsed (and the React UI renders after that), so wait for the
  // app to have put its status text on the page before touching the controls.
  await waitFor("(() => { const s = document.getElementById('state-text'); return !!s && s.textContent !== 'Starting…'; })()", "the app to initialise");
  const react = await evaluate("!!document.getElementById('nav-system')");
  const openPage = async (hash) => {
    if (!react) return;
    await evaluate(`location.hash = '${hash}'`);
    await sleep(400);
  };
  if (process.env.E2E_GEMINI_KEY && react) {
    await openPage("#/system");
    await evaluate("document.getElementById('gemini-key').focus()");
    await send("Input.insertText", { text: process.env.E2E_GEMINI_KEY });
    await evaluate("document.getElementById('key-save').click()");
    await waitFor("!!document.querySelector('.key-saved') || !!document.querySelector('.key-message[data-ok=false]')", "the key check", 20000);
    console.log("key check:", await evaluate("document.querySelector('.key-message')?.textContent ?? ''"));
    await shot("00-key");
    await openPage("#/");
  }
  if (transport === "upload") {
    // The React UI has a button on its System page; the classic page has a <select>.
    await openPage("#/system");
    await evaluate(`(() => {
      const button = document.getElementById('transport-upload');
      if (button) return button.click();
      const select = document.getElementById('transport');
      select.value = 'upload';
      select.dispatchEvent(new Event('change'));
    })()`);
    await openPage("#/");
  }
  await waitFor("(() => { const m = document.getElementById('mic'); return !!m && !m.disabled; })()", "mic button to enable (server connected and models ready)", 90000);
  if (args.view) await evaluate(`document.getElementById('view-${args.view}')?.click()`); // React UI only
  if (args.language) {
    await waitFor(`!!document.getElementById('language-${args.language}')`, `the ${args.language} language button`, 10000);
    await evaluate(`document.getElementById('language-${args.language}').click()`);
  }
  if (args.attach) {
    // The real file input, as a person picking the file would use it.
    const { result: { root } } = await send("DOM.getDocument", {});
    const { result: { nodeId } } = await send("DOM.querySelector", { nodeId: root.nodeId, selector: "#file-input" });
    await send("DOM.setFileInputFiles", { nodeId, files: [resolve(args.attach)] });
    await waitFor("document.querySelectorAll('.file-chip[data-status=ready]').length === 1", "the attached file to be read", 30000);
    console.log("attached:", await evaluate("document.querySelector('.file-chip').textContent"));
  }
  console.log(`page ready (${transport} mode)`);
  await sleep(1800); // let the start-up animation finish before the first picture
  await shot("01-ready");
  if (gifPath) await send("Page.startScreencast", { format: "jpeg", quality: 60, maxWidth: 800, maxHeight: 520, everyNthFrame: 6 });

  if (scenario === "deny-mic") {
    // Error path: the browser refuses the microphone. The UI must say so and stay usable.
    await send("Browser.setPermission", { permission: { name: "microphone" }, setting: "denied", origin: new URL(url).origin });
    await evaluate("document.getElementById('mic').click()");
    await waitFor("(() => { const b = document.getElementById('banner'); return !!b && !b.hidden; })()", "the error banner", 10000);
    await sleep(600);
    await shot("05-mic-denied");
    const result = await evaluate(`({ banner: document.getElementById('banner-text').textContent, phase: document.getElementById('mic').dataset.phase })`);
    console.log(JSON.stringify(result, null, 2));
    socket.close();
    const denyChecks = {
      "banner explains the denied permission": /permission was denied/i.test(result.banner),
      "mic returns to idle": result.phase === "idle",
      "no page errors": problems.length === 0,
    };
    for (const [name, ok] of Object.entries(denyChecks)) console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
    return Object.values(denyChecks).every(Boolean);
  }

  await evaluate("document.getElementById('mic').click()");
  await waitFor("document.getElementById('mic').dataset.phase === 'recording'", "recording to start", 10000);
  console.log("recording…");
  await sleep(1400);
  await shot("02-listening");
  await sleep(Math.max(0, speakSeconds * 1000 - 1400)); // the fake microphone plays the WAV from the beginning once
  await evaluate("document.getElementById('mic').click()");
  console.log("stopped; waiting for transcript, reply and audio…");

  const finished = `document.querySelectorAll('.msg.assistant').length > 0 && !document.querySelector('.msg.pending') && !['speaking', 'thinking', 'transcribing'].includes(document.getElementById('mic').dataset.phase)`;
  const seen = new Set();
  const shownWhileSpeaking = []; // characters of the reply on screen, sampled while it is spoken
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    const phase = await evaluate("document.getElementById('mic').dataset.phase");
    if (phase === "speaking") {
      shownWhileSpeaking.push(await evaluate("([...document.querySelectorAll('.msg.assistant p')].at(-1)?.textContent ?? '').length"));
    }
    if (["transcribing", "thinking", "speaking"].includes(phase) && !seen.has(phase)) {
      seen.add(phase);
      await sleep(350); // a moment into the phase, so the visuals have reacted
      await shot(`03-${phase}`);
    }
    if (await evaluate(finished)) break;
    await sleep(150);
  }
  if (!(await evaluate(finished))) throw new Error("timed out waiting for: turn to finish");
  await sleep(900);
  await shot("04-done");
  if (gifPath) {
    await sleep(1500); // let the final state sit for a moment before the loop restarts
    await send("Page.stopScreencast");
    encodeGif();
  }

  const conversation = await evaluate(`(() => {
    const banner = document.getElementById('banner');
    return {
      user: [...document.querySelectorAll('.msg.user p')].map(p => p.textContent),
      assistant: [...document.querySelectorAll('.msg.assistant p')].map(p => p.textContent),
      replayAudio: [...document.querySelectorAll('.msg.assistant audio')].map(a => a.src.slice(0, 30)),
      banner: banner && !banner.hidden ? document.getElementById('banner-text').textContent : null,
      state: document.getElementById('state-text').textContent,
      mode: document.getElementById('mode-pill')?.textContent ?? null,
    };
  })()`);
  await openPage("#/system"); // the React UI keeps timings and diagnostics on its System page
  await shot("05-system");
  const measured = await evaluate(`({
    clientLatency: document.getElementById('client-latency')?.textContent ?? '',
    diagnostics: [...document.querySelectorAll('#diag-body tr')].map(r => [...r.children].map(c => c.textContent).join(' | ')),
  })`);
  const snapshot = { ...conversation, ...measured };
  console.log(JSON.stringify(snapshot, null, 2));
  const finalLength = snapshot.assistant.at(-1)?.length ?? 0;
  console.log(`reply characters on screen while speaking: ${[...new Set(shownWhileSpeaking)].join(", ")} (final ${finalLength})`);
  socket.close();

  const checks = {
    "transcript shown": snapshot.user.length > 0 && snapshot.user[0].toLowerCase().includes(expected),
    "reply text shown": snapshot.assistant.length > 0 && snapshot.assistant[0].length > 0,
    "replay audio attached": snapshot.replayAudio.length > 0,
    // Each transport writes a different latency label, so this proves which path really ran.
    [`${transport} transport was used`]: snapshot.clientLatency.includes(transport === "upload" ? "response received" : "first audio"),
    "no error banner": snapshot.banner === null,
    // The text is revealed as it is heard: part of it was on screen while the voice was speaking.
    ...(transport === "stream" && snapshot.replayAudio.length
      ? {
          "reply text followed the voice":
            shownWhileSpeaking[0] < finalLength && shownWhileSpeaking.some((n) => n > 0 && n < finalLength),
        }
      : {}),
    "no page errors": problems.length === 0,
  };
  for (const [name, ok] of Object.entries(checks)) console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  if (problems.length) console.log(problems.join("\n"));
  return Object.values(checks).every(Boolean);
}

function encodeGif() {
  if (frames.length < 2) return console.log("no screencast frames were captured");
  const dir = mkdtempSync(join(tmpdir(), "voice-agent-gif-"));
  const lines = [];
  frames.forEach((frame, i) => {
    const name = join(dir, `f${String(i).padStart(5, "0")}.jpg`).replaceAll("\\", "/");
    writeFileSync(name, Buffer.from(frame.data, "base64"));
    const next = frames[i + 1]?.time ?? frame.time + 1.5;
    lines.push(`file '${name}'`, `duration ${Math.min(1, Math.max(0.01, next - frame.time)).toFixed(3)}`);
  });
  lines.push(`file '${join(dir, `f${String(frames.length - 1).padStart(5, "0")}.jpg`).replaceAll("\\", "/")}'`);
  const list = join(dir, "list.txt");
  writeFileSync(list, lines.join(String.fromCharCode(10)));
  mkdirSync(join(gifPath, ".."), { recursive: true });
  const filter = "fps=10,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=3";
  const result = spawnSync("ffmpeg", ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list, "-vf", filter, "-loop", "0", gifPath], { encoding: "utf8" });
  console.log(result.status === 0 ? `gif: ${gifPath} (${frames.length} frames captured)` : `ffmpeg failed: ${result.stderr || result.error}`);
  rmSync(dir, { recursive: true, force: true });
}

let ok = false;
try { ok = await main(); } catch (error) { console.error(`FAILED: ${error.message}`); if (problems.length) console.error(problems.join("\n")); }
browser.kill();
await sleep(500);
try { rmSync(profile, { recursive: true, force: true }); } catch { /* the browser may still hold files */ }
process.exit(ok ? 0 : 1);
