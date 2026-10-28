// Microphone capture. Two implementations behind the two transports:
//  * streamCapture: an AudioWorklet downsamples to 16 kHz PCM16 and hands over 100 ms chunks;
//  * recorderCapture: MediaRecorder produces one WebM/MP4 blob for the HTTP upload path.
// Both expose an analyser so the HUD can draw the real microphone signal.
import { UnsupportedError } from "../lib/errors.ts";
import { SPECTRUM_BINS } from "./player.ts";

export interface StreamCapture {
  analyser: AnalyserNode;
  begin(): void;
  /** Resolves once the last audio chunk has been handed to `send`. */
  stop(): Promise<void>;
  abort(): void;
}

export interface RecorderCapture {
  analyser: AnalyserNode;
  begin(): void;
  stop(): Promise<Blob>;
  abort(): void;
}

const AUDIO_CONSTRAINTS = { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true };

function makeAnalyser(ctx: AudioContext, source: MediaStreamAudioSourceNode): AnalyserNode {
  const analyser = ctx.createAnalyser();
  analyser.fftSize = SPECTRUM_BINS * 2;
  analyser.smoothingTimeConstant = 0.7;
  source.connect(analyser);
  return analyser;
}

export async function openStreamCapture(send: (pcm: ArrayBuffer) => void): Promise<StreamCapture> {
  if (!("AudioWorkletNode" in window)) {
    throw new UnsupportedError("this browser has no AudioWorklet; use Upload mode");
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS });
  const ctx = new AudioContext(); // device rate; the worklet downsamples to 16 kHz
  await ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}pcm-worklet.js`);
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-capture");
  const silent = ctx.createGain(); // keeps the node in the graph without playing the mic back
  silent.gain.value = 0;
  source.connect(node).connect(silent).connect(ctx.destination);
  const analyser = makeAnalyser(ctx, source);

  let sending = false;
  let onFlushed = () => {};
  node.port.onmessage = (event: MessageEvent<{ pcm?: ArrayBuffer; flushed?: boolean }>) => {
    if (event.data.flushed) return onFlushed();
    if (event.data.pcm && sending) send(event.data.pcm);
  };
  const release = () => {
    source.disconnect();
    node.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    void ctx.close();
  };
  return {
    analyser,
    begin() {
      sending = true;
    },
    async stop() {
      const flushed = new Promise<void>((resolve) => {
        onFlushed = resolve;
        setTimeout(resolve, 500); // never hang if the worklet already stopped
      });
      node.port.postMessage("flush");
      await flushed;
      sending = false;
      release();
    },
    abort() {
      sending = false;
      release();
    },
  };
}

export async function openRecorderCapture(): Promise<RecorderCapture> {
  if (!("MediaRecorder" in window)) throw new UnsupportedError("MediaRecorder is not supported");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS });
  const recorder = new MediaRecorder(stream); // the browser picks WebM/Opus or MP4; the server decodes either
  const parts: Blob[] = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size) parts.push(event.data);
  };
  const ctx = new AudioContext();
  const analyser = makeAnalyser(ctx, ctx.createMediaStreamSource(stream));
  const release = () => {
    stream.getTracks().forEach((track) => track.stop());
    void ctx.close();
  };
  return {
    analyser,
    begin() {
      recorder.start();
    },
    async stop() {
      const stopped = new Promise<void>((resolve) => {
        recorder.onstop = () => resolve();
      });
      recorder.stop();
      await stopped;
      release();
      return new Blob(parts, { type: recorder.mimeType });
    },
    abort() {
      recorder.onstop = null;
      if (recorder.state !== "inactive") recorder.stop();
      release();
    },
  };
}
