// Gapless playback of the per-sentence PCM chunks the server streams, with an analyser so the
// HUD can draw what is actually being played.

export const SPECTRUM_BINS = 128; // fftSize 256 -> 128 frequency bins

export class Player {
  private ctx: AudioContext | null = null;
  private nextTime = 0;
  private readonly active = new Set<AudioBufferSourceNode>();
  private readonly onIdle: () => void;
  analyser: AnalyserNode | null = null;

  constructor(onIdle: () => void) {
    this.onIdle = onIdle;
  }

  get busy(): boolean {
    return this.active.size > 0;
  }

  /** The playback position that is audible now, in seconds on the clock `enqueue` reports times in. */
  get audibleTime(): number {
    if (!this.ctx) return 0;
    return this.ctx.currentTime - (this.ctx.outputLatency || 0);
  }

  /** True while the browser keeps audio paused (no user gesture yet): nothing scheduled will be heard. */
  get suspended(): boolean {
    return this.ctx?.state === "suspended";
  }

  /** Must run inside a user gesture, or the browser keeps the AudioContext suspended. */
  unlock(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = SPECTRUM_BINS * 2;
      this.analyser.smoothingTimeConstant = 0.78;
      this.analyser.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  /** Schedules the samples right after what is already queued and returns when they will play. */
  enqueue(samples: Int16Array, sampleRate: number): { start: number; end: number } {
    const ctx = this.unlock();
    const buffer = ctx.createBuffer(1, samples.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) channel[i] = (samples[i] ?? 0) / 32768;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.analyser ?? ctx.destination);
    // Chain each sentence directly after the previous one so playback has no gaps.
    const startAt = Math.max(ctx.currentTime + 0.05, this.nextTime);
    source.start(startAt);
    this.nextTime = startAt + buffer.duration;
    this.active.add(source);
    source.onended = () => {
      this.active.delete(source);
      if (!this.busy) this.onIdle();
    };
    return { start: startAt, end: this.nextTime };
  }

  stop(): void {
    for (const source of this.active) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* already ended */
      }
    }
    this.active.clear();
    this.nextTime = 0;
  }

  close(): void {
    this.stop();
    void this.ctx?.close();
    this.ctx = null;
    this.analyser = null;
  }
}
