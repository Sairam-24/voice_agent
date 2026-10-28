// Converts microphone audio (Float32 at the device's rate, usually 44.1 or 48 kHz) into the
// 16 kHz signed 16-bit PCM the server expects. Pure logic, no browser APIs, so it runs in the
// AudioWorklet and in Node tests.
//
// Method: average all input samples that fall inside each output sample's time slot (a box
// filter). That is a crude low-pass filter, which is enough for speech recognition and avoids
// the worst aliasing of plain sample-dropping.

export class Downsampler {
  constructor(inputRate, outputRate = 16000, chunkSamples = 1600) {
    this.step = inputRate / outputRate; // input samples per output sample (may be fractional)
    this.chunkSamples = chunkSamples; // 1600 samples = 100 ms at 16 kHz
    this.phase = 0;
    this.sum = 0;
    this.count = 0;
    this.chunk = new Int16Array(chunkSamples);
    this.filled = 0;
  }

  // Feed a block of Float32 samples; returns zero or more complete Int16Array chunks.
  push(samples) {
    const ready = [];
    for (let i = 0; i < samples.length; i++) {
      this.sum += samples[i];
      this.count++;
      this.phase += 1;
      if (this.phase >= this.step) {
        this.phase -= this.step;
        this.chunk[this.filled++] = toInt16(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
        if (this.filled === this.chunkSamples) {
          ready.push(this.chunk);
          this.chunk = new Int16Array(this.chunkSamples);
          this.filled = 0;
        }
      }
    }
    return ready;
  }

  // Whatever is left over at the end of a recording (shorter than a full chunk).
  flush() {
    const rest = this.chunk.slice(0, this.filled);
    this.filled = 0;
    return rest;
  }
}

function toInt16(x) {
  const clamped = Math.max(-1, Math.min(1, x));
  return clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767);
}
