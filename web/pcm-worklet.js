// AudioWorklet that runs on the audio thread: downsample the microphone to 16 kHz PCM16 and
// hand 100 ms chunks to the main thread, together with a real input level for the meter.
import { Downsampler } from "./downsampler.js";

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.downsampler = new Downsampler(sampleRate); // `sampleRate` is a global of the worklet scope
    this.peak = 0;
    this.stopped = false;
    this.port.onmessage = (event) => {
      if (event.data === "flush") {
        const rest = this.downsampler.flush();
        if (rest.length) this.port.postMessage({ pcm: rest.buffer }, [rest.buffer]);
        // Messages arrive in order, so once the main thread sees this, the last audio is already sent.
        this.port.postMessage({ flushed: true });
        this.stopped = true;
      }
    };
  }

  process(inputs) {
    if (this.stopped) return false;
    const channel = inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) this.peak = Math.max(this.peak, Math.abs(channel[i]));
    for (const chunk of this.downsampler.push(channel)) {
      this.port.postMessage({ pcm: chunk.buffer, level: this.peak }, [chunk.buffer]);
      this.peak = 0;
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
