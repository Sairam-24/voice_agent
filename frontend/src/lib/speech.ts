// Shows a spoken reply's text as it is heard instead of before. The server sends, with each audio
// chunk, the text that chunk speaks; that text is revealed word by word over the time its audio
// plays. Word timings are not known, so within one piece of text the words are spread evenly over
// its audio. Pure (the clock is passed in), so Node can test it.

/** A typical speaking rate, used only while a piece's audio is still arriving and its length is unknown. */
export const CHARS_PER_SECOND = 14;

interface Piece {
  id: number;
  text: string;
  start: number;
  end: number;
  complete: boolean;
}

export class SpeechTimeline {
  private readonly pieces: Piece[] = [];

  /** Audio of piece `id` (saying `text`) was scheduled to play from `start` to `end`, in seconds on the player's clock. */
  add(id: number, text: string, start: number, end: number): void {
    const last = this.pieces.at(-1);
    if (last && last.id === id) {
      last.end = Math.max(last.end, end);
      return;
    }
    if (last) last.complete = true; // a new piece started, so the previous one has all its audio
    this.pieces.push({ id, text, start, end, complete: false });
  }

  /** No more audio is coming, so every piece's length is known. */
  complete(): void {
    for (const piece of this.pieces) piece.complete = true;
  }

  /** The text heard by `now`: the pieces already played, plus the words of the one playing. */
  heard(now: number): string {
    const parts: string[] = [];
    for (const piece of this.pieces) {
      if (now < piece.start) break;
      const end = piece.complete ? piece.end : Math.max(piece.end, piece.start + piece.text.length / CHARS_PER_SECOND);
      const progress = end > piece.start ? (now - piece.start) / (end - piece.start) : 1;
      if (progress >= 1) {
        parts.push(piece.text);
        continue;
      }
      const partial = wordsUpTo(piece.text, Math.floor(piece.text.length * progress));
      if (partial) parts.push(partial);
      break;
    }
    return parts.join(" ");
  }
}

/** The whole words within the first `chars` characters. Text written without spaces is cut anywhere. */
export function wordsUpTo(text: string, chars: number): string {
  if (chars <= 0) return "";
  if (chars >= text.length) return text;
  if (!text.includes(" ")) return text.slice(0, chars);
  const space = text.lastIndexOf(" ", chars);
  return space > 0 ? text.slice(0, space) : "";
}
