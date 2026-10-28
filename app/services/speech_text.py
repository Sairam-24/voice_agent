"""Prepare LLM output for speech: strip markup and cut a token stream into sentences."""

import re

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING_OR_QUOTE = re.compile(r"^\s*(?:#+|>)\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-•]\s+", re.MULTILINE)
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿️]")
_EMPHASIS_CHARS = re.compile(r"[*_`~]+")


def clean_for_speech(text: str) -> str:
    """Remove markdown and emoji that a TTS engine would read aloud ("asterisk asterisk")."""
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _HEADING_OR_QUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMOJI.sub("", text)
    text = _EMPHASIS_CHARS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# A sentence ends at . ! ? (or the Devanagari danda and CJK full stops), plus closing quotes or
# brackets, followed by whitespace; or at a newline.
_BOUNDARY = re.compile(r"[.!?।॥。！？]+[\"')\]]*\s|\n+")
_CLAUSE_BOUNDARY = re.compile(r"[,;:]\s")
_LAST_WORD = re.compile(r"(\S+)$")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no"}


class SentenceChunker:
    """Turns a stream of text deltas into speakable sentences.

    Speaking sentence by sentence lets TTS start while the LLM is still generating.
    A boundary needs the whitespace after the punctuation, so a sentence is only
    released once the next delta confirms it ended; flush() releases the tail.
    """

    def __init__(self, max_chars: int = 200, first_clause_chars: int = 0):
        self._buffer = ""
        self._max_chars = max_chars
        # If > 0, the very first chunk may end at a comma/semicolon/colon once at least this many
        # characters precede it, so a long opening sentence does not delay the first sound.
        self._first_clause_chars = first_clause_chars
        self._emitted_any = False

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        sentences = []
        while (cut := self._find_cut()) is not None:
            sentence, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:]
            if sentence:
                sentences.append(sentence)
                self._emitted_any = True
        # A run-on with no punctuation would delay speech indefinitely: cut it at a comma or space.
        while len(self._buffer) > self._max_chars:
            cut = self._soft_cut()
            sentences.append(self._buffer[:cut].strip())
            self._buffer = self._buffer[cut:]
        return sentences

    def flush(self) -> str | None:
        tail, self._buffer = self._buffer.strip(), ""
        return tail or None

    def _find_cut(self) -> int | None:
        cut = None
        for match in _BOUNDARY.finditer(self._buffer):
            if match.group().startswith("\n") or not self._is_abbreviation(match.start()):
                cut = match.end()
                break
        if self._first_clause_chars and not self._emitted_any:
            for match in _CLAUSE_BOUNDARY.finditer(self._buffer):
                if match.start() >= self._first_clause_chars:
                    return match.end() if cut is None else min(cut, match.end())
        return cut

    def _is_abbreviation(self, punctuation_start: int) -> bool:
        word = _LAST_WORD.search(self._buffer[:punctuation_start])
        if not word:
            return False
        token = word.group(1).lower().strip("\"'([")
        # "Dr." / "e.g." are abbreviations; so is a lone initial as in "J. K. Rowling".
        return token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())

    def _soft_cut(self) -> int:
        window = self._buffer[: self._max_chars]
        for separator in (", ", "; ", " "):
            index = window.rfind(separator)
            if index > 0:
                return index + len(separator)
        return self._max_chars
