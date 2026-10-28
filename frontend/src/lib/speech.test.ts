import assert from "node:assert/strict";
import { test } from "node:test";
import { CHARS_PER_SECOND, SpeechTimeline, wordsUpTo } from "./speech.ts";

test("pieces are revealed word by word while their audio plays", () => {
  const timeline = new SpeechTimeline();
  timeline.add(0, "The capital of France is Paris.", 1, 3);
  timeline.add(1, "It is a lovely city.", 3, 4);
  timeline.complete();
  assert.equal(timeline.heard(0.5), "", "nothing before the audio starts");
  assert.equal(timeline.heard(2), "The capital of", "half of the first piece, whole words only");
  assert.equal(timeline.heard(3.05), "The capital of France is Paris.");
  assert.equal(timeline.heard(3.2), "The capital of France is Paris. It", "the next piece follows at once");
  assert.equal(timeline.heard(10), "The capital of France is Paris. It is a lovely city.");
});

test("chunks of one piece extend it instead of starting a new one", () => {
  const timeline = new SpeechTimeline();
  timeline.add(0, "one two three four", 0, 1);
  timeline.add(0, "one two three four", 1, 2);
  timeline.complete();
  assert.equal(timeline.heard(1.1), "one two", "the piece lasts two seconds, not one");
});

test("while a piece's audio is still arriving, its length is estimated from a speaking rate", () => {
  const text = "a ".repeat(70).trim(); // 139 characters: about 10 s at the estimated rate
  const timeline = new SpeechTimeline();
  timeline.add(0, text, 0, 1); // only the first second has arrived
  const heardAfterOne = timeline.heard(1);
  assert.ok(heardAfterOne.length < text.length / 5, "it does not race through the text");
  assert.ok(Math.abs(heardAfterOne.length - CHARS_PER_SECOND) <= 2);
});

test("words are never cut, except in text written without spaces", () => {
  assert.equal(wordsUpTo("నమస్కారం మీకు ఎలా", 11), "నమస్కారం");
  assert.equal(wordsUpTo("hello world", 3), "");
  assert.equal(wordsUpTo("你好世界", 2), "你好");
  assert.equal(wordsUpTo("short", 99), "short");
});
