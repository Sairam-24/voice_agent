import pytest

from app.services.speech_text import SentenceChunker, clean_for_speech


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**Paris** is the capital.", "Paris is the capital."),
        ("# Heading\nSome text", "Heading Some text"),
        ("- one\n- two", "one two"),
        ("See [the docs](https://example.com/x) now", "See the docs now"),
        ("Great job 🎉!", "Great job !"),
        ("use `print()` here", "use print() here"),
        ("C# and F# are languages", "C# and F# are languages"),  # '#' only stripped as a heading marker
        ("  lots   of\n\n whitespace ", "lots of whitespace"),
    ],
)
def test_clean_for_speech(raw, expected):
    assert clean_for_speech(raw) == expected


def feed_all(chunker, deltas):
    out = []
    for delta in deltas:
        out.extend(chunker.feed(delta))
    return out, chunker.flush()


def test_sentences_are_released_only_once_the_next_text_confirms_the_boundary():
    chunker = SentenceChunker()
    assert chunker.feed("Hello there.") == []  # could still be "Hello there.5"
    assert chunker.feed(" How") == ["Hello there."]
    assert chunker.feed(" are you?") == []
    assert chunker.flush() == "How are you?"


def test_token_by_token_stream_gives_the_same_sentences_as_one_big_chunk():
    text = "The capital of France is Paris. It is a lovely city! Have you been there? I hope so."
    one_shot, tail_a = feed_all(SentenceChunker(), [text])
    per_char, tail_b = feed_all(SentenceChunker(), list(text))
    assert one_shot + [tail_a] == per_char + [tail_b]
    assert per_char + [tail_b] == [
        "The capital of France is Paris.",
        "It is a lovely city!",
        "Have you been there?",
        "I hope so.",
    ]


def test_decimals_and_abbreviations_do_not_split_sentences():
    text = "Dr. Smith paid 3.5 dollars. J. K. Rowling wrote it, e.g. in 1997. Done."
    sentences, tail = feed_all(SentenceChunker(), [text])
    assert sentences + [tail] == [
        "Dr. Smith paid 3.5 dollars.",
        "J. K. Rowling wrote it, e.g. in 1997.",
        "Done.",
    ]


def test_newline_ends_a_sentence():
    sentences, tail = feed_all(SentenceChunker(), ["first line\nsecond line"])
    assert sentences == ["first line"]
    assert tail == "second line"


RUN_ON = "this is a very long sentence without any full stop, but it keeps going and going and going"


def test_long_run_on_text_is_cut_at_a_comma_when_one_is_in_range():
    sentences, tail = feed_all(SentenceChunker(max_chars=60), [RUN_ON])
    assert sentences[0] == "this is a very long sentence without any full stop,"
    assert " ".join(sentences + [tail]) == RUN_ON


def test_long_run_on_text_falls_back_to_a_space_when_no_comma_is_in_range():
    sentences, tail = feed_all(SentenceChunker(max_chars=40), [RUN_ON])
    assert sentences, "a run-on must be released before the end of the stream"
    assert all(len(s) <= 40 for s in sentences)
    assert " ".join(sentences + [tail]) == RUN_ON


LONG_OPENING = "A rainbow forms when sunlight passes through water droplets in the air at a specific angle, causing refraction. Then more."


def test_first_clause_option_releases_a_long_opening_sentence_early():
    chunker = SentenceChunker(first_clause_chars=40)
    emitted_early = chunker.feed("A rainbow forms when sunlight passes through water droplets in the air at a specific angle, ")
    assert emitted_early == ["A rainbow forms when sunlight passes through water droplets in the air at a specific angle,"]
    rest, tail = feed_all(chunker, ["causing refraction. Then more."])
    assert rest + [tail] == ["causing refraction.", "Then more."]  # only the first chunk is cut early


def test_first_clause_option_is_off_by_default():
    sentences, tail = feed_all(SentenceChunker(), [LONG_OPENING])
    assert sentences + [tail] == [
        "A rainbow forms when sunlight passes through water droplets in the air at a specific angle, causing refraction.",
        "Then more.",
    ]


def test_first_clause_option_ignores_commas_that_come_too_early():
    chunker = SentenceChunker(first_clause_chars=40)
    # The only comma so far is at index 4, far below the 40-character minimum: no cut.
    assert chunker.feed("Well, that is a long story") == []
    # A second comma now appears after more than 40 characters, so the chunk is released there.
    assert chunker.feed(" to tell about rainbows and light, ok") == [
        "Well, that is a long story to tell about rainbows and light,"
    ]


def test_first_clause_option_prefers_an_earlier_sentence_end():
    sentences, tail = feed_all(SentenceChunker(first_clause_chars=10), ["Short one. Then a longer part, with a comma. End."])
    assert sentences[0] == "Short one."


def test_empty_input_yields_nothing():
    chunker = SentenceChunker()
    assert chunker.feed("") == []
    assert chunker.flush() is None
    assert chunker.feed("   ") == []
    assert chunker.flush() is None


def test_devanagari_danda_ends_a_sentence():
    chunker = SentenceChunker()
    sentences = chunker.feed("भारत की राजधानी नई दिल्ली है। यह बड़ा शहर है।")
    assert sentences == ["भारत की राजधानी नई दिल्ली है।"]
    assert chunker.flush() == "यह बड़ा शहर है।"
