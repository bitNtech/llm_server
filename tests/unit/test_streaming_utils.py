from __future__ import annotations

from app.core.segmenter import SentenceSegmenter
from app.core.toolcalls import ToolCallAccumulator


def feed(seg: SentenceSegmenter, text: str, step: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), step):
        out += seg.push(text[i:i + step])
    tail = seg.flush()
    return out + ([tail] if tail else [])


def test_sentences_are_cut_at_terminators():
    seg = SentenceSegmenter(min_chars=5)
    assert feed(seg, "Hello there. How are you today? I am fine!") == [
        "Hello there.", "How are you today?", "I am fine!"]


def test_segment_emitted_before_stream_ends():
    seg = SentenceSegmenter(min_chars=5)
    assert seg.push("Your appointment is confirmed.") == []  # no whitespace yet: may be "confirmed.5"
    assert seg.push(" See you") == ["Your appointment is confirmed."]


def test_short_fragments_merge_with_the_next_sentence():
    seg = SentenceSegmenter(min_chars=10)
    assert feed(seg, "Ok. Let me check that for you.") == ["Ok. Let me check that for you."]


def test_abbreviations_and_decimals_do_not_split():
    seg = SentenceSegmenter(min_chars=4)
    assert feed(seg, "Dr. Kumar sees you at 3.30 today. Bye now") == ["Dr. Kumar sees you at 3.30 today.", "Bye now"]


def test_tamil_and_devanagari_text():
    seg = SentenceSegmenter(min_chars=4)
    assert feed(seg, "வணக்கம். உங்களுக்கு எப்படி உதவலாம்? नमस्ते। धन्यवाद") == [
        "வணக்கம்.", "உங்களுக்கு எப்படி உதவலாம்?", "नमस्ते।", "धन्यवाद"]


def test_long_text_without_punctuation_is_split_at_soft_break():
    seg = SentenceSegmenter(min_chars=4, max_chars=30)
    parts = feed(seg, "one two three four five six seven eight nine ten eleven twelve", step=5)
    assert all(len(p) <= 30 for p in parts)
    assert " ".join(parts) == "one two three four five six seven eight nine ten eleven twelve"


def test_newlines_are_boundaries():
    seg = SentenceSegmenter(min_chars=3)
    assert feed(seg, "First line\nSecond line") == ["First line", "Second line"]


def test_tool_call_fragments_assemble():
    acc = ToolCallAccumulator()
    started = acc.add([{"index": 0, "id": "call_1", "type": "function",
                        "function": {"name": "lookup", "arguments": ""}}])
    assert [c.name for c in started] == ["lookup"]
    assert acc.add([{"index": 0, "function": {"arguments": '{"id": '}}]) == []
    acc.add([{"index": 0, "function": {"arguments": "7}"}}])
    acc.add([{"index": 1, "function": {"name": "other", "arguments": "{}"}}])
    calls = acc.finish()
    assert [(c.id, c.name, c.arguments) for c in calls][0] == ("call_1", "lookup", '{"id": 7}')
    assert calls[1].id.startswith("call_")  # generated when the engine sends none
    assert calls[0].to_message()["function"] == {"name": "lookup", "arguments": '{"id": 7}'}


def test_empty_accumulator_is_falsy():
    acc = ToolCallAccumulator()
    assert not acc and acc.finish() == []
