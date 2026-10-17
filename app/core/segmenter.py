"""Incremental sentence segmentation for streaming TTS.

TTS engines sound best when fed whole phrases, and the first phrase should leave as soon as it
is complete. `push()` returns every segment completed by a delta; `flush()` returns the tail.
Handles Latin, Devanagari (danda) and CJK terminators; Tamil uses Latin punctuation.
"""
from __future__ import annotations

import re

_TERMINATORS = ".!?…。！？।॥"
# a terminator (optionally followed by closing quotes/brackets) followed by whitespace, or a newline
_BOUNDARY = re.compile(rf"[{re.escape(_TERMINATORS)}]+[\"'”’)\]]*(?=\s)|\n+")
_SOFT_BREAK = re.compile(r"[,;:—–]\s|\s")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "approx", "dept",
}


class SentenceSegmenter:
    def __init__(self, min_chars: int = 8, max_chars: int = 220) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""
        self.count = 0

    def push(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._buf += delta
        out: list[str] = []
        while True:
            seg = self._cut()
            if seg is None:
                break
            out.append(seg)
        return out

    def flush(self) -> str | None:
        tail, self._buf = self._buf.strip(), ""
        if tail:
            self.count += 1
            return tail
        return None

    def _emit(self, end: int) -> str | None:
        seg, self._buf = self._buf[:end].strip(), self._buf[end:].lstrip()
        if not seg:
            return None
        self.count += 1
        return seg

    def _cut(self) -> str | None:
        for match in _BOUNDARY.finditer(self._buf):
            end = match.end()
            candidate = self._buf[:end].strip()
            if len(candidate) < self.min_chars:
                continue
            last_word = candidate.rstrip(_TERMINATORS + "\"'”’)]").rsplit(None, 1)[-1].lower()
            if candidate.endswith(".") and last_word.rstrip(".") in _ABBREVIATIONS:
                continue
            return self._emit(end)
        if len(self._buf) > self.max_chars:
            # no sentence end in sight: break at the last soft boundary inside the window
            window = self._buf[: self.max_chars]
            breaks = [m.end() for m in _SOFT_BREAK.finditer(window)]
            return self._emit(breaks[-1] if breaks else self.max_chars)
        return None
