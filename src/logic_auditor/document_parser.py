from __future__ import annotations

from src.logic_auditor.models import TextSpan


KOREAN_SENTENCE_ENDINGS = (
    "했습니다",
    "합니다",
    "습니다",
    "입니다",
    "였습니다",
    "하였습니다",
    "수행했습니다",
    "개선했습니다",
    "달성했습니다",
)


def sentence_spans(text: str) -> list[TextSpan]:
    """Split Korean/English resume text while preserving original offsets."""
    spans: list[TextSpan] = []
    start = 0
    for index, char in enumerate(text):
        if char == "\n":
            _append_trimmed_span(text, start, index, spans)
            start = index + 1
            continue
        if char in ".!?。":
            prev = text[index - 1] if index > 0 else ""
            nxt = text[index + 1] if index + 1 < len(text) else ""
            if char == "." and prev.isalnum() and nxt.isalnum():
                continue
            if nxt and not nxt.isspace():
                continue
            _append_trimmed_span(text, start, index + 1, spans)
            start = index + 1
            continue
        if char.isspace() and _has_korean_ending_before(text, index):
            _append_trimmed_span(text, start, index, spans)
            start = index + 1
    _append_trimmed_span(text, start, len(text), spans)
    return spans


def _append_trimmed_span(
    text: str,
    raw_start: int,
    raw_end: int,
    spans: list[TextSpan],
) -> None:
    while raw_start < raw_end and text[raw_start].isspace():
        raw_start += 1
    while raw_end > raw_start and text[raw_end - 1].isspace():
        raw_end -= 1
    if raw_start < raw_end:
        spans.append(TextSpan(raw_start, raw_end))


def _has_korean_ending_before(text: str, index: int) -> bool:
    previous = text[max(0, index - 18) : index]
    return any(previous.endswith(ending) for ending in KOREAN_SENTENCE_ENDINGS)
