from __future__ import annotations

import pytest

from src.config import DATA_DIR
from src.data.loaders import (
    iter_linkareer,
    iter_naver_cafe,
    iter_resumes,
    load_ceragem_positions,
)


def _has(path_name: str) -> bool:
    return (DATA_DIR / path_name).exists()


@pytest.mark.skipif(
    not _has("ceragem_job_posting.json"), reason="ceragem dataset absent"
)
def test_load_ceragem_positions_yields_27() -> None:
    positions = load_ceragem_positions()
    assert len(positions) == 27
    sample = positions[0]
    assert sample.company
    assert sample.job_title
    assert sample.requirements


@pytest.mark.skipif(
    not _has("linkareer_cover_letter_parsed.jsonl"),
    reason="linkareer dataset absent",
)
def test_iter_linkareer_first_record_has_sections() -> None:
    it = iter_linkareer()
    first = next(it)
    assert first.source == "linkareer"
    assert first.company
    assert first.sections
    assert first.full_text


@pytest.mark.skipif(
    not _has("naver_cafe_passassay_parsed.jsonl"),
    reason="naver cafe dataset absent",
)
def test_iter_naver_cafe_first_record_loads() -> None:
    it = iter_naver_cafe()
    first = next(it)
    assert first.source == "naver_cafe"
    assert first.url


@pytest.mark.skipif(
    not (
        _has("linkareer_cover_letter_parsed.jsonl")
        or _has("naver_cafe_passassay_parsed.jsonl")
    ),
    reason="resume datasets absent",
)
def test_iter_resumes_skips_empty_sections() -> None:
    count = 0
    for r in iter_resumes():
        assert r.sections
        count += 1
        if count >= 5:
            break
    assert count > 0
