from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import DATA_DIR

JOB_CATEGORY_MAP_PATH = DATA_DIR / "job_category_map.json"
UNCATEGORIZED = "미분류"
OTHER = "기타"


def _compile_pattern(raw: str) -> tuple[str, re.Pattern[str] | None]:
    """ASCII 패턴은 word-boundary regex 로, 한글 등은 substring 매칭."""
    p = raw.lower()
    if all(c.isascii() for c in p):
        return p, re.compile(
            rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", re.IGNORECASE
        )
    return p, None


class JobCategoryNormalizer:
    """자유서식 직무 문자열을 표준 카테고리 라벨로 변환.

    규칙은 `data/job_category_map.json` 의 우선순위 순서.
    ASCII 패턴(it, hr, cs 등)은 단어 경계로 매칭되어 부분 일치 노이즈를 피하고,
    한글 패턴(개발, 마케팅 등)은 substring 으로 합성어를 폭넓게 잡는다.
    """

    def __init__(self, path: Path = JOB_CATEGORY_MAP_PATH) -> None:
        self.path = path
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.rules: list[tuple[str, list[tuple[str, re.Pattern[str] | None]]]] = []
        for item in raw:
            compiled = [_compile_pattern(p) for p in item["patterns"]]
            self.rules.append((item["category"], compiled))

    def normalize(self, raw: str | None) -> str:
        if not raw:
            return UNCATEGORIZED
        lowered = raw.lower()
        for category, patterns in self.rules:
            for pat, regex in patterns:
                if regex is not None:
                    if regex.search(raw):
                        return category
                elif pat in lowered:
                    return category
        return OTHER


_DEFAULT: JobCategoryNormalizer | None = None


def normalize_job(raw: str | None) -> str:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = JobCategoryNormalizer()
    return _DEFAULT.normalize(raw)
