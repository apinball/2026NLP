from __future__ import annotations

import re
from dataclasses import dataclass

from src.logic_auditor.ontology import OntologyDB, TechEntry

# 4자리 숫자가 명확히 "연도" 컨텍스트에 있을 때만 잡는다.
# - 직전 자릿수가 없어야 함 (12345 안의 1234 차단)
# - 범위 1980~2029
# - 직후가 "년" 또는 공백/구두점/끝일 때만 (1900%, 2000건 같은 비-연도 차단)
_YEAR_RE = re.compile(
    r"(?<!\d)(19[8-9]\d|20[0-2]\d)(?=년|[\s.\-/,)\]]|$)"
)
_VERSION_RE = re.compile(r"\bv?(\d+(?:\.\d+){0,2})\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")

# 한국어 자모·완성형 한글 범위. 매칭 대상이 한국어 단어일 때 직전이 다른 한국어
# 글자/영문/숫자면 합성어(예: "온라인" 안의 "라인")이라 차단한다.
_KOREAN_CHAR = r"ㄱ-㆏가-힯"
# 자주 쓰이는 한국어 조사 시작 글자. "잔디로", "토스페이먼츠와", "카카오톡으로"
# 처럼 직후에 조사가 1~2글자 붙는 경우는 매칭을 허용해야 하므로 lookahead
# 화이트리스트로 처리한다 (조사 길이는 최대 2글자까지 인정).
_KOREAN_PARTICLES = "은는이가을를와과의도만에로으야서"

_ASCII_RE_CACHE: dict[str, re.Pattern[str]] = {}
_KOREAN_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _name_pattern(name: str) -> re.Pattern[str]:
    """ontology name/alias 매칭용 word-boundary regex 를 캐시해 반환."""
    if all(c.isascii() for c in name):
        cache = _ASCII_RE_CACHE
        if name not in cache:
            cache[name] = re.compile(
                rf"(?<![a-zA-Z0-9]){re.escape(name)}(?![a-zA-Z0-9])",
                re.IGNORECASE,
            )
        return cache[name]
    cache = _KOREAN_RE_CACHE
    if name not in cache:
        # 직전: 한국어/영문/숫자 아니어야 함 (합성어 차단)
        # 직후: 한국어가 아니거나 / 조사 한 글자 후 한국어가 아닐 때만 허용
        pattern = (
            rf"(?<![{_KOREAN_CHAR}a-zA-Z0-9])"
            rf"{re.escape(name)}"
            rf"(?:[{_KOREAN_PARTICLES}]{{1,2}}(?![{_KOREAN_CHAR}])|(?![{_KOREAN_CHAR}]))"
        )
        cache[name] = re.compile(pattern, re.IGNORECASE)
    return cache[name]


@dataclass
class RuleViolation:
    tech: str
    claimed_year: int
    actual_year: int
    snippet: str
    kind: str
    message: str


class RuleBasedAuditor:
    """온톨로지 DB를 참조해 연도/버전 팩트 오류를 탐지."""

    def __init__(self, ontology: OntologyDB | None = None) -> None:
        self.ontology = ontology or OntologyDB()

    def audit(self, text: str) -> list[RuleViolation]:
        violations: list[RuleViolation] = []
        for sent in self._split_sentences(text):
            years = [int(m.group(1)) for m in _YEAR_RE.finditer(sent)]
            if not years:
                continue
            claimed = min(years)
            for entry in self._matched_entries(sent):
                violations.extend(self._check_entry(entry, sent, claimed))
        return violations

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

    def _matched_entries(self, sentence: str) -> list[TechEntry]:
        matched: list[TechEntry] = []
        seen: set[str] = set()
        for entry in self.ontology.all_entries():
            if entry.name in seen:
                continue
            for name in (entry.name, *entry.aliases):
                if not name:
                    continue
                if _name_pattern(name).search(sentence):
                    matched.append(entry)
                    seen.add(entry.name)
                    break
        return matched

    @staticmethod
    def _check_entry(
        entry: TechEntry, sentence: str, claimed_year: int
    ) -> list[RuleViolation]:
        out: list[RuleViolation] = []
        if claimed_year < entry.released_year:
            out.append(
                RuleViolation(
                    tech=entry.name,
                    claimed_year=claimed_year,
                    actual_year=entry.released_year,
                    snippet=sentence,
                    kind="release_year",
                    message=(
                        f"{entry.name}는 {entry.released_year}년에 공개되었으나"
                        f" 문장은 {claimed_year}년을 주장합니다."
                    ),
                )
            )

        for match in _VERSION_RE.finditer(sentence):
            version = match.group(1)
            vyear = entry.version_year(version)
            if vyear is not None and claimed_year < vyear:
                out.append(
                    RuleViolation(
                        tech=entry.name,
                        claimed_year=claimed_year,
                        actual_year=vyear,
                        snippet=sentence,
                        kind="version_year",
                        message=(
                            f"{entry.name} {version}는 {vyear}년에 출시되었으나"
                            f" 문장은 {claimed_year}년을 주장합니다."
                        ),
                    )
                )
        return out