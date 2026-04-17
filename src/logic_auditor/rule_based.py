from __future__ import annotations

import re
from dataclasses import dataclass

from src.logic_auditor.ontology import OntologyDB, TechEntry

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_VERSION_RE = re.compile(r"\bv?(\d+(?:\.\d+){0,2})\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


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
            years = [int(m.group()) for m in _YEAR_RE.finditer(sent)]
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
        lowered = sentence.lower()
        matched: list[TechEntry] = []
        seen: set[str] = set()
        for entry in self.ontology.all_entries():
            names = [entry.name, *entry.aliases]
            for name in names:
                if name.lower() in lowered and entry.name not in seen:
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
