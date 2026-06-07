from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class DetectedEntity:
    entity_type: str
    text: str
    normalized: str
    span: TextSpan
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim_text: str
    span: TextSpan
    entities: tuple[DetectedEntity, ...]
    claim_type: str = "general"

    @property
    def tech_mentions(self) -> tuple[str, ...]:
        return tuple(
            entity.normalized
            for entity in self.entities
            if entity.entity_type == "TECH"
        )

    @property
    def year_mentions(self) -> tuple[int, ...]:
        return tuple(
            int(entity.normalized)
            for entity in self.entities
            if entity.entity_type == "YEAR" and entity.normalized.isdigit()
        )

    @property
    def duration_months(self) -> int | None:
        durations = [
            int(entity.metadata["months"])
            for entity in self.entities
            if entity.entity_type == "DURATION" and "months" in entity.metadata
        ]
        return min(durations) if durations else None


@dataclass(frozen=True)
class AuditIssue:
    issue_id: str
    claim_id: str
    issue_type: str
    severity: str
    confidence: float
    span: TextSpan
    highlight_text: str
    message: str
    evidence: dict[str, Any]
    penalty: int


@dataclass(frozen=True)
class RuleAuditResult:
    claims: tuple[Claim, ...]
    issues: tuple[AuditIssue, ...]
    trust_score: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
