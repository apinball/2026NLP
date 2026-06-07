from __future__ import annotations

import re
from dataclasses import dataclass

from src.logic_auditor.claim_extractor import ClaimExtractor
from src.logic_auditor.entity_extractor import EntityExtractor
from src.logic_auditor.models import AuditIssue, Claim, DetectedEntity, RuleAuditResult
from src.logic_auditor.ontology import OntologyDB, TechEntry
from src.logic_auditor.trust_score import calculate_trust_score

_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


@dataclass
class RuleViolation:
    tech: str
    claimed_year: int
    actual_year: int
    snippet: str
    kind: str
    message: str


SEVERITY_PENALTIES = {
    ("FACTUAL_ERROR", "HIGH"): 25,
    ("VERSION_ERROR", "HIGH"): 20,
    ("EXAGGERATION_SUSPECTED", "HIGH"): 12,
    ("EXAGGERATION_SUSPECTED", "MEDIUM"): 7,
}


class RuleBasedAuditor:
    """온톨로지 DB를 참조해 클레임 단위의 논리 오류를 탐지.

    `audit()`는 기존 호출부와 호환되는 RuleViolation 리스트를 반환하고,
    `audit_detailed()`는 claim/entity/issue/score가 포함된 상세 리포트를 반환한다.
    """

    def __init__(
        self,
        ontology: OntologyDB | None = None,
        claim_extractor: ClaimExtractor | None = None,
    ) -> None:
        self.ontology = ontology or OntologyDB()
        self.claim_extractor = claim_extractor or ClaimExtractor(
            EntityExtractor(self.ontology)
        )

    def audit(self, text: str) -> list[RuleViolation]:
        return self.issues_to_violations(self.audit_detailed(text).issues)

    @staticmethod
    def issues_to_violations(issues: tuple[AuditIssue, ...]) -> list[RuleViolation]:
        return [_issue_to_violation(issue) for issue in issues]

    def audit_detailed(self, text: str) -> RuleAuditResult:
        claims = self.claim_extractor.extract(text)
        issues: list[AuditIssue] = []
        for claim in claims:
            issues.extend(self._timeline_issues(claim))
            issues.extend(self._feature_issues(claim))
            issues.extend(self._exaggeration_issues(claim))
        numbered = _renumber_issues(tuple(_dedupe_issues(issues)))
        return RuleAuditResult(
            claims=claims,
            issues=numbered,
            trust_score=calculate_trust_score(numbered),
        )

    def _timeline_issues(self, claim: Claim) -> list[AuditIssue]:
        years = claim.year_mentions
        if not years:
            return []
        claim_year = min(years)
        issues: list[AuditIssue] = []
        for tech in _entities(claim, "TECH"):
            release_year = int(tech.metadata.get("release_year") or 0)
            if release_year and claim_year < release_year:
                issues.append(
                    _issue(
                        claim=claim,
                        issue_type="FACTUAL_ERROR",
                        severity="HIGH",
                        confidence=0.98,
                        message=(
                            f"{tech.normalized}는 {release_year}년에 공개되었으므로 "
                            f"{claim_year}년에 사용했다는 주장은 시간상 불가능합니다."
                        ),
                        evidence={
                            "tech": tech.normalized,
                            "claim_year": claim_year,
                            "release_year": release_year,
                            "kind": "release_year",
                        },
                    )
                )
            entry = self.ontology.lookup(tech.normalized)
            if entry:
                issues.extend(
                    self._version_timeline_issues(claim, tech, entry, claim_year)
                )
        return issues

    def _version_timeline_issues(
        self,
        claim: Claim,
        tech: DetectedEntity,
        entry: TechEntry,
        claim_year: int,
    ) -> list[AuditIssue]:
        issues: list[AuditIssue] = []
        for version in _versions_near_tech(claim, tech):
            release_year = entry.version_year(version.normalized)
            if release_year is not None and claim_year < release_year:
                issues.append(
                    _issue(
                        claim=claim,
                        issue_type="VERSION_ERROR",
                        severity="HIGH",
                        confidence=0.97,
                        message=(
                            f"{entry.name} {version.normalized}는 {release_year}년에 "
                            f"출시되었으므로 {claim_year}년 사용 주장은 버전 시점과 맞지 않습니다."
                        ),
                        evidence={
                            "tech": entry.name,
                            "version": version.normalized,
                            "claim_year": claim_year,
                            "version_release_year": release_year,
                            "kind": "version_year",
                        },
                    )
                )
        return issues

    def _feature_issues(self, claim: Claim) -> list[AuditIssue]:
        years = claim.year_mentions
        issues: list[AuditIssue] = []
        for feature in _entities(claim, "FEATURE"):
            tech_name = str(feature.metadata.get("tech") or "")
            introduced_year = int(feature.metadata.get("introduced_year") or 0)
            introduced_version = str(feature.metadata.get("introduced_version") or "")
            if years and introduced_year and min(years) < introduced_year:
                claim_year = min(years)
                issues.append(
                    _issue(
                        claim=claim,
                        issue_type="VERSION_ERROR",
                        severity="HIGH",
                        confidence=0.95,
                        message=(
                            f"{tech_name} {feature.normalized}는 {introduced_year}년 이후 "
                            f"도입되어 {claim_year}년 사용 주장은 호환되지 않습니다."
                        ),
                        evidence={
                            "tech": tech_name,
                            "feature": feature.normalized,
                            "claim_year": claim_year,
                            "introduced_year": introduced_year,
                            "introduced_version": introduced_version,
                            "kind": "feature_year",
                        },
                    )
                )
            for version in _versions_for_feature(claim, feature):
                if (
                    introduced_version
                    and _compare_versions(version.normalized, introduced_version) < 0
                ):
                    issues.append(
                        _issue(
                            claim=claim,
                            issue_type="VERSION_ERROR",
                            severity="HIGH",
                            confidence=0.93,
                            message=(
                                f"{feature.normalized}는 {tech_name} {introduced_version} 이후 "
                                f"기능이므로 {version.normalized} 버전과 호환되지 않습니다."
                            ),
                            evidence={
                                "tech": tech_name,
                                "feature": feature.normalized,
                                "claimed_version": version.normalized,
                                "introduced_version": introduced_version,
                                "kind": "feature_version",
                            },
                        )
                    )
        return issues

    @staticmethod
    def _exaggeration_issues(claim: Claim) -> list[AuditIssue]:
        duration = claim.duration_months
        if duration is None:
            return []
        role_values = {entity.normalized for entity in _entities(claim, "ROLE")}
        scale_values = {entity.normalized for entity in _entities(claim, "SCALE")}
        solo_or_owner = bool(
            role_values & {"solo", "full_ownership", "architect", "leadership"}
        )
        broad_role = bool(
            role_values & {"design", "operation", "architect", "leadership"}
        )
        large_scale = bool(scale_values)

        if duration <= 3 and solo_or_owner and broad_role and large_scale:
            return [
                _issue(
                    claim=claim,
                    issue_type="EXAGGERATION_SUSPECTED",
                    severity="HIGH",
                    confidence=0.84,
                    message="짧은 기간, 단독/리딩 역할, 대규모 표현이 함께 등장해 실제 기여 범위 확인이 필요합니다.",
                    evidence={
                        "duration_months": duration,
                        "role_signals": sorted(role_values),
                        "scale_signals": sorted(scale_values),
                        "kind": "experience_scale",
                    },
                )
            ]
        if duration <= 3 and solo_or_owner and large_scale:
            return [
                _issue(
                    claim=claim,
                    issue_type="EXAGGERATION_SUSPECTED",
                    severity="MEDIUM",
                    confidence=0.72,
                    message="매우 짧은 기간에 비해 주장한 시스템 규모와 담당 범위가 큽니다.",
                    evidence={
                        "duration_months": duration,
                        "role_signals": sorted(role_values),
                        "scale_signals": sorted(scale_values),
                        "kind": "experience_scale",
                    },
                )
            ]
        return []


def _entities(claim: Claim, entity_type: str) -> tuple[DetectedEntity, ...]:
    return tuple(entity for entity in claim.entities if entity.entity_type == entity_type)


def _versions_for_feature(
    claim: Claim,
    feature: DetectedEntity,
) -> tuple[DetectedEntity, ...]:
    tech_name = str(feature.metadata.get("tech") or "")
    related_versions: list[DetectedEntity] = []
    matched_feature_tech = False
    for tech in _entities(claim, "TECH"):
        if tech.normalized.lower() == tech_name.lower():
            matched_feature_tech = True
            related_versions.extend(_versions_near_tech(claim, tech))
    if matched_feature_tech:
        return tuple(related_versions)
    return tuple(
        version
        for version in _entities(claim, "VERSION")
        if 0 <= feature.span.start - version.span.end <= 12
        or 0 <= version.span.start - feature.span.end <= 12
    )


def _versions_near_tech(claim: Claim, tech: DetectedEntity) -> tuple[DetectedEntity, ...]:
    versions = _entities(claim, "VERSION")
    if not versions:
        return ()
    # Korean/English resume phrasing usually keeps the version within a short window:
    # "React 16", "Next.js 13", "TensorFlow 2.0".
    near = [
        version
        for version in versions
        if 0 <= version.span.start - tech.span.end <= 8
    ]
    return tuple(near)


def _issue_to_violation(issue: AuditIssue) -> RuleViolation:
    evidence = issue.evidence
    return RuleViolation(
        tech=str(evidence.get("tech") or evidence.get("feature") or "claim"),
        claimed_year=int(
            evidence.get("claim_year") or evidence.get("duration_months") or 0
        ),
        actual_year=int(
            evidence.get("release_year")
            or evidence.get("version_release_year")
            or evidence.get("introduced_year")
            or 0
        ),
        snippet=issue.highlight_text,
        kind=str(evidence.get("kind") or issue.issue_type.lower()),
        message=issue.message,
    )


def _issue(
    claim: Claim,
    issue_type: str,
    severity: str,
    confidence: float,
    message: str,
    evidence: dict,
) -> AuditIssue:
    return AuditIssue(
        issue_id="",
        claim_id=claim.claim_id,
        issue_type=issue_type,
        severity=severity,
        confidence=confidence,
        span=claim.span,
        highlight_text=claim.claim_text,
        message=message,
        evidence=evidence,
        penalty=SEVERITY_PENALTIES[(issue_type, severity)],
    )


def _renumber_issues(issues: tuple[AuditIssue, ...]) -> tuple[AuditIssue, ...]:
    return tuple(
        AuditIssue(
            issue_id=f"i{index}",
            claim_id=issue.claim_id,
            issue_type=issue.issue_type,
            severity=issue.severity,
            confidence=issue.confidence,
            span=issue.span,
            highlight_text=issue.highlight_text,
            message=issue.message,
            evidence=issue.evidence,
            penalty=issue.penalty,
        )
        for index, issue in enumerate(issues, start=1)
    )


def _dedupe_issues(issues: list[AuditIssue]) -> list[AuditIssue]:
    best: dict[tuple[str, str, str], AuditIssue] = {}
    for issue in issues:
        key = (
            issue.claim_id,
            issue.issue_type,
            str(sorted(issue.evidence.items())),
        )
        if key not in best or issue.confidence > best[key].confidence:
            best[key] = issue
    return list(best.values())


def _compare_versions(left: str, right: str) -> int:
    left_parts = _version_parts(left)
    right_parts = _version_parts(right)
    max_len = max(len(left_parts), len(right_parts))
    left_parts.extend([0] * (max_len - len(left_parts)))
    right_parts.extend([0] * (max_len - len(right_parts)))
    if left_parts < right_parts:
        return -1
    if left_parts > right_parts:
        return 1
    return 0


def _version_parts(version: str) -> list[int]:
    return [int(part) for part in version.split(".") if part.isdigit()]
