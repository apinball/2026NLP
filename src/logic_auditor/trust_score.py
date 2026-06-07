from __future__ import annotations

from collections import defaultdict

from src.logic_auditor.models import AuditIssue


SAME_CLAIM_MAX_PENALTY = 35


def calculate_trust_score(issues: tuple[AuditIssue, ...]) -> int:
    penalties_by_claim: dict[str, int] = defaultdict(int)
    for issue in issues:
        penalties_by_claim[issue.claim_id] += issue.penalty
    total = sum(min(SAME_CLAIM_MAX_PENALTY, p) for p in penalties_by_claim.values())
    return max(0, min(100, 100 - total))
