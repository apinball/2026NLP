from __future__ import annotations

from dataclasses import dataclass, field

from src.logic_auditor.ml_detector import AIDetectionResult, AIGenerationDetector
from src.logic_auditor.rule_based import RuleBasedAuditor, RuleViolation


@dataclass
class AuditReport:
    violations: list[RuleViolation] = field(default_factory=list)
    ai_detection: AIDetectionResult | None = None
    trust_score: int = 100
    suspect_spans: list[str] = field(default_factory=list)


class LogicAuditorPipeline:
    """Module B — Logic Auditor 오케스트레이터."""

    PENALTY_PER_VIOLATION = 15
    MAX_RULE_PENALTY = 60
    MAX_ML_PENALTY = 30

    def __init__(
        self,
        rule_auditor: RuleBasedAuditor | None = None,
        ai_detector: AIGenerationDetector | None = None,
    ) -> None:
        self.rule_auditor = rule_auditor or RuleBasedAuditor()
        self.ai_detector = ai_detector

    def run(self, text: str) -> AuditReport:
        violations = self.rule_auditor.audit(text)
        ai_result = self.ai_detector.score(text) if self.ai_detector else None

        score = 100
        score -= min(
            self.MAX_RULE_PENALTY, self.PENALTY_PER_VIOLATION * len(violations)
        )
        if ai_result is not None:
            score -= int(round(ai_result.ai_probability * self.MAX_ML_PENALTY))
        score = max(0, score)

        spans = sorted({v.snippet for v in violations})

        return AuditReport(
            violations=violations,
            ai_detection=ai_result,
            trust_score=score,
            suspect_spans=spans,
        )
