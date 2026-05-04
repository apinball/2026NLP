from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.logic_auditor.pipeline import AuditReport, LogicAuditorPipeline
from src.reverse_job.pipeline import ReverseJobPipeline, ReverseJobResult


@dataclass
class IntegrationReport:
    """채용공고 분석 + 포트폴리오 검증을 합친 통합 리포트."""

    job_analysis: ReverseJobResult
    resume_audit: AuditReport
    skill_gap: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": {
                "explicit": self.job_analysis.explicit.tech_stack,
                "implicit_arm": self.job_analysis.implicit_arm,
                "implicit_llm": self.job_analysis.implicit_llm,
                "implicit_union": self.job_analysis.implicit_union,
            },
            "resume": {
                "trust_score": self.resume_audit.trust_score,
                "violations": [
                    {
                        "tech": v.tech,
                        "kind": v.kind,
                        "claimed_year": v.claimed_year,
                        "actual_year": v.actual_year,
                        "snippet": v.snippet,
                        "message": v.message,
                    }
                    for v in self.resume_audit.violations
                ],
                "ai_signal": (
                    {
                        "perplexity": self.resume_audit.ai_detection.perplexity,
                        "burstiness": self.resume_audit.ai_detection.burstiness,
                        "ai_probability": self.resume_audit.ai_detection.ai_probability,
                    }
                    if self.resume_audit.ai_detection is not None
                    else None
                ),
            },
            "skill_gap": self.skill_gap,
        }


class IntegrationPipeline:
    """Module A → Module B 통합. 채용공고 분석으로 요구 역량을 추출하고,
    이력서·포트폴리오 검증으로 신뢰도를 산출한 뒤, 두 결과를 비교해
    지원자가 갖춰야 할 미보유 역량(skill_gap)을 도출한다.
    """

    def __init__(
        self,
        reverse_job: ReverseJobPipeline | None = None,
        audit: LogicAuditorPipeline | None = None,
    ) -> None:
        self.reverse_job = reverse_job or ReverseJobPipeline(load_ner=False)
        self.audit = audit or LogicAuditorPipeline()

    def run(
        self,
        job_text: str,
        resume_text: str,
        corpus: list[str] | None = None,
    ) -> IntegrationReport:
        analysis = self.reverse_job.run(job_text, corpus=corpus)
        audit_report = self.audit.run(resume_text)

        required = set(analysis.explicit.tech_stack) | set(analysis.implicit_union)
        # 이력서에서 자연어로 언급된 기술과 비교 (find_tech_keywords 재사용)
        from src.reverse_job.extractor import find_tech_keywords

        possessed = set(find_tech_keywords(resume_text))
        gap = sorted(required - possessed)

        return IntegrationReport(
            job_analysis=analysis,
            resume_audit=audit_report,
            skill_gap=gap,
        )