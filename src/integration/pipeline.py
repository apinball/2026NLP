from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.logic_auditor.pipeline import AuditReport, LogicAuditorPipeline
from src.reverse_job.extractor import find_tech_keywords, normalize_skill_token
from src.reverse_job.pipeline import ReverseJobPipeline, ReverseJobResult


@dataclass
class SkillMatch:
    """채용공고 요구 역량 ↔ 이력서 보유 역량 비교 결과."""

    required: list[str] = field(default_factory=list)
    possessed: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    bonus: list[str] = field(default_factory=list)

    @property
    def match_score(self) -> int:
        """매칭 점수 0~100. 요구역량 중 보유 비율 × 100."""
        if not self.required:
            return 100
        return int(round(100 * len(self.matched) / len(self.required)))


@dataclass
class IntegrationReport:
    """채용공고 분석 + 포트폴리오 검증을 합친 통합 리포트."""

    job_analysis: ReverseJobResult
    resume_audit: AuditReport
    skill_match: SkillMatch = field(default_factory=SkillMatch)

    # 하위 호환: 기존 skill_gap 은 missing 의 별칭
    @property
    def skill_gap(self) -> list[str]:
        return self.skill_match.missing

    @property
    def match_score(self) -> int:
        return self.skill_match.match_score

    @property
    def trust_score(self) -> int:
        return self.resume_audit.trust_score

    def verdict(self) -> tuple[str, str]:
        """종합 평결 (라벨, 색상hex)."""
        m = self.match_score
        t = self.trust_score
        if t < 60:
            return ("신뢰도 부족 — 재검토", "#dc3545")
        if m >= 80 and t >= 85:
            return ("강한 후보", "#28a745")
        if m >= 60 and t >= 70:
            return ("적합", "#62B864")
        if m >= 40:
            return ("부분 적합", "#ffc107")
        return ("부적합", "#dc3545")

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_score": self.match_score,
            "trust_score": self.trust_score,
            "verdict": self.verdict()[0],
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
            "skill_match": {
                "required": self.skill_match.required,
                "possessed": self.skill_match.possessed,
                "matched": self.skill_match.matched,
                "missing": self.skill_match.missing,
                "bonus": self.skill_match.bonus,
            },
        }


def _normalize_skills(skills: list[str]) -> set[str]:
    out: set[str] = set()
    for s in skills:
        n = normalize_skill_token(s)
        if n:
            out.add(n)
    return out


class IntegrationPipeline:
    """Module A → Module B 통합. 채용공고 분석으로 요구 역량을 추출하고,
    이력서·포트폴리오 검증으로 신뢰도를 산출한 뒤, 두 결과를 비교해
    매칭·부족·보너스 스킬과 매칭 점수를 도출한다.
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
        transactions: list[list[str]] | None = None,
    ) -> IntegrationReport:
        analysis = self.reverse_job.run(
            job_text, corpus=corpus, transactions=transactions
        )
        audit_report = self.audit.run(resume_text)

        # 요구 = 명시적 + 암묵적(합집합), 정규화 후 중복 제거
        required = _normalize_skills(
            list(analysis.explicit.tech_stack) + list(analysis.implicit_union)
        )
        # 보유 = 이력서 본문 키워드 매칭
        possessed = _normalize_skills(find_tech_keywords(resume_text))

        matched = sorted(required & possessed)
        missing = sorted(required - possessed)
        bonus = sorted(possessed - required)

        return IntegrationReport(
            job_analysis=analysis,
            resume_audit=audit_report,
            skill_match=SkillMatch(
                required=sorted(required),
                possessed=sorted(possessed),
                matched=matched,
                missing=missing,
                bonus=bonus,
            ),
        )
