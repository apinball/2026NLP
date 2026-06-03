from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Reverse Job Engineering 에 사용할 LLM 의 최소 계약.

    구현체는 `src/reverse_job/adapters/` 참고
    (Ollama / Groq / Gemini / Mock).
    """

    def complete(self, prompt: str) -> str: ...


@dataclass
class LLMImplicitSkill:
    """LLM 이 추론한 암묵적 역량 한 건 (구조화 출력)."""

    skill: str
    reason: str = ""
    confidence: float = 0.5


@dataclass
class LLMImplicitResult:
    implicit_skills: list[LLMImplicitSkill] = field(default_factory=list)
    job_analysis: str = ""
    actual_tasks: list[str] = field(default_factory=list)
    raw_response: str = ""


SYSTEM_PROMPT = """당신은 10년 경력의 IT 채용 전문가입니다.
채용공고를 분석하여 공고에 명시되지 않았지만 실무에서 반드시 필요한 암묵적 역량을 추론합니다.

[중요 규칙]
1. implicit_skills 에는 반드시 공고에 없는 기술만 포함하세요.
2. 명시된 기술과 유사하거나 포함 관계인 기술도 제외하세요.
3. skill 필드는 반드시 구체적인 기술/도구 이름으로 작성하세요.
   좋은 예: "Git", "FastAPI", "Pandas", "Jira", "Linux", "Kubernetes"
   나쁜 예: "버전 관리 시스템", "협업 도구", "커뮤니케이션 스킬"
4. confidence 는 실무에서 실제로 필요할 확률 (0.0~1.0).
5. 최소 3개, 최대 6개의 암묵적 기술을 추론하세요.
6. 소프트 스킬(커뮤니케이션·문제해결 등)은 절대 포함하지 마세요.
7. reason 필드는 공고 원문의 특정 조건을 직접 인용하여 연결하세요.

반드시 아래 JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON 만 출력:
{{
  "job_analysis": "이 직무가 실제로 하는 일 (1-2문장)",
  "actual_tasks": ["실제 업무 1", "실제 업무 2", "실제 업무 3"],
  "implicit_skills": [
    {{"skill": "기술명", "reason": "이 기술이 왜 실무에서 반드시 필요한지", "confidence": 0.0~1.0}}
  ]
}}"""

USER_TEMPLATE = """다음 채용공고를 분석하여 암묵적 요구 역량을 추론해주세요.

[채용공고]
{job_text}

[공고에 이미 명시된 기술 - 절대 implicit_skills 에 포함하지 마세요]
{explicit_skills}

[통계 분석(ARM) 이 도출한 연관 기술 후보 - 참고용]
{arm_candidates}

[추론 단계]
1단계: 직무가 실제로 어떤 업무를 하는지 파악
2단계: 공고의 추상 표현을 구체 기술명으로 변환
       "딥러닝 프레임워크" → PyTorch, TensorFlow
       "서버 인터페이스 + Python" → FastAPI, Flask
       "클라우드 환경" → Docker, Kubernetes
       "데이터 분석" → Pandas, SQL
       "버전 관리" → Git
       "CI/CD" → Jenkins, GitHub Actions
3단계: 명시된 기술과 겹치는 항목 제거
4단계: ARM 후보와 대조하여 최종 목록 확정
5단계: 각 기술의 reason 을 공고 조건과 직접 연결

JSON 만 출력하세요.
"""

_LEGACY_RE = re.compile(r"IMPLICIT:\s*(\[.*?\])", re.DOTALL)


def _strip_code_block(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text).strip()
    return text.rstrip("`").strip()


class LLMReasoner:
    """Chain-of-Thought 프롬프팅으로 암묵적 역량을 추론.

    `infer(...)` → 구조화된 LLMImplicitResult (skill + reason + confidence)
    `infer_implicit(...)` → list[str] (하위 호환 / 기존 파이프라인용)
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def infer(
        self,
        job_text: str,
        explicit_skills: list[str],
        arm_candidates: list[str] | None = None,
    ) -> LLMImplicitResult:
        prompt = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
            job_text=job_text.strip()[:1000],
            explicit_skills=", ".join(explicit_skills) or "없음",
            arm_candidates=", ".join(arm_candidates or []) or "없음",
        )
        raw = self.client.complete(prompt)
        return self._parse(raw)

    def infer_implicit(
        self,
        job_text: str,
        explicit_skills: list[str],
        arm_candidates: list[str] | None = None,
    ) -> list[str]:
        result = self.infer(job_text, explicit_skills, arm_candidates)
        return [s.skill for s in result.implicit_skills]

    @staticmethod
    def _parse(text: str) -> LLMImplicitResult:
        result = LLMImplicitResult(raw_response=text)

        clean = _strip_code_block(text)
        try:
            data = json.loads(clean)
            result.job_analysis = str(data.get("job_analysis", ""))
            result.actual_tasks = [
                str(t) for t in (data.get("actual_tasks") or []) if t
            ]
            for item in data.get("implicit_skills") or []:
                if not isinstance(item, dict):
                    continue
                skill = str(item.get("skill", "")).strip()
                if not skill:
                    continue
                try:
                    conf = float(item.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                result.implicit_skills.append(
                    LLMImplicitSkill(
                        skill=skill,
                        reason=str(item.get("reason", "")).strip(),
                        confidence=conf,
                    )
                )
            if result.implicit_skills:
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 레거시 포맷: IMPLICIT: ["a", "b"]
        match = _LEGACY_RE.search(text)
        if match:
            try:
                items = json.loads(match.group(1))
                for s in items:
                    skill = str(s).strip()
                    if skill:
                        result.implicit_skills.append(
                            LLMImplicitSkill(skill=skill)
                        )
            except json.JSONDecodeError:
                pass

        return result
