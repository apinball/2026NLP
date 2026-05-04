from __future__ import annotations

import json
import re
from typing import Protocol


class LLMClient(Protocol):
    """Reverse Job Engineering 에 사용할 LLM 의 최소 계약.

    TODO(team): 사용할 LLM(Anthropic / OpenAI / 로컬 HF / Ollama 등) 선정 후
    이 Protocol 을 만족하는 구현체를 `LLMReasoner` 에 주입한다.
    """

    def complete(self, prompt: str) -> str: ...


COT_PROMPT = """당신은 채용공고를 분석하여 공고에 명시되지 않은 암묵적 요구 *기술/도구* 를 추론하는 전문가입니다.

규칙:
1. 출력은 반드시 **구체적인 기술/도구/플랫폼/언어/프레임워크 이름** 만 포함하세요.
2. 소프트 스킬(예: "팀워크", "커뮤니케이션", "문제 해결")은 제외합니다.
3. 명시된 요구사항과 의미가 같은 항목은 제외합니다.
4. 영문 기술명은 영문(소문자)으로, 한국어 고유명사(카카오톡 등)는 한국어로 통일.
5. 추론이 끝나면 마지막 줄에 JSON 배열로만 출력.

좋은 출력 예: ["docker", "kubernetes", "redis", "kafka", "github actions"]
나쁜 출력 예: ["팀워크", "기술 트렌드 파악", "고객 이해 능력"]

## 채용공고
{job_text}

## 명시된 요구사항
{explicit_skills}

출력 형식(마지막 줄):
IMPLICIT: ["기술명1", "기술명2", ...]
"""

_IMPLICIT_RE = re.compile(r"IMPLICIT:\s*(\[.*?\])", re.DOTALL)


class LLMReasoner:
    """Chain-of-Thought 프롬프팅으로 암묵적 역량을 추론한다.

    LLM 은 외부에서 `LLMClient` 로 주입받는다. 구현체는 아직 미정.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def infer_implicit(self, job_text: str, explicit_skills: list[str]) -> list[str]:
        prompt = COT_PROMPT.format(
            job_text=job_text.strip(),
            explicit_skills=", ".join(explicit_skills) or "(없음)",
        )
        text = self.client.complete(prompt)
        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> list[str]:
        match = _IMPLICIT_RE.search(text)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []
        return [str(x).strip() for x in parsed if str(x).strip()]
