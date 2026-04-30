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


COT_PROMPT = """당신은 채용공고를 분석하여 명시되지 않은 암묵적 요구 역량을 추론하는 전문가입니다.

아래 채용공고와 명시된 요구사항을 바탕으로:
1. 실무 맥락상 필연적으로 요구될 기술/역량을 단계별로 추론하세요.
2. 회사가 공고에 적지 않았지만 합격자는 당연히 알고 있어야 할 역량에 초점을 맞추세요.
3. 추론이 끝나면 마지막 줄에 JSON 배열로 암묵적 역량만 출력하세요.

## 채용공고
{job_text}

## 명시된 요구사항
{explicit_skills}

출력 형식(마지막 줄):
IMPLICIT: ["역량1", "역량2", ...]
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
