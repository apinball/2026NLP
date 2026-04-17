from __future__ import annotations

import json
import re

from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, LLM_MODEL

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
    """Chain-of-Thought 프롬프팅으로 암묵적 역량을 추론한다."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or ANTHROPIC_API_KEY
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        self.client = Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens

    def infer_implicit(self, job_text: str, explicit_skills: list[str]) -> list[str]:
        prompt = COT_PROMPT.format(
            job_text=job_text.strip(),
            explicit_skills=", ".join(explicit_skills) or "(없음)",
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
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
