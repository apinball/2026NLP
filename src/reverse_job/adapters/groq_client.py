from __future__ import annotations

import os

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqLLMClient:
    """Groq API 어댑터 (무료 티어).

    무료 한도:
      llama-3.3-70b-versatile: 30 req/min, 14,400/day

    API 키 발급: https://console.groq.com → API Keys
    환경변수: GROQ_API_KEY
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set — pass api_key= or set env variable"
            )
        from groq import Groq

        self.client = Groq(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
