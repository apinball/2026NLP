from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiLLMClient:
    """Google Gemini API 어댑터.

    무료 한도:
      gemini-2.0-flash: 15 req/min, 1,500/day

    API 키 발급: https://aistudio.google.com → Get API Key
    환경변수: GEMINI_API_KEY
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set — pass api_key= or set env variable"
            )
        import google.generativeai as genai

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

    def complete(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text or ""
