from __future__ import annotations

import os

import httpx

DEFAULT_MODEL = "qwen2.5:14b-instruct-q4_K_M"


class OllamaLLMClient:
    """Ollama HTTP API 를 LLMClient Protocol 로 감싼 어댑터.

    Docker compose 에서는 OLLAMA_HOST=http://ollama:11434 환경변수가 자동 주입되어
    같은 네트워크의 ollama 서비스로 요청한다. 호스트에서 직접 호출하려면
    OLLAMA_HOST=http://localhost:11434 으로 설정.

    사용:
        from src.reverse_job.adapters import OllamaLLMClient
        from src.reverse_job.llm_reasoner import LLMReasoner

        reasoner = LLMReasoner(client=OllamaLLMClient())
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str | None = None,
        timeout: float = 180.0,
        keep_alive: str = "5m",
        options: dict | None = None,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.options = options or {"temperature": 0.7, "num_predict": 1024}

    def complete(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data.get("response", "")

    def healthcheck(self) -> bool:
        """ollama 서비스 도달성 확인. /api/tags 호출."""
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.host}/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False