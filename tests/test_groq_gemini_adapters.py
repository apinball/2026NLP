from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_groq(monkeypatch) -> MagicMock:
    """`groq` 모듈을 stub 으로 주입해 실 SDK 없이 테스트."""
    fake = types.ModuleType("groq")

    class Groq:
        last_init_kwargs: dict | None = None

        def __init__(self, api_key: str | None = None) -> None:
            Groq.last_init_kwargs = {"api_key": api_key}
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="GROQ-OK"))]
            )

    fake.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", fake)
    return Groq


def _install_fake_genai(monkeypatch) -> MagicMock:
    fake = types.ModuleType("google.generativeai")
    fake.configure = MagicMock()

    class GenerativeModel:
        def __init__(self, model_name: str, generation_config: dict | None = None) -> None:
            self.model_name = model_name
            self.generation_config = generation_config

        def generate_content(self, prompt: str):
            return MagicMock(text=f"GEMINI-OK:{prompt[:5]}")

    fake.GenerativeModel = GenerativeModel
    # google.generativeai 는 `google` 패키지의 서브모듈처럼 임포트됨
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)
    return GenerativeModel


def test_groq_client_complete(monkeypatch) -> None:
    _install_fake_groq(monkeypatch)
    from src.reverse_job.adapters.groq_client import GroqLLMClient

    client = GroqLLMClient(api_key="test-key", model="llama-3.3-70b-versatile")
    out = client.complete("hello")
    assert out == "GROQ-OK"


def test_groq_client_missing_key(monkeypatch) -> None:
    _install_fake_groq(monkeypatch)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src.reverse_job.adapters.groq_client import GroqLLMClient

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqLLMClient()


def test_gemini_client_complete(monkeypatch) -> None:
    _install_fake_genai(monkeypatch)
    from src.reverse_job.adapters.gemini_client import GeminiLLMClient

    client = GeminiLLMClient(api_key="test-key")
    out = client.complete("hello world")
    assert out.startswith("GEMINI-OK:")


def test_gemini_client_missing_key(monkeypatch) -> None:
    _install_fake_genai(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from src.reverse_job.adapters.gemini_client import GeminiLLMClient

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiLLMClient()


def test_mock_client_returns_valid_json() -> None:
    import json

    from src.reverse_job.adapters.mock_client import MockLLMClient

    client = MockLLMClient()
    out = json.loads(client.complete("백엔드 채용공고"))
    assert "implicit_skills" in out
    assert len(out["implicit_skills"]) >= 1
