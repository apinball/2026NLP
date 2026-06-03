from __future__ import annotations

import json
from typing import Any

from src.reverse_job.adapters.mock_client import MockLLMClient
from src.reverse_job.llm_reasoner import (
    LLMImplicitResult,
    LLMImplicitSkill,
    LLMReasoner,
)


class _FixedClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.captured_prompt: str | None = None

    def complete(self, prompt: str) -> str:
        self.captured_prompt = prompt
        return self.response


def test_parse_structured_json_output() -> None:
    payload: dict[str, Any] = {
        "job_analysis": "백엔드 직무",
        "actual_tasks": ["API 설계", "DB 운영"],
        "implicit_skills": [
            {"skill": "Git", "reason": "협업 필수", "confidence": 0.95},
            {"skill": "Linux", "reason": "서버 환경", "confidence": 0.9},
        ],
    }
    reasoner = LLMReasoner(client=_FixedClient(json.dumps(payload, ensure_ascii=False)))
    result = reasoner.infer("백엔드 채용", ["Python"])

    assert isinstance(result, LLMImplicitResult)
    assert result.job_analysis == "백엔드 직무"
    assert result.actual_tasks == ["API 설계", "DB 운영"]
    assert [s.skill for s in result.implicit_skills] == ["Git", "Linux"]
    assert result.implicit_skills[0].confidence == 0.95


def test_parse_json_wrapped_in_code_block() -> None:
    raw = """```json
{"implicit_skills": [{"skill": "Docker", "reason": "배포 표준", "confidence": 0.8}]}
```"""
    reasoner = LLMReasoner(client=_FixedClient(raw))
    result = reasoner.infer("public job text", ["Python"])
    assert [s.skill for s in result.implicit_skills] == ["Docker"]


def test_parse_legacy_implicit_format() -> None:
    raw = '추론 결과:\nIMPLICIT: ["aws", "docker", "git"]'
    reasoner = LLMReasoner(client=_FixedClient(raw))
    result = reasoner.infer("job", ["python"])
    assert [s.skill for s in result.implicit_skills] == ["aws", "docker", "git"]
    assert all(s.confidence == 0.5 for s in result.implicit_skills)


def test_infer_implicit_returns_list_str_backward_compat() -> None:
    payload = {
        "implicit_skills": [
            {"skill": "Kubernetes", "reason": "운영", "confidence": 0.8},
            {"skill": "Redis", "reason": "캐시", "confidence": 0.7},
        ]
    }
    reasoner = LLMReasoner(client=_FixedClient(json.dumps(payload)))
    skills = reasoner.infer_implicit("job", ["python"])
    assert skills == ["Kubernetes", "Redis"]


def test_prompt_includes_explicit_and_arm_candidates() -> None:
    client = _FixedClient('{"implicit_skills":[]}')
    LLMReasoner(client=client).infer(
        "백엔드 채용",
        explicit_skills=["python", "django"],
        arm_candidates=["aws", "redis"],
    )
    assert client.captured_prompt is not None
    assert "python" in client.captured_prompt
    assert "django" in client.captured_prompt
    assert "aws" in client.captured_prompt
    assert "redis" in client.captured_prompt


def test_invalid_json_returns_empty_result() -> None:
    reasoner = LLMReasoner(client=_FixedClient("this is not json"))
    result = reasoner.infer("job", ["python"])
    assert result.implicit_skills == []


def test_mock_client_integration() -> None:
    reasoner = LLMReasoner(client=MockLLMClient())
    result = reasoner.infer("백엔드 채용공고", ["python"])
    assert len(result.implicit_skills) >= 1
    assert all(isinstance(s, LLMImplicitSkill) for s in result.implicit_skills)
    assert "백엔드" in result.job_analysis
