from __future__ import annotations

import json

_RESPONSES: dict[str, dict] = {
    "backend": {
        "job_analysis": "백엔드 시스템 개발 및 운영 직무",
        "actual_tasks": ["API 설계 및 구현", "DB 스키마 관리", "장애 대응"],
        "implicit_skills": [
            {"skill": "Git", "reason": "팀 협업 코드 관리에 필수", "confidence": 0.95},
            {"skill": "Linux", "reason": "서버 운영 환경 기본 역량", "confidence": 0.90},
            {"skill": "REST API", "reason": "백엔드 핵심 업무", "confidence": 0.85},
        ],
    },
    "frontend": {
        "job_analysis": "프론트엔드 화면/상태 개발",
        "actual_tasks": ["UI 구현", "상태 관리", "성능 최적화"],
        "implicit_skills": [
            {"skill": "Git", "reason": "버전 관리 필수", "confidence": 0.95},
            {"skill": "CSS", "reason": "프론트엔드 기본 역량", "confidence": 0.92},
            {"skill": "TypeScript", "reason": "현대 FE 개발 표준", "confidence": 0.80},
        ],
    },
    "default": {
        "job_analysis": "일반 개발 직무",
        "actual_tasks": ["설계", "구현", "운영"],
        "implicit_skills": [
            {"skill": "Git", "reason": "모든 개발 직군 필수", "confidence": 0.95},
            {"skill": "Linux", "reason": "기본 개발 환경", "confidence": 0.80},
            {"skill": "Docker", "reason": "배포/실행 표준", "confidence": 0.75},
        ],
    },
}


class MockLLMClient:
    """오프라인 테스트용 가짜 LLM. 프롬프트 키워드로 응답 선택."""

    def complete(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "백엔드" in prompt or "backend" in lowered:
            payload = _RESPONSES["backend"]
        elif "프론트엔드" in prompt or "frontend" in lowered:
            payload = _RESPONSES["frontend"]
        else:
            payload = _RESPONSES["default"]
        return json.dumps(payload, ensure_ascii=False)
