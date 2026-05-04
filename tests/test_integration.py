from __future__ import annotations

from src.integration.pipeline import IntegrationPipeline


def test_integration_returns_combined_report() -> None:
    pipeline = IntegrationPipeline()
    job = "[백엔드 채용] Python, Django, MySQL 3년 경력"
    resume = (
        "저는 2010년부터 Docker 로 인프라를 구축했고, "
        "Python 과 Django 로 백엔드를 개발했습니다."
    )
    corpus = [
        "Python Django MySQL AWS Docker Git",
        "Python Django Redis AWS Docker Git",
        "Python FastAPI MySQL AWS Docker Git",
    ]
    report = pipeline.run(job, resume, corpus=corpus)

    # Module A: 명시적 + 암묵적
    assert "python" in report.job_analysis.explicit.tech_stack
    assert "django" in report.job_analysis.explicit.tech_stack
    # Module B: Docker(2013) 위반 탐지 → 신뢰도 100 미만
    assert report.resume_audit.trust_score < 100
    docker_violation = any(
        v.tech == "Docker" for v in report.resume_audit.violations
    )
    assert docker_violation

    # 통합: skill gap 계산됨
    assert isinstance(report.skill_gap, list)


def test_to_dict_serializable_json() -> None:
    import json

    pipeline = IntegrationPipeline()
    report = pipeline.run(
        "Python Django 채용",
        "Python 으로 개발했습니다.",
        corpus=None,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "trust_score" in payload
    assert "explicit" in payload