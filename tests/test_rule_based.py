from __future__ import annotations

from src.logic_auditor.pipeline import LogicAuditorPipeline
from src.logic_auditor.rule_based import RuleBasedAuditor


def test_release_year_violation_detected() -> None:
    text = "저는 2010년부터 Docker를 활용해 인프라를 구축했습니다."
    violations = RuleBasedAuditor().audit(text)
    techs = {v.tech for v in violations}
    assert "Docker" in techs
    docker = next(v for v in violations if v.tech == "Docker")
    assert docker.claimed_year == 2010
    assert docker.actual_year == 2013


def test_version_year_violation_detected() -> None:
    text = "2015년 프로젝트에서 React 16과 Next.js 13을 적용했습니다."
    violations = RuleBasedAuditor().audit(text)
    kinds = {(v.tech, v.kind) for v in violations}
    assert ("React", "version_year") in kinds or ("Next.js", "release_year") in kinds


def test_clean_text_has_no_violations() -> None:
    text = "2022년 프로젝트에서 Docker와 Kubernetes를 활용했습니다."
    violations = RuleBasedAuditor().audit(text)
    assert violations == []


def test_pipeline_trust_score_drops_on_violation() -> None:
    pipeline = LogicAuditorPipeline()
    bad = pipeline.run("2010년에 Docker로 인프라를 구축했습니다.")
    clean = pipeline.run("2022년에 Docker로 인프라를 구축했습니다.")
    assert bad.trust_score < clean.trust_score
    assert clean.trust_score == 100
