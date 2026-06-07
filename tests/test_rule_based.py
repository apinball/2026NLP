from __future__ import annotations

import json

import pytest

from src.logic_auditor.entity_extractor import EntityExtractor
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


def test_feature_version_incompatibility_detected() -> None:
    text = "React 16에서 Server Components를 사용했습니다."
    report = RuleBasedAuditor().audit_detailed(text)

    assert report.trust_score < 100
    assert any(
        issue.issue_type == "VERSION_ERROR"
        and issue.evidence["kind"] == "feature_version"
        for issue in report.issues
    )


def test_feature_version_ignores_unrelated_technology_version() -> None:
    text = "Next.js 14와 React Hooks를 활용했습니다."
    report = RuleBasedAuditor().audit_detailed(text)

    assert not any(
        issue.evidence.get("kind") == "feature_version"
        for issue in report.issues
    )


def test_feature_keyword_does_not_match_inside_word() -> None:
    entities = EntityExtractor().extract("React 18에서 webhook 이벤트를 처리했습니다.")

    assert not any(entity.entity_type == "FEATURE" for entity in entities)


def test_invalid_feature_rule_config_fails_fast(tmp_path) -> None:
    bad_rules = tmp_path / "bad_feature_rules.json"
    bad_rules.write_text(
        json.dumps(
            [
                {
                    "feature": "Broken Feature",
                    "tech": "React",
                    "introduced_year": "not-a-year",
                    "introduced_version": "18",
                    "keywords": ["broken"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="introduced_year"):
        EntityExtractor(feature_rules_path=bad_rules)


def test_exaggerated_scope_claim_detected() -> None:
    text = "3개월 만에 대규모 금융권 시스템 아키텍처를 단독으로 설계하고 운영까지 완료했습니다."
    report = RuleBasedAuditor().audit_detailed(text)

    assert report.trust_score < 100
    assert any(issue.issue_type == "EXAGGERATION_SUSPECTED" for issue in report.issues)


def test_pipeline_exposes_claims_and_issues() -> None:
    report = LogicAuditorPipeline().run(
        "2010년 Docker를 사용했습니다.\nReact 16에서 Server Components를 사용했습니다."
    )

    assert len(report.claims) == 2
    assert len(report.issues) >= 2
    assert {claim.claim_id for claim in report.claims} == {"c1", "c2"}
