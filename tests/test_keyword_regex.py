from __future__ import annotations

from src.reverse_job.extractor import find_tech_keywords


def test_python_matches_with_korean_particle() -> None:
    assert "python" in find_tech_keywords("Python을 사용해 개발했습니다.")
    assert "python" in find_tech_keywords("python의 장점")


def test_no_match_inside_word() -> None:
    # going / ago 안의 'go' 가 매칭되면 false positive
    assert "go" not in find_tech_keywords("I am going to ago")
    assert "java" not in find_tech_keywords("javafication")
    assert "rest" not in find_tech_keywords("restoration project")


def test_compound_keywords_match() -> None:
    hits = find_tech_keywords("Spring Boot 와 Node.js 를 함께 사용했습니다.")
    assert "spring boot" in hits
    assert "node.js" in hits


def test_alias_priority_longest_first() -> None:
    # spring boot 가 spring 보다 먼저 매칭되어 양쪽 다 잡혀야 함
    hits = find_tech_keywords("Spring Boot 환경")
    assert "spring boot" in hits


def test_typescript_not_just_javascript() -> None:
    hits = find_tech_keywords("TypeScript 와 JavaScript 모두 사용")
    assert "typescript" in hits
    assert "javascript" in hits


def test_punctuation_boundaries() -> None:
    hits = find_tech_keywords("(REST) API, gRPC, CI/CD")
    assert "rest" in hits
    assert "grpc" in hits
    assert "ci/cd" in hits


def test_case_insensitive() -> None:
    assert "aws" in find_tech_keywords("AWS 인프라")
    assert "aws" in find_tech_keywords("aws 인프라")
