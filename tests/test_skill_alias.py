from __future__ import annotations

from src.reverse_job.extractor import normalize_skill_token


def test_spring_framework_collapses_to_spring() -> None:
    assert normalize_skill_token("Spring Framework") == "spring"
    assert normalize_skill_token("spring framework") == "spring"


def test_react_variants() -> None:
    assert normalize_skill_token("React.js") == "react"
    assert normalize_skill_token("ReactJS") == "react"


def test_node_variants() -> None:
    assert normalize_skill_token("Node") == "node.js"
    assert normalize_skill_token("nodejs") == "node.js"


def test_kubernetes_alias() -> None:
    assert normalize_skill_token("k8s") == "kubernetes"


def test_unknown_token_returns_lowered() -> None:
    assert normalize_skill_token("Photoshop") == "photoshop"


def test_empty_returns_empty() -> None:
    assert normalize_skill_token("") == ""
    assert normalize_skill_token("   ") == ""