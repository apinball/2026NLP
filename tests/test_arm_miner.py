from __future__ import annotations

from src.reverse_job.arm_miner import ARMMiner


def test_find_implicit_detects_co_occurring_skill() -> None:
    transactions = [
        ["python", "django", "docker", "aws"],
        ["python", "django", "docker", "git"],
        ["python", "fastapi", "docker", "aws"],
        ["python", "flask", "docker", "aws"],
        ["python", "django", "docker", "aws"],
    ]
    miner = ARMMiner(min_support=0.4, min_confidence=0.7)
    rules = miner.mine(transactions)

    target = {"python", "django"}
    implicit = miner.find_implicit(target, rules)
    assert "docker" in implicit


def test_empty_corpus_returns_empty() -> None:
    miner = ARMMiner()
    assert miner.mine([]).empty
    assert miner.find_implicit({"python"}, miner.mine([])) == []
