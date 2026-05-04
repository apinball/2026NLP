from __future__ import annotations

from src.data.loaders import CoverLetterSection, PassSpec, Resume
from src.data.skill_index import SkillIndex, build_skill_index


def _make_resume(text: str, certs: str = "") -> Resume:
    return Resume(
        source="test",
        url="",
        company="",
        job="",
        apply_period="",
        pass_spec=PassSpec(certifications=certs),
        sections=[CoverLetterSection(number="1", question="", answer=text)],
    )


def test_build_skill_index_counts_occurrences() -> None:
    resumes = [
        _make_resume("Python과 Django 로 백엔드를 구현했습니다."),
        _make_resume("AWS 위에 Docker 와 Python 을 함께 사용했습니다."),
        _make_resume("자바스크립트 프론트엔드만 다뤘습니다."),
    ]
    idx = build_skill_index(resumes=resumes)
    assert idx.total_docs == 3
    assert idx.counts["python"] == 2
    assert idx.counts["django"] == 1
    assert idx.counts["docker"] == 1


def test_recall_proxy_matches_threshold() -> None:
    idx = SkillIndex(
        counts={"python": 5, "django": 1, "rust": 0},
        total_docs=10,
    )
    # python 50%, django 10%, rust 0% — min_freq=0.05 면 python+django 통과 → 2/3
    assert idx.recall_proxy(["python", "django", "rust"], min_freq=0.05) == 2 / 3
