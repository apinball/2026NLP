from __future__ import annotations

from src.reverse_job.extractor import JobExtractor


def test_keyword_matching_without_ner() -> None:
    ext = JobExtractor(load_ner=False)
    req = ext.extract("Python과 Django, AWS 경험 3년 이상 요구됩니다.")
    assert "python" in req.tech_stack
    assert "django" in req.tech_stack
    assert "aws" in req.tech_stack
    assert 3 in req.experience_years
