from __future__ import annotations

from src.data.job_category import OTHER, UNCATEGORIZED, normalize_job


def test_office_admin_variants_collapse() -> None:
    for raw in ["사무", "사무직", "일반행정", "사무행정", "행정"]:
        assert normalize_job(raw) == "사무/행정", raw


def test_marketing_variants() -> None:
    assert normalize_job("마케팅") == "마케팅"
    assert normalize_job("콘텐츠마케팅") == "마케팅"
    assert normalize_job("브랜드마케팅") == "마케팅"


def test_it_dev_takes_priority_over_others() -> None:
    # 백엔드개발자 가 기획·사무 등으로 잘못 흡수되지 않아야 함
    assert normalize_job("백엔드개발자") == "IT/개발"
    assert normalize_job("프론트엔드 엔지니어") == "IT/개발"
    assert normalize_job("데이터분석") == "IT/개발"


def test_planning_caught_after_marketing() -> None:
    # 콘텐츠마케팅기획 같은 합성어는 마케팅이 우선
    assert normalize_job("콘텐츠마케팅기획") == "마케팅"
    assert normalize_job("전략기획") == "기획"
    assert normalize_job("서비스기획") == "기획"


def test_sales_overseas() -> None:
    assert normalize_job("해외영업") == "영업"
    assert normalize_job("기술영업") == "영업"


def test_empty_returns_uncategorized() -> None:
    assert normalize_job("") == UNCATEGORIZED
    assert normalize_job(None) == UNCATEGORIZED


def test_unknown_returns_other() -> None:
    assert normalize_job("점성술사") == OTHER
