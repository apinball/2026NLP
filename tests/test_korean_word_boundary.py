from __future__ import annotations

from src.logic_auditor.pipeline import LogicAuditorPipeline


def _violations(text: str) -> int:
    return len(LogicAuditorPipeline().run(text).violations)


def test_korean_compound_words_not_matched() -> None:
    # "라인" 이 합성어 일부일 때 매칭되지 않아야 함
    assert _violations("2005년 온라인 게임을 즐겼습니다") == 0
    assert _violations("2010년 파이프라인 구축 경험") == 0
    assert _violations("2009년 오프라인 캠페인 기획") == 0


def test_compound_noun_not_matched() -> None:
    # "잔디" + "밭" 같이 한국어 합성어
    assert _violations("2010년 잔디밭에서 산책") == 0
    # "라인" + "너" 처럼 한국어 글자가 이어지면 차단
    assert _violations("2005년 라이너 노트") == 0


def test_year_must_be_followed_by_year_marker() -> None:
    # "1900%", "2000건" 등 비-연도 숫자는 잡지 말 것
    assert _violations("1900% 매출 상승을 달성했습니다") == 0
    assert _violations("2000건의 데이터를 처리했습니다") == 0


def test_korean_particles_allowed() -> None:
    # 1글자 조사
    assert _violations("2010년에 잔디로 협업했습니다") == 1
    # 2글자 조사
    assert _violations("2008년 카카오톡으로 회의했습니다") == 1
    assert _violations("2008년 카카오톡에서 토론했습니다") == 1


def test_normal_resume_text_clean() -> None:
    # 정상 케이스
    assert _violations("2022년 HyperCLOVA 모델을 활용") == 0
    assert _violations("2014년에 카카오페이로 결제") == 0
    assert _violations("2020년 빅데이터분석기사 자격증 취득") == 0