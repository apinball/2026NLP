from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.config import DATA_DIR

CERAGEM_PATH = DATA_DIR / "ceragem_job_posting.json"
WANTED_JOBS_PATH = DATA_DIR / "wanted_jobs.jsonl"
LINKAREER_PARSED = DATA_DIR / "linkareer_cover_letter_parsed.jsonl"
NAVER_PARSED = DATA_DIR / "naver_cafe_passassay_parsed.jsonl"


@dataclass
class JobPosting:
    """채용공고 단위. 회사 내 여러 position 이면 각 position 이 하나의 JobPosting."""

    source: str
    company: str
    category: str
    job_title: str
    responsibilities: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    preferred: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    work_location: str = ""
    raw_text: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class PassSpec:
    """자소서 작성자의 합격스펙."""

    school: str = ""
    major: str = ""
    gpa: str = ""
    language: str = ""
    certifications: str = ""
    activities: str = ""
    career: str = ""


@dataclass
class CoverLetterSection:
    number: str
    question: str
    answer: str


@dataclass
class Resume:
    source: str
    url: str
    company: str
    job: str
    apply_period: str
    pass_spec: PassSpec
    sections: list[CoverLetterSection] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"[{s.number}] {s.question}\n{s.answer}" for s in self.sections
        )


def load_ceragem_positions(path: Path = CERAGEM_PATH) -> list[JobPosting]:
    """ceragem_job_posting.json 의 27개 position 을 JobPosting 으로 평탄화.

    job_posting.skills 는 27개 position 전체 union 이라 position 별로 복사하면
    모든 공고가 같은 스킬셋을 가진 것처럼 ARM 빈출 결과를 왜곡한다 → 여기서는
    skills 를 비워 두고 raw_text 의 키워드 매칭만으로 추출하게 한다.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    company = raw.get("company", {}).get("name", "")
    out: list[JobPosting] = []
    for pos in raw.get("positions", []):
        responsibilities = pos.get("responsibilities", []) or []
        requirements = pos.get("requirements", []) or []
        preferred = pos.get("preferred", []) or []
        out.append(
            JobPosting(
                source="jobkorea",
                company=company,
                category=pos.get("category", ""),
                job_title=pos.get("job_title", ""),
                responsibilities=responsibilities,
                requirements=requirements,
                preferred=preferred,
                skills=[],
                work_location=pos.get("work_location", ""),
                raw_text="\n".join(
                    [pos.get("job_title", ""), *responsibilities, *requirements, *preferred]
                ),
                meta={
                    "headcount": pos.get("headcount", ""),
                    "url": raw.get("url", ""),
                },
            )
        )
    return out


def iter_wanted_jobs(path: Path = WANTED_JOBS_PATH) -> Iterator[JobPosting]:
    """scripts/crawl_wanted.py 가 작성한 JobPosting jsonl 을 순회."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield JobPosting(
                source=d.get("source", "wanted"),
                company=d.get("company", "") or "",
                category=d.get("category", "") or "",
                job_title=d.get("job_title", "") or "",
                responsibilities=list(d.get("responsibilities") or []),
                requirements=list(d.get("requirements") or []),
                preferred=list(d.get("preferred") or []),
                skills=list(d.get("skills") or []),
                work_location=d.get("work_location", "") or "",
                raw_text=d.get("raw_text", "") or "",
                meta=dict(d.get("meta") or {}),
            )


def iter_jobs() -> Iterator[JobPosting]:
    """현재 보유한 모든 소스의 JobPosting 을 순회."""
    if CERAGEM_PATH.exists():
        yield from load_ceragem_positions()
    if WANTED_JOBS_PATH.exists():
        yield from iter_wanted_jobs()


def _parse_pass_spec(raw: dict) -> PassSpec:
    return PassSpec(
        school=raw.get("학교", "") or "",
        major=raw.get("학과", "") or "",
        gpa=raw.get("학점", "") or "",
        language=raw.get("어학점수", "") or "",
        certifications=raw.get("자격증", "") or "",
        activities=raw.get("대외활동", "") or raw.get("대외활동 경력 및 인턴", "") or "",
        career=raw.get("경력", "") or "",
    )


def _parse_sections(raw: list[dict] | None) -> list[CoverLetterSection]:
    out: list[CoverLetterSection] = []
    if not raw:
        return out
    for s in raw:
        out.append(
            CoverLetterSection(
                number=str(s.get("번호", "") or ""),
                question=s.get("질문", "") or "",
                answer=s.get("답변", "") or "",
            )
        )
    return out


def _resume_from_jsonl_record(d: dict, source: str) -> Resume:
    return Resume(
        source=source,
        url=d.get("url", "") or "",
        company=d.get("기업명", "") or "",
        job=d.get("직무", "") or "",
        apply_period=d.get("지원시기", "") or "",
        pass_spec=_parse_pass_spec(d.get("합격스펙", {}) or {}),
        sections=_parse_sections(d.get("자소서")),
        meta={
            "id": d.get("id"),
            "title": d.get("title", ""),
            "crawledAt": d.get("crawledAt", ""),
            "scrapCount": d.get("scrapCount"),
            "viewCount": d.get("viewCount"),
            "types": d.get("types"),
        },
    )


def iter_linkareer(path: Path = LINKAREER_PARSED) -> Iterator[Resume]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield _resume_from_jsonl_record(json.loads(line), source="linkareer")


def iter_naver_cafe(path: Path = NAVER_PARSED) -> Iterator[Resume]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield _resume_from_jsonl_record(json.loads(line), source="naver_cafe")


def iter_resumes() -> Iterator[Resume]:
    """linkareer + naver_cafe 통합 순회. 본문 비어있는 레코드는 스킵."""
    for r in iter_linkareer():
        if r.sections:
            yield r
    for r in iter_naver_cafe():
        if r.sections:
            yield r


def load_resumes(limit: int | None = None) -> list[Resume]:
    out: list[Resume] = []
    for i, r in enumerate(iter_resumes()):
        if limit is not None and i >= limit:
            break
        out.append(r)
    return out
