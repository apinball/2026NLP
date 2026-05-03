"""원티드 비공식 JSON API 크롤러.

엔드포인트(2026-05 기준 실측):
  list:   GET https://www.wanted.co.kr/api/v4/jobs
  detail: GET https://www.wanted.co.kr/api/v4/jobs/{id}

list 응답에서 id 만 모은 뒤 detail 을 호출해 풍부한 필드를 채운다. 결과는
JobPosting (src/data/loaders.py) 직렬화 형태의 jsonl 로 저장된다.

사용 예:
  python scripts/crawl_wanted.py --job-group 518 --pages 10 --sleep 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

LIST_URL = "https://www.wanted.co.kr/api/v4/jobs"
DETAIL_URL = "https://www.wanted.co.kr/api/v4/jobs/{id}"

# job_group_id 참고: 518=개발, 1610=디자인, 523=마케팅·광고 (사이트 UI 에서 확인 가능)
DEFAULT_JOB_GROUP = 518

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "2026NLP-research-crawler "
        "(+https://github.com/apinball/2026NLP)"
    ),
    "Accept": "application/json",
}


def fetch_list(
    client: httpx.Client, *, job_group: int, offset: int, limit: int
) -> list[dict[str, Any]]:
    params = {
        "country": "kr",
        "job_group_id": job_group,
        "years": -1,
        "locations": "all",
        "sort": "job.latest_order",
        "limit": limit,
        "offset": offset,
    }
    resp = client.get(LIST_URL, params=params)
    resp.raise_for_status()
    return resp.json().get("data", []) or []


def fetch_detail(client: httpx.Client, job_id: int) -> dict[str, Any]:
    resp = client.get(DETAIL_URL.format(id=job_id))
    resp.raise_for_status()
    return resp.json().get("job", {}) or {}


def _split_lines(text: str) -> list[str]:
    if not text:
        return []
    out = []
    for line in text.splitlines():
        cleaned = line.strip(" -•·*\t·")
        if cleaned:
            out.append(cleaned)
    return out


def to_job_posting(job: dict[str, Any]) -> dict[str, Any]:
    company = job.get("company") or {}
    address = job.get("address") or {}
    detail = job.get("detail") or {}
    skill_tags = [t.get("title", "") for t in job.get("skill_tags") or [] if t]
    category_tags = [
        t.get("title", "") if isinstance(t, dict) else (t or "")
        for t in job.get("category_tags") or []
    ]

    requirements = (detail.get("requirements") or "").strip()
    main_tasks = (detail.get("main_tasks") or "").strip()
    preferred = (detail.get("preferred_points") or "").strip()
    intro = (detail.get("intro") or "").strip()

    return {
        "source": "wanted",
        "company": company.get("name", ""),
        "category": ",".join(s for s in category_tags if s),
        "job_title": job.get("position", ""),
        "responsibilities": _split_lines(main_tasks),
        "requirements": _split_lines(requirements),
        "preferred": _split_lines(preferred),
        "skills": [s for s in skill_tags if s],
        "work_location": address.get("full_location") or address.get("country") or "",
        "raw_text": "\n".join(
            [job.get("position", ""), intro, main_tasks, requirements, preferred]
        ),
        "meta": {
            "id": job.get("id"),
            "url": f"https://www.wanted.co.kr/wd/{job.get('id')}",
            "due_time": job.get("due_time"),
            "annual_from": job.get("annual_from"),
            "annual_to": job.get("annual_to"),
            "company_id": company.get("id"),
        },
    }


def crawl(
    *,
    job_group: int,
    pages: int,
    page_size: int,
    sleep: float,
    out_path: Path,
    raw_path: Path,
    resume: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[int] = set()
    if resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(int(rec.get("meta", {}).get("id", -1)))
                except json.JSONDecodeError:
                    continue
        print(f"[resume] already collected {len(seen)} jobs")

    with httpx.Client(headers=DEFAULT_HEADERS, timeout=20.0) as client:
        ids: list[int] = []
        for page in range(pages):
            offset = page * page_size
            try:
                items = fetch_list(
                    client, job_group=job_group, offset=offset, limit=page_size
                )
            except httpx.HTTPError as e:
                print(f"[list offset={offset}] {e}", file=sys.stderr)
                break
            if not items:
                break
            for item in items:
                jid = item.get("id")
                if isinstance(jid, int) and jid not in seen:
                    ids.append(jid)
            time.sleep(sleep)

        print(f"[list] {len(ids)} new ids (skipped {len(seen)} already-seen)")

        out_mode = "a" if resume else "w"
        raw_mode = "a" if resume else "w"
        with raw_path.open(raw_mode, encoding="utf-8") as raw_f, out_path.open(
            out_mode, encoding="utf-8"
        ) as out_f:
            for job_id in tqdm(ids, desc="detail"):
                try:
                    job = fetch_detail(client, job_id)
                except httpx.HTTPError as e:
                    print(f"[detail {job_id}] {e}", file=sys.stderr)
                    continue
                if not job:
                    continue
                raw_f.write(json.dumps(job, ensure_ascii=False) + "\n")
                raw_f.flush()
                out_f.write(json.dumps(to_job_posting(job), ensure_ascii=False) + "\n")
                out_f.flush()
                time.sleep(sleep)


def main() -> None:
    ap = argparse.ArgumentParser(description="Wanted unofficial API crawler")
    ap.add_argument("--job-group", type=int, default=DEFAULT_JOB_GROUP)
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument("--page-size", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("data/wanted_jobs.jsonl"))
    ap.add_argument("--raw", type=Path, default=Path("data/wanted_raw.jsonl"))
    ap.add_argument(
        "--resume",
        action="store_true",
        help="기존 out 파일에 있는 id 는 건너뛰고 append",
    )
    args = ap.parse_args()

    crawl(
        job_group=args.job_group,
        pages=args.pages,
        page_size=args.page_size,
        sleep=args.sleep,
        out_path=args.out,
        raw_path=args.raw,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
