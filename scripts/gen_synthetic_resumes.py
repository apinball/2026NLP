"""Module B 학습/평가용 합성 이력서 생성기.

Ollama LLM 으로 두 종류의 텍스트를 생성:
  - normal: 정상 자소서/이력서 (위반 없음)
  - error : 의도적 연도 모순이 들어간 텍스트 (자동 라벨)

출력 jsonl 스키마:
  {
    "text": "...",
    "kind": "normal" | "error",
    "tech": "Docker",
    "year": 2010,
    "violations": [{"tech": ..., "kind": "release_year",
                    "claimed_year": Y, "actual_year": Y'}],
    "ai_generated": true
  }

전제: ollama 서비스가 동작 중이고 모델이 pull 되어 있어야 함.
  docker compose up -d ollama
  docker compose exec ollama ollama pull qwen2.5:14b-instruct-q4_K_M
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

from src.logic_auditor.ontology import OntologyDB, TechEntry
from src.reverse_job.adapters.ollama_client import DEFAULT_MODEL, OllamaLLMClient

NORMAL_PROMPT = """다음 조건에 맞는 한국어 자기소개서/이력서 한 단락을 작성하세요.

- 분량: 4~6 문장
- 직무 경험을 구체적인 연도와 함께 기술
- "{tech}" 를 {year}년 시점에 활용한 경험을 자연스럽게 포함
- 1인칭("저는", "제가") 사용
- 모든 연도와 기술 출시 시점이 일치해야 함

출력은 본문만, 다른 설명·접두어 없이.
"""

ERROR_PROMPT = """다음 조건에 맞는 한국어 자기소개서/이력서 한 단락을 작성하세요.

- 분량: 4~6 문장
- 반드시 다음 표현을 자연스럽게 포함: "{year}년에 {tech}를(을) 활용하여"
  (참고: "{tech}" 의 실제 출시 연도는 {year}년 이후이므로 사실상 불가능한 진술입니다.
   하지만 본 데이터는 학습용 의도적 오류 샘플이므로 그대로 넣어주세요.)
- 그 외 내용은 합리적인 직무 경험으로 채움
- 1인칭("저는", "제가") 사용
- 출시 모순을 직접적으로 언급하거나 사과하지 말 것 (자연스럽게 서술)

출력은 본문만, 다른 설명·접두어 없이.
"""


def _normal_pick(
    ontology: OntologyDB, rng: random.Random
) -> tuple[TechEntry, int]:
    entry = rng.choice(ontology.all_entries())
    year = rng.randint(max(entry.released_year, 2000), 2024)
    return entry, year


def _error_pick(
    ontology: OntologyDB, rng: random.Random
) -> tuple[TechEntry, int]:
    # 출시연도가 너무 이르면 의도적 오류 만들기 어려움 → 2005년 이후 항목만
    candidates = [e for e in ontology.all_entries() if e.released_year >= 2005]
    entry = rng.choice(candidates)
    # 출시 1~8년 전 사이에서 한 해를 임의 선택 (1995 이상 보장)
    earliest = max(1995, entry.released_year - 8)
    year = rng.randint(earliest, entry.released_year - 1)
    return entry, year


def gen_one(
    client: OllamaLLMClient,
    kind: str,
    ontology: OntologyDB,
    rng: random.Random,
) -> dict:
    if kind == "normal":
        entry, year = _normal_pick(ontology, rng)
        prompt = NORMAL_PROMPT.format(tech=entry.name, year=year)
        violations: list[dict] = []
    else:
        entry, year = _error_pick(ontology, rng)
        prompt = ERROR_PROMPT.format(tech=entry.name, year=year)
        violations = [
            {
                "tech": entry.name,
                "kind": "release_year",
                "claimed_year": year,
                "actual_year": entry.released_year,
            }
        ]

    text = client.complete(prompt).strip()
    return {
        "text": text,
        "kind": kind,
        "tech": entry.name,
        "year": year,
        "violations": violations,
        "ai_generated": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_resumes.jsonl"))
    ap.add_argument("--normal", type=int, default=50, help="정상 샘플 수")
    ap.add_argument("--error", type=int, default=50, help="오류 샘플 수")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true", help="기존 파일에 이어 쓰기")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ontology = OntologyDB()
    if not ontology.all_entries():
        print("ontology empty — run validate_ontology.py first", file=sys.stderr)
        return 1

    client = OllamaLLMClient(model=args.model)
    if not client.healthcheck():
        print(
            f"Ollama not reachable at {client.host}.\n"
            "  docker compose up -d ollama\n"
            f"  docker compose exec ollama ollama pull {args.model}",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    plan = [("normal", args.normal), ("error", args.error)]
    total = sum(c for _, c in plan)

    with args.out.open(mode, encoding="utf-8") as f, tqdm(
        total=total, desc="generate"
    ) as pbar:
        for kind, count in plan:
            for _ in range(count):
                rec = gen_one(client, kind, ontology, rng)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                pbar.update(1)

    print(f"\nwrote {total} records → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())