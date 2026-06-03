"""Module B 정합성 분류기 학습/평가용 합성 이력서 생성기.

6가지 위반 유형을 균등하게 분포시켜 BERT 가 *rule-based 가 못 잡는*
의미적 정합성 위반까지 학습할 수 있도록 한다.

위반 유형:
  1) release_year       : 출시연도 이전 사용 — rule-based 도 잡음
  2) version_year       : 버전 출시연도 이전 도입 — rule-based 도 잡음
  3) experience_scale   : 짧은 기간 / 비현실적 규모 — **rule 불가, BERT 만**
  4) role_scope         : 신입·인턴인데 경영 책임 — **rule 불가, BERT 만**
  5) indirect_time      : 활동 시작 연도 < 기술 출시 연도, 단 다른 문장에 분리
                          → 동일 문장 매칭하는 rule 은 못 잡음
  6) tech_combo         : 호환되지 않는 기술 조합 — **rule 불가, BERT 만**

전제: ollama 서비스 + 모델 pull 완료.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from src.logic_auditor.ontology import OntologyDB
from src.reverse_job.adapters.ollama_client import DEFAULT_MODEL, OllamaLLMClient

NORMAL_PROMPT = """다음 조건에 맞는 한국어 자기소개서/이력서 한 단락을 작성하세요.

- 분량: 4~6 문장
- "{tech}" 를 {year}년 시점에 활용한 경험을 자연스럽게 포함
- 1인칭("저는", "제가") 사용
- 모든 연도와 기술 출시 시점이 일치해야 함

출력은 본문만, 다른 설명·접두어 없이.
"""

ERROR_TEMPLATES: dict[str, str] = {
    "release_year": """다음 조건의 한국어 자기소개서 한 단락 작성:
- 4~6 문장, 1인칭
- 반드시 포함: "{year}년에 {tech}를(을) 활용하여"
  (참고: {tech} 실제 출시는 {actual_year}년 — 학습용 의도된 오류 샘플)
- 다른 내용은 합리적 직무 경험으로 채움
출력은 본문만.""",

    "version_year": """다음 조건의 한국어 자기소개서 한 단락 작성:
- 4~6 문장, 1인칭
- 반드시 포함: "{year}년 프로젝트에서 {tech} {version}을 도입하여 개발했습니다"
  (참고: {tech} {version} 실제 출시는 {actual_year}년 — 의도된 오류)
- 다른 내용은 합리적으로 채움
출력은 본문만.""",

    "experience_scale": """다음 조건의 한국어 자기소개서 한 단락 작성:
- 4~6 문장, 1인칭
- 반드시 포함: "입사 후 {period} 만에 {scale} 규모의 {system}을 단독 설계·구현했습니다"
- 다른 내용은 합리적으로 채움 (의도된 과장 — 기간 대비 규모 비현실적)
출력은 본문만.""",

    "role_scope": """다음 조건의 한국어 자기소개서 한 단락 작성:
- 4~6 문장, 1인칭
- 반드시 포함: "{role} {month}개월 차에 {senior_task}를 단독으로 주도했습니다"
- 다른 내용은 합리적으로 채움 (의도된 부조화 — 직급 대비 책임 비현실적)
출력은 본문만.""",

    "indirect_time": """다음 조건의 한국어 자기소개서 한 단락 작성:
- 4~6 문장, 1인칭
- 첫 문장에 반드시 포함: "저는 {wrong_year}년부터 IT 업계에서 활동해왔습니다"
- 두 번째 문장에 반드시 포함: "{tech} 가 처음 등장했을 때부터 즉시 도입하여 운영했습니다"
- 두 문장 사이에 구체적 연도 숫자를 절대 쓰지 마세요
  (참고: {tech} 실제 출시 {actual_year}년 > 활동 시작 {wrong_year}년 — 시기 모순)
- 나머지 문장은 합리적으로 채움
출력은 본문만.""",

    "tech_combo": """다음 조건의 한국어 자기소개서 한 단락 작성:
- 4~6 문장, 1인칭
- 반드시 포함: "{tech_a}로 {task_b}를 구현하여 운영했습니다"
- 다른 내용은 합리적으로 채움 (의도된 부조화 — 두 기술/역할은 사실상 호환 불가)
출력은 본문만.""",
}

# experience_scale 파라미터 풀
SHORT_PERIODS = ["2주", "3주", "1개월", "2개월"]
HUGE_SCALES = [
    "10만 TPS", "100만 TPS", "1억 TPS",
    "DAU 100만", "DAU 1천만",
    "전사 1만명 동시 사용", "글로벌 7개 리전",
]
SYSTEMS = [
    "대규모 마이크로서비스 아키텍처",
    "전사 데이터 플랫폼",
    "글로벌 결제 시스템",
    "AI 모델 서빙 인프라",
    "분산 트랜잭션 시스템",
    "실시간 스트리밍 파이프라인",
]

# role_scope 파라미터 풀
JUNIOR_ROLES = ["신입", "인턴", "수습", "신규 입사자"]
SENIOR_TASKS = [
    "전사 KPI 설계",
    "임원진 대상 비즈니스 전략 보고",
    "조직 전체 개발 표준 수립",
    "글로벌 본사 협상",
    "100명 이상 규모 조직 개편",
    "C레벨 의사결정 단독 수행",
    "전사 정보보안 정책 수립",
]

# tech_combo 파라미터 풀 — (도구, 호환 안 되는 임무)
TECH_COMBO_PAIRS = [
    ("HWP", "분산 메시지 큐 시스템"),
    ("한컴오피스", "쿠버네티스 클러스터 운영"),
    ("카카오톡", "PostgreSQL 인덱스 최적화"),
    ("React Native", "백엔드 마이크로서비스 인프라"),
    ("jQuery", "대규모 Kafka 스트림 처리"),
    ("Photoshop", "데이터베이스 트랜잭션 격리 수준 조정"),
    ("Excel", "쿠버네티스 오토스케일링"),
    ("아프리카TV", "분산 데이터베이스 트랜잭션 처리"),
    ("멜론", "ML 모델 서빙 파이프라인"),
    ("카카오페이", "리눅스 커널 드라이버 개발"),
]


def _pick_release_year(ontology: OntologyDB, rng: random.Random) -> dict:
    cands = [e for e in ontology.all_entries() if e.released_year >= 2005]
    entry = rng.choice(cands)
    earliest = max(1995, entry.released_year - 8)
    year = rng.randint(earliest, entry.released_year - 1)
    return {"tech": entry.name, "year": year, "actual_year": entry.released_year}


def _pick_version_year(ontology: OntologyDB, rng: random.Random) -> dict:
    cands = [e for e in ontology.all_entries() if e.versions]
    if not cands:
        return _pick_release_year(ontology, rng)
    entry = rng.choice(cands)
    version, vyear = rng.choice(list(entry.versions.items()))
    earliest = max(1995, vyear - 6)
    year = rng.randint(earliest, vyear - 1)
    return {
        "tech": entry.name,
        "version": version,
        "year": year,
        "actual_year": vyear,
    }


def _pick_experience_scale(_ontology: OntologyDB, rng: random.Random) -> dict:
    return {
        "period": rng.choice(SHORT_PERIODS),
        "scale": rng.choice(HUGE_SCALES),
        "system": rng.choice(SYSTEMS),
    }


def _pick_role_scope(_ontology: OntologyDB, rng: random.Random) -> dict:
    return {
        "role": rng.choice(JUNIOR_ROLES),
        "month": rng.randint(1, 3),
        "senior_task": rng.choice(SENIOR_TASKS),
    }


def _pick_indirect_time(ontology: OntologyDB, rng: random.Random) -> dict:
    cands = [e for e in ontology.all_entries() if e.released_year >= 2010]
    entry = rng.choice(cands)
    earliest = max(1995, entry.released_year - 12)
    wrong_year = rng.randint(earliest, entry.released_year - 3)
    return {
        "tech": entry.name,
        "wrong_year": wrong_year,
        "actual_year": entry.released_year,
    }


def _pick_tech_combo(_ontology: OntologyDB, rng: random.Random) -> dict:
    tech_a, task_b = rng.choice(TECH_COMBO_PAIRS)
    return {"tech_a": tech_a, "task_b": task_b}


PICKERS: dict[str, Callable] = {
    "release_year": _pick_release_year,
    "version_year": _pick_version_year,
    "experience_scale": _pick_experience_scale,
    "role_scope": _pick_role_scope,
    "indirect_time": _pick_indirect_time,
    "tech_combo": _pick_tech_combo,
}


def _violations_for(type_name: str, params: dict) -> list[dict]:
    if type_name == "release_year":
        return [
            {
                "tech": params["tech"],
                "kind": "release_year",
                "claimed_year": params["year"],
                "actual_year": params["actual_year"],
            }
        ]
    if type_name == "version_year":
        return [
            {
                "tech": params["tech"],
                "version": params["version"],
                "kind": "version_year",
                "claimed_year": params["year"],
                "actual_year": params["actual_year"],
            }
        ]
    if type_name == "experience_scale":
        return [
            {
                "kind": "experience_scale",
                "period": params["period"],
                "scale": params["scale"],
                "system": params["system"],
            }
        ]
    if type_name == "role_scope":
        return [
            {
                "kind": "role_scope",
                "role": params["role"],
                "month": params["month"],
                "senior_task": params["senior_task"],
            }
        ]
    if type_name == "indirect_time":
        return [
            {
                "tech": params["tech"],
                "kind": "indirect_time",
                "claimed_year": params["wrong_year"],
                "actual_year": params["actual_year"],
            }
        ]
    if type_name == "tech_combo":
        return [
            {
                "kind": "tech_combo",
                "tech_a": params["tech_a"],
                "task_b": params["task_b"],
            }
        ]
    return []


def _normal_pick(ontology: OntologyDB, rng: random.Random) -> dict:
    entry = rng.choice(ontology.all_entries())
    year = rng.randint(max(entry.released_year, 2000), 2024)
    return {"tech": entry.name, "year": year}


def gen_one(
    client: OllamaLLMClient,
    kind: str,
    error_type: str | None,
    ontology: OntologyDB,
    rng: random.Random,
) -> dict:
    if kind == "normal":
        params = _normal_pick(ontology, rng)
        prompt = NORMAL_PROMPT.format(tech=params["tech"], year=params["year"])
        violations: list[dict] = []
    else:
        assert error_type is not None
        params = PICKERS[error_type](ontology, rng)
        prompt = ERROR_TEMPLATES[error_type].format(**params)
        violations = _violations_for(error_type, params)

    text = client.complete(prompt).strip()
    return {
        "text": text,
        "kind": kind,
        "error_type": error_type if kind == "error" else None,
        "params": params,
        "violations": violations,
        "ai_generated": True,
    }


def _build_plan(normal: int, error: int) -> list[tuple[str, str | None]]:
    plan: list[tuple[str, str | None]] = [("normal", None)] * normal
    types = list(ERROR_TEMPLATES.keys())
    per_type = error // len(types)
    remainder = error - per_type * len(types)
    for i, t in enumerate(types):
        count = per_type + (1 if i < remainder else 0)
        plan.extend([("error", t)] * count)
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_resumes.jsonl"))
    ap.add_argument("--normal", type=int, default=500, help="정상 샘플 수")
    ap.add_argument("--error", type=int, default=500, help="오류 샘플 수 (6 유형 균등)")
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
            f"Ollama not reachable at {client.host}\n"
            f"  docker compose up -d ollama && \\\n"
            f"  docker compose exec ollama ollama pull {args.model}",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    plan = _build_plan(args.normal, args.error)
    rng.shuffle(plan)
    total = len(plan)

    # 유형별 카운트 미리 출력
    from collections import Counter
    type_counts = Counter(t or "normal" for _, t in plan)
    print("계획:")
    for t, c in type_counts.most_common():
        print(f"  {t:20s} {c}")

    with args.out.open(mode, encoding="utf-8") as f, tqdm(
        total=total, desc="generate"
    ) as pbar:
        for kind, error_type in plan:
            rec = gen_one(client, kind, error_type, ontology, rng)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)

    print(f"\nwrote {total} records → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
