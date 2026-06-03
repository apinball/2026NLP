"""Module A 추론 변형 비교: ARM-only vs LLM-only vs Ensemble.

각 채용공고에 대해 세 변형이 출력한 암묵적 역량 목록을, 자소서 코퍼스 기반
SkillIndex (Recall 대리지표, 기획서 §3.3) 로 정량 비교한다.

전제: ollama 서비스가 동작 중이고 모델이 pull 되어 있어야 함.
  docker compose up -d ollama
  docker compose exec ollama ollama pull qwen2.5:14b-instruct-q4_K_M
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from scripts.run_arm import is_it_job, job_skills
from src.data.loaders import iter_jobs, iter_resumes
from src.data.skill_index import build_skill_index
from src.reverse_job.adapters.ollama_client import DEFAULT_MODEL, OllamaLLMClient
from src.reverse_job.arm_miner import ARMMiner
from src.reverse_job.extractor import find_tech_keywords
from src.reverse_job.llm_reasoner import LLMReasoner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=15, help="평가할 IT 공고 수")
    ap.add_argument("--resume-sample", type=int, default=2000, help="SkillIndex 표본 자소서 수")
    ap.add_argument("--min-support", type=float, default=0.05)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", type=Path, default=Path("data/eval_module_a.jsonl"), help="개별 결과 저장"
    )
    ap.add_argument("--no-llm", action="store_true", help="LLM 호출 건너뛰기 (ARM-only 측정)")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("[1/4] IT 공고 + ARM 코퍼스 빌드…")
    it_jobs = [j for j in iter_jobs() if is_it_job(j)]
    transactions = [job_skills(j) for j in it_jobs]
    transactions = [t for t in transactions if len(t) >= 2]
    print(f"      it_jobs={len(it_jobs)}, transactions={len(transactions)}")

    miner = ARMMiner(min_support=args.min_support, min_confidence=args.min_confidence)
    rules = miner.mine(transactions)
    print(f"      arm_rules={len(rules)}")

    print(f"[2/4] SkillIndex 빌드 (자소서 표본 {args.resume_sample})…")
    sample_resumes = list(islice(iter_resumes(), args.resume_sample))
    skill_index = build_skill_index(resumes=sample_resumes)
    print(f"      indexed_docs={skill_index.total_docs}")

    print(f"[3/4] 평가 대상 IT 공고 {args.sample}건 샘플링…")
    eligible = [j for j in it_jobs if len(j.raw_text) > 200]
    rng.shuffle(eligible)
    target_jobs = eligible[: args.sample]

    reasoner: LLMReasoner | None = None
    if not args.no_llm:
        print("      Ollama 헬스체크…")
        client = OllamaLLMClient(model=args.model)
        if not client.healthcheck():
            print(f"      Ollama not reachable at {client.host}", file=sys.stderr)
            return 1
        reasoner = LLMReasoner(client=client)

    print("[4/4] 변형별 추론 + Recall 대리지표 산출…")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_job: list[dict] = []

    with args.out.open("w", encoding="utf-8") as f:
        for job in tqdm(target_jobs, desc="eval"):
            explicit = sorted(set(find_tech_keywords(job.raw_text) + [s.lower() for s in job.skills]))
            arm_implicit = miner.find_implicit(set(explicit), rules)

            llm_implicit: list[str] = []
            if reasoner is not None:
                try:
                    llm_implicit = reasoner.infer_implicit(
                        job.raw_text, explicit, arm_candidates=arm_implicit
                    )
                except Exception as e:
                    print(f"  [llm error] {e}", file=sys.stderr)

            llm_normalized = sorted({s.lower().strip() for s in llm_implicit if s})
            ensemble = sorted(set(arm_implicit) | set(llm_normalized))

            row = {
                "company": job.company,
                "job_title": job.job_title,
                "explicit": explicit,
                "arm_implicit": arm_implicit,
                "llm_implicit": llm_normalized,
                "ensemble": ensemble,
                "recall_arm": skill_index.recall_proxy(arm_implicit),
                "recall_llm": skill_index.recall_proxy(llm_normalized),
                "recall_ensemble": skill_index.recall_proxy(ensemble),
                "size_arm": len(arm_implicit),
                "size_llm": len(llm_normalized),
                "size_ensemble": len(ensemble),
            }
            per_job.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    if not per_job:
        return 0

    def _avg(field: str) -> float:
        vs = [r[field] for r in per_job if r[field] is not None]
        return mean(vs) if vs else 0.0

    print("\n=== 변형별 평균 (sample={} jobs) ===".format(len(per_job)))
    print(f"{'variant':10s}  {'avg recall':>12s}  {'avg size':>10s}")
    print(f"{'-'*36}")
    print(f"{'ARM':10s}  {_avg('recall_arm'):>12.3f}  {_avg('size_arm'):>10.1f}")
    if reasoner:
        print(f"{'LLM':10s}  {_avg('recall_llm'):>12.3f}  {_avg('size_llm'):>10.1f}")
        print(f"{'Ensemble':10s}  {_avg('recall_ensemble'):>12.3f}  {_avg('size_ensemble'):>10.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())