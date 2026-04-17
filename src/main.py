from __future__ import annotations

import argparse
import json

from src.logic_auditor.pipeline import LogicAuditorPipeline
from src.reverse_job.pipeline import ReverseJobPipeline

SAMPLE_JOB = """[백엔드 개발자 채용]
- Python, Django, MySQL 3년 이상 경력
- REST API 설계 경험 필수
- AWS 기반 서비스 운영 경험 우대
"""

SAMPLE_CORPUS = [
    "Python Django AWS Docker Git PostgreSQL REST",
    "Python FastAPI Docker Kubernetes Redis Git",
    "Java Spring MySQL AWS Docker Jenkins Git",
    "Python Django MySQL Redis Docker Git CI/CD",
    "Python FastAPI MySQL Docker AWS Git REST",
]

SAMPLE_RESUME = """2010년부터 Docker를 활용하여 대규모 인프라를 구축했습니다.
3개월 만에 대규모 시스템 아키텍처를 단독으로 설계하고 10만 TPS를 처리하는 서비스를 완성했습니다.
React 16과 Next.js 13을 2015년 프로젝트에 적용했습니다.
"""


def demo_reverse_job() -> None:
    # LLMReasoner 는 사용할 LLM 미정이므로 주입하지 않고 ARM 단독으로 데모.
    pipeline = ReverseJobPipeline(reasoner=None, load_ner=False)
    result = pipeline.run(SAMPLE_JOB, corpus=SAMPLE_CORPUS)

    print("=== Reverse Job Engineering ===")
    print(
        json.dumps(
            {
                "explicit_tech": result.explicit.tech_stack,
                "implicit_arm": result.implicit_arm,
                "implicit_llm": result.implicit_llm,
                "implicit_union": result.implicit_union,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def demo_logic_auditor(use_ml: bool) -> None:
    detector = None
    if use_ml:
        from src.logic_auditor.ml_detector import AIGenerationDetector

        detector = AIGenerationDetector()

    pipeline = LogicAuditorPipeline(ai_detector=detector)
    report = pipeline.run(SAMPLE_RESUME)

    print("\n=== Logic Auditor ===")
    print(f"Trust score: {report.trust_score}/100")
    for v in report.violations:
        print(
            f"- [{v.tech}] claimed={v.claimed_year} actual={v.actual_year} :: {v.snippet}"
        )
    if report.ai_detection is not None:
        ai = report.ai_detection
        print(
            f"AI-gen signal: perplexity={ai.perplexity:.2f} "
            f"burstiness={ai.burstiness:.2f} p(ai)={ai.ai_probability:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reverse Job + Logic Auditor demo")
    parser.add_argument(
        "--module",
        choices=["rje", "audit", "all"],
        default="all",
        help="which demo to run",
    )
    parser.add_argument(
        "--use-ml", action="store_true", help="enable perplexity-based AI detector"
    )
    args = parser.parse_args()

    if args.module in {"rje", "all"}:
        demo_reverse_job()
    if args.module in {"audit", "all"}:
        demo_logic_auditor(args.use_ml)


if __name__ == "__main__":
    main()
