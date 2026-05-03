"""Logic Auditor (Module B) Rule-based 검증을 합격 자소서 코퍼스에 돌린다.

합격자가 작성한 글이므로 진짜 사실 오류는 거의 없어야 한다 → 위반 비율은
온톨로지·정규식의 false positive 율을 가늠하는 신호로 해석한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice

from src.data.loaders import iter_resumes
from src.logic_auditor.pipeline import LogicAuditorPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="검사할 자소서 수")
    ap.add_argument(
        "--show-violations", type=int, default=15, help="샘플 위반 출력 개수"
    )
    args = ap.parse_args()

    pipeline = LogicAuditorPipeline()  # ai_detector 는 None — Rule-based 만

    inspected = 0
    docs_with_violation = 0
    tech_counter: Counter[str] = Counter()
    kind_counter: Counter[str] = Counter()
    score_buckets: Counter[str] = Counter()
    violation_samples: list[tuple[str, str, str]] = []  # (tech, snippet, message)

    for resume in islice(iter_resumes(), args.limit):
        inspected += 1
        report = pipeline.run(resume.full_text)
        if report.violations:
            docs_with_violation += 1
            for v in report.violations:
                tech_counter[v.tech] += 1
                kind_counter[v.kind] += 1
                if len(violation_samples) < args.show_violations:
                    violation_samples.append((v.tech, v.snippet, v.message))

        bucket = "100" if report.trust_score == 100 else (
            "85-99" if report.trust_score >= 85 else (
                "70-84" if report.trust_score >= 70 else "<70"
            )
        )
        score_buckets[bucket] += 1

    print(f"inspected: {inspected}")
    print(f"docs with violation: {docs_with_violation} ({docs_with_violation/inspected*100:.1f}%)")

    print("\n--- trust score 분포 ---")
    for bucket in ("100", "85-99", "70-84", "<70"):
        n = score_buckets.get(bucket, 0)
        print(f"  {bucket:>6s}  {n:>4d}  ({n/inspected*100:.1f}%)")

    print("\n--- 위반 기술 top ---")
    for tech, n in tech_counter.most_common(15):
        print(f"  {tech:25s} {n}")

    print("\n--- 위반 종류 ---")
    for kind, n in kind_counter.most_common():
        print(f"  {kind:15s} {n}")

    if violation_samples:
        print("\n--- 위반 샘플 (false positive 점검) ---")
        for tech, snippet, message in violation_samples:
            short = snippet[:120].replace("\n", " ")
            print(f"  [{tech}] {message}")
            print(f"      ...{short}...")


if __name__ == "__main__":
    main()