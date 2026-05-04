"""합성 이력서 데이터셋으로 Module B (RuleBasedAuditor) 정량 평가.

각 합성 레코드는 라벨(violations: list)을 가지고 있으므로
- error 샘플 (의도된 연도 모순): 검출되어야 정상 → Recall
- normal 샘플 (정상 진술): 검출되지 않아야 정상 → False Positive Rate

라벨된 tech 와 detector 가 잡은 tech 가 같은지로 strict 매칭,
"error 샘플에서 어떤 위반이라도 잡혔는지"로 lenient 매칭도 함께 보고.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.logic_auditor.pipeline import LogicAuditorPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, default=Path("data/synthetic_resumes.jsonl")
    )
    ap.add_argument("--show-misses", type=int, default=10)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"missing: {args.input}")
        return 1

    pipeline = LogicAuditorPipeline()

    n_normal = n_error = 0
    # error 샘플 평가
    error_strict_hit = 0   # 라벨된 tech 가 detector 결과에 포함됨
    error_lenient_hit = 0  # detector 가 위반을 1개 이상 잡음
    error_missed: list[dict] = []
    error_wrong_tech: list[dict] = []  # 위반은 잡았으나 다른 tech 로

    # normal 샘플 평가
    normal_clean = 0
    normal_false_positive: list[dict] = []
    fp_tech_counter: Counter[str] = Counter()

    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec["kind"]
            text = rec["text"]
            label_techs = {v["tech"] for v in rec.get("violations", [])}

            report = pipeline.run(text)
            detected_techs = {v.tech for v in report.violations}

            if kind == "normal":
                n_normal += 1
                if not detected_techs:
                    normal_clean += 1
                else:
                    for t in detected_techs:
                        fp_tech_counter[t] += 1
                    if len(normal_false_positive) < args.show_misses:
                        normal_false_positive.append(
                            {
                                "tech": rec.get("tech"),
                                "year": rec.get("year"),
                                "detected": sorted(detected_techs),
                                "snippet": text[:120],
                            }
                        )
            else:  # error
                n_error += 1
                if detected_techs & label_techs:
                    error_strict_hit += 1
                if detected_techs:
                    error_lenient_hit += 1
                else:
                    if len(error_missed) < args.show_misses:
                        error_missed.append(
                            {
                                "tech": rec.get("tech"),
                                "year": rec.get("year"),
                                "snippet": text[:160],
                            }
                        )
                if detected_techs and not (detected_techs & label_techs):
                    if len(error_wrong_tech) < 5:
                        error_wrong_tech.append(
                            {
                                "label_tech": rec.get("tech"),
                                "detected": sorted(detected_techs),
                                "snippet": text[:120],
                            }
                        )

    print(f"corpus: {n_normal + n_error} ({n_normal} normal / {n_error} error)\n")

    print("=== Module B (RuleBasedAuditor) 평가 ===")
    if n_error:
        recall_strict = error_strict_hit / n_error
        recall_lenient = error_lenient_hit / n_error
        print(f"Recall (strict, 라벨된 tech 일치):  {recall_strict:.3f} ({error_strict_hit}/{n_error})")
        print(f"Recall (lenient, 위반 1+):          {recall_lenient:.3f} ({error_lenient_hit}/{n_error})")
    if n_normal:
        fp_rate = (n_normal - normal_clean) / n_normal
        print(f"False positive rate (normal):        {fp_rate:.3f} ({n_normal - normal_clean}/{n_normal})")
        print(f"Specificity:                          {normal_clean / n_normal:.3f}")

    if fp_tech_counter:
        print("\n--- normal 샘플에서 잘못 잡힌 tech top ---")
        for tech, cnt in fp_tech_counter.most_common(10):
            print(f"  {tech:25s} {cnt}")

    if error_missed:
        print(f"\n--- 검출 실패한 error 샘플 (최대 {args.show_misses}건) ---")
        for m in error_missed:
            print(f"  [{m['tech']} / {m['year']}] {m['snippet']}")

    if normal_false_positive:
        print(f"\n--- normal 샘플 false positive (최대 {args.show_misses}건) ---")
        for m in normal_false_positive:
            print(f"  [라벨 tech={m['tech']} / 검출={m['detected']}] {m['snippet']}")

    if error_wrong_tech:
        print("\n--- 위반은 잡았으나 다른 tech 로 검출 ---")
        for m in error_wrong_tech:
            print(f"  [라벨={m['label_tech']} / 검출={m['detected']}] {m['snippet']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())