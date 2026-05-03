"""수집된 채용공고에 Reverse Job Engineering 파이프라인을 돌려본다.

ARM(Apriori) 만 사용 — LLM 미정 상태이므로 reasoner 는 None.
수집된 wanted 공고 + ceragem positions 를 모두 코퍼스로 사용한다.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from src.data.loaders import iter_jobs
from src.reverse_job.arm_miner import ARMMiner
from src.reverse_job.extractor import find_tech_keywords


def build_transactions(min_skills: int = 2) -> list[list[str]]:
    """JobPosting → transaction list. requirements/responsibilities 본문 + API skills 결합."""
    transactions: list[list[str]] = []
    for job in iter_jobs():
        text = " ".join(
            [
                job.raw_text,
                " ".join(job.responsibilities),
                " ".join(job.requirements),
                " ".join(job.preferred),
                " ".join(job.skills),
            ]
        )
        skills = find_tech_keywords(text)
        # API skill_tags 도 합쳐 표준 키로 정규화
        for s in job.skills:
            normalized = s.lower().strip()
            if normalized and normalized not in skills:
                # extractor 에 없는 항목은 그대로 추가
                skills.append(normalized)
        if len(skills) >= min_skills:
            transactions.append(sorted(set(skills)))
    return transactions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=float, default=0.05)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--top-rules", type=int, default=20)
    ap.add_argument("--target", default="python", help="암묵적 역량 추출 기준 스킬")
    args = ap.parse_args()

    transactions = build_transactions()
    print(f"transactions: {len(transactions)}")
    if not transactions:
        return

    skill_counts: Counter[str] = Counter()
    for t in transactions:
        skill_counts.update(t)
    print("\n--- top 15 skills ---")
    for s, c in skill_counts.most_common(15):
        print(f"  {s:25s} {c:>4d}  ({c/len(transactions)*100:.1f}%)")

    miner = ARMMiner(
        min_support=args.min_support, min_confidence=args.min_confidence
    )
    rules = miner.mine(transactions)
    print(f"\nrules: {len(rules)}")
    if rules.empty:
        print("(no rules at given thresholds — try lower min-support)")
        return

    rules = rules.sort_values(["confidence", "lift"], ascending=False).head(args.top_rules)
    print(f"\n--- top {len(rules)} rules ---")
    for _, row in rules.iterrows():
        ant = sorted(row["antecedents"])
        cons = sorted(row["consequents"])
        print(
            f"  {ant} -> {cons}  "
            f"(supp={row['support']:.3f}, conf={row['confidence']:.3f}, lift={row['lift']:.2f})"
        )

    target = args.target.lower()
    implicit = miner.find_implicit({target}, rules)
    print(f"\n--- implicit skills for {{{target}}} ---")
    print(json.dumps(implicit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
