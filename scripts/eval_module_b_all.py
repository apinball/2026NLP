"""Module B 종합 평가 — Rule vs BERT vs Ensemble.

합성 이력서 테스트 셋에서 세 방식의 Precision/Recall/F1 을 비교.

세 방식:
  1) Rule-based  : RuleBasedAuditor (출시연도 + 한국어 word-boundary)
  2) BERT        : KLUE-RoBERTa 일관성 분류기 (학습 모델)
  3) Ensemble    : OR (둘 중 하나라도 위반 검출하면 violation)

문서 단위 평가 (label = error 여부, prediction = violation 검출 여부).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.logic_auditor.pipeline import LogicAuditorPipeline
from src.logic_auditor.rule_based import RuleBasedAuditor


def load_records(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _print_metrics(name: str, y_true: list[int], y_pred: list[int]) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n=== {name} ===")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  F1:        {f1:.3f}")
    print("  Confusion matrix:")
    print(f"               pred=normal  pred=violation")
    print(f"  true=normal       {cm[0][0]:>4d}            {cm[0][1]:>4d}")
    print(f"  true=violation    {cm[1][0]:>4d}            {cm[1][1]:>4d}")
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, default=Path("data/synthetic_resumes.jsonl")
    )
    ap.add_argument(
        "--bert-model-dir", type=Path, default=Path("data/bert_classifier")
    )
    ap.add_argument(
        "--bert-threshold", type=float, default=0.5
    )
    ap.add_argument(
        "--use-test-split",
        type=Path,
        default=Path("data/bert_splits.json"),
        help="train/val/test 분할이 있으면 test 부분만 평가",
    )
    ap.add_argument("--out", type=Path, default=Path("data/eval_module_b_all.json"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1

    records = load_records(args.input)
    print(f"총 합성 레코드: {len(records)}")

    # test 분할 사용: 학습에 안 본 문서만 평가
    test_sentences: set[str] | None = None
    if args.use_test_split and args.use_test_split.exists():
        splits = json.loads(args.use_test_split.read_text(encoding="utf-8"))
        test_sentences = set(splits.get("test", {}).get("texts", []))
        print(f"test 분할 문장 set: {len(test_sentences)}")

    # 문서 단위 평가셋 구성
    eval_records: list[dict] = []
    for rec in records:
        if test_sentences is None:
            eval_records.append(rec)
        else:
            # 이 문서의 어떤 문장이든 test split 에 들어있으면 평가에 포함
            text = rec.get("text", "")
            if any(s in text for s in test_sentences if len(s) > 20):
                eval_records.append(rec)
    print(f"평가 대상 문서: {len(eval_records)}")

    y_true: list[int] = [1 if r.get("kind") == "error" else 0 for r in eval_records]

    print("\n[Rule-based] 실행…")
    rule_pipeline = LogicAuditorPipeline()
    y_rule = [
        1 if rule_pipeline.run(r.get("text", "")).violations else 0
        for r in eval_records
    ]

    print("[BERT] 실행…")
    y_bert: list[int]
    if not args.bert_model_dir.exists():
        print(
            f"   BERT 모델 디렉토리 없음 ({args.bert_model_dir}) — 학습부터 진행 필요\n"
            "   python scripts/train_bert_classifier.py",
            file=sys.stderr,
        )
        y_bert = [0] * len(eval_records)
        bert_available = False
    else:
        from src.logic_auditor.bert_detector import BertConsistencyDetector

        detector = BertConsistencyDetector(
            model_dir=args.bert_model_dir, threshold=args.bert_threshold
        )
        y_bert = [
            1 if detector.predict(r.get("text", "")).n_violations > 0 else 0
            for r in eval_records
        ]
        bert_available = True

    # Ensemble: OR
    y_ens = [1 if (rb or bb) else 0 for rb, bb in zip(y_rule, y_bert)]

    results: dict = {"n": len(eval_records)}
    results["rule"] = _print_metrics("Rule-based", y_true, y_rule)
    if bert_available:
        results["bert"] = _print_metrics(
            f"BERT (threshold={args.bert_threshold})", y_true, y_bert
        )
        results["ensemble_or"] = _print_metrics(
            "Ensemble (Rule OR BERT)", y_true, y_ens
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
