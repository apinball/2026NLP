"""Module B 종합 평가 — Rule vs BERT vs Ensemble (유형별 breakdown 포함).

학습에 사용한 train_indices 와 동일한 test_indices 만 평가하여 데이터 누수 방지.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.logic_auditor.pipeline import LogicAuditorPipeline


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
    print(f"  Precision: {p:.3f}    Recall: {r:.3f}    F1: {f1:.3f}")
    print(f"  Confusion matrix:")
    print(f"              pred=normal  pred=error")
    print(f"  normal           {cm[0][0]:>4d}        {cm[0][1]:>4d}")
    print(f"  error            {cm[1][0]:>4d}        {cm[1][1]:>4d}")
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def _per_type_recall(name: str, y_true, y_pred, types) -> dict[str, float]:
    print(f"\n--- {name} per error_type recall ---")
    by_type: dict[str, tuple[int, int]] = {}
    for yt, yp, t in zip(y_true, y_pred, types):
        if yt != 1:
            continue
        key = t or "unknown"
        tp, tot = by_type.get(key, (0, 0))
        if yp == 1:
            tp += 1
        by_type[key] = (tp, tot + 1)
    result: dict[str, float] = {}
    for t in sorted(by_type):
        tp, tot = by_type[t]
        recall = tp / tot if tot else 0.0
        result[t] = recall
        print(f"  {t:20s} {tp:>3d}/{tot:>3d}  ({recall*100:.1f}%)")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, default=Path("data/synthetic_resumes.jsonl")
    )
    ap.add_argument(
        "--bert-model-dir", type=Path, default=Path("data/bert_classifier")
    )
    ap.add_argument("--bert-threshold", type=float, default=0.5)
    ap.add_argument(
        "--splits", type=Path, default=Path("data/bert_splits.json"),
        help="train_bert_classifier.py 가 저장한 인덱스 (test_indices 사용)",
    )
    ap.add_argument("--out", type=Path, default=Path("data/eval_module_b_all.json"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1

    records = load_records(args.input)
    print(f"전체 합성 레코드: {len(records)}")

    # test split 만 평가
    if args.splits.exists():
        splits = json.loads(args.splits.read_text(encoding="utf-8"))
        test_idx = splits.get("test_indices", [])
        eval_records = [records[i] for i in test_idx]
        print(f"test split: {len(eval_records)} docs")
    else:
        print("splits 파일 없음 — 전체 사용")
        eval_records = records

    y_true = [1 if r.get("kind") == "error" else 0 for r in eval_records]
    types = [r.get("error_type") for r in eval_records]
    texts = [r.get("text", "") for r in eval_records]

    print("\n[Rule-based] 실행…")
    rule_pipeline = LogicAuditorPipeline()
    y_rule = [1 if rule_pipeline.run(t).violations else 0 for t in texts]

    print("[BERT] 실행…")
    bert_available = args.bert_model_dir.exists()
    if bert_available:
        from src.logic_auditor.bert_detector import BertConsistencyDetector

        detector = BertConsistencyDetector(
            model_dir=args.bert_model_dir, threshold=args.bert_threshold
        )
        # BERT 는 학습과 동일하게 문서 단위로 추론
        y_bert = [
            1 if detector.predict(t).is_violation else 0 for t in texts
        ]
    else:
        print(
            f"   BERT 모델 디렉토리 없음 ({args.bert_model_dir})", file=sys.stderr
        )
        y_bert = [0] * len(eval_records)

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

    # 유형별 recall
    results["per_type_recall"] = {
        "rule": _per_type_recall("Rule-based", y_true, y_rule, types),
    }
    if bert_available:
        results["per_type_recall"]["bert"] = _per_type_recall(
            "BERT", y_true, y_bert, types
        )
        results["per_type_recall"]["ensemble"] = _per_type_recall(
            "Ensemble", y_true, y_ens, types
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nsaved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
