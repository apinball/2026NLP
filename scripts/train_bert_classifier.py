"""Module B 의 ML 축 — KLUE-RoBERTa 일관성 분류기 학습.

합성 이력서를 문장 단위로 분해해 "위반 포함" / "정상" 이진 분류 학습.
의도된 모순(연도+기술명 동시 등장)을 BERT 가 일반화해서 잡는지 평가.

사용:
  docker compose run --rm app bash -c \\
    "cd /app && PYTHONPATH=. python scripts/train_bert_classifier.py \\
     --input data/synthetic_resumes.jsonl --epochs 3"

전제: 합성 이력서 jsonl 이 존재해야 함.
  python scripts/gen_synthetic_resumes.py --normal 500 --error 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.logic_auditor.rule_based import _SENT_SPLIT


def load_records(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def build_sentence_examples(
    records: Iterable[dict],
) -> tuple[list[str], list[int], list[dict]]:
    """문장 단위 (text, label) 쌍 생성.

    label=1 if 문장에 (tech name) + (claimed_year) 가 동시에 등장 → 위반 문장
    label=0 그 외 모든 문장 (정상 문서의 모든 문장 + error 문서 중 비-위반 문장)
    """
    texts: list[str] = []
    labels: list[int] = []
    meta: list[dict] = []

    for rec in records:
        sentences = split_sentences(rec.get("text", ""))
        violations = rec.get("violations") or []

        for sent in sentences:
            is_violation = False
            matched_tech = None
            if rec.get("kind") == "error":
                for v in violations:
                    tech = (v.get("tech") or "").lower()
                    year = str(v.get("claimed_year") or "")
                    if tech and year and tech in sent.lower() and year in sent:
                        is_violation = True
                        matched_tech = v.get("tech")
                        break
            texts.append(sent)
            labels.append(1 if is_violation else 0)
            meta.append(
                {
                    "kind": rec.get("kind"),
                    "doc_tech": rec.get("tech"),
                    "matched_tech": matched_tech,
                }
            )

    return texts, labels, meta


class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = self.labels[idx]
        return enc


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/synthetic_resumes.jsonl"))
    ap.add_argument("--model-name", default="klue/roberta-base")
    ap.add_argument("--output-dir", type=Path, default=Path("data/bert_classifier"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save-splits",
        type=Path,
        default=Path("data/bert_splits.json"),
        help="train/val/test 분할 텍스트와 라벨을 저장 (eval 재현용)",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        print(
            "  python scripts/gen_synthetic_resumes.py --normal 500 --error 500",
            file=sys.stderr,
        )
        return 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[1/5] {args.input} 로딩…")
    records = load_records(args.input)
    print(f"      레코드 {len(records)} (normal {sum(1 for r in records if r.get('kind')=='normal')} "
          f"/ error {sum(1 for r in records if r.get('kind')=='error')})")

    print("[2/5] 문장 단위 데이터셋 변환…")
    texts, labels, meta = build_sentence_examples(records)
    n_pos = sum(labels)
    print(f"      문장 {len(texts)} (positive {n_pos} / negative {len(texts)-n_pos})")
    if n_pos < 10:
        print("      WARNING: positive 샘플이 너무 적음. error 데이터를 더 생성하세요.",
              file=sys.stderr)

    print("[3/5] stratified 70/15/15 split…")
    tr_x, tmp_x, tr_y, tmp_y = train_test_split(
        texts, labels, test_size=0.30, stratify=labels, random_state=args.seed
    )
    val_x, te_x, val_y, te_y = train_test_split(
        tmp_x, tmp_y, test_size=0.50, stratify=tmp_y, random_state=args.seed
    )
    print(f"      train {len(tr_x)} / val {len(val_x)} / test {len(te_x)}")

    if args.save_splits:
        args.save_splits.parent.mkdir(parents=True, exist_ok=True)
        args.save_splits.write_text(
            json.dumps(
                {
                    "train": {"texts": tr_x, "labels": tr_y},
                    "val": {"texts": val_x, "labels": val_y},
                    "test": {"texts": te_x, "labels": te_y},
                    "seed": args.seed,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"      splits → {args.save_splits}")

    print(f"[4/5] {args.model_name} 로딩…")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    train_ds = SentenceDataset(tr_x, tr_y, tokenizer, args.max_length)
    val_ds = SentenceDataset(val_x, val_y, tokenizer, args.max_length)
    test_ds = SentenceDataset(te_x, te_y, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=20,
        report_to="none",
        seed=args.seed,
    )

    print("[5/5] 학습 시작…")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=_metrics,
    )
    trainer.train()

    print("\n=== Test set evaluation ===")
    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=1)
    probs = torch.softmax(torch.tensor(pred_output.predictions), dim=1)[:, 1].numpy()

    metrics = _metrics((pred_output.predictions, np.array(te_y)))
    metrics["auc"] = float(roc_auc_score(te_y, probs))
    print(json.dumps(metrics, indent=2))

    print("\n=== Confusion matrix ===")
    cm = confusion_matrix(te_y, preds)
    print(cm)
    print("\n=== Classification report ===")
    print(classification_report(te_y, preds, target_names=["normal", "violation"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\n모델 저장: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
