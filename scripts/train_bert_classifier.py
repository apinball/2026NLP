"""Module B 의 ML 축 — KLUE-RoBERTa 일관성 분류기 학습 (문서 단위).

합성 이력서 전체 본문을 입력으로 정상 vs 위반 이진 분류 학습.
이전 문장 단위 학습은 (a) 다양한 위반 유형 중 일부 패턴이 한 문장 안에 안 들어옴,
(b) indirect_time 같은 cross-sentence 모순을 학습 불가 → 문서 단위로 전환.

사용:
  docker compose run --rm app bash -c \\
    "cd /app && PYTHONPATH=. python scripts/train_bert_classifier.py \\
     --input data/synthetic_resumes.jsonl --epochs 3"
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

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


def load_records(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class DocumentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save-splits",
        type=Path,
        default=Path("data/bert_splits.json"),
        help="train/val/test 분할 인덱스 + error_type 메타 저장",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"missing: {args.input}", file=sys.stderr)
        return 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[1/5] {args.input} 로딩…")
    records = load_records(args.input)
    n_normal = sum(1 for r in records if r.get("kind") == "normal")
    n_error = sum(1 for r in records if r.get("kind") == "error")
    print(f"      레코드 {len(records)} (normal {n_normal} / error {n_error})")

    type_counts: dict[str, int] = {}
    for r in records:
        if r.get("kind") == "error":
            t = r.get("error_type") or "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
    if type_counts:
        print("      error_type 분포:")
        for t, c in sorted(type_counts.items()):
            print(f"        {t:20s} {c}")

    print("[2/5] 문서 단위 데이터셋 변환…")
    texts = [r.get("text", "") for r in records]
    labels = [1 if r.get("kind") == "error" else 0 for r in records]
    types = [r.get("error_type") for r in records]
    print(f"      문서 {len(texts)} (positive {sum(labels)} / negative {len(labels)-sum(labels)})")

    print("[3/5] stratified 70/15/15 split…")
    idx = list(range(len(texts)))
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=args.seed
    )
    val_idx, te_idx = train_test_split(
        tmp_idx,
        test_size=0.50,
        stratify=[labels[i] for i in tmp_idx],
        random_state=args.seed,
    )
    print(f"      train {len(tr_idx)} / val {len(val_idx)} / test {len(te_idx)}")

    def _by(indices, src):
        return [src[i] for i in indices]

    tr_x = _by(tr_idx, texts); tr_y = _by(tr_idx, labels)
    val_x = _by(val_idx, texts); val_y = _by(val_idx, labels)
    te_x = _by(te_idx, texts); te_y = _by(te_idx, labels)
    te_types = _by(te_idx, types)

    if args.save_splits:
        args.save_splits.parent.mkdir(parents=True, exist_ok=True)
        args.save_splits.write_text(
            json.dumps(
                {
                    "train_indices": tr_idx,
                    "val_indices": val_idx,
                    "test_indices": te_idx,
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

    train_ds = DocumentDataset(tr_x, tr_y, tokenizer, args.max_length)
    val_ds = DocumentDataset(val_x, val_y, tokenizer, args.max_length)
    test_ds = DocumentDataset(te_x, te_y, tokenizer, args.max_length)
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
    print(confusion_matrix(te_y, preds))
    print("\n=== Classification report ===")
    print(classification_report(te_y, preds, target_names=["normal", "error"]))

    # 유형별 Recall
    print("\n=== Error type 별 Recall (test set) ===")
    per_type: dict[str, tuple[int, int]] = {}
    for i, lbl in enumerate(te_y):
        if lbl != 1:
            continue
        t = te_types[i] or "unknown"
        tp, total = per_type.get(t, (0, 0))
        if preds[i] == 1:
            tp += 1
        per_type[t] = (tp, total + 1)
    for t, (tp, total) in sorted(per_type.items()):
        recall = tp / total if total else 0.0
        print(f"  {t:20s} {tp:>3d}/{total:>3d}  ({recall*100:.1f}%)")

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
