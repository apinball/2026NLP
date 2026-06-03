"""학습된 KLUE-RoBERTa 일관성 분류기 추론기.

Module B 의 ML 축. 문장 단위로 위반 확률을 산출하고 문서 단위로 집계한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.logic_auditor.rule_based import _SENT_SPLIT


@dataclass
class BertSentencePrediction:
    sentence: str
    probability: float
    is_violation: bool


@dataclass
class BertDetectionResult:
    sentence_predictions: list[BertSentencePrediction] = field(default_factory=list)
    max_probability: float = 0.0
    mean_probability: float = 0.0
    n_violations: int = 0


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class BertConsistencyDetector:
    """KLUE-RoBERTa 기반 문장 일관성 분류기.

    학습은 scripts/train_bert_classifier.py 참고. 출력 폴더(`data/bert_classifier`)
    를 model_dir 로 주입.
    """

    def __init__(
        self,
        model_dir: str | Path = "data/bert_classifier",
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.threshold = threshold
        self._torch = torch

    def predict(self, text: str, max_length: int = 128) -> BertDetectionResult:
        sentences = _split_sentences(text)
        if not sentences:
            return BertDetectionResult()

        torch = self._torch
        enc = self.tokenizer(
            sentences,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

        preds = [
            BertSentencePrediction(
                sentence=sent,
                probability=float(prob),
                is_violation=prob >= self.threshold,
            )
            for sent, prob in zip(sentences, probs)
        ]

        return BertDetectionResult(
            sentence_predictions=preds,
            max_probability=max(probs),
            mean_probability=sum(probs) / len(probs),
            n_violations=sum(1 for p in preds if p.is_violation),
        )
