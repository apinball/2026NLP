"""학습된 KLUE-RoBERTa 일관성 분류기 추론기.

Module B 의 ML 축. 학습은 문서 단위로 진행되었으므로 추론도 문서 단위가 기본.
필요 시 `predict_sentences()` 로 문장 단위 분해도 가능.
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
    document_probability: float = 0.0
    is_violation: bool = False
    sentence_predictions: list[BertSentencePrediction] = field(default_factory=list)
    # 문장 단위 호출용 보조 통계
    max_sentence_probability: float = 0.0
    mean_sentence_probability: float = 0.0
    n_violations: int = 0


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class BertConsistencyDetector:
    """KLUE-RoBERTa 기반 정합성 분류기.

    기본은 문서 단위 추론 (train_bert_classifier.py 와 동일).
    """

    def __init__(
        self,
        model_dir: str | Path = "data/bert_classifier",
        threshold: float = 0.5,
        max_length: int = 256,
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
        self.max_length = max_length
        self._torch = torch

    def _score(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        torch = self._torch
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        return probs

    def predict(self, text: str) -> BertDetectionResult:
        """문서 단위 추론. 학습과 동일한 입력 분포."""
        doc_prob = self._score([text])[0] if text.strip() else 0.0
        return BertDetectionResult(
            document_probability=float(doc_prob),
            is_violation=doc_prob >= self.threshold,
        )

    def predict_sentences(self, text: str) -> BertDetectionResult:
        """문장 단위 분해 후 각각 추론. 위반 위치 하이라이트용."""
        sentences = _split_sentences(text)
        if not sentences:
            return BertDetectionResult()
        probs = self._score(sentences)
        preds = [
            BertSentencePrediction(
                sentence=sent,
                probability=float(prob),
                is_violation=prob >= self.threshold,
            )
            for sent, prob in zip(sentences, probs)
        ]
        max_p = max(probs)
        return BertDetectionResult(
            document_probability=max_p,
            is_violation=max_p >= self.threshold,
            sentence_predictions=preds,
            max_sentence_probability=max_p,
            mean_sentence_probability=sum(probs) / len(probs),
            n_violations=sum(1 for p in preds if p.is_violation),
        )
