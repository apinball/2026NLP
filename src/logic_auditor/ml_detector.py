from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean, stdev

from src.config import HF_PPL_MODEL

_SENT_SPLIT = re.compile(r"[.!?。\n]+")


@dataclass
class AIDetectionResult:
    perplexity: float
    burstiness: float
    ai_probability: float


class AIGenerationDetector:
    """문장별 perplexity와 burstiness로 AI 생성 가능성을 추정한다.

    LLM 생성 텍스트는 perplexity 가 낮고 분산(burstiness)도 낮은 경향을 보인다는
    관찰에 기반한 휴리스틱 스코어. 정식 classifier 로 교체하기 전 placeholder.
    """

    def __init__(
        self,
        model_name: str = HF_PPL_MODEL,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._torch = torch

    def score(self, text: str) -> AIDetectionResult:
        sentences = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 5]
        if not sentences:
            return AIDetectionResult(float("nan"), float("nan"), 0.0)

        ppls = []
        for s in sentences:
            p = self._sentence_perplexity(s)
            if not math.isnan(p):
                ppls.append(p)

        if not ppls:
            return AIDetectionResult(float("nan"), float("nan"), 0.0)

        overall = mean(ppls)
        burst = stdev(ppls) if len(ppls) > 1 else 0.0
        ai_prob = self._heuristic(overall, burst)
        return AIDetectionResult(overall, burst, ai_prob)

    def _sentence_perplexity(self, text: str) -> float:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = enc["input_ids"].to(self.device)
        if input_ids.size(1) < 2:
            return float("nan")
        with self._torch.no_grad():
            out = self.model(input_ids, labels=input_ids)
        return math.exp(out.loss.item())

    @staticmethod
    def _heuristic(perplexity: float, burstiness: float) -> float:
        ppl_score = max(0.0, min(1.0, (60.0 - perplexity) / 60.0))
        burst_score = max(0.0, min(1.0, (15.0 - burstiness) / 15.0))
        return round(0.6 * ppl_score + 0.4 * burst_score, 3)
