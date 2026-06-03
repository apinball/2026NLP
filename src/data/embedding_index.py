"""의미 기반 SkillIndex — Sentence-BERT 임베딩으로 자소서 코퍼스와의 유사도 계산.

기존 SkillIndex(키워드 사전 매칭) 의 한계:
- LLM 이 출력한 "팀 협업 능력", "디지털 시스템 설계" 같은 의미적 역량은
  TECH_KEYWORDS 사전에 없으면 0% Recall 로 잡힘
- 사실 자소서에 비슷한 표현이 있을 수 있음

해결: 한국어 SBERT (jhgan/ko-sroberta-multitask) 로 자소서 sections 를 임베딩,
LLM 출력 스킬 텍스트와 코사인 유사도 측정. threshold 이상이면 "발견".
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.data.loaders import Resume, iter_resumes

DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_CACHE = Path("data/embedding_cache.npz")

# 단일 키워드("python") 만 임베딩하면 자소서 코퍼스와의 의미 정렬이 약하다.
# 같은 키워드를 여러 한국어 문맥에 끼워 넣어 query 다양화 → 각 query 별 최대
# 코사인 유사도의 최댓값을 채택한다 (앙상블 query).
QUERY_TEMPLATES: list[str] = [
    "{skill}",
    "{skill}을 활용한 프로젝트 경험",
    "{skill} 기반 개발 경험",
    "{skill} 사용 능력",
]


def _expand_queries(skill: str) -> list[str]:
    return [tmpl.format(skill=skill) for tmpl in QUERY_TEMPLATES]


@dataclass
class EmbeddingSkillIndex:
    """자소서 코퍼스의 SBERT 임베딩 인덱스 + 유사도 기반 Recall 대리지표."""

    model_name: str = DEFAULT_MODEL
    embeddings: np.ndarray | None = None  # shape (N, D)
    texts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    _model: object | None = None  # SentenceTransformer (lazy)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(
        self,
        resumes: Iterable[Resume] | None = None,
        max_docs: int = 1500,
        min_len: int = 50,
        max_len: int = 600,
        batch_size: int = 32,
    ) -> "EmbeddingSkillIndex":
        """자소서 sections 를 임베딩해 인덱스 구축."""
        if resumes is None:
            resumes = iter_resumes()

        chunks: list[str] = []
        chunk_sources: list[str] = []
        for resume in resumes:
            for sec in resume.sections:
                t = (sec.answer or "").strip()
                if not (min_len <= len(t) <= max_len):
                    continue
                chunks.append(t)
                chunk_sources.append(f"{resume.source}/{resume.company}/{resume.job}")
                if len(chunks) >= max_docs:
                    break
            if len(chunks) >= max_docs:
                break

        model = self._load_model()
        embs = model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 코사인 유사도 = dot product
        )
        self.embeddings = embs
        self.texts = chunks
        self.sources = chunk_sources
        return self

    def save(self, path: Path = DEFAULT_CACHE) -> None:
        if self.embeddings is None:
            raise RuntimeError("index not built")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=self.embeddings,
            texts=np.array(self.texts, dtype=object),
            sources=np.array(self.sources, dtype=object),
            model_name=np.array([self.model_name]),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_CACHE) -> "EmbeddingSkillIndex":
        data = np.load(path, allow_pickle=True)
        idx = cls(
            model_name=str(data["model_name"][0]),
            embeddings=data["embeddings"],
            texts=list(data["texts"]),
            sources=list(data["sources"]),
        )
        return idx

    def similarities(self, skills: list[str], batch_size: int = 32) -> np.ndarray:
        """각 skill 의 코퍼스 내 최대 코사인 유사도. shape (S,).

        각 skill 을 4개의 한국어 query 템플릿으로 확장한 뒤, 모든
        (query × 코퍼스) 쌍의 코사인 유사도 중 최댓값을 채택한다.
        """
        if not skills or self.embeddings is None:
            return np.array([])
        model = self._load_model()

        # skill 별 query 범위
        expanded: list[str] = []
        ranges: list[tuple[int, int]] = []
        for s in skills:
            queries = _expand_queries(s)
            start = len(expanded)
            expanded.extend(queries)
            ranges.append((start, start + len(queries)))

        q_embs = model.encode(
            expanded,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sims_all = q_embs @ self.embeddings.T  # (Q, N)

        out = np.zeros(len(skills), dtype=np.float32)
        for i, (start, end) in enumerate(ranges):
            out[i] = sims_all[start:end].max()
        return out

    def recall_proxy(self, skills: list[str], threshold: float = 0.55) -> float:
        """유사도 ≥ threshold 인 스킬의 비율 (의미 기반 Recall 대리지표)."""
        skills = [s for s in skills if s]
        if not skills:
            return 0.0
        max_sims = self.similarities(skills)
        hit = int((max_sims >= threshold).sum())
        return hit / len(skills)

    def top_matches(
        self, skill: str, k: int = 3, threshold: float = 0.0
    ) -> list[tuple[float, str]]:
        """디버깅용: 한 skill 의 top-k 유사 자소서 문장."""
        if self.embeddings is None:
            return []
        model = self._load_model()
        emb = model.encode([skill], convert_to_numpy=True, normalize_embeddings=True)
        sims = (emb @ self.embeddings.T).ravel()
        order = np.argsort(-sims)
        out = []
        for i in order[:k]:
            if sims[i] < threshold:
                break
            out.append((float(sims[i]), self.texts[i]))
        return out


def build_or_load(
    cache: Path = DEFAULT_CACHE,
    max_docs: int = 1500,
) -> EmbeddingSkillIndex:
    """캐시 있으면 로드, 없으면 빌드."""
    if cache.exists():
        print(f"[embedding-index] load cache {cache}")
        return EmbeddingSkillIndex.load(cache)
    print(f"[embedding-index] build (max_docs={max_docs})…")
    idx = EmbeddingSkillIndex().build(max_docs=max_docs)
    idx.save(cache)
    print(f"[embedding-index] saved → {cache}  (N={len(idx.texts)})")
    return idx
