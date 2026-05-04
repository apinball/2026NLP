from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.config import HF_NER_MODEL_KO

TECH_KEYWORDS: set[str] = {
    "python", "java", "javascript", "typescript", "golang", "rust", "kotlin",
    "react", "vue", "angular", "next.js", "nuxt", "svelte",
    "spring", "spring boot", "django", "flask", "fastapi", "node.js", "express",
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch", "oracle",
    "docker", "kubernetes", "k8s", "aws", "gcp", "azure",
    "git", "github", "gitlab", "jenkins", "ci/cd",
    "rest", "restful", "graphql", "grpc",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "kafka", "rabbitmq", "airflow", "spark", "hadoop",
}

# 표기 변형 → 표준 토큰 매핑. wanted API skill_tags 같이 외부에서 들어오는
# 자유서식 스킬명을 ARM 등에서 합치기 위한 정규화 테이블.
SKILL_ALIASES: dict[str, str] = {
    "spring framework": "spring",
    "spring-boot": "spring boot",
    "springboot": "spring boot",
    "react.js": "react",
    "reactjs": "react",
    "react native": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "nextjs": "next.js",
    "next js": "next.js",
    "node": "node.js",
    "nodejs": "node.js",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
}


def normalize_skill_token(raw: str) -> str:
    """외부 입력 스킬 문자열을 SKILL_ALIASES + lower 로 정규화."""
    if not raw:
        return ""
    norm = raw.lower().strip()
    return SKILL_ALIASES.get(norm, norm)


EXPERIENCE_RE = re.compile(r"(\d+)\s*년\s*(?:이상|이하|차)?")


def build_keyword_regex(keywords: Iterable[str]) -> re.Pattern[str]:
    """ASCII 단어 경계 가드를 적용한 키워드 매칭 regex.

    `\\b` 는 한국어 텍스트(예: "Python을")에서는 동작하지 않으므로
    영문/숫자가 아닌 문자만 경계로 인정하는 lookaround 를 사용한다.
    이렇게 해야 "going" 안에서 "go" 를 거짓 매칭하지 않으면서
    "Python을" 의 "python" 은 정상 매칭된다.
    """
    sorted_kws = sorted({k.lower() for k in keywords if k}, key=len, reverse=True)
    if not sorted_kws:
        return re.compile(r"(?!x)x")
    escaped = "|".join(re.escape(k) for k in sorted_kws)
    return re.compile(
        rf"(?<![a-zA-Z0-9])({escaped})(?![a-zA-Z0-9])",
        re.IGNORECASE,
    )


KEYWORD_REGEX = build_keyword_regex(TECH_KEYWORDS)


def find_tech_keywords(text: str, regex: re.Pattern[str] | None = None) -> list[str]:
    pattern = regex or KEYWORD_REGEX
    hits: set[str] = set()
    for m in pattern.finditer(text):
        hits.add(m.group(1).lower())
    return sorted(hits)


@dataclass
class JobRequirements:
    tech_stack: list[str] = field(default_factory=list)
    experience_years: list[int] = field(default_factory=list)
    qualifications: list[str] = field(default_factory=list)
    raw_entities: list[dict[str, Any]] = field(default_factory=list)


class JobExtractor:
    """채용공고에서 명시적 요구사항을 추출한다.

    load_ner=False 로 초기화하면 HF 모델을 내려받지 않고 키워드 사전 기반으로만
    동작한다. 오프라인 스모크 테스트에 유용.
    """

    def __init__(
        self,
        model_name: str = HF_NER_MODEL_KO,
        load_ner: bool = True,
    ) -> None:
        self.model_name = model_name
        self._ner = None
        if load_ner:
            self._ner = self._build_ner(model_name)

    @staticmethod
    def _build_ner(model_name: str):
        from transformers import pipeline

        return pipeline("ner", model=model_name, aggregation_strategy="simple")

    def extract(self, text: str) -> JobRequirements:
        req = JobRequirements()
        req.tech_stack = find_tech_keywords(text)
        req.experience_years = [int(m.group(1)) for m in EXPERIENCE_RE.finditer(text)]

        if self._ner is not None:
            raw = self._ner(text)
            req.raw_entities = [dict(e) for e in raw]
            for ent in raw:
                group = ent.get("entity_group", "")
                word = ent.get("word", "").strip()
                if not word:
                    continue
                if group in {"CVL", "OGG_EDUCATION", "QT"}:
                    req.qualifications.append(word)

        return req
