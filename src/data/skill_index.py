from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from src.data.loaders import Resume, iter_resumes
from src.reverse_job.extractor import (
    KEYWORD_REGEX,
    TECH_KEYWORDS,
    build_keyword_regex,
    find_tech_keywords,
)


@dataclass
class SkillIndex:
    """후기(자소서) 코퍼스에서 어떤 스킬이 얼마나 언급되는지 집계.

    Module A 의 Recall 대리지표(§3.3) 용. ARM/LLM 이 추론한 암묵적 역량이
    실제 합격자 자소서 본문에서 등장하는 빈도로 검증한다.
    """

    counts: Counter[str] = field(default_factory=Counter)
    total_docs: int = 0

    def frequency(self, skill: str) -> float:
        if self.total_docs == 0:
            return 0.0
        return self.counts.get(skill.lower(), 0) / self.total_docs

    def recall_proxy(self, skills: Iterable[str], min_freq: float = 0.005) -> float:
        skills = [s for s in skills if s]
        if not skills:
            return 0.0
        hit = sum(1 for s in skills if self.frequency(s) >= min_freq)
        return hit / len(skills)

    def top(self, k: int = 30) -> list[tuple[str, int]]:
        return self.counts.most_common(k)


def build_skill_index(
    resumes: Iterable[Resume] | None = None,
    vocab: Iterable[str] | None = None,
) -> SkillIndex:
    if resumes is None:
        resumes = iter_resumes()
    if vocab is None:
        regex = KEYWORD_REGEX
    else:
        regex = build_keyword_regex(vocab)

    idx = SkillIndex()
    for resume in resumes:
        text = resume.full_text + " " + resume.pass_spec.certifications
        if not text.strip():
            continue
        idx.total_docs += 1
        for kw in find_tech_keywords(text, regex):
            idx.counts[kw] += 1
    return idx
