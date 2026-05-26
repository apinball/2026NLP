"""
pipeline.py — Module A 통합 파이프라인 (Reverse Job Engineering)

[전체 파이프라인 흐름]
  채용공고 입력
      │
      ▼
  [1] JobExtractor           → 명시적 요구사항 추출 (기술스택, 경력, 자격증)
      │
      ▼
  [2] ARMMiner.fit()         → 코퍼스 전체로 연관 규칙 학습
      │
      ▼
  [3] ARMMiner.predict()     → 특정 공고의 누락 기술 예측
      │
      ▼
  [4] LLMReasoner.reason()   → CoT 기반 실무 맥락 추론
      │
      ▼
  [5] Ensemble               → ARM 결과 + LLM 결과 가중 합산
      │
      ▼
  RJEResult (최종 암묵적 역량 + 신뢰도 점수)

[앙상블 전략]
  - ARM 결과: confidence를 그대로 점수로 사용
  - LLM 결과: confidence를 그대로 점수로 사용
  - 두 소스 모두에서 나온 기술: 점수 보너스 (+0.15) — 합의에 의한 신뢰도 강화
  - 최종 점수 = min(1.0, 개별점수 + 합의보너스)
  - 정렬 후 상위 top_k개만 반환
"""

from dataclasses import dataclass, field
from typing import Optional

from extractor import JobExtractor, ExtractedRequirements
from arm_miner import ARMMiner, DomainARMMiner, ImplicitSkill
from llm_reasoner import LLMReasoner, LLMClient, LLMImplicitResult


# ── 최종 결과 데이터 클래스 ─────────────────────────────────────
@dataclass
class ImplicitSkillResult:
    skill: str
    score: float              # 0~1, 높을수록 암묵적으로 요구될 가능성 높음
    sources: list[str]        # ["ARM", "LLM"] 중 해당되는 소스
    reason: str = ""          # LLM이 제공한 근거 (있을 경우)
    arm_confidence: float = 0.0
    llm_confidence: float = 0.0


@dataclass
class RJEResult:
    """Reverse Job Engineering 최종 출력"""
    job_title: str
    explicit_skills: list[str]              # 명시적 추출 기술
    experience_years: Optional[int]
    certifications: list[str]
    implicit_skills: list[ImplicitSkillResult]  # 암묵적 역량 (점수 내림차순)
    used_mock_llm: bool = False


# ── 앙상블 로직 ────────────────────────────────────────────────
class SkillEnsemble:
    """
    ARM과 LLM 두 소스의 결과를 합산.

    [설계 원칙]
    - ARM은 통계적 근거, LLM은 의미론적 근거
    - 두 소스가 동의하면 신뢰도 보너스
    - 명시적 기술은 제거 (이미 공고에 있으므로)
    """

    CONSENSUS_BONUS = 0.15  # 두 소스 합의 시 보너스
    ALPHA = 0.6  # LLM 가중치 (맥락 이해 우선)
    BETA  = 0.4  # ARM 가중치 (통계적 신뢰도)

    def _normalize(self, scores: list[float]) -> list[float]:
        """
        소프트 정규화: 최댓값 기준으로 스케일 통일
        Min-Max 대신 max 기준 나누기 사용
        → 최솟값이 0으로 죽는 문제 방지
        예) [0.67, 0.67, 0.33] → [1.0, 1.0, 0.49]
        """
        if not scores:
            return scores
        max_s = max(scores)
        if max_s == 0:
            return [0.0 for _ in scores]
        return [s / max_s for s in scores]

    def merge(
        self,
        explicit_skills: list[str],
        arm_results: list[ImplicitSkill],
        llm_result: LLMImplicitResult,
        top_k: int = 10,
    ) -> list[ImplicitSkillResult]:

        explicit_lower = {s.lower() for s in explicit_skills}
        merged: dict[str, ImplicitSkillResult] = {}

        # ARM confidence 정규화
        arm_confs = [item.confidence for item in arm_results]
        arm_normalized = self._normalize(arm_confs)

        # LLM confidence 정규화
        llm_confs = [item.confidence for item in llm_result.implicit_skills]
        llm_normalized = self._normalize(llm_confs)

        # ARM 결과 수집 (정규화된 값 사용)
        for item, norm_conf in zip(arm_results, arm_normalized):
            if item.skill.lower() in explicit_lower:
                continue
            merged[item.skill] = ImplicitSkillResult(
                skill=item.skill,
                score=self.BETA * norm_conf,
                sources=["ARM"],
                arm_confidence=norm_conf,
            )

        # LLM 결과 수집 및 합산 (정규화된 값 사용)
        for item, norm_conf in zip(llm_result.implicit_skills, llm_normalized):
            if item.skill.lower() in explicit_lower:
                continue
            if item.skill in merged:
                # 합의: Final = α×LLM + β×ARM + 보너스
                existing = merged[item.skill]
                existing.sources.append("LLM")
                existing.llm_confidence = norm_conf
                existing.reason = item.reason
                existing.score = min(
                    1.0,
                    self.ALPHA * norm_conf + self.BETA * existing.arm_confidence + self.CONSENSUS_BONUS
                )
            else:
                # LLM only: Final = α×LLM
                merged[item.skill] = ImplicitSkillResult(
                    skill=item.skill,
                    score=self.ALPHA * norm_conf,
                    sources=["LLM"],
                    reason=item.reason,
                    llm_confidence=norm_conf,
                )

        # 점수 내림차순 정렬 후 top_k 반환
        results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ── 메인 파이프라인 ─────────────────────────────────────────────
class ReverseJobEngineeringPipeline:
    """
    Module A 통합 파이프라인.

    [초기화 파라미터]
    - ontology_path: 기술 온톨로지 JSON 경로
    - use_ner: KoELECTRA NER 사용 여부 (GPU 없으면 False)
    - llm_client: LLM 어댑터 (None이면 Mock 사용)
    - min_support: ARM 최소 지지도
    - min_confidence: ARM 최소 신뢰도
    - top_k: 반환할 암묵적 역량 최대 개수

    [데이터 교체 방법]
    pipeline.run(target_job, corpus_data_path="data/실제파일.json")
    → corpus_data_path 인자 하나만 바꾸면 됨
    """

    def __init__(
        self,
        ontology_path: str = "data/ontology_seed.json",
        use_ner: bool = False,
        llm_client: Optional[LLMClient] = None,
        min_support: float = 0.2,
        min_confidence: float = 0.4,
        top_k: int = 10,
    ):
        self.extractor = JobExtractor(ontology_path=ontology_path, use_ner=use_ner)
        self.miner = DomainARMMiner(min_support=0.05, min_confidence=0.3, min_transactions=8)
        self.reasoner = LLMReasoner(client=llm_client)
        self.ensemble = SkillEnsemble()
        self.top_k = top_k
        self._fitted = False  # ARM 학습 완료 여부

    def fit(self, corpus_data_path=None) -> "ReverseJobEngineeringPipeline":
        """
        [STEP 1+2] 코퍼스 로딩 및 ARM 학습.

        corpus_data_path=None → 샘플 데이터 사용
        corpus_data_path="경로" → 실제 데이터 사용  ← 팀원 데이터 받으면 여기 변경
        """
        print("[Pipeline] 코퍼스 로딩 및 특징 추출 시작...")
        corpus_results = self.extractor.extract_corpus(corpus_data_path)
        self._corpus_results = corpus_results
        extracted_list = [req for _, req in corpus_results]

        print(f"[Pipeline] {len(extracted_list)}건 추출 완료. 직군별 ARM 학습 시작...")
        self.miner.fit(corpus_results)  # DomainARMMiner: (job, req) 튜플 전달
        self._fitted = True
        return self

    def run(self, target_job: dict) -> RJEResult:
        """
        [STEP 3+4+5] 단일 채용공고에 대한 암묵적 역량 추론.

        target_job: {"title": "...", "description": "...", "company": "..."}
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 실행하세요.")

        # STEP 3: 대상 공고 명시적 추출
        req: ExtractedRequirements = self.extractor.extract(target_job)
        current_skills = set(req.tech_stack)

        # STEP 4: ARM으로 누락 기술 예측 (직군별 파티셔닝)
        arm_implicit = self.miner.predict_implicit(
            current_skills,
            job_title=target_job.get("title", ""),
            job_description=target_job.get("description", ""),
        )

        # ARM 상위 5개를 LLM 프롬프트에 후보로 전달
        arm_candidates = [s.skill for s in arm_implicit[:5]]

        # STEP 5: LLM Chain-of-Thought 추론
        llm_result = self.reasoner.reason(
            title=target_job.get("title", ""),
            description=target_job.get("description", ""),
            explicit_skills=list(current_skills),
            arm_candidates=arm_candidates,
        )

        # STEP 6: 앙상블
        implicit_skills = self.ensemble.merge(
            explicit_skills=list(current_skills),
            arm_results=arm_implicit,
            llm_result=llm_result,
            top_k=self.top_k,
        )

        return RJEResult(
            job_title=target_job.get("title", ""),
            explicit_skills=sorted(current_skills),
            experience_years=req.experience_years,
            certifications=req.certifications,
            implicit_skills=implicit_skills,
            used_mock_llm=llm_result.used_mock,
        )

    def run_batch(
        self,
        target_jobs: list[dict],
        corpus_data_path: Optional[str] = None,
    ) -> list[RJEResult]:
        """
        여러 공고를 한 번에 처리.
        fit()이 아직 안 됐으면 자동으로 fit() 수행.
        """
        if not self._fitted:
            self.fit(corpus_data_path)
        return [self.run(job) for job in target_jobs]

    def print_result(self, result: RJEResult) -> None:
        """결과 콘솔 출력 (디버깅/데모용)"""
        print(f"\n{'='*60}")
        print(f"직무: {result.job_title}")
        print(f"{'='*60}")
        print(f"[명시적 기술] {', '.join(result.explicit_skills) or '없음'}")
        if result.experience_years:
            print(f"[경력] {result.experience_years}년 이상")
        else:
            print("[경력] 명시 없음")
        print(f"[자격증] {', '.join(result.certifications) or '없음'}")

        print(f"\n[ARM 연관 규칙 매칭]")
        arm_matched = [s for s in result.implicit_skills if "ARM" in s.sources]
        if arm_matched:
            for s in arm_matched:
                print(f"  ✓ {s.skill} — ARM 규칙으로 도출 (conf={s.arm_confidence:.2f})")
        else:
            print("  현재 공고 기술과 매칭되는 ARM 규칙 없음")
            print("  (채용공고 데이터가 많아질수록 ARM 규칙이 풍부해집니다)")

        print(f"\n[암묵적 요구 역량] (LLM Mock: {result.used_mock_llm})")
        print("-" * 60)
        for i, skill in enumerate(result.implicit_skills, 1):
            sources = "+".join(skill.sources)
            print(f"{i}. [{sources}] {skill.skill} (점수: {skill.score:.3f})")
            if skill.reason:
                words = skill.reason.split()
                line = "   "
                for word in words:
                    if len(line) + len(word) + 1 > 80:
                        print(line)
                        line = "   " + word
                    else:
                        line += (" " if line.strip() else "") + word
                if line.strip():
                    print(line)
        print("-" * 60)
        arm_cnt = len([s for s in result.implicit_skills if "ARM" in s.sources and "LLM" not in s.sources])
        llm_cnt = len([s for s in result.implicit_skills if s.sources == ["LLM"]])
        both_cnt = len([s for s in result.implicit_skills if "ARM" in s.sources and "LLM" in s.sources])
        print(f"  ARM 도출: {arm_cnt}개  LLM 도출: {llm_cnt}개  합의(ARM+LLM): {both_cnt}개")


# ── Colab 엔트리포인트 ──────────────────────────────────────────
def run_demo(
    corpus_data_path: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
):
    """
    Colab에서 바로 실행 가능한 데모 함수.

    사용법:
        # 샘플 데이터 + Mock LLM
        run_demo()

        # 실제 데이터 + Mock LLM  ← 팀원 데이터 받으면 경로만 변경
        run_demo(corpus_data_path="data/실제파일.json")

        # 실제 데이터 + 실제 LLM
        run_demo(corpus_data_path="data/실제파일.json", llm_client=OpenAIAdapter(...))
    """
    pipeline = ReverseJobEngineeringPipeline(
        use_ner=False,       # GPU 없을 경우 False
        llm_client=llm_client,
    )

    # ★ 데이터 교체 시 corpus_data_path 하나만 바꾸면 됨 ★
    pipeline.fit(corpus_data_path)

    # 테스트 공고들
    test_jobs = [
        {
            "title": "백엔드 개발자",
            "description": "Python, FastAPI, PostgreSQL 경험자 우대. Docker 활용 경험. 경력 3년 이상.",
            "company": "테스트A",
        },
        {
            "title": "프론트엔드 개발자",
            "description": "React, TypeScript 필수. 경력 2년 이상.",
            "company": "테스트B",
        },
    ]

    for job in test_jobs:
        result = pipeline.run(job)
        pipeline.print_result(result)

    return pipeline


# ── 단독 실행 ───────────────────────────────────────────────────
if __name__ == "__main__":
    run_demo()