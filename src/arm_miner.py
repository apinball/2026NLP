"""
arm_miner.py — Association Rule Mining으로 암묵적 요구 역량을 도출하는 모듈

[아키텍처 요약]
  입력: [(채용공고, ExtractedRequirements), ...] — extractor.py 출력
  처리: ① 트랜잭션 행렬 구성  ② Apriori 알고리즘  ③ 누락 기술 도출
  출력: {기술명: 신뢰도 점수} — 특정 공고에서 누락된 암묵적 역량 후보

[핵심 알고리즘: Apriori + Association Rule Mining]

  Apriori란?
  - 마트 장바구니 분석에서 유래 ("기저귀 사는 사람은 맥주도 산다")
  - 채용공고에 적용하면: "Python 공고에는 보통 Docker도 있다"를 학습
  - 특정 공고에 Python은 있는데 Docker가 없으면 → Docker가 암묵적 요구사항!

  주요 지표:
  - Support(A):      전체 공고 중 A가 등장하는 비율
  - Confidence(A→B): A가 있는 공고 중 B도 있는 비율  ← 핵심 지표
  - Lift(A→B):       Confidence / Support(B)  → 1보다 크면 양의 상관관계
"""

from collections import defaultdict
from itertools import combinations
from dataclasses import dataclass, field

from extractor import ExtractedRequirements


# ── 데이터 클래스 ──────────────────────────────────────────────
@dataclass
class AssociationRule:
    antecedent: frozenset  # 선행 기술 집합 (A)
    consequent: frozenset  # 결과 기술 집합 (B)
    support: float
    confidence: float
    lift: float


@dataclass
class ImplicitSkill:
    skill: str
    confidence: float   # 해당 공고의 기존 기술들로부터 이 기술을 예측하는 신뢰도
    lift: float
    source_rules: list[str] = field(default_factory=list)  # 근거가 된 규칙


# ── Apriori 구현 ───────────────────────────────────────────────
class ARMMiner:
    """
    [동작 방식 상세]

    1단계: 트랜잭션 구성
       각 채용공고 = 하나의 트랜잭션 (장바구니)
       예) ["Python", "FastAPI", "Docker", "PostgreSQL"]

    2단계: Frequent Itemset 탐색 (Apriori)
       - support ≥ min_support인 아이템 집합만 유지
       - Apriori 원리: 비빈번 집합의 상위 집합은 항상 비빈번
         → 가지치기로 탐색 공간 대폭 축소
       - 예) {Python}이 비빈번이면 {Python, Docker}도 탐색 불필요

    3단계: 규칙 생성
       빈번 집합 {A, B}에서 A→B, B→A 두 방향 모두 생성
       confidence, lift 계산 후 임계값 필터링

    4단계: 암묵적 역량 도출
       특정 공고의 기술 집합과 규칙을 대조
       공고에 A는 있지만 B가 없고 A→B의 confidence가 높으면
       → B를 암묵적 요구사항으로 추천
    """

    def __init__(
        self,
        min_support: float = 0.2,     # 전체 공고의 최소 20%에서 등장해야 빈번 아이템
        min_confidence: float = 0.5,  # A→B 신뢰도 최소 50%
        min_lift: float = 1.2,        # 우연 이상의 상관관계
        max_itemset_size: int = 3,    # 탐색할 최대 집합 크기 (3이면 3-itemset까지)
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_itemset_size = max_itemset_size
        self.rules: list[AssociationRule] = []
        self.item_support: dict[str, float] = {}  # 개별 아이템 support 저장

    def fit(self, extracted_list: list[ExtractedRequirements]) -> "ARMMiner":
        """
        코퍼스 전체로 연관 규칙 학습.
        extracted_list: extractor.py의 extract_corpus() 결과 중 ExtractedRequirements 부분
        """
        transactions = [set(req.tech_stack) for req in extracted_list if req.tech_stack]
        n = len(transactions)
        if n == 0:
            print("[ARMMiner] 트랜잭션이 없습니다.")
            return self

        # ── 1단계: 개별 아이템 support 계산 ──────────────────
        item_counts: dict[str, int] = defaultdict(int)
        for t in transactions:
            for item in t:
                item_counts[item] += 1

        self.item_support = {item: cnt / n for item, cnt in item_counts.items()}

        # ── 2단계: Apriori — Frequent Itemsets 탐색 ──────────
        freq_itemsets: dict[frozenset, float] = {}

        # 빈번 1-itemset
        freq_1 = {
            frozenset([item]): sup
            for item, sup in self.item_support.items()
            if sup >= self.min_support
        }
        freq_itemsets.update(freq_1)

        current_freq = freq_1
        for size in range(2, self.max_itemset_size + 1):
            if not current_freq:
                break
            candidates = self._generate_candidates(current_freq, size)
            new_freq = {}
            for cand in candidates:
                cnt = sum(1 for t in transactions if cand.issubset(t))
                sup = cnt / n
                if sup >= self.min_support:
                    new_freq[cand] = sup
            freq_itemsets.update(new_freq)
            current_freq = new_freq

        # ── 3단계: 연관 규칙 생성 ─────────────────────────────
        self.rules = []
        for itemset, sup in freq_itemsets.items():
            if len(itemset) < 2:
                continue
            # itemset의 모든 비공집합 분할을 antecedent/consequent로 사용
            items = list(itemset)
            for r in range(1, len(items)):
                for ant_items in combinations(items, r):
                    ant = frozenset(ant_items)
                    con = itemset - ant
                    ant_sup = freq_itemsets.get(ant, 0)
                    if ant_sup == 0:
                        continue
                    conf = sup / ant_sup
                    # consequent 개별 support로 lift 계산
                    con_sup = min(
                        self.item_support.get(c, 1e-9) for c in con
                    )
                    lift = conf / con_sup if con_sup > 0 else 0

                    if conf >= self.min_confidence and lift >= self.min_lift:
                        self.rules.append(AssociationRule(
                            antecedent=ant,
                            consequent=con,
                            support=sup,
                            confidence=conf,
                            lift=lift,
                        ))

        print(f"[ARMMiner] 학습 완료 — 트랜잭션 {n}건, 규칙 {len(self.rules)}개 생성")
        return self

    def _generate_candidates(self, freq_prev: dict, size: int) -> list[frozenset]:
        """
        이전 단계의 빈번 집합에서 크기+1 후보 생성.
        Apriori 원리: 후보의 모든 (size-1) 부분집합이 빈번해야 후보로 인정.
        """
        prev_keys = list(freq_prev.keys())
        candidates = set()
        for i in range(len(prev_keys)):
            for j in range(i + 1, len(prev_keys)):
                union = prev_keys[i] | prev_keys[j]
                if len(union) == size:
                    # Apriori 가지치기: 모든 (size-1) 부분집합이 빈번한지 확인
                    subsets_ok = all(
                        frozenset(sub) in freq_prev
                        for sub in combinations(union, size - 1)
                    )
                    if subsets_ok:
                        candidates.add(union)
        return list(candidates)

    def predict_implicit(self, current_skills: set[str]) -> list[ImplicitSkill]:
        """
        특정 공고의 기술 집합을 받아 누락된 암묵적 역량 예측.

        [동작 원리]
        학습된 규칙 중 antecedent ⊆ current_skills 이면서
        consequent ∩ current_skills == ∅ 인 규칙을 찾음.
        → "이 공고에 A가 있고 A→B 규칙이 있는데 B가 없다 = B가 암묵적 요구사항"

        동일 기술을 여러 규칙이 가리키면 confidence 최댓값 사용.
        """
        skill_scores: dict[str, dict] = {}

        for rule in self.rules:
            # antecedent가 현재 기술의 부분집합이고
            if not rule.antecedent.issubset(current_skills):
                continue
            # consequent가 현재 기술에 없어야 암묵적
            missing = rule.consequent - current_skills
            if not missing:
                continue

            for skill in missing:
                ant_str = ", ".join(sorted(rule.antecedent))
                rule_str = f"{ant_str} → {skill} (conf={rule.confidence:.2f})"
                if skill not in skill_scores:
                    skill_scores[skill] = {
                        "confidence": rule.confidence,
                        "lift": rule.lift,
                        "rules": [rule_str],
                    }
                else:
                    # confidence 최댓값 갱신
                    if rule.confidence > skill_scores[skill]["confidence"]:
                        skill_scores[skill]["confidence"] = rule.confidence
                        skill_scores[skill]["lift"] = rule.lift
                    # 중복 규칙 추가 방지
                    if rule_str not in skill_scores[skill]["rules"]:
                        skill_scores[skill]["rules"].append(rule_str)

        results = [
            ImplicitSkill(
                skill=skill,
                confidence=v["confidence"],
                lift=v["lift"],
                source_rules=v["rules"],
            )
            for skill, v in skill_scores.items()
        ]
        # confidence 내림차순 정렬
        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def get_top_rules(self, n: int = 10) -> list[AssociationRule]:
        """신뢰도 상위 n개 규칙 반환 (분석/디버깅용)"""
        return sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:n]


# ── 단독 실행 테스트 ────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from extractor import JobExtractor

    extractor = JobExtractor(use_ner=False)
    corpus_results = extractor.extract_corpus()
    extracted_list = [req for _, req in corpus_results]

    miner = ARMMiner(min_support=0.2, min_confidence=0.4)
    miner.fit(extracted_list)

    print("\n[상위 연관 규칙]")
    for rule in miner.get_top_rules(5):
        ant = ", ".join(sorted(rule.antecedent))
        con = ", ".join(sorted(rule.consequent))
        print(f"  {ant} → {con}  (conf={rule.confidence:.2f}, lift={rule.lift:.2f})")

    print("\n[백엔드 개발자 공고 암묵적 역량 예측]")
    test_skills = {"Python", "FastAPI", "PostgreSQL"}
    implicit = miner.predict_implicit(test_skills)
    for s in implicit:
        print(f"  {s.skill}: confidence={s.confidence:.2f}, lift={s.lift:.2f}")
        for r in s.source_rules:
            print(f"    근거: {r}")


# ── 직무 카테고리 분류기 ────────────────────────────────────────
class JobCategoryClassifier:
    """
    채용공고 직무명/설명에서 직군 카테고리를 분류.
    직무별 파티셔닝 ARM에 사용.
    """
    CATEGORY_KEYWORDS = {
        "backend":    ["백엔드", "back", "서버", "spring", "django", "fastapi", "flask", "node"],
        "frontend":   ["프론트", "front", "react", "vue", "angular", "next", "typescript", "ui", "ux"],
        "ml":         ["ml", "ai", "머신러닝", "딥러닝", "데이터사이언", "pytorch", "tensorflow", "nlp"],
        "data":       ["데이터 엔지니어", "data engineer", "etl", "pipeline", "spark", "airflow"],
        "devops":     ["devops", "인프라", "sre", "kubernetes", "docker", "ci/cd", "클라우드"],
        "security":   ["보안", "security", "isms", "취약점"],
        "planning":   ["기획", "pm", "po", "서비스기획", "it기획", "플래닝"],
    }

    def classify(self, title: str, description: str = "") -> str:
        text = (title + " " + description).lower()
        scores = {cat: 0 for cat in self.CATEGORY_KEYWORDS}
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[cat] += 1
        best = max(scores, key=lambda c: scores[c])
        return best if scores[best] > 0 else "other"


# ── 직무별 파티셔닝 ARM ─────────────────────────────────────────
class DomainARMMiner:
    """
    직군별로 분리된 ARM 학습기.

    [동작 원리]
    전체 코퍼스를 직군(백엔드/프론트/ML 등)으로 분류한 뒤
    각 직군별로 별도의 ARMMiner를 학습.
    예측 시 현재 공고의 직군을 먼저 파악하고 해당 직군의 규칙만 사용.

    [장점]
    - ML 공고에는 ML 직군 규칙만 적용 → Java/Spring 같은 엉뚱한 결과 제거
    - 직군별 특화된 규칙 생성

    [데이터 부족 대응]
    직군별 트랜잭션이 min_transactions 미만이면
    전체 코퍼스 ARM(fallback)으로 대체.
    """

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.3,
        min_transactions: int = 10,  # 직군별 최소 트랜잭션 수
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_transactions = min_transactions
        self.classifier = JobCategoryClassifier()
        self.domain_miners: dict[str, ARMMiner] = {}  # 직군별 ARMMiner
        self.fallback_miner: ARMMiner = None           # 전체 코퍼스 ARMMiner
        self._fitted = False

    def fit(self, corpus_results: list[tuple]) -> "DomainARMMiner":
        """
        corpus_results: [(job_dict, ExtractedRequirements), ...]
        """
        # 직군별로 분류
        domain_data: dict[str, list] = defaultdict(list)
        for job, req in corpus_results:
            if not req.tech_stack:
                continue
            category = self.classifier.classify(
                job.get("title", ""),
                job.get("description", "")
            )
            domain_data[category].append(req)

        print(f"[DomainARMMiner] 직군별 트랜잭션 수:")
        for cat, reqs in domain_data.items():
            print(f"  {cat:<15}: {len(reqs)}건")

        # 직군별 ARMMiner 학습
        all_reqs = []
        for category, reqs in domain_data.items():
            miner = ARMMiner(
                min_support=self.min_support,
                min_confidence=self.min_confidence,
            )
            if len(reqs) >= self.min_transactions:
                miner.fit(reqs)
                self.domain_miners[category] = miner
                print(f"  [{category}] ARM 학습 완료 — 규칙 {len(miner.rules)}개")
            else:
                print(f"  [{category}] 데이터 부족({len(reqs)}건) → fallback 사용")
            all_reqs.extend(reqs)

        # 전체 코퍼스 fallback ARM
        self.fallback_miner = ARMMiner(
            min_support=self.min_support,
            min_confidence=self.min_confidence,
        )
        self.fallback_miner.fit(all_reqs)
        self._fitted = True
        return self

    def predict_implicit(
        self,
        current_skills: set[str],
        job_title: str = "",
        job_description: str = "",
    ) -> list:
        """
        직군을 파악하여 해당 직군의 ARM 규칙으로 예측.
        직군 규칙이 없으면 fallback 사용.
        """
        category = self.classifier.classify(job_title, job_description)
        miner = self.domain_miners.get(category, self.fallback_miner)
        source = category if category in self.domain_miners else "fallback"
        print(f"  [DomainARM] 직군: {category} → {source} 규칙 사용")
        return miner.predict_implicit(current_skills)

    def get_top_rules(self, n: int = 10, category: str = None):
        """직군별 또는 전체 상위 규칙 반환"""
        if category and category in self.domain_miners:
            return self.domain_miners[category].get_top_rules(n)
        return self.fallback_miner.get_top_rules(n)