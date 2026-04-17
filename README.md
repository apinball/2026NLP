# Reverse Job Engineering + Logic Auditor

채용공고의 **암묵적 요구 역량**을 추론하고, 포트폴리오·이력서의 **논리적 모순**을 탐지하는 통합 NLP 파이프라인.

## 구조

```
├── Dockerfile / docker-compose.yml    # 실행 환경
├── requirements.txt / .env.example    # 의존성 & 설정
├── data/ontology_seed.json            # 기술 온톨로지 시드 (30개)
├── src/
│   ├── config.py                      # 환경변수 로딩
│   ├── main.py                        # 데모 엔트리
│   ├── reverse_job/                   # Module A
│   │   ├── extractor.py               # 키워드 사전 + KoELECTRA NER
│   │   ├── arm_miner.py               # Apriori 공동출현 분석
│   │   ├── llm_reasoner.py            # Anthropic Chain-of-Thought
│   │   └── pipeline.py
│   └── logic_auditor/                 # Module B
│       ├── ontology.py                # TechEntry 인덱스
│       ├── rule_based.py              # 연도/버전 팩트 검증
│       ├── ml_detector.py             # Perplexity + Burstiness
│       └── pipeline.py                # 신뢰도 점수 산출
└── tests/                             # pytest 스모크 테스트
```

## 빠른 시작

```bash
cp .env.example .env            # ANTHROPIC_API_KEY 채우기 (선택)
docker compose build
docker compose run --rm app                                    # 전체 데모
docker compose run --rm app python -m src.main --module rje    # Module A만
docker compose run --rm app python -m src.main --module audit  # Module B만
docker compose run --rm app python -m src.main --use-llm       # LLM 추론 활성화
docker compose run --rm app pytest -q                          # 테스트 실행
```

`.env` 의 `ANTHROPIC_API_KEY` 를 비워두면 LLM 추론은 건너뛰고 ARM + Rule 기반 모듈만 동작한다.

## 설정 항목 (`.env`)

| 변수 | 기본값 | 용도 |
|------|--------|------|
| `ANTHROPIC_API_KEY` | — | Module A LLM 추론 |
| `LLM_MODEL` | `claude-sonnet-4-6` | Anthropic 모델 |
| `HF_NER_MODEL_KO` | `monologg/koelectra-base-v3-finetuned-naver-ner` | 한국어 NER |
| `HF_PPL_MODEL` | `skt/kogpt2-base-v2` | Perplexity 계산 |

## 파이프라인 요약

**Module A — Reverse Job Engineering**
1. `JobExtractor` 가 채용공고 → 명시적 tech stack + 경력 추출
2. `ARMMiner` 가 동일 직군 코퍼스로 Apriori 규칙 학습 → 누락된 공통 역량 도출
3. `LLMReasoner` 가 Chain-of-Thought 로 실무 맥락 추론 → 앙상블

**Module B — Logic Auditor**
1. `RuleBasedAuditor` 가 온톨로지 DB 대조로 출시연도/버전 모순 탐지
2. `AIGenerationDetector` 가 perplexity·burstiness 로 AI 생성 신호 측정
3. `LogicAuditorPipeline` 이 위반 수 × 페널티 + AI 확률로 **신뢰도 점수(0–100)** 산출
