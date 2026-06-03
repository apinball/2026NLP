# Reverse Job Engineering + Logic Auditor

채용공고의 **암묵적 요구 역량**을 추론(Module A)하고, 이력서·포트폴리오의 **정합성**을 Rule + BERT 앙상블로 검증(Module B)하는 통합 NLP 파이프라인.

## 핵심 결과

**Module A — Reverse Job Engineering** (627 → 2,423 IT 공고)
- ARM Recall (proxy): **0.74**
- LLM CoT (Ollama Qwen2.5-14B / Groq / Gemini 어댑터)
- `{python, aws, docker}` → `[java, kubernetes, spring, …]` 추론 동작

**Module B — Logic Auditor** (1,000 합성 / 150 test 문서)

| 방식 | Precision | Recall | F1 |
|---|---|---|---|
| Rule-based | 1.000 | 0.187 | 0.315 |
| BERT (KLUE-RoBERTa) | 0.986 | 0.960 | 0.973 |
| **Ensemble (Rule∨BERT)** | **0.987** | **0.987** | **0.987** |

유형별 recall — Rule 과 BERT 의 **보완성** 확인:

| 유형 | Rule | BERT | Ensemble |
|---|---|---|---|
| release_year | 100% | 84.6% | 100% |
| version_year | 7.7% | 92.3% | 92.3% |
| experience_scale | 0% | 100% | 100% |
| role_scope | 0% | 100% | 100% |
| indirect_time | 0% | 100% | 100% |
| tech_combo | 0% | 100% | 100% |

## 구조

```
├── Dockerfile / docker-compose.yml      # 실행 환경 (app / ollama / demo)
├── data/
│   ├── ontology_seed.json               # 128 엔트리 (한국 특화 98)
│   ├── job_category_map.json            # 17 카테고리 매핑
│   ├── wanted_jobs.jsonl                # 2,396 (gitignore)
│   ├── synthetic_resumes.jsonl          # 1,000 (gitignore)
│   ├── bert_classifier/                 # 학습된 모델 (gitignore)
│   └── eval_module_b_all.json           # 평가 결과 (gitignore)
├── src/
│   ├── reverse_job/                     # Module A
│   │   ├── extractor.py                 # KoELECTRA NER + 키워드
│   │   ├── arm_miner.py                 # Apriori
│   │   ├── llm_reasoner.py              # CoT (4단계, JSON 강제)
│   │   └── adapters/                    # LLM 백엔드 4종
│   │       ├── ollama_client.py         # 로컬 GPU
│   │       ├── groq_client.py           # Llama 3.3 70B 무료
│   │       ├── gemini_client.py         # Gemini 2.0 Flash 무료
│   │       └── mock_client.py           # 오프라인 테스트
│   ├── logic_auditor/                   # Module B
│   │   ├── ontology.py                  # TechEntry 인덱스
│   │   ├── rule_based.py                # 출시연도·버전·한국어 word-boundary
│   │   ├── ml_detector.py               # perplexity + burstiness 휴리스틱
│   │   ├── bert_detector.py             # KLUE-RoBERTa 정합성 분류기
│   │   └── pipeline.py
│   ├── data/                            # 로더·인덱스·카테고리 정규화
│   ├── integration/                     # A → B end-to-end
│   └── demo/app.py                      # Streamlit UI
├── scripts/                             # 크롤러·생성·학습·평가·시각화
└── tests/                               # pytest 56건
```

## 빠른 시작

### 1. 환경 준비

```bash
cp .env.example .env
# 필요한 키만 채우기:
#   HF_TOKEN=hf_xxxxx          (HF 모델 다운로드 안정성)
#   GROQ_API_KEY=...           (Groq LLM 사용 시)
#   GEMINI_API_KEY=...         (Gemini LLM 사용 시)
# Ollama 로컬 사용 시 키 불필요

docker compose build
```

### 2. 데모 UI

```bash
docker compose up -d ollama          # LLM 추론 사용 시
docker compose up demo
# → http://localhost:8501
```

사이드바에서 토글:
- **LLM 추론** (Ollama / 미사용)
- **BERT 정합성 검출** (학습된 모델 있을 때만)
- BERT 임계값 슬라이더
- 샘플 데이터 한 번 클릭 로드

### 3. 테스트

```bash
docker compose run --rm app bash -c "cd /app && PYTHONPATH=. python -m pytest tests -q"
# 56 passed
```

## 풀 파이프라인 (재현)

```bash
# A) Ollama 모델 받기 (9GB, 1회)
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5:14b-instruct-q4_K_M

# B) 채용공고 수집 (~30분)
docker compose run --rm app bash -c \
  "cd /app && PYTHONPATH=. python scripts/crawl_wanted.py --pages 40 --sleep 1.0 --resume"

# C) 합성 이력서 1,000건 생성 (~40분, 6 위반 유형 균등)
docker compose run --rm app bash -c \
  "cd /app && PYTHONPATH=. python scripts/gen_synthetic_resumes.py --normal 500 --error 500"

# D) BERT 일관성 분류기 학습 (~1분, GPU 필요)
docker compose run --rm app bash -c \
  "cd /app && PYTHONPATH=. python scripts/train_bert_classifier.py --input data/synthetic_resumes.jsonl --epochs 3"

# E) Module A 평가 (ARM vs LLM vs Ensemble)
docker compose run --rm app bash -c \
  "cd /app && PYTHONPATH=. python scripts/eval_module_a.py --sample 15"

# F) Module B 평가 (Rule vs BERT vs Ensemble, 유형별)
docker compose run --rm app bash -c \
  "cd /app && PYTHONPATH=. python scripts/eval_module_b_all.py"

# G) 발표용 그림 생성
docker compose run --rm app bash -c \
  "cd /app && PYTHONPATH=. python scripts/plot_eval_results.py"
# → data/figures/{per_type_recall,overall_f1,confusion_matrices}.png

# H) 온톨로지 스키마 검증
docker compose run --rm app bash -c \
  "cd /app && PYTHONPATH=. python scripts/validate_ontology.py"
```

## 모듈별 사용

### Module A — Reverse Job Engineering

```python
from src.reverse_job.pipeline import ReverseJobPipeline
from src.reverse_job.llm_reasoner import LLMReasoner
from src.reverse_job.adapters import OllamaLLMClient  # 또는 GroqLLMClient, GeminiLLMClient

reasoner = LLMReasoner(client=OllamaLLMClient(model="qwen2.5:14b-instruct-q4_K_M"))
pipeline = ReverseJobPipeline(reasoner=reasoner, load_ner=False)

result = pipeline.run(
    target_job="[백엔드] Python, Django, MySQL 3년 경력...",
    transactions=corpus_transactions,  # IT 직군 코퍼스의 스킬 리스트들
)
print(result.implicit_union)  # ARM ∪ LLM 합집합
```

### Module B — Logic Auditor

```python
from src.logic_auditor.pipeline import LogicAuditorPipeline
from src.logic_auditor.bert_detector import BertConsistencyDetector

# Rule-based
pipeline = LogicAuditorPipeline()
report = pipeline.run("저는 2010년부터 Docker를 활용...")
print(report.trust_score, report.violations)

# Rule + BERT 앙상블
bert = BertConsistencyDetector(model_dir="data/bert_classifier")
bert_result = bert.predict_sentences(text)
ensemble_violation = bool(report.violations) or bert_result.is_violation
```

## 데이터

대용량 raw 파일은 `.gitignore` 처리. 별도 공유 채널(드라이브 등)로 받아 `data/` 에 배치:

| 파일 | 건수 | 출처 |
|---|---|---|
| `wanted_jobs.jsonl` | 2,396 | 원티드 비공식 API |
| `wanted_raw.jsonl` | 2,396 | (원본) |
| `linkareer_cover_letter_parsed.jsonl` | 7,300 | linkareer 합격 자소서 |
| `naver_cafe_passassay_parsed.jsonl` | 3,250 | 네이버카페 합격 자소서 |
| `ceragem_job_posting.json` | 27 positions | 잡코리아 단일 공고 |
| `synthetic_resumes.jsonl` | 1,000 | LLM 생성 (6 위반 유형 균등) |

깃 포함 (시드 자산):
- `data/ontology_seed.json` — **128 엔트리** (글로벌 OSS 30 + 한국 특화 98: service-kr / certification-kr / stack-kr / fintech-kr / recruit-kr / education-kr / game-kr / community-kr 등)
- `data/job_category_map.json` — 17개 표준 카테고리

## 팀 분담

| 역할 | 담당 |
|---|---|
| Module A 모델 (NER, ARM, LLM CoT 프롬프팅) | 팀원 A |
| Module B 모델 (BERT 일관성 분류기, AI 생성 휴리스틱) | 팀원 B |
| 데이터·통합·UI·인프라·평가 | 풀스택 |

`feature/module-a` (팀원 A) 작업은 `feat/baseline-scaffold` 로 흡수 완료 — Groq/Gemini 어댑터, 4단계 CoT 프롬프트, `LLMImplicitSkill(skill, reason, confidence)` 구조화 출력 모두 채택.

## 의존성 / 환경

```
Python      3.11
PyTorch     2.x + CUDA
Transformers 4.46+
KLUE-RoBERTa-base  (110M, 자동 다운로드)
Ollama      qwen2.5:14b-instruct-q4_K_M  (9GB Q4)
Docker      NVIDIA Container Toolkit 필요 (GPU passthrough)
```

GPU 메모리 16GB 권장 (RTX 5070 Ti / 4080 등). 7B Q4 모델로 다운그레이드 시 8GB 도 가능.

## 라이선스

LICENSE 파일 참고.
