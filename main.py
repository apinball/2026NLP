# ============================================================
#  Module A — Reverse Job Engineering
#  main.py  (로컬 실행용: VS Code / PyCharm)
# ============================================================

import sys
import os
from dotenv import load_dotenv

# ── 경로 설정 ────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 이 파일 기준 프로젝트 루트

sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

ONTOLOGY_PATH    = os.path.join(ROOT_DIR, "data", "ontology_seed.json")
CATEGORY_PATH    = os.path.join(ROOT_DIR, "data", "job_category_map.json")
CORPUS_DATA_PATH = [
    os.path.join(ROOT_DIR, "data", "ceragem_job_posting.json"),
    os.path.join(ROOT_DIR, "data", "wanted_jobs.jsonl"),
]

print("경로 설정 완료")
print(f"  ontology_seed.json 존재: {os.path.exists(ONTOLOGY_PATH)}")
print(f"  job_category_map.json 존재: {os.path.exists(CATEGORY_PATH)}")
for p in CORPUS_DATA_PATH:
    print(f"  {os.path.basename(p)} 존재: {os.path.exists(p)}")

# ── API 키 로드 ──────────────────────────────────────────────
load_dotenv(os.path.join(ROOT_DIR, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(".env 파일에 GROQ_API_KEY가 없습니다. .env.example을 참고하세요.")

# ── Groq API 연동 및 파이프라인 초기화 ──────────────────────
from pipeline import ReverseJobEngineeringPipeline
from llm_reasoner import GroqAdapter

groq_client = GroqAdapter(api_key=GROQ_API_KEY)

pipeline = ReverseJobEngineeringPipeline(
    ontology_path=ONTOLOGY_PATH,
    use_ner=False,
    llm_client=groq_client,
    min_support=0.05,
    min_confidence=0.3,
    top_k=10,
)
print("파이프라인 초기화 완료")

# ── 코퍼스 학습 (ARM) ────────────────────────────────────────
pipeline.fit(corpus_data_path=CORPUS_DATA_PATH)

# ── 샘플 공고 분석 (ARM + LLM 앙상블) ───────────────────────
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
    {
        "title": "ML 엔지니어",
        "description": "PyTorch 기반 모델 학습 및 서빙. AWS 인프라 활용. 경력 3년 이상.",
        "company": "테스트C",
    },
]

for job in test_jobs:
    result = pipeline.run(job)
    pipeline.print_result(result)

# ── 사용자 입력 기반 공고 분석 (ARM + LLM 앙상블) ───────────
print("=" * 60)
print("채용공고 암묵적 역량 분석기 (ARM + LLM 앙상블)")
print("=" * 60)

title = input("직무명을 입력하세요: ")
company = input("회사명을 입력하세요 (없으면 엔터): ")
print("채용공고 내용을 입력하세요 (입력 완료 후 빈 줄에서 엔터):")

lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)
description = "\n".join(lines)

my_job = {
    "title": title,
    "description": description,
    "company": company or "미입력",
}

print("\n분석 중...")
result = pipeline.run(my_job)
pipeline.print_result(result)