NLP_Final/
├── main.py                 # 메인 실행 파일
├── requirements.txt        # 패키지 목록
├── .env                    # API 키 (GitHub 업로드 X)
├── .env.example            # API 키 형식 안내
├── .gitignore
├── src/
│   ├── pipeline.py         # 통합 파이프라인
│   ├── arm_miner.py        # 연관 규칙 마이닝
│   ├── llm_reasoner.py     # LLM 추론
│   └── extractor.py        # 채용공고 특징 추출
└── data/
    ├── ontology_seed.json
    ├── job_category_map.json
    ├── ceragem_job_posting.json
    └── wanted_jobs.jsonl