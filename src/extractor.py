"""
extractor.py — 채용공고에서 명시적 요구사항을 추출하는 모듈
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedRequirements:
    tech_stack:       list = field(default_factory=list)
    experience_years: Optional[int] = None
    certifications:   list = field(default_factory=list)
    roles:            list = field(default_factory=list)
    raw_text:         str = ""


class KeywordMatcher:
    def __init__(self, ontology_path="data/ontology_seed.json"):
        with open(ontology_path, encoding="utf-8") as f:
            ontology = json.load(f)
        self.patterns = {}
        for entry in ontology:
            names = [entry["name"]] + entry.get("aliases", [])
            for n in names:
                escaped = re.escape(n)
                if re.search(r"[가-힣]", n):
                    pat = re.compile(f"(?<![가-힣]){escaped}(?![가-힣])", re.IGNORECASE)
                else:
                    pat = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                self.patterns[entry["name"]] = pat

    def match(self, text):
        found = []
        for name, pat in self.patterns.items():
            if pat.search(text):
                found.append(name)
        return found


class NERExtractor:
    def __init__(self, model_name="monologg/koelectra-base-v3-finetuned-naver-ner"):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("ner", model=self.model_name, aggregation_strategy="simple")
                print(f"[NERExtractor] 모델 로드 완료: {self.model_name}")
            except Exception as e:
                print(f"[NERExtractor] 모델 로드 실패: {e}")
                self._pipeline = None

    def extract(self, text):
        self._load()
        if self._pipeline is None:
            return []
        try:
            entities = self._pipeline(text[:512])
            return [e["word"].strip() for e in entities if e.get("entity_group") in {"OG", "TI"} and len(e["word"].strip()) > 1]
        except Exception as e:
            print(f"[NERExtractor] 추출 오류: {e}")
            return []


class MetaExtractor:
    EXPERIENCE_PATTERNS = [
        r"경력\s*(\d+)년\s*이상",
        r"(\d+)년\s*이상\s*경력",
        r"(\d+)\+?\s*years?\s*of\s*experience",
        r"최소\s*(\d+)년",
    ]
    CERTIFICATION_KEYWORDS = [
        "정보처리기사", "정보처리산업기사", "SQLD", "SQLP",
        "ADsP", "ADP", "빅데이터분석기사", "정보보안기사",
        "AICE", "TOPCIT", "컴퓨터활용능력", "컴활", "AWS", "GCP", "Azure",
    ]
    ROLE_PATTERNS = [
        r"(프론트엔드|백엔드|풀스택|데이터\s*엔지니어|ML\s*엔지니어|DevOps|SRE|QA|보안|클라우드)\s*(개발자|엔지니어)?",
        r"(시니어|주니어|리드|CTO|테크리드)\s*(개발자|엔지니어)?",
    ]

    def extract_experience(self, text):
        for pat in self.EXPERIENCE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    def extract_certifications(self, text):
        return [c for c in self.CERTIFICATION_KEYWORDS if c.lower() in text.lower()]

    def extract_roles(self, text):
        roles = []
        for pat in self.ROLE_PATTERNS:
            roles += re.findall(pat, text)
        return [" ".join(r).strip() if isinstance(r, tuple) else r for r in roles]


class JobExtractor:
    def __init__(self, ontology_path="data/ontology_seed.json", use_ner=True):
        self.keyword_matcher = KeywordMatcher(ontology_path)
        self.ner_extractor = NERExtractor() if use_ner else None
        self.meta_extractor = MetaExtractor()

    def load_corpus(self, data_path=None):
        if data_path:
            return self._load_real_data(data_path)
        return self._load_sample_data()

    def _load_real_data(self, path):
        """
        두 가지 포맷 자동 감지 및 여러 파일 리스트 지원.

        포맷 A: ceragem_job_posting.json
        포맷 B: wanted_jobs.jsonl
        """
        IT_KEYWORDS = [
            "개발", "엔지니어", "데이터", "ML", "AI", "DevOps", "devops",
            "백엔드", "프론트", "풀스택", "클라우드", "인프라", "SW", "QA",
            "보안", "architect", "아키텍트", "engineer", "developer"
        ]
        IT_CATEGORIES = {"IT", "정보보안", "기술연구소", "디자인"}

        # 경로가 리스트면 여러 파일 합치기
        paths = path if isinstance(path, list) else [path]

        result = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                content = f.read().strip()

            if p.endswith(".jsonl"):
                # 포맷 B: wanted_jobs.jsonl
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        job = json.loads(line)
                    except Exception:
                        continue
                    title = job.get("job_title", "")
                    if not any(kw.lower() in title.lower() for kw in IT_KEYWORDS):
                        continue
                    description = " ".join([
                        str(x) for x in
                        job.get("requirements", []) +
                        job.get("preferred", []) +
                        job.get("responsibilities", []) +
                        job.get("skills", [])
                        if x
                    ])
                    result.append({
                        "title":       title,
                        "description": description,
                        "company":     job.get("company", ""),
                        "category":    job.get("category", "IT"),
                    })
            else:
                # 포맷 A: ceragem_job_posting.json
                raw = json.loads(content)
                files = raw if isinstance(raw, list) else [raw]
                for file_data in files:
                    if isinstance(file_data.get("company"), dict):
                        company_name = file_data["company"].get("name", "")
                    else:
                        company_name = file_data.get("company", "")
                    for pos in file_data.get("positions", []):
                        category = pos.get("category", "")
                        if not any(it_cat in category for it_cat in IT_CATEGORIES):
                            continue
                        description = " ".join([
                            str(x) for x in
                            pos.get("requirements", []) +
                            pos.get("preferred", []) +
                            pos.get("responsibilities", [])
                            if x
                        ])
                        result.append({
                            "title":       pos.get("job_title", ""),
                            "description": description,
                            "company":     company_name,
                            "category":    category,
                        })

        print(f"[JobExtractor] 로딩 완료 — IT 직군 {len(result)}건 추출")
        return result

    def _load_sample_data(self):
        return [
            {"title": "백엔드 개발자", "description": "Python, FastAPI, PostgreSQL 경험자 우대. Docker, Kubernetes 활용 경험. 경력 3년 이상.", "company": "샘플A"},
            {"title": "프론트엔드 개발자", "description": "React, TypeScript 필수. Next.js 경험 우대. GraphQL, REST API 연동 경험.", "company": "샘플B"},
            {"title": "ML 엔지니어", "description": "PyTorch, TensorFlow 경험. Transformers 라이브러리 활용. AWS SageMaker 경험 우대. 경력 2년 이상.", "company": "샘플C"},
            {"title": "DevOps 엔지니어", "description": "Kubernetes, Docker 필수. AWS 인프라 구축 경험. Kafka, Redis 활용 경험. CI/CD 파이프라인 구성 경험.", "company": "샘플D"},
            {"title": "데이터 엔지니어", "description": "Python, Spark 경험. PostgreSQL, MongoDB 필수. Kafka 스트리밍 경험 우대. 경력 3년 이상. SQLD 자격증 보유자 우대.", "company": "샘플E"},
        ]

    def extract(self, job):
        text = f"{job.get('title', '')} {job.get('description', '')}"
        tech_kw = set(self.keyword_matcher.match(text))
        tech_ner = set(self.ner_extractor.extract(text)) if self.ner_extractor else set()
        tech_stack = sorted(tech_kw | tech_ner)
        return ExtractedRequirements(
            tech_stack=tech_stack,
            experience_years=self.meta_extractor.extract_experience(text),
            certifications=self.meta_extractor.extract_certifications(text),
            roles=self.meta_extractor.extract_roles(text),
            raw_text=text,
        )

    def extract_corpus(self, data_path=None):
        corpus = self.load_corpus(data_path)
        return [(job, self.extract(job)) for job in corpus]


if __name__ == "__main__":
    extractor = JobExtractor(use_ner=False)
    results = extractor.extract_corpus()
    for job, req in results:
        print(f"\n[{job['title']}]")
        print(f"  기술스택: {req.tech_stack}")
        print(f"  경력: {req.experience_years}년")
        