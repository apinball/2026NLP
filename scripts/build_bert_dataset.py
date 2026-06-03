"""BERT 일관성 분류기 학습용 통합 데이터셋 빌더.

합성 위반 데이터(LLM 생성, 6 유형) + 실제 합격 자소서(linkareer / naver) 의
*정상* 샘플을 결합하여 BERT 가 학습 분포 외 입력(실 자소서·CV)에서도 정상을
정확히 판단할 수 있게 한다.

이전 합성-only 학습은 Qwen 의 특정 자기소개서 문체에 과적합되어, 실제 CV 의
섹션 헤더("EDUCATION"), 단편 정보("이메일: ...", "GPA: 4.08"), 학술 인용
스타일 등을 모두 위반으로 잘못 판단하는 문제가 있었음.

출력 스키마는 기존 `synthetic_resumes.jsonl` 과 호환:
  {text, kind, error_type, params?, violations, ai_generated, source}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

from src.data.loaders import Resume, iter_resumes


def load_synthetic(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _extract_normal_text(resume: Resume, rng: random.Random) -> str | None:
    """이 자소서에서 학습에 적합한 단락 한 개를 뽑아 반환."""
    if not resume.sections:
        return None
    rng.shuffle(resume.sections)
    for sec in resume.sections:
        text = (sec.answer or "").strip()
        # 길이 필터 (BERT max_length 256 토큰 ≈ 한국어 700자 안팎)
        if 120 <= len(text) <= 1200:
            return text
    return None


def gather_real_normal(n: int, seed: int) -> list[dict]:
    """실제 합격 자소서에서 정상 샘플 n 건 수집."""
    rng = random.Random(seed)
    out: list[dict] = []
    seen_texts: set[str] = set()

    iterator = iter_resumes()
    with tqdm(total=n, desc="real-normal") as pbar:
        for resume in iterator:
            text = _extract_normal_text(resume, rng)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            out.append(
                {
                    "text": text,
                    "kind": "normal",
                    "error_type": None,
                    "params": {
                        "source": resume.source,
                        "company": resume.company,
                        "job": resume.job,
                    },
                    "violations": [],
                    "ai_generated": False,
                    "source": resume.source,
                }
            )
            pbar.update(1)
            if len(out) >= n:
                break

    return out


def make_cv_style_normal(rng: random.Random) -> list[dict]:
    """실제 CV 형식(섹션 헤더 / bullet / 약식 정보) 정상 샘플 수작업 예시.

    이 카테고리는 합성 자소서·linkareer 자소서 모두에 없어 학습에서 OOD 가 됨.
    소수 안전 샘플로 보강.
    """
    cv_samples = [
        "학력 (EDUCATION)\n한성대학교 (서울, 대한민국) | 2021년 – 현재\nAI응용학과 학사 (사이버보안 트랙), 2027년 2월 졸업 예정\n학점 (GPA): 4.08 / 4.5",
        "이메일: example@gmail.com | 전화번호: +82-10-1234-5678 | GitHub: github.com/example",
        "연구 관심 분야 (RESEARCH INTERESTS)\n- 표현 학습 및 생성 모델 (Representation learning and generative models)\n- 자가지도 사전학습 (Self-supervised pre-training)\n- 3D 비전 및 3D 복원 (3D vision and 3D reconstruction)",
        "기술 스택 (TECHNICAL SKILLS)\n사용 언어: Python, C++, C\n프레임워크 / 툴: PyTorch, NumPy, OpenCV, Open3D, Git\n전문 분야: 자가지도 학습, 생성 모델(GAN, MAE), 시맨틱 세그멘테이션",
        "논문 및 출판 실적 (PUBLICATIONS)\n홍길동, 김철수. \"라만 분광 데이터 노이즈 제거를 위한 자가지도 하이브리드 MAE-CNN.\" 한국정보처리학회(KIPS) 논문지, 제14권, 제10호, 2025년 10월.",
        "경력 (WORK EXPERIENCE)\n네이버 클라우드 플랫폼 | 백엔드 엔지니어 인턴 | 2024.06 - 2024.08\n- Spring Boot 기반 API 설계 및 구현\n- AWS RDS 마이그레이션 보조\n- 코드리뷰 및 단위테스트 작성",
        "프로젝트 경험\n해커톤 1위 수상작 - AI 기반 교육 콘텐츠 추천 시스템\n사용 기술: Python, FastAPI, PostgreSQL, Docker\nGitHub: github.com/example/proj",
        "수상 및 활동 (AWARDS)\n- 2024 SW 마에스트로 인증\n- 2023 정보보호 대회 장려상\n- 2022 한성대 캡스톤디자인 우수상",
        "어학 능력\n영어: TOEIC 920 (2024.03)\n일본어: JLPT N2 (2023.07)\n자격증: 정보처리기사, SQLD, ADsP",
        "관심 분야 및 자기개발\n오픈소스 컨트리뷰션, 알고리즘 문제풀이(백준 골드 1), 기술 블로그 운영, AI 논문 리뷰",
    ]
    out: list[dict] = []
    for text in cv_samples:
        out.append(
            {
                "text": text,
                "kind": "normal",
                "error_type": None,
                "params": {"source": "cv_format"},
                "violations": [],
                "ai_generated": False,
                "source": "cv_format",
            }
        )
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--synthetic",
        type=Path,
        default=Path("data/synthetic_resumes.jsonl"),
        help="기존 합성 데이터 (정상 + 6유형 오류)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("data/bert_training.jsonl")
    )
    ap.add_argument(
        "--real-normal", type=int, default=1500,
        help="실제 자소서에서 추가로 가져올 정상 샘플 수",
    )
    ap.add_argument(
        "--add-cv-style", action="store_true", default=True,
        help="CV 형식(섹션 헤더·약식 정보) 수작업 샘플 추가",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.synthetic.exists():
        print(f"missing synthetic: {args.synthetic}", file=sys.stderr)
        return 1

    print(f"[1/3] 합성 데이터 로드 {args.synthetic}…")
    synthetic = load_synthetic(args.synthetic)
    n_syn_normal = sum(1 for r in synthetic if r.get("kind") == "normal")
    n_syn_error = sum(1 for r in synthetic if r.get("kind") == "error")
    print(f"      합성: normal {n_syn_normal} / error {n_syn_error}")
    # source 필드가 없으면 'synthetic' 으로 표시
    for r in synthetic:
        r.setdefault("source", "synthetic")

    print(f"[2/3] 실제 자소서 정상 샘플 {args.real_normal}건 수집…")
    real_normal = gather_real_normal(args.real_normal, seed=args.seed)
    print(f"      수집됨: {len(real_normal)}")

    cv_normal: list[dict] = []
    if args.add_cv_style:
        cv_normal = make_cv_style_normal(random.Random(args.seed))
        print(f"      CV 형식 샘플: {len(cv_normal)}")

    print("[3/3] 통합 + 셔플 + 저장…")
    rng = random.Random(args.seed)
    combined = synthetic + real_normal + cv_normal
    rng.shuffle(combined)

    # 통계
    from collections import Counter

    kinds = Counter(r.get("kind") for r in combined)
    sources = Counter(r.get("source") for r in combined)
    print(f"      총 {len(combined)} 건")
    print(f"      kind: {dict(kinds)}")
    print(f"      source: {dict(sources)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in combined:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nwrote → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
