from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ONTOLOGY_PATH = DATA_DIR / "ontology_seed.json"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

HF_NER_MODEL_KO = os.getenv(
    "HF_NER_MODEL_KO", "monologg/koelectra-base-v3-finetuned-naver-ner"
)
HF_NER_MODEL_EN = os.getenv("HF_NER_MODEL_EN", "dslim/bert-base-NER")
HF_PPL_MODEL = os.getenv("HF_PPL_MODEL", "skt/kogpt2-base-v2")
