"""LLMClient 어댑터 모음.

지원 백엔드 (LLMClient Protocol: complete(prompt) -> str):
- OllamaLLMClient  : 로컬 GPU 서비스 (Qwen2.5-14B 등)
- GroqLLMClient    : Groq API (llama-3.3-70b 무료 티어)
- GeminiLLMClient  : Google Gemini API (2.0 Flash 무료 티어)
- MockLLMClient    : 오프라인 테스트용 (실 호출 없음)
"""

from src.reverse_job.adapters.gemini_client import GeminiLLMClient
from src.reverse_job.adapters.groq_client import GroqLLMClient
from src.reverse_job.adapters.mock_client import MockLLMClient
from src.reverse_job.adapters.ollama_client import OllamaLLMClient

__all__ = [
    "OllamaLLMClient",
    "GroqLLMClient",
    "GeminiLLMClient",
    "MockLLMClient",
]
