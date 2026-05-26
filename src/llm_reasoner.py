"""
llm_reasoner.py — LLM Chain-of-Thought으로 실무 맥락 기반 암묵적 역량 추론

[아키텍처 요약]
  입력: 채용공고 텍스트 + ARM이 도출한 암묵적 역량 후보
  처리: Chain-of-Thought 프롬프팅 → LLM 추론 → 구조화된 JSON 파싱
  출력: LLMImplicitResult (추론된 역량 목록 + 각 근거)

[LLM 연동 설계]
  LLMClient Protocol로 추상화 → LLM 미확정 상태에서도 코드 완성 가능
  확정 후 MyLLMAdapter 하나만 구현하면 나머지 코드 변경 없음

  지원 예정 LLM:
  - OpenAI GPT-4o
  - Anthropic Claude
  - 로컬 HuggingFace 모델 (Ollama 등)

[Chain-of-Thought 전략]
  "직접 답 달라"가 아니라 단계별 사고를 유도:
  1) 공고의 직무 파악
  2) 해당 직무의 실제 업무 추론
  3) 업무 수행에 필요한 기술 도출
  4) 명시되지 않은 기술 중 필수인 것 선별
  → 모델이 근거를 먼저 생성하므로 답변 품질 향상
"""

import json
import re
from dataclasses import dataclass, field
from typing import Protocol, Optional, runtime_checkable


# ── Protocol: LLM 추상 인터페이스 ──────────────────────────────
@runtime_checkable
class LLMClient(Protocol):
    """
    LLM 어댑터가 구현해야 하는 인터페이스.
    LLM이 확정되면 이 Protocol을 구현하는 클래스 하나만 작성하면 됨.

    사용 예:
        class OpenAIAdapter:
            def complete(self, prompt: str) -> str:
                response = openai.chat.completions.create(...)
                return response.choices[0].message.content

        reasoner = LLMReasoner(client=OpenAIAdapter())
    """
    def complete(self, prompt: str) -> str:
        ...


# ── 어댑터 구현 예시 (LLM 확정 후 주석 해제) ───────────────────
# class OpenAIAdapter:
#     def __init__(self, api_key: str, model: str = "gpt-4o"):
#         import openai
#         self.client = openai.OpenAI(api_key=api_key)
#         self.model = model
#
#     def complete(self, prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#         )
#         return response.choices[0].message.content

# class AnthropicAdapter:
#     def __init__(self, api_key: str, model: str = "claude-opus-4-6"):
#         import anthropic
#         self.client = anthropic.Anthropic(api_key=api_key)
#         self.model = model
#
#     def complete(self, prompt: str) -> str:
#         message = self.client.messages.create(
#             model=self.model,
#             max_tokens=1024,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         return message.content[0].text


class GeminiAdapter:
    """
    Google Gemini API 어댑터 (무료 티어 사용 가능)

    [무료 한도]
    - gemini-2.0-flash: 분당 15회, 일 1500회
    - 채용공고 분석 용도로는 충분

    [설치]
    !pip install google-generativeai

    [API 키 발급]
    https://aistudio.google.com → Get API Key → Create API Key

    [사용법]
    from llm_reasoner import GeminiAdapter
    client = GeminiAdapter(api_key="your-api-key")
    pipeline = ReverseJobEngineeringPipeline(llm_client=client, ...)
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
            }
        )
        print(f"[GeminiAdapter] 모델 로드 완료: {model}")

    def complete(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class GroqAdapter:
    """
    Groq API 어댑터 (완전 무료)

    [무료 한도]
    - llama-3.3-70b-versatile: 분당 30회, 일 14,400회
    - 한도 초과 시 돈 청구 없이 잠깐 대기 후 자동 해제

    [설치]
    !pip install groq

    [API 키 발급]
    https://console.groq.com → API Keys → Create API Key

    [사용법]
    from llm_reasoner import GroqAdapter
    client = GroqAdapter(api_key="your-api-key")
    pipeline = ReverseJobEngineeringPipeline(llm_client=client, ...)
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"[GroqAdapter] 모델 로드 완료: {model}")

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ── 출력 데이터 클래스 ──────────────────────────────────────────
@dataclass
class LLMImplicitSkill:
    skill: str
    reason: str          # LLM이 생성한 추론 근거
    confidence: float    # LLM 자체 추정 신뢰도 (0~1)


@dataclass
class LLMImplicitResult:
    implicit_skills: list[LLMImplicitSkill] = field(default_factory=list)
    raw_response: str = ""    # 디버깅용 원본 응답
    used_mock: bool = False   # Mock 사용 여부 (LLM 미연동 시 True)


# ── 프롬프트 빌더 ───────────────────────────────────────────────
class PromptBuilder:
    """
    Chain-of-Thought 프롬프트 생성.

    [CoT 구조]
    STEP 1: 직무 분석  → 어떤 일을 하는 포지션인가?
    STEP 2: 실무 추론  → 실제로 어떤 업무를 수행하는가?
    STEP 3: 역량 도출  → 그 업무에 필요한 기술은?
    STEP 4: 갭 분석    → 공고에 없는데 실무상 필수인 기술은?

    JSON 응답 강제로 파싱 안정성 확보.
    """

    SYSTEM_TEMPLATE = """당신은 10년 경력의 IT 채용 전문가입니다.
채용공고를 분석하여 공고에 명시되지 않았지만 실무에서 반드시 필요한 암묵적 역량을 추론합니다.

[중요 규칙]
1. implicit_skills에는 반드시 공고에 없는 기술만 포함하세요
2. 명시된 기술과 유사하거나 포함 관계인 기술도 제외하세요
3. skill 필드는 반드시 구체적인 기술/도구 이름으로 작성하세요
   좋은 예: "Git", "FastAPI", "Pandas", "Jira", "Tableau", "Linux"
   나쁜 예: "버전 관리 시스템", "협업 도구", "커뮤니케이션 스킬"
4. confidence는 실무에서 실제로 필요할 확률로 설정하세요 (0.0~1.0)
5. 최소 3개, 최대 6개의 암묵적 기술을 추론하세요
6. 소프트 스킬(커뮤니케이션, 문제해결 등)은 절대 포함하지 마세요
7. reason 필드는 반드시 공고 원문의 특정 조건을 직접 인용하여 연결하세요
   좋은 예: "공고에서 서버 연동 인터페이스 구축 경험을 요구하므로 FastAPI 역량이 필요합니다"
   나쁜 예: "FastAPI는 백엔드 개발에 필수적인 기술입니다"

반드시 아래 JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 출력하세요:
{{
  "job_analysis": "이 직무가 실제로 하는 일 (1-2문장)",
  "actual_tasks": ["실제 업무 1", "실제 업무 2", "실제 업무 3"],
  "implicit_skills": [
    {{
      "skill": "기술명",
      "reason": "이 기술이 왜 실무에서 반드시 필요한지 구체적 이유 (1-2문장)",
      "confidence": 0.0~1.0
    }}
  ]
}}"""

    USER_TEMPLATE = """다음 채용공고를 분석하여 암묵적 요구 역량을 추론해주세요.

[채용공고]
직무: {title}
요구사항: {description}

[공고에 이미 명시된 기술 - 이것들은 절대 implicit_skills에 포함하지 마세요]
{explicit_skills}

[통계 분석(ARM)이 도출한 연관 기술 후보 - 참고용]
{arm_candidates}

[추론 단계]
1단계: 직무명과 요구사항을 읽고 이 직무가 실제로 어떤 업무를 하는지 파악
2단계: 공고 원문에서 추상적 표현을 반드시 구체적 기술명으로 변환
       [필수 변환 규칙 - 반드시 적용]
       "딥러닝 프레임워크" → PyTorch와 TensorFlow 둘 다 (명시 안 된 것만)
       "서버 연동 인터페이스 + Python" → FastAPI 또는 Flask
       "클라우드 환경" → Docker, Kubernetes
       "데이터 분석" → Pandas, SQL
       "버전 관리" → Git
       "CI/CD" → Jenkins 또는 GitHub Actions
       "컨테이너" → Docker
       주의: 우대사항이라도 직무 핵심 역량이면 반드시 포함할 것
3단계: 변환된 기술 중 이미 명시된 기술과 겹치는 것 제거
4단계: 남은 기술을 ARM 후보와 대조하여 최종 암묵적 역량 목록 확정
5단계: 각 기술의 reason을 공고 원문 조건과 직접 연결하여 작성

위 단계로 사고한 뒤 JSON만 출력하세요."""

    def build(
        self,
        title: str,
        description: str,
        explicit_skills: list[str],
        arm_candidates: list[str],
    ) -> str:
        explicit_str = ", ".join(explicit_skills) if explicit_skills else "없음"
        arm_str = ", ".join(arm_candidates) if arm_candidates else "없음"
        user_msg = self.USER_TEMPLATE.format(
            title=title,
            description=description[:500],  # 토큰 절약
            explicit_skills=explicit_str,
            arm_candidates=arm_str,
        )
        return f"{self.SYSTEM_TEMPLATE}\n\n{user_msg}"


# ── 응답 파서 ──────────────────────────────────────────────────
class ResponseParser:
    """
    LLM 응답 JSON 파싱.
    LLM이 가끔 JSON 앞뒤에 마크다운 코드블록을 붙이는 경우 처리.
    """

    def parse(self, response: str) -> list[LLMImplicitSkill]:
        # ```json ... ``` 블록 제거
        clean = re.sub(r"```(?:json)?", "", response).strip().rstrip("`").strip()
        try:
            data = json.loads(clean)
            skills = []
            for item in data.get("implicit_skills", []):
                skills.append(LLMImplicitSkill(
                    skill=item.get("skill", ""),
                    reason=item.get("reason", ""),
                    confidence=float(item.get("confidence", 0.5)),
                ))
            return skills
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[ResponseParser] JSON 파싱 실패: {e}\n응답: {response[:200]}")
            return []


# ── Mock LLM (연동 전 테스트용) ─────────────────────────────────
class MockLLMClient:
    """
    LLM API 미연동 상태에서 파이프라인 전체를 테스트하기 위한 Mock.
    실제 API 연동 후 교체 예정.
    """

    MOCK_RESPONSES = {
        "백엔드": {
            "implicit_skills": [
                {"skill": "Git", "reason": "팀 협업 코드 관리에 필수", "confidence": 0.95},
                {"skill": "Linux", "reason": "서버 운영 환경 기본 역량", "confidence": 0.90},
                {"skill": "REST API 설계", "reason": "백엔드 핵심 업무", "confidence": 0.85},
                {"skill": "단위 테스트", "reason": "코드 품질 유지를 위한 기본 소양", "confidence": 0.80},
            ]
        },
        "프론트엔드": {
            "implicit_skills": [
                {"skill": "Git", "reason": "버전 관리 필수", "confidence": 0.95},
                {"skill": "CSS/HTML", "reason": "프론트엔드 기본 역량", "confidence": 0.92},
                {"skill": "크로스브라우저 대응", "reason": "실제 서비스 배포 시 필수", "confidence": 0.75},
                {"skill": "웹 성능 최적화", "reason": "사용자 경험 개선 핵심", "confidence": 0.70},
            ]
        },
        "default": {
            "implicit_skills": [
                {"skill": "Git", "reason": "모든 개발 직군 필수 도구", "confidence": 0.95},
                {"skill": "커뮤니케이션", "reason": "팀 협업 기본 역량", "confidence": 0.85},
                {"skill": "문서화", "reason": "지식 공유 및 유지보수 필수", "confidence": 0.75},
            ]
        },
    }

    def complete(self, prompt: str) -> str:
        # 프롬프트에서 직무 키워드 감지하여 적절한 Mock 응답 선택
        if "백엔드" in prompt:
            data = self.MOCK_RESPONSES["백엔드"]
        elif "프론트엔드" in prompt:
            data = self.MOCK_RESPONSES["프론트엔드"]
        else:
            data = self.MOCK_RESPONSES["default"]
        return json.dumps(data, ensure_ascii=False)


# ── 메인 LLMReasoner ───────────────────────────────────────────
class LLMReasoner:
    """
    Chain-of-Thought 기반 암묵적 역량 추론기.

    [사용법]
    # LLM 미확정 시 (Mock 사용)
    reasoner = LLMReasoner()

    # LLM 확정 후
    reasoner = LLMReasoner(client=OpenAIAdapter(api_key="..."))
    reasoner = LLMReasoner(client=AnthropicAdapter(api_key="..."))
    """

    def __init__(self, client: Optional[LLMClient] = None):
        if client is None:
            print("[LLMReasoner] LLM 미연동 → MockLLMClient 사용")
            self.client = MockLLMClient()
            self._using_mock = True
        else:
            self.client = client
            self._using_mock = False
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()

    def reason(
        self,
        title: str,
        description: str,
        explicit_skills: list[str],
        arm_candidates: list[str],
    ) -> LLMImplicitResult:
        """
        단일 채용공고에 대한 LLM 추론 실행.
        """
        prompt = self.prompt_builder.build(
            title=title,
            description=description,
            explicit_skills=explicit_skills,
            arm_candidates=arm_candidates,
        )
        try:
            response = self.client.complete(prompt)
            skills = self.parser.parse(response)
            # 명시적 기술 + 설명에 포함된 기술 모두 제거
            explicit_lower = {s.lower() for s in explicit_skills}
            # description에서 언급된 기술도 필터링
            desc_lower = description.lower()
            filtered = []
            # description 텍스트에서 언급된 기술도 명시적으로 간주
            desc_skills = set()
            for exp in explicit_skills:
                desc_skills.add(exp.lower())
                # 부분 매칭도 포함 (예: "PostgreSQL" ↔ "postgres")
                if len(exp) > 4:
                    desc_skills.add(exp[:5].lower())

            for s in skills:
                skill_lower = s.skill.lower()
                # 명시적 기술과 정확히 일치
                if skill_lower in explicit_lower:
                    continue
                # 부분 문자열 포함 관계
                if any(skill_lower in exp.lower() or exp.lower() in skill_lower
                       for exp in explicit_skills):
                    continue
                # description 원문에 직접 언급된 기술
                if any(skill_lower in desc.lower() for desc in [description]):
                    continue
                filtered.append(s)
            return LLMImplicitResult(
                implicit_skills=filtered,
                raw_response=response,
                used_mock=self._using_mock,
            )
        except Exception as e:
            print(f"[LLMReasoner] 추론 오류: {e}")
            return LLMImplicitResult(used_mock=self._using_mock)


# ── 단독 실행 테스트 ────────────────────────────────────────────
if __name__ == "__main__":
    reasoner = LLMReasoner()  # Mock 사용
    result = reasoner.reason(
        title="백엔드 개발자",
        description="Python, FastAPI, PostgreSQL 경험자. Docker 활용 경험 우대. 경력 3년 이상.",
        explicit_skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
        arm_candidates=["Kubernetes", "Redis"],
    )
    print(f"Mock 사용: {result.used_mock}")
    print("\n[LLM 추론 암묵적 역량]")
    for s in result.implicit_skills:
        print(f"  {s.skill} (conf={s.confidence:.2f}): {s.reason}")
        