"""Streamlit 데모 UI — 채용공고 분석 + 이력서 검증 통합.

실행:
  docker compose up -d ollama   # LLM 사용 시
  docker compose up demo
  → http://localhost:8501
"""
from __future__ import annotations

from html import escape

import streamlit as st

from src.data.loaders import iter_jobs
from src.integration.pipeline import IntegrationPipeline, IntegrationReport
from src.logic_auditor.pipeline import LogicAuditorPipeline
from src.reverse_job.adapters.ollama_client import DEFAULT_MODEL, OllamaLLMClient
from src.reverse_job.llm_reasoner import LLMReasoner
from src.reverse_job.pipeline import ReverseJobPipeline

SAMPLE_JOB = """[백엔드 개발자 채용]
- Python, Django, MySQL 3년 이상 경력
- REST API 설계 경험 필수
- AWS 기반 서비스 운영 경험 우대
- Git 기반 협업 가능자
"""

SAMPLE_RESUME = """저는 2010년부터 Docker를 활용하여 대규모 인프라를 구축했습니다.
3개월 만에 대규모 시스템 아키텍처를 단독으로 설계하고 10만 TPS를 처리하는 서비스를 완성했습니다.
React 16과 Next.js 13을 2015년 프로젝트에 적용했습니다.
이후 Python과 Django 로 백엔드를 개발했고, AWS 환경에서 운영했습니다.
"""


@st.cache_data
def load_arm_transactions() -> list[list[str]]:
    """IT 직무 채용공고 → 이미 스킬 추출된 ARM transactions. 시작 시 1회 빌드."""
    from scripts.run_arm import is_it_job, job_skills

    return [
        job_skills(j)
        for j in iter_jobs()
        if is_it_job(j) and len(job_skills(j)) >= 2
    ]


def build_pipeline(use_llm: bool, model: str) -> IntegrationPipeline:
    rj = ReverseJobPipeline(load_ner=False)
    if use_llm:
        client = OllamaLLMClient(model=model)
        rj.reasoner = LLMReasoner(client=client)
    audit = LogicAuditorPipeline()
    return IntegrationPipeline(reverse_job=rj, audit=audit)


def highlight_violations(text: str, violations) -> str:
    """위반된 문장 부분을 <mark> 로 감싸 HTML 반환."""
    if not violations:
        return f"<div style='line-height:1.7;'>{escape(text).replace(chr(10), '<br>')}</div>"
    spans = sorted({v.snippet for v in violations}, key=len, reverse=True)
    out = escape(text)
    for s in spans:
        out = out.replace(
            escape(s),
            f"<mark style='background-color:#ffe066;'>{escape(s)}</mark>",
        )
    return f"<div style='line-height:1.7;'>{out.replace(chr(10), '<br>')}</div>"


def render_module_a(report: IntegrationReport) -> None:
    explicit = report.job_analysis.explicit.tech_stack
    arm = report.job_analysis.implicit_arm
    llm = report.job_analysis.implicit_llm
    union = report.job_analysis.implicit_union

    cols = st.columns(4)
    cols[0].metric("명시적", len(explicit))
    cols[1].metric("ARM 추론", len(arm))
    cols[2].metric("LLM 추론", len(llm))
    cols[3].metric("합집합", len(union))

    st.markdown("**명시적 역량 (NER + 키워드 사전)**")
    st.write(", ".join(explicit) if explicit else "_없음_")

    st.markdown("**ARM 추론 (코퍼스 공동출현)**")
    st.write(", ".join(arm) if arm else "_없음_")

    st.markdown("**LLM 추론 (Chain-of-Thought)**")
    st.write(", ".join(llm) if llm else "_(LLM 미사용)_")

    st.markdown("**암묵적 역량 합집합**")
    st.write(", ".join(union) if union else "_없음_")


def render_module_b(resume_text: str, report: IntegrationReport) -> None:
    score = report.resume_audit.trust_score
    color = "#28a745" if score >= 85 else ("#ffc107" if score >= 70 else "#dc3545")

    st.markdown(
        f"<div style='font-size:42px;font-weight:bold;color:{color};'>"
        f"신뢰도 {score} / 100</div>",
        unsafe_allow_html=True,
    )
    st.progress(score / 100)

    if report.resume_audit.violations:
        st.error(f"위반 {len(report.resume_audit.violations)}건 탐지")
        for v in report.resume_audit.violations:
            with st.expander(f"[{v.tech}] {v.kind} — {v.message}"):
                st.write(f"**문장**: {v.snippet}")
                st.write(
                    f"**주장 연도**: {v.claimed_year}  /  "
                    f"**실제 출시**: {v.actual_year}"
                )
    else:
        st.success("탐지된 사실 모순 없음 ✓")

    if report.resume_audit.ai_detection is not None:
        ai = report.resume_audit.ai_detection
        st.write(
            f"**AI 생성 신호**  perplexity={ai.perplexity:.2f}  "
            f"burstiness={ai.burstiness:.2f}  p(ai)={ai.ai_probability:.2f}"
        )

    st.markdown("**위반 위치 하이라이트**")
    st.markdown(
        highlight_violations(resume_text, report.resume_audit.violations),
        unsafe_allow_html=True,
    )


def render_integration(report: IntegrationReport) -> None:
    if report.skill_gap:
        st.warning(f"**미보유 역량 ({len(report.skill_gap)})**")
        st.write(", ".join(report.skill_gap))
    else:
        st.success("채용공고 요구역량을 모두 보유하고 있습니다 ✓")

    with st.expander("전체 JSON 리포트"):
        st.json(report.to_dict())


def main() -> None:
    st.set_page_config(
        page_title="Reverse Job + Logic Auditor",
        layout="wide",
        page_icon="📄",
    )
    st.title("📄 Reverse Job Engineering + Logic Auditor")
    st.caption(
        "채용공고에서 암묵적 요구 역량을 추론(Module A)하고, "
        "이력서·포트폴리오의 사실 모순을 탐지(Module B)하는 통합 NLP 파이프라인"
    )

    with st.sidebar:
        st.header("⚙️ 설정")
        use_llm = st.checkbox(
            "LLM 추론 활성화 (Ollama)", value=False,
            help="활성화하면 ARM 결과에 LLM Chain-of-Thought 결과를 합집합. 응답 5~15초 소요.",
        )
        llm_model = st.text_input("LLM 모델", DEFAULT_MODEL)

        st.divider()
        st.header("📚 샘플")
        if st.button("백엔드 채용 샘플", use_container_width=True):
            st.session_state["job_text"] = SAMPLE_JOB
        if st.button("이력서 (오류 포함) 샘플", use_container_width=True):
            st.session_state["resume_text"] = SAMPLE_RESUME

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.subheader("🧾 채용공고")
        job_text = st.text_area(
            "내용",
            key="job_text",
            height=320,
            placeholder=SAMPLE_JOB,
            label_visibility="collapsed",
        )
    with col_r:
        st.subheader("👤 이력서·포트폴리오")
        resume_text = st.text_area(
            "내용",
            key="resume_text",
            height=320,
            placeholder=SAMPLE_RESUME,
            label_visibility="collapsed",
        )

    if not st.button("🚀 분석", type="primary", use_container_width=True):
        st.info("채용공고와 이력서를 입력하고 분석 버튼을 눌러주세요.")
        return

    if not job_text.strip() or not resume_text.strip():
        st.error("두 입력란을 모두 채워주세요.")
        return

    with st.spinner("ARM 코퍼스 로딩…"):
        transactions = load_arm_transactions()

    if use_llm:
        client = OllamaLLMClient(model=llm_model)
        if not client.healthcheck():
            st.warning(
                f"Ollama 서비스에 연결 실패 ({client.host}). "
                "LLM 없이 ARM-only 로 진행합니다."
            )
            use_llm = False

    pipeline = build_pipeline(use_llm=use_llm, model=llm_model)

    with st.spinner("분석 중…"):
        try:
            report = pipeline.run(
                job_text, resume_text, transactions=transactions
            )
        except Exception as e:
            st.error(f"분석 실패: {e}")
            return

    tabs = st.tabs(
        ["📋 채용공고 분석 (Module A)", "🔍 이력서 검증 (Module B)", "🎯 통합 리포트"]
    )
    with tabs[0]:
        render_module_a(report)
    with tabs[1]:
        render_module_b(resume_text, report)
    with tabs[2]:
        render_integration(report)


if __name__ == "__main__":
    main()