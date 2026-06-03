"""Streamlit 데모 UI — 채용공고 분석 + 이력서 정합성 검증 통합.

실행:
  docker compose up -d ollama     # LLM 추론 사용 시
  docker compose up demo
  → http://localhost:8501
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from src.data.loaders import iter_jobs
from src.integration.pipeline import IntegrationPipeline, IntegrationReport
from src.logic_auditor.pipeline import LogicAuditorPipeline
from src.reverse_job.adapters.ollama_client import DEFAULT_MODEL, OllamaLLMClient
from src.reverse_job.llm_reasoner import LLMReasoner
from src.reverse_job.pipeline import ReverseJobPipeline

BERT_MODEL_DIR = Path("data/bert_classifier")

SAMPLE_JOB = """[백엔드 개발자 채용]
- Python, Django, MySQL 3년 이상 경력
- REST API 설계 경험 필수
- AWS 기반 서비스 운영 경험 우대
- Git 기반 협업 가능자
"""

SAMPLE_RESUME = """저는 2010년부터 Docker를 활용하여 대규모 인프라를 구축했습니다.
신입 1개월 차에 전사 KPI 설계를 단독으로 주도했습니다.
입사 후 2주 만에 10만 TPS 규모의 대규모 마이크로서비스 아키텍처를 단독 설계·구현했습니다.
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


@st.cache_resource
def load_bert_detector(model_dir: str):
    from src.logic_auditor.bert_detector import BertConsistencyDetector

    return BertConsistencyDetector(model_dir=model_dir)


def build_reverse_job_pipeline(use_llm: bool, model: str) -> ReverseJobPipeline:
    rj = ReverseJobPipeline(load_ner=False)
    if use_llm:
        client = OllamaLLMClient(model=model)
        rj.reasoner = LLMReasoner(client=client)
    return rj


def highlight_violations(text: str, snippets: list[str]) -> str:
    """문자열 리스트의 각 구간을 본문에서 <mark> 로 감싸 HTML 반환."""
    if not snippets:
        return f"<div style='line-height:1.7;'>{escape(text).replace(chr(10), '<br>')}</div>"
    spans = sorted({s for s in snippets if s}, key=len, reverse=True)
    out = escape(text)
    for s in spans:
        out = out.replace(
            escape(s),
            f"<mark style='background-color:#ffe066;'>{escape(s)}</mark>",
        )
    return f"<div style='line-height:1.7;'>{out.replace(chr(10), '<br>')}</div>"


def highlight_bert(text: str, bert_result) -> str:
    """BERT 가 위반으로 판정한 문장만 빨간색 하이라이트."""
    if not bert_result or not bert_result.sentence_predictions:
        return f"<div style='line-height:1.7;'>{escape(text).replace(chr(10), '<br>')}</div>"
    violators = sorted(
        (p.sentence for p in bert_result.sentence_predictions if p.is_violation),
        key=len, reverse=True,
    )
    out = escape(text)
    for s in violators:
        out = out.replace(
            escape(s),
            f"<mark style='background-color:#ffadad;'>{escape(s)}</mark>",
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


def render_module_b(
    resume_text: str,
    report: IntegrationReport,
    bert_result=None,
    bert_threshold: float = 0.5,
) -> None:
    rule_violation = bool(report.resume_audit.violations)
    bert_violation = bert_result.is_violation if bert_result is not None else False
    ensemble_violation = rule_violation or bert_violation

    # 점수: Rule 기반은 신뢰도, BERT 는 확률
    score = report.resume_audit.trust_score
    color = "#28a745" if score >= 85 else ("#ffc107" if score >= 70 else "#dc3545")

    st.markdown(
        f"<div style='font-size:42px;font-weight:bold;color:{color};'>"
        f"Rule 신뢰도 {score} / 100</div>",
        unsafe_allow_html=True,
    )
    st.progress(score / 100)

    if bert_result is not None:
        pct = bert_result.document_probability * 100
        bert_color = "#dc3545" if bert_violation else "#28a745"
        st.markdown(
            f"<div style='font-size:24px;font-weight:bold;color:{bert_color};margin-top:8px;'>"
            f"BERT 위반 확률 {pct:.1f}% (임계값 {bert_threshold*100:.0f}%)</div>",
            unsafe_allow_html=True,
        )

    # Ensemble 평결
    verdict_color = "#dc3545" if ensemble_violation else "#28a745"
    verdict = "위반 감지" if ensemble_violation else "통과"
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background:{verdict_color}15;"
        f"border-left:6px solid {verdict_color};margin:12px 0;'>"
        f"<b>앙상블 평결:</b> <span style='color:{verdict_color};font-weight:bold;'>{verdict}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    cols = st.columns(2)

    with cols[0]:
        st.markdown("##### 📐 Rule-based")
        if report.resume_audit.violations:
            for v in report.resume_audit.violations:
                st.error(f"[{v.tech}] {v.kind}")
                st.caption(v.message)
                st.code(v.snippet, language=None)
        else:
            st.success("위반 없음")

    with cols[1]:
        st.markdown("##### 🤖 BERT")
        if bert_result is None:
            st.info("BERT 비활성화 (사이드바에서 활성화)")
        elif bert_result.is_violation:
            st.error(f"문서 위반 확률 {bert_result.document_probability*100:.1f}%")
            st.caption("문장별 점수(참고용, 학습은 문서 단위로 진행):")
            for p in bert_result.sentence_predictions:
                pct = p.probability * 100
                if p.is_violation:
                    st.write(f"🔴 {pct:>5.1f}% — {p.sentence[:80]}…")
                else:
                    st.write(f"⚪ {pct:>5.1f}% — {p.sentence[:80]}…")
        else:
            st.success(f"정상 (위반 확률 {bert_result.document_probability*100:.1f}%)")
            if bert_result.sentence_predictions:
                st.caption("문장별 점수(참고용):")
                for p in bert_result.sentence_predictions:
                    pct = p.probability * 100
                    st.write(f"⚪ {pct:>5.1f}% — {p.sentence[:80]}…")

    st.divider()
    if report.resume_audit.ai_detection is not None:
        ai = report.resume_audit.ai_detection
        st.caption(
            f"AI 생성 휴리스틱  perplexity={ai.perplexity:.2f}  "
            f"burstiness={ai.burstiness:.2f}  p(ai)={ai.ai_probability:.2f}"
        )

    st.markdown("##### 본문 하이라이트")
    rule_html_col, bert_html_col = st.columns(2)
    with rule_html_col:
        st.caption("Rule-based (🟡 노란색)")
        rule_snippets = [v.snippet for v in report.resume_audit.violations]
        st.markdown(highlight_violations(resume_text, rule_snippets), unsafe_allow_html=True)
    with bert_html_col:
        st.caption("BERT (🔴 빨간색)")
        if bert_result is not None:
            st.markdown(highlight_bert(resume_text, bert_result), unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='line-height:1.7;color:#888;'>{escape(resume_text).replace(chr(10), '<br>')}</div>",
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
        "이력서·포트폴리오의 정합성을 Rule + BERT 앙상블로 검증(Module B)하는 통합 NLP 파이프라인"
    )

    with st.sidebar:
        st.header("⚙️ 설정")
        use_llm = st.checkbox(
            "LLM 추론 활성화 (Ollama)", value=False,
            help="ARM 결과에 LLM Chain-of-Thought 결과를 합집합. 응답 5~15초.",
        )
        llm_model = st.text_input("LLM 모델", DEFAULT_MODEL)

        st.divider()
        bert_available = BERT_MODEL_DIR.exists()
        use_bert = st.checkbox(
            "BERT 정합성 검출 활성화 (KLUE-RoBERTa)",
            value=bert_available,
            disabled=not bert_available,
            help=(
                "Rule 이 못 잡는 의미적 정합성 위반(경험-기간, 신입-경영, "
                "간접 시간 참조, 호환 불가 기술 조합)을 BERT 가 검출."
            ),
        )
        if not bert_available:
            st.caption("⚠️ data/bert_classifier 없음 — train_bert_classifier.py 먼저 실행")
        bert_threshold = st.slider(
            "BERT 임계값", min_value=0.1, max_value=0.95, value=0.5, step=0.05,
            disabled=not (bert_available and use_bert),
        )

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
                f"Ollama 서비스 연결 실패 ({client.host}). LLM 없이 ARM-only 로 진행합니다."
            )
            use_llm = False

    bert_detector = None
    if use_bert and bert_available:
        with st.spinner("BERT 모델 로딩…"):
            try:
                bert_detector = load_bert_detector(str(BERT_MODEL_DIR))
                bert_detector.threshold = bert_threshold
            except Exception as e:
                st.warning(f"BERT 로딩 실패: {e}")
                bert_detector = None

    rj = build_reverse_job_pipeline(use_llm=use_llm, model=llm_model)
    audit = LogicAuditorPipeline()
    pipeline = IntegrationPipeline(reverse_job=rj, audit=audit)

    with st.spinner("분석 중…"):
        try:
            report = pipeline.run(job_text, resume_text, transactions=transactions)
        except Exception as e:
            st.error(f"분석 실패: {e}")
            return

    bert_result = None
    if bert_detector is not None:
        with st.spinner("BERT 정합성 추론 중…"):
            # 학습과 동일한 문서 단위 추론 + 문장별 보조 점수 (참고용)
            bert_result = bert_detector.predict_sentences(resume_text)
            doc_result = bert_detector.predict(resume_text)
            bert_result.document_probability = doc_result.document_probability
            bert_result.is_violation = doc_result.is_violation

    tabs = st.tabs(
        ["📋 채용공고 분석 (Module A)", "🔍 이력서 검증 (Module B)", "🎯 통합 리포트"]
    )
    with tabs[0]:
        render_module_a(report)
    with tabs[1]:
        render_module_b(resume_text, report, bert_result, bert_threshold)
    with tabs[2]:
        render_integration(report)


if __name__ == "__main__":
    main()
