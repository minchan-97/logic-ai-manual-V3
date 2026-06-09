"""
manual_app.py — 교사 업무 매뉴얼 RAG Q&A 도구
================================================

흐름:
  1. 사용자가 매뉴얼 PDF 1~5개 업로드
  2. 파싱 → 청킹 → OpenAI 임베딩 인덱스 (1회)
  3. 사용자가 질문 입력
  4. 관련 청크 top-k 검색
  5. 두 관점 LLM 호출 (절차 안내 / 주의사항)
  6. 두 응답의 mismatch가 크면 "매뉴얼이 모호함, 직접 확인하세요" 신호
  7. 답변에 출처 (파일명·페이지) 강제 인용

원칙:
  - LLM은 업로드된 PDF 내용만으로 답변 (system prompt 강제)
  - PDF에 없는 내용은 "제공된 자료에 없습니다"라고 답하도록 지시
  - 출처 인용 마커가 컨텍스트에 박혀 있어 LLM이 자연스럽게 사용

v4 코어 재사용:
  - 두 관점 일관성 비교
  - safety 누적 차단 (선택)
"""
from __future__ import annotations
import os, json, time, sqlite3, hashlib, io
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

from rag import (
    Chunk, RagIndex,
    parse_pdf_pages, build_chunks_from_pdf, build_index,
    search_index, format_context_for_llm,
    save_index, load_index, get_index_path,
)

try:
    from safety import (
        SafetyConfig, SafetyState,
        record_verdict as safety_record,
        request_release as safety_request_release,
        reset_state as safety_reset,
    )
    SAFETY_AVAILABLE = True
except Exception:
    SAFETY_AVAILABLE = False

# CoreAI NeuralMarkov (선택)
try:
    from neural_markov_engine import NeuralMarkovEngine
    from guardrail_loop import run_guardrail_loop
    COREAI_AVAILABLE = True
except Exception:
    COREAI_AVAILABLE = False


load_dotenv()
st.set_page_config(page_title="교사 매뉴얼 Q&A", layout="wide")

CHAT_URL_DEFAULT = "https://api.openai.com/v1/chat/completions"
EMBEDDING_URL_DEFAULT = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"


# =========================================================
# Two-perspective system prompts (매뉴얼 도메인 특화)
# =========================================================

PROCEDURE_SYSTEM = (
    "당신은 학교 교직원의 업무 매뉴얼 질의응답 보조자입니다. "
    "**제공된 자료에 적힌 내용으로만** 답하세요. "
    "다음 한 가지 일만 하세요: 사용자가 물은 업무의 **절차를 단계 순서로 정리**.\n\n"
    "엄격한 규칙:\n"
    "- 자료에 없는 내용은 절대 추측해서 채우지 마세요.\n"
    "- 자료에 답이 없으면 '제공된 자료에 해당 정보가 없습니다'라고 답하세요.\n"
    "- 자료에 있는 내용은 반드시 [출처: 파일명 p.N] 형식으로 인용하세요.\n"
    "- 법적 판단·결과 예측·자문은 하지 마세요. 자료를 인용한 정리만 하세요.\n"
    "- 매뉴얼 버전·시점이 다를 수 있으니, 답변 끝에 '실제 시스템 화면과 다를 수 있으니 행정실/교무부에 확인하세요'를 항상 부착하세요."
)

CAVEATS_SYSTEM = (
    "당신은 학교 교직원의 업무 매뉴얼 질의응답 보조자입니다. "
    "**제공된 자료에 적힌 내용으로만** 답하세요. "
    "다음 한 가지 일만 하세요: 사용자가 물은 업무에서 **주의해야 할 점, 흔한 실수, 빠뜨리기 쉬운 단계**를 자료에서 찾아 짚기.\n\n"
    "엄격한 규칙:\n"
    "- 자료에 없는 주의사항을 추측해서 만들지 마세요.\n"
    "- 자료에 명시된 주의·경고·필수 조건만 추출하세요.\n"
    "- 자료에 그런 정보가 없으면 '제공된 자료에 명시된 주의사항이 없습니다'라고 답하세요.\n"
    "- 인용은 [출처: 파일명 p.N] 형식.\n"
    "- 법적 판단·결과 예측·자문은 하지 마세요.\n"
    "- 답변 끝에 '실제 시스템 화면과 다를 수 있으니 행정실/교무부에 확인하세요'를 항상 부착하세요."
)


# =========================================================
# DB
# =========================================================
DB_PATH = "manual_qa.db"

@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            doc_set_hash TEXT,
            query TEXT,
            mismatch REAL,
            verdict TEXT
        )
    """)
    conn.commit()
    return conn


conn = get_db()


# =========================================================
# Helpers
# =========================================================
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


def call_chat(prompt: str, system: str, model: str, api_key: str,
              url: str = CHAT_URL_DEFAULT, temperature: float = 0.1) -> str:
    """매뉴얼 도메인은 temperature 낮게 (사실성 우선)"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    res = requests.post(url, headers=headers, json=body, timeout=90)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()


@lru_cache(maxsize=128)
def _cached_query_emb(key: str, text: str, model: str, url: str, api_key: str) -> Tuple[float, ...]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    res = requests.post(url, headers=headers,
                         json={"model": model, "input": text}, timeout=60)
    res.raise_for_status()
    return tuple(res.json()["data"][0]["embedding"])


def compute_response_mismatch(a: str, b: str, api_key: str, model: str, url: str) -> float:
    """두 응답의 의미 거리"""
    h_a = hashlib.sha256(f"{model}::{a}".encode()).hexdigest()
    h_b = hashlib.sha256(f"{model}::{b}".encode()).hexdigest()
    va = np.array(_cached_query_emb(h_a, a, model, url, api_key))
    vb = np.array(_cached_query_emb(h_b, b, model, url, api_key))
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 100.0
    cs = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return (1.0 - cs) / 2.0 * 100.0


def build_user_message(query: str, context: str) -> str:
    """LLM에 보낼 user 메시지: 컨텍스트 + 질문"""
    return (
        f"# 제공된 자료\n\n{context}\n\n"
        f"# 사용자 질문\n\n{query}\n\n"
        f"위 자료에서 근거를 찾아 답하세요. 자료에 없으면 '제공된 자료에 없습니다'라고 답하세요."
    )


# =========================================================
# Session state
# =========================================================
if "rag_index" not in st.session_state:
    st.session_state.rag_index = RagIndex()
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "doc_set_hash" not in st.session_state:
    st.session_state.doc_set_hash = ""

# 저장된 인덱스 자동 로드 (페이지 새로고침 후에도 유지)
if not st.session_state.rag_index.is_built() and st.session_state.doc_set_hash:
    cached = load_index(get_index_path(st.session_state.doc_set_hash))
    if cached:
        st.session_state.rag_index = cached
if "threshold" not in st.session_state:
    st.session_state.threshold = 25.0
if "last_qa" not in st.session_state:
    st.session_state.last_qa = None
if SAFETY_AVAILABLE:
    if "safety_state" not in st.session_state:
        st.session_state.safety_state = SafetyState()
    if "safety_cfg" not in st.session_state:
        st.session_state.safety_cfg = SafetyConfig()


# =========================================================
# Header
# =========================================================
st.title("📚 교사 업무 매뉴얼 Q&A")
st.caption("나이스·K-에듀파인·회계 매뉴얼 등을 업로드하고 한국어 질문으로 답을 찾는 도구")

st.error(
    "⚠️ **이 도구는 업로드된 자료의 내용만 검색·정리합니다.** "
    "법적·행정적 자문이 아니며, 매뉴얼 버전이 실제 시스템과 다를 수 있습니다. "
    "최종 절차는 **반드시 행정실·교무부·해당 부서**에 확인하세요. "
    "민감 정보(학생 이름, 주민번호 등) 포함 PDF는 업로드하지 마세요."
)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("⚙️ API 설정")
    api_key = st.text_input("OpenAI API Key", value=get_secret("OPENAI_API_KEY"),
                             type="password")
    chat_model = st.text_input("Chat model",
                                value=get_secret("OPENAI_MODEL", "gpt-4o-mini"))
    emb_model = st.text_input("Embedding model",
                               value=get_secret("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT))

    st.divider()
    st.header("🔍 검색 설정")
    top_k = st.slider("Top-K 청크 수", 3, 10, 5, 1,
                       help="질문당 검색할 청크 수. 많으면 풍부하지만 비용↑")
    max_chars = st.slider("청크 최대 길이 (자)", 400, 1500, 800, 100)
    overlap = st.slider("청크 겹침 (자)", 0, 300, 100, 50)
    st.session_state.threshold = st.slider(
        "두 관점 mismatch 임계값 (%)",
        5.0, 60.0, float(st.session_state.threshold), 1.0,
        help="이 값 초과면 '매뉴얼이 모호하니 직접 확인' 신호",
    )

    if SAFETY_AVAILABLE:
        st.divider()
        safety_mode = st.checkbox("🚨 누적 안전 차단 (선택)", value=False)
    else:
        safety_mode = False

    # CoreAI NeuralMarkov
    if COREAI_AVAILABLE:
        st.divider()
        st.markdown("### 🎯 CoreAI 가드레일")
        coreai_mode = st.checkbox(
            "CoreAI 사용",
            value=False,
            help="매뉴얼 코퍼스로 NeuralMarkov 학습 → 답변이 매뉴얼 도메인 안에 있는지 검증 → 이탈 시 재생성",
        )
        # 기본값 먼저 설정 (coreai_mode=False여도 변수 존재해야 함)
        coreai_epochs = 10
        coreai_retry  = 2
        coreai_logp   = -11.5

        if coreai_mode:
            if "coreai_engine" not in st.session_state:
                st.session_state.coreai_engine = NeuralMarkovEngine()
            if "coreai_trained" not in st.session_state:
                st.session_state.coreai_trained = False

            st.caption("PDF 인덱스 빌드 후 자동 학습 또는 별도 txt 업로드")
            coreai_corpus_file = st.file_uploader(
                "추가 코퍼스 (.txt, 선택)",
                type=["txt"],
                key="coreai_manual_corpus",
            )
            coreai_epochs  = st.slider("학습 Epochs", 5, 20, 10, key="coreai_ep")
            coreai_retry   = st.slider("최대 재생성 횟수", 1, 3, 2, key="coreai_retry")
            coreai_logp    = st.slider("logP 임계값", -15.0, -5.0, -11.5, 0.5, key="coreai_logp")

            if coreai_corpus_file and st.button("🎯 CoreAI 학습", use_container_width=True):
                with st.spinner("NeuralMarkov 학습 중..."):
                    try:
                        corpus_text = coreai_corpus_file.read().decode("utf-8", errors="ignore")
                        st.session_state.coreai_engine.train(
                            corpus_text, embedding_dim=32, epochs=coreai_epochs
                        )
                        st.session_state.coreai_trained = True
                        st.success(f"✓ 학습 완료 ({len(st.session_state.coreai_engine.idx2word)}어휘)")
                    except Exception as e:
                        st.error(f"학습 실패: {e}")

            if st.session_state.get("coreai_trained"):
                st.success("✓ CoreAI 학습됨")
    else:
        coreai_mode   = False
        coreai_epochs = 10
        coreai_retry  = 2
        coreai_logp   = -11.5

    st.divider()
    st.markdown("### 🔒 개인정보")
    st.caption(
        "PDF는 OpenAI API로 전송됩니다. 학생 개인정보가 포함된 자료는 "
        "업로드하지 마세요. DB에는 본문 저장 안 함."
    )


# =========================================================
# Step 1: PDF 업로드
# =========================================================
st.subheader("1단계: 매뉴얼 PDF 업로드")
st.caption("나이스·K-에듀파인 매뉴얼, 시도교육청 업무매뉴얼, 학교 자체 가이드 등. "
            "한 번에 1~5개 권장. 큰 매뉴얼은 관심 부분만 잘라 업로드하세요.")

uploaded = st.file_uploader(
    "PDF 파일 (여러 개 선택 가능)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded:
    file_summary = [(f.name, len(f.getvalue())) for f in uploaded]
    new_hash = hashlib.sha256(
        json.dumps(sorted(file_summary), ensure_ascii=False).encode()
    ).hexdigest()[:16]
    st.session_state.uploaded_files = file_summary

    # 이미 인덱스된 파일 셋과 다르면 새로 빌드 필요
    needs_rebuild = (new_hash != st.session_state.doc_set_hash)
    st.session_state.doc_set_hash = new_hash

    cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        with cols[i % len(cols)]:
            st.metric(f.name[:25], f"{len(f.getvalue())//1024} KB")

    if needs_rebuild or not st.session_state.rag_index.is_built():
        if st.button("📊 인덱스 빌드", type="primary"):
            if not api_key:
                st.error("OpenAI API Key가 필요합니다.")
            else:
                with st.spinner("PDF 파싱 중..."):
                    all_chunks: List[Chunk] = []
                    parse_errors = []
                    for f in uploaded:
                        try:
                            chunks = build_chunks_from_pdf(
                                f.getvalue(), f.name,
                                max_chars=max_chars, overlap=overlap,
                            )
                            all_chunks.extend(chunks)
                            st.caption(f"✓ {f.name}: {len(chunks)}개 청크")
                        except Exception as e:
                            parse_errors.append(f"{f.name}: {e}")

                    for err in parse_errors:
                        st.warning(err)

                    if not all_chunks:
                        st.error("파싱된 청크가 없습니다. PDF가 텍스트 추출이 가능한 형식인지 확인하세요.")
                    else:
                        st.info(f"총 {len(all_chunks)}개 청크. 임베딩 중...")
                        prog = st.progress(0)
                        def cb(done, total):
                            prog.progress(min(done / total, 1.0))
                        try:
                            emb_url = get_secret("OPENAI_EMBEDDING_URL", EMBEDDING_URL_DEFAULT)
                            idx = build_index(all_chunks, api_key, emb_model, emb_url, progress_cb=cb)
                            st.session_state.rag_index = idx

                            # 인덱스 자동 저장 (페이지 새로고침 후에도 유지)
                            save_path = get_index_path(st.session_state.doc_set_hash)
                            if save_index(idx, save_path):
                                st.success(f"인덱스 빌드 완료: {idx.size()}개 청크 (저장됨 — 재빌드 불필요)")
                            else:
                                st.success(f"인덱스 빌드 완료: {idx.size()}개 청크")

                            # CoreAI 자동 학습 — 청크 텍스트를 코퍼스로 사용
                            if coreai_mode and COREAI_AVAILABLE and not st.session_state.get("coreai_trained"):
                                with st.spinner("CoreAI NeuralMarkov 자동 학습 중..."):
                                    try:
                                        corpus_text = "\n".join(c.text for c in all_chunks)
                                        st.session_state.coreai_engine.train(
                                            corpus_text, embedding_dim=32, epochs=coreai_epochs
                                        )
                                        st.session_state.coreai_trained = True
                                        st.success(f"✓ CoreAI 자동 학습 완료 ({len(st.session_state.coreai_engine.idx2word)}어휘)")
                                    except Exception as e:
                                        st.warning(f"CoreAI 학습 실패 (계속 진행): {e}")
                            # 비용 추정
                            total_chars = sum(len(c.text) for c in all_chunks)
                            est_tokens = total_chars // 2  # 한국어는 char당 ~0.5 토큰
                            est_cost_won = est_tokens / 1_000_000 * 20 * 1400  # text-embedding-3-small 기준
                            st.caption(f"임베딩 비용 추정: 약 {est_cost_won:.1f}원 (한 번만)")
                        except Exception as e:
                            st.error(f"인덱스 빌드 실패: {e}")

# 인덱스 상태
import os, pickle, io

if st.session_state.rag_index.is_built():
    st.success(f"✓ 인덱스 준비 완료 ({st.session_state.rag_index.size()}개 청크)")

    # 인덱스 다운로드 버튼
    idx_bytes = pickle.dumps({
        "chunks": st.session_state.rag_index.chunks,
        "embeddings": st.session_state.rag_index.embeddings,
    })
    st.download_button(
        label="💾 인덱스 다운로드 (로컬 저장용)",
        data=idx_bytes,
        file_name=f"rag_index_{st.session_state.doc_set_hash[:8]}.pkl",
        mime="application/octet-stream",
        help="다운로드 후 다음 접속 시 업로드하면 재빌드 불필요 (비용 0원)",
    )
else:
    # 저장된 캐시 있는지 확인
    if st.session_state.doc_set_hash:
        cached_path = get_index_path(st.session_state.doc_set_hash)
        if os.path.exists(cached_path):
            if st.button("💾 저장된 인덱스 불러오기", type="primary"):
                loaded = load_index(cached_path)
                if loaded:
                    st.session_state.rag_index = loaded
                    st.success(f"✓ 저장된 인덱스 로드 완료 ({loaded.size()}개 청크)")
                    st.rerun()

    # 인덱스 파일 업로드로 불러오기
    idx_file = st.file_uploader(
        "💾 저장된 인덱스 업로드 (.pkl) — 재빌드 없이 바로 사용",
        type=["pkl"],
        key="idx_uploader",
    )
    if idx_file:
        try:
            data = pickle.loads(idx_file.read())
            loaded = RagIndex(chunks=data["chunks"], embeddings=data["embeddings"])
            if loaded.is_built():
                st.session_state.rag_index = loaded
                st.success(f"✓ 인덱스 로드 완료 ({loaded.size()}개 청크) — 비용 0원")
                st.rerun()
            else:
                st.error("유효하지 않은 인덱스 파일이에요.")
        except Exception as e:
            st.error(f"인덱스 로드 실패: {e}")

    st.info("PDF를 업로드하고 '인덱스 빌드'를 누르세요.")


# =========================================================
# Step 2: 질의
# =========================================================
st.divider()
st.subheader("2단계: 질문 입력")

query = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 나이스에서 학생 전입 처리는 어떻게 하나요?",
)

ready = st.session_state.rag_index.is_built() and query.strip()
run = st.button("🔍 질문 검색 및 답변", type="primary", disabled=not ready)

if run:
    if not api_key:
        st.error("OpenAI API Key가 필요합니다.")
        st.stop()
    if safety_mode and SAFETY_AVAILABLE and st.session_state.safety_state.locked:
        st.error(f"🚨 안전 잠금: {st.session_state.safety_state.locked_reason}")
        st.stop()

    chat_url = get_secret("OPENAI_CHAT_URL", CHAT_URL_DEFAULT)
    emb_url = get_secret("OPENAI_EMBEDDING_URL", EMBEDDING_URL_DEFAULT)

    with st.spinner("관련 자료 검색 중..."):
        try:
            results = search_index(
                st.session_state.rag_index, query, api_key,
                top_k=top_k, model=emb_model, url=emb_url,
            )
        except Exception as e:
            st.error(f"검색 실패: {e}")
            st.stop()

    if not results:
        st.warning("관련 자료를 찾지 못했습니다.")
        st.stop()

    context = format_context_for_llm(results)
    user_msg = build_user_message(query, context)
    # RAG 컨텍스트 앞부분을 CoreAI 가이드라인 힌트로 사용
    guideline_hint = context[:400] if context else ""

    with st.spinner("두 관점으로 답변 생성 중..."):
        try:
            if (coreai_mode and COREAI_AVAILABLE
                    and st.session_state.get("coreai_trained")):
                engine = st.session_state.coreai_engine

                proc_r = run_guardrail_loop(
                    question=user_msg,
                    llm_fn=lambda p: call_chat(p, PROCEDURE_SYSTEM, chat_model, api_key, chat_url),
                    engine=engine, max_attempts=coreai_retry,
                    logp_thr=coreai_logp, guideline_hint=guideline_hint,
                )
                cav_r = run_guardrail_loop(
                    question=user_msg,
                    llm_fn=lambda p: call_chat(p, CAVEATS_SYSTEM, chat_model, api_key, chat_url),
                    engine=engine, max_attempts=coreai_retry,
                    logp_thr=coreai_logp, guideline_hint=guideline_hint,
                )
                proc = proc_r.answer
                cav  = cav_r.answer
                coreai_proc_status   = proc_r.status
                coreai_cav_status    = cav_r.status
                coreai_proc_attempts = proc_r.attempts
                coreai_cav_attempts  = cav_r.attempts
            else:
                proc = call_chat(user_msg, PROCEDURE_SYSTEM, chat_model, api_key, chat_url)
                cav  = call_chat(user_msg, CAVEATS_SYSTEM, chat_model, api_key, chat_url)
                coreai_proc_status = coreai_cav_status = None
                coreai_proc_attempts = coreai_cav_attempts = 0
        except Exception as e:
            st.error(f"LLM 호출 실패: {e}")
            st.stop()

    # mismatch
    try:
        mm = compute_response_mismatch(proc, cav, api_key, emb_model, emb_url)
    except Exception:
        mm = float("nan")

    verdict = "ALIGNED" if (not np.isnan(mm) and mm <= st.session_state.threshold) else "DIVERGENT"

    st.session_state.last_qa = {
        "query": query,
        "results": results,
        "procedure": proc,
        "caveats": cav,
        "mismatch": mm,
        "verdict": verdict,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "coreai_proc_status":   coreai_proc_status,
        "coreai_cav_status":    coreai_cav_status,
        "coreai_proc_attempts": coreai_proc_attempts,
        "coreai_cav_attempts":  coreai_cav_attempts,
    }

    conn.execute(
        "INSERT INTO queries (doc_set_hash, query, mismatch, verdict) VALUES (?,?,?,?)",
        (st.session_state.doc_set_hash, query,
         mm if not np.isnan(mm) else 0.0, verdict),
    )
    conn.commit()

    if safety_mode and SAFETY_AVAILABLE:
        safety_record(st.session_state.safety_state,
                       "PASS_SAFE" if verdict == "ALIGNED" else "MISMATCH_STEERED",
                       st.session_state.safety_cfg)


# =========================================================
# Step 3: 답변 표시
# =========================================================
qa = st.session_state.last_qa
if qa:
    st.divider()
    st.subheader("3단계: 답변")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📋 절차 안내")
        st.markdown(qa["procedure"])
    with col2:
        st.markdown("### ⚠️ 주의사항")
        st.markdown(qa["caveats"])

    # 일관성 안내
    mm = qa["mismatch"]
    if np.isnan(mm):
        st.warning("두 관점 차이 측정 실패")
    elif qa["verdict"] == "ALIGNED":
        st.info(f"📊 두 관점이 비교적 일치합니다 (mismatch {mm:.1f}%).")
    else:
        st.warning(
            f"📊 두 관점이 갈립니다 (mismatch {mm:.1f}%). "
            "매뉴얼이 모호하거나 자료 간 차이가 있을 수 있습니다. "
            "**행정실/교무부에 직접 확인하시는 것을 강력히 권장합니다.**"
        )

    # CoreAI 가드레일 판정
    if qa.get("coreai_proc_status"):
        st.markdown("---")
        st.markdown("### 🎯 CoreAI 가드레일 판정")
        icon_map = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}
        ca1, ca2 = st.columns(2)
        ps = qa["coreai_proc_status"]
        cs = qa["coreai_cav_status"]
        ca1.markdown(
            f"절차 안내: {icon_map.get(ps,'⬜')} **{ps}** "
            f"({qa['coreai_proc_attempts']}회 시도)"
        )
        ca2.markdown(
            f"주의사항: {icon_map.get(cs,'⬜')} **{cs}** "
            f"({qa['coreai_cav_attempts']}회 시도)"
        )
        if ps == "FATAL" or cs == "FATAL":
            st.error("🔴 CoreAI: 매뉴얼 도메인 이탈 — 답변 신뢰도 낮음. 원본 매뉴얼 직접 확인 필요")
        elif ps == "WARNING" or cs == "WARNING":
            st.warning("🟡 CoreAI: 경계 수준 — 매뉴얼 내용과 일부 다를 수 있음")
        else:
            st.success("✅ CoreAI: 매뉴얼 도메인 안 답변")

    # 검색된 출처
    with st.expander(f"🔍 검색된 자료 ({len(qa['results'])}개)", expanded=False):
        for i, (c, score) in enumerate(qa["results"], 1):
            st.markdown(f"**자료 {i}** — `{c.source_file}` p.{c.page} (유사도 {score:.3f})")
            st.text(c.text[:500] + ("..." if len(c.text) > 500 else ""))
            st.divider()


# =========================================================
# Safety lock
# =========================================================
if safety_mode and SAFETY_AVAILABLE and st.session_state.safety_state.locked:
    st.divider()
    st.markdown("## 🚨 안전 잠금 관리")
    s = st.session_state.safety_state
    cfg = st.session_state.safety_cfg
    st.error(f"잠금 사유: {s.locked_reason}")
    signer = st.text_input("관리자 ID", key="signer_in")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("서명 제출", use_container_width=True):
            if signer:
                r = safety_request_release(s, signer.strip(), cfg)
                if r.get("released"):
                    st.success(r["msg"]); time.sleep(0.5); st.rerun()
                elif r.get("ok"):
                    st.info(r["msg"]); st.rerun()
                else:
                    st.error(r["msg"])
    with c2:
        if st.button("강제 초기화", use_container_width=True):
            safety_reset(s); st.warning("초기화됨"); time.sleep(0.5); st.rerun()


# =========================================================
# 새로 시작
# =========================================================
st.divider()
if st.button("🔄 인덱스/세션 초기화"):
    st.session_state.rag_index = RagIndex()
    st.session_state.uploaded_files = []
    st.session_state.doc_set_hash = ""
    st.session_state.last_qa = None
    st.rerun()
