import os
import json
import time
import sqlite3
import hashlib
from functools import lru_cache
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

# 클라우드 모드용 (선택 기능) - 모듈이 없거나 실패해도 기본 모드는 동작해야 함
try:
    from cloud_features import all_features as cloud_all_features
    from cloud_features import FEATURE_ORDER as CLOUD_FEATURE_ORDER
    CLOUD_AVAILABLE = True
except Exception as _e:
    CLOUD_AVAILABLE = False
    _CLOUD_IMPORT_ERROR = str(_e)

# 안전 차단 모듈 (선택 기능)
try:
    from safety import (
        SafetyConfig, SafetyState,
        record_verdict as safety_record,
        request_release as safety_request_release,
        status as safety_status,
        reset_state as safety_reset,
    )
    SAFETY_AVAILABLE = True
except Exception as _e:
    SAFETY_AVAILABLE = False
    _SAFETY_IMPORT_ERROR = str(_e)

load_dotenv()

DB_PATH = "logic_ai_data.db"

st.set_page_config(
    page_title="Logic AI Trust Verification",
    layout="wide",
)

# -----------------------------
# Database
# -----------------------------
@st.cache_resource
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trajectory_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT,
            free_output TEXT,
            steered_output TEXT,
            output TEXT,
            mismatch_rate REAL,
            threshold REAL,
            c_value REAL,
            status TEXT,
            provider TEXT,
            model TEXT,
            raw_response TEXT
        )
        """
    )
    conn.commit()
    return conn

conn = get_connection()


def save_log(
    prompt: str,
    free_output: str,
    steered_output: str,
    final_output: str,
    mismatch_rate: float,
    threshold: float,
    status: str,
    provider: str,
    model: str,
    raw_response: Optional[Dict[str, Any]] = None,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO trajectory_logs
        (prompt, free_output, steered_output, output, mismatch_rate, threshold, c_value, status, provider, model, raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prompt,
            free_output,
            steered_output,
            final_output,
            mismatch_rate,
            threshold,
            threshold,
            status,
            provider,
            model,
            json.dumps(raw_response or {}, ensure_ascii=False),
        ),
    )
    conn.commit()


# -----------------------------
# API callers
# -----------------------------
def call_openai_compatible(
    prompt: str,
    system_prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    res = requests.post(api_url, headers=headers, json=payload, timeout=90)
    res.raise_for_status()
    data = res.json()
    text = data["choices"][0]["message"]["content"]
    return text.strip(), data


def call_custom_json_api(
    prompt: str,
    system_prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    res = requests.post(api_url, headers=headers, json=payload, timeout=90)
    res.raise_for_status()
    data = res.json()

    # 커스텀 API가 반환할 수 있는 여러 필드명을 허용
    text = (
        data.get("output")
        or data.get("answer")
        or data.get("text")
        or data.get("content")
        or data.get("response")
    )
    if not text:
        raise ValueError("Custom API 응답에서 output/answer/text/content/response 필드를 찾지 못했습니다.")
    return str(text).strip(), data


def mock_answer(prompt: str, mode: str) -> Tuple[str, Dict[str, Any]]:
    if mode == "free":
        return f"자유 추론 응답: '{prompt}'에 대해 일반적인 방향으로 답변을 생성했습니다.", {"mock": True}
    return f"제어 추론 응답: '{prompt}'에 대해 안전 검증 기준을 적용해 답변을 생성했습니다.", {"mock": True}


# -----------------------------
# Mismatch logic (OpenAI embedding 기반)
# -----------------------------
# 이전 버전의 'pseudo-embedding'(바이트 빈도 합)은 의미를 보지 못하고
# 문자 분포만 봤기 때문에 "안전하다 vs 안전하지 않다" 같은 의미 반전을
# 거의 0% 불일치로 판정하는 결정적 결함이 있었음. OpenAI 임베딩으로 교체.
EMBEDDING_URL_DEFAULT = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"


def _emb_cache_key(text: str, model: str) -> str:
    h = hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()
    return h


@lru_cache(maxsize=512)
def _cached_embedding(cache_key: str, text: str, model: str, url: str, api_key: str) -> Tuple[float, ...]:
    """단일 텍스트의 임베딩 벡터를 가져온다. lru_cache로 같은 텍스트 재호출 방지.
    캐시 키에 model이 포함되므로 모델이 바뀌면 자동으로 새로 호출된다."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {"model": model, "input": text}
    res = requests.post(url, headers=headers, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()
    vec = data["data"][0]["embedding"]
    return tuple(vec)  # lru_cache 호환을 위해 hashable


def get_embedding(
    text: str,
    api_key: str,
    model: str = EMBEDDING_MODEL_DEFAULT,
    url: str = EMBEDDING_URL_DEFAULT,
) -> np.ndarray:
    if not text.strip():
        raise ValueError("빈 텍스트는 임베딩할 수 없습니다.")
    if not api_key:
        raise ValueError("임베딩 API Key가 비어 있습니다.")
    key = _emb_cache_key(text, model)
    vec = _cached_embedding(key, text, model, url, api_key)
    return np.array(vec, dtype=float)


def _hash_fallback_vector(text: str, dim: int = 128) -> np.ndarray:
    """Mock 모드 등 임베딩 API를 쓸 수 없는 환경의 폴백.
    의미를 보지 못한다는 점을 분명히 알 수 있도록 별도 함수로 분리해 둠."""
    vec = np.zeros(dim, dtype=float)
    encoded = text.encode("utf-8", errors="ignore")
    for i, b in enumerate(encoded):
        vec[i % dim] += (b - 127.5) / 127.5
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def calculate_mismatch_embedding(
    free_output: str,
    steered_output: str,
    api_key: str,
    model: str,
    url: str,
) -> float:
    """OpenAI 임베딩 cosine 거리를 0~100 스케일로 환산.
    cosine similarity in [-1, 1] -> distance (1 - cs)/2 in [0, 1] -> * 100.
    이전 버전은 (1 - cs) * 100 으로 0~200% 범위가 나오던 표시 버그도 함께 수정."""
    v_free = get_embedding(free_output, api_key, model, url)
    v_steered = get_embedding(steered_output, api_key, model, url)
    denom = np.linalg.norm(v_free) * np.linalg.norm(v_steered)
    if denom == 0:
        return 100.0
    cs = float(np.clip(np.dot(v_free, v_steered) / denom, -1.0, 1.0))
    return float((1.0 - cs) / 2.0 * 100.0)


def calculate_mismatch_fallback(free_output: str, steered_output: str) -> float:
    """임베딩을 쓸 수 없을 때만 사용. 의미가 아닌 글자 분포 기반이며,
    의미 반전을 못 잡으므로 결과는 참고용일 뿐이라는 점을 호출부에서 명시한다."""
    v_free = _hash_fallback_vector(free_output)
    v_steered = _hash_fallback_vector(steered_output)
    denom = np.linalg.norm(v_free) * np.linalg.norm(v_steered)
    if denom == 0:
        return 100.0
    cs = float(np.clip(np.dot(v_free, v_steered) / denom, -1.0, 1.0))
    return float((1.0 - cs) / 2.0 * 100.0)


# =========================================================
# Cloud Mode (실험적) — N회 호출 → 임베딩 클라우드 → 위상 특성
# =========================================================
# 가설: 같은 질문을 N번 호출해 만든 N개 임베딩의 점 클라우드 위상 구조가,
#       단순 두 점 거리보다 일관성/환각 신호를 더 풍부하게 담는다.
#
# 솔직한 한계 (UI에도 명시):
#   - 비용이 N/2배로 늘어남 (N=5면 호출 2.5배)
#   - 시뮬레이션 실험(별도 평가 도구)에서 가설이 일단 살아남았지만,
#     진짜 LLM N회 호출 데이터로 검증은 아직 안 됨.
#   - 따라서 'EXPERIMENTAL' 라벨을 유지함.
#
# 통합 원칙:
#   - 기본 모드(두 회 호출)는 절대 건드리지 않음.
#   - 사용자가 토글을 켰을 때만 동작.
#   - 호출 실패 시 기본 모드로 자동 폴백.

def call_n_times(
    prompt: str,
    system_prompt: str,
    n: int,
    provider: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    """같은 (prompt, system) 쌍으로 N번 호출.
    temperature가 0이면 결과가 같아지므로 호출부에서 temperature > 0 보장 권장."""
    results = []
    for _ in range(n):
        if provider == "Mock":
            results.append(mock_answer(prompt, "free"))
        elif provider == "OpenAI-compatible":
            results.append(call_openai_compatible(
                prompt, system_prompt, api_url, api_key, model, temperature, max_tokens
            ))
        else:
            results.append(call_custom_json_api(
                prompt, system_prompt, api_url, api_key, model, temperature, max_tokens
            ))
    return results


def cloud_mismatch_from_texts(
    texts: List[str],
    api_key: str,
    emb_model: str,
    emb_url: str,
) -> Tuple[float, Dict[str, float]]:
    """N개 텍스트의 임베딩 클라우드 → 위상 특성 → 단일 mismatch 점수로 환원.

    환원 방식: 클라우드 평균 pairwise distance를 0~100으로 (기존 v2와 같은 스케일).
    다만 전체 위상 특성 dict도 함께 반환해 UI에서 보조 정보로 보여줌.
    """
    if not CLOUD_AVAILABLE:
        raise RuntimeError(f"cloud_features 모듈을 불러올 수 없습니다: {_CLOUD_IMPORT_ERROR}")
    if len(texts) < 2:
        raise ValueError("클라우드 모드는 N>=2 응답이 필요합니다.")

    # 각 텍스트를 임베딩 (lru_cache로 중복 호출 절감)
    vectors = []
    for t in texts:
        vectors.append(get_embedding(t, api_key, emb_model, emb_url))
    embeddings = np.array(vectors, dtype=float)

    # 클라우드 특성 (gudhi 없으면 topo_*는 NaN으로 들어옴, 표시에서 제외)
    feats = cloud_all_features(embeddings, include_topology=True)
    # NaN 제거(표시용)
    feats_clean = {k: v for k, v in feats.items() if not (isinstance(v, float) and np.isnan(v))}

    # 단일 점수로 환원: basic_mean을 0~100 스케일로
    # (실험적이므로 단순 환원만; 학습된 융합은 별도 평가 도구에서)
    mismatch = float(feats.get("basic_mean", 0.0)) * 100.0
    return mismatch, feats_clean


# -----------------------------
# UI state
# -----------------------------
if "threshold" not in st.session_state:
    # 새 스케일: cosine distance를 [0, 100]으로 정규화한 값.
    # 0 = 완전 동일, 100 = 정반대. OpenAI 임베딩에서 무관한 문장 쌍은
    # 보통 30~50 사이에 분포하므로 보수적으로 15를 기본값으로 둠.
    # (이전 버전 기본 30은 0~200 스케일 가정이라 더 이상 맞지 않음)
    st.session_state.threshold = 15.0

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Safety 누적 상태 (안전 차단 옵션 — Grav Prison에서 추출한 패턴)
if SAFETY_AVAILABLE:
    if "safety_state" not in st.session_state:
        st.session_state.safety_state = SafetyState()
    if "safety_cfg" not in st.session_state:
        st.session_state.safety_cfg = SafetyConfig()

st.title("🛡️ Logic AI: 실시간 불일치율 기반 신뢰도 검증 시스템")
st.caption("API 호출 기반 free answer / steered answer 비교 + 불일치율 판정 + SQLite 피드백 누적")
with st.expander("이 도구의 한계 (꼭 한 번 읽어주세요)", expanded=False):
    st.markdown(
        """
- 이 도구는 **같은 질문에 대한 두 응답의 의미 유사도**를 OpenAI 임베딩으로 재는 LLM consistency 체크입니다.
- 임베딩으로 측정 방식을 교체했지만, **임계값은 여전히 휴리스틱**입니다.
  의료·법률·금융 같은 고위험 도메인에서 이 판정을 **단독 근거로 사용하지 마세요.**
- 두 응답이 의미상 비슷해도 둘 다 같이 틀릴 수 있고(공통 환각),
  의미상 달라도 둘 다 옳을 수 있습니다(서로 다른 측면 강조).
- 임베딩 API 키가 없거나 호출이 실패하면 글자 분포 폴백으로 동작하며,
  이 경우는 **의미 반전을 감지하지 못합니다.** 결과 패널에 그 사실이 표시됩니다.
        """
    )

with st.sidebar:
    st.write(f"### ⚙️ 현재 허용 임계값 τ: `{st.session_state.threshold:.1f}%`")
    st.session_state.threshold = st.slider(
        "허용 임계값 직접 조정",
        min_value=1.0,
        max_value=80.0,
        value=float(st.session_state.threshold),
        step=1.0,
        help="cosine distance를 0~100으로 정규화한 값. 0=동일, 100=정반대. "
             "두 응답의 임베딩 차이가 이 값 이하면 자유 응답을 채택합니다.",
    )

    st.write("---")
    provider = st.selectbox(
        "API Provider",
        ["Mock", "OpenAI-compatible", "Custom JSON API"],
        index=0,
    )

    default_openai_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    api_url = st.text_input("Chat API URL", value=default_openai_url)
    api_key = st.text_input("API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)
    max_tokens = st.slider("Max tokens", 128, 4096, 512, 128)

    st.write("---")
    st.markdown("**🧭 임베딩(불일치 측정용)**")
    embedding_url = st.text_input(
        "Embedding API URL",
        value=os.getenv("OPENAI_EMBEDDING_URL", EMBEDDING_URL_DEFAULT),
    )
    embedding_model = st.text_input(
        "Embedding Model",
        value=os.getenv("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT),
    )
    embedding_key = st.text_input(
        "Embedding API Key (비우면 Chat API Key를 재사용)",
        value=os.getenv("OPENAI_EMBEDDING_KEY", ""),
        type="password",
    )

    st.write("---")
    st.caption("OpenAI 공식 URL: https://api.openai.com/v1/chat/completions")

    # ----- Cloud Mode (실험적) -----
    st.write("---")
    st.markdown("### 🧪 클라우드 모드 (실험적)")
    if not CLOUD_AVAILABLE:
        st.caption(f"⚠️ cloud_features 모듈 로드 실패. 비활성화됨.")
        cloud_mode = False
    else:
        cloud_mode = st.checkbox(
            "N회 호출 클라우드 모드 사용",
            value=False,
            help="같은 system prompt로 N번 호출해 만든 임베딩 클라우드의 위상 구조로 "
                 "일관성을 잰다. 비용이 N/2배 늘어남.",
        )
        if cloud_mode:
            st.warning(
                "이 모드는 실험적입니다. 시뮬레이션 평가에선 가설이 살아남았지만, "
                "진짜 LLM N회 호출 데이터로는 아직 검증되지 않았습니다."
            )
            cloud_n = st.slider("N (호출 횟수)", 3, 8, 5, 1,
                                help="3~8 권장. 클수록 신호가 풍부하지만 비용도 증가.")
            cloud_temperature = st.slider(
                "Cloud Temperature", 0.3, 1.5, 0.8, 0.1,
                help="temperature가 너무 낮으면 N번 호출이 모두 같아져 신호가 없음. "
                     "0.7 이상 권장.",
            )
        else:
            cloud_n = 5
            cloud_temperature = 0.8

    # ----- Safety Accumulator (안전 누적 차단) -----
    st.write("---")
    st.markdown("### 🚨 안전 누적 차단")
    if not SAFETY_AVAILABLE:
        st.caption(f"⚠️ safety 모듈 로드 실패. 비활성화됨.")
        safety_mode = False
    else:
        safety_mode = st.checkbox(
            "누적 INCONSISTENT 감시 사용",
            value=False,
            help="최근 N회 호출 중 INCONSISTENT 비율이 임계값을 넘으면 "
                 "시스템을 자동으로 잠그고 관리자 다중 승인이 있어야 해제.",
        )
        if safety_mode:
            cfg = st.session_state.safety_cfg
            cfg.window_size = st.slider("감시 window (최근 N회)", 5, 50, cfg.window_size, 1)
            cfg.inconsistent_threshold = st.slider(
                "INCONSISTENT 임계 비율", 0.2, 0.9, cfg.inconsistent_threshold, 0.05,
                help="이 비율을 넘으면 자동 잠금. 0.5 = 절반 이상",
            )
            cfg.min_samples = st.slider(
                "최소 표본 수", 3, 20, cfg.min_samples, 1,
                help="이 수보다 적으면 잠금 판정 안 함 (작은 표본 보호)",
            )
            cfg.required_signatures = st.slider(
                "해제 필요 승인 수", 1, 5, cfg.required_signatures, 1,
            )
            signers_text = st.text_input(
                "승인 가능 ID (쉼표 구분)",
                value=",".join(cfg.authorized_signers),
            )
            cfg.authorized_signers = [s.strip() for s in signers_text.split(",") if s.strip()]

prompt = st.text_input("질문을 입력하세요:", placeholder="예: 대한민국의 수도는 어디인가요?")

free_system_prompt = st.text_area(
    "Free Answer System Prompt",
    value="너는 일반적인 답변 모델이다. 사용자의 질문에 자연스럽고 직접적으로 답하라.",
    height=90,
)

steered_system_prompt = st.text_area(
    "Steered Answer System Prompt",
    value=(
        "너는 Logic AI 검증 시스템의 제어 응답 모델이다. "
        "답변 전 사실성, 안전성, 논리 일관성을 우선 검토하고, "
        "불확실하면 단정하지 말고 보수적으로 답하라."
    ),
    height=110,
)

run_col, reset_col = st.columns([1, 1])

with run_col:
    run = st.button("검증 추론 실행", type="primary", use_container_width=True)

with reset_col:
    if st.button("세션 결과 초기화", use_container_width=True):
        st.session_state.last_result = None
        st.rerun()

if run and prompt:
    # ============== Safety 잠금 가드 ==============
    if safety_mode and SAFETY_AVAILABLE and st.session_state.safety_state.locked:
        st.error(
            "🚨 시스템이 안전 잠금 상태입니다. 추론을 진행할 수 없습니다.\n\n"
            f"잠금 사유: {st.session_state.safety_state.locked_reason}\n\n"
            "아래 '안전 잠금 관리' 패널에서 관리자 승인을 받아 해제하세요."
        )
        st.stop()

    with st.spinner("API 호출 및 불일치율 계산 중..."):
        try:
            # =============================================================
            # 분기: 기본 모드 (2회 호출) vs 클라우드 모드 (N회 호출)
            # =============================================================
            cloud_features_dict = None       # 클라우드 모드일 때만 채워짐
            cloud_responses: List[str] = []  # 클라우드 모드일 때만 채워짐
            using_cloud = cloud_mode and CLOUD_AVAILABLE

            if using_cloud:
                # ----- 클라우드 모드 -----
                if provider == "Mock":
                    # Mock은 같은 응답을 반환하므로 노이즈를 살짝 줘서 N개 흉내
                    cloud_responses = [
                        mock_answer(prompt, "free")[0] + f" (variant {i+1})"
                        for i in range(cloud_n)
                    ]
                    raw_free = {"mock": True, "variants": cloud_n}
                elif provider == "OpenAI-compatible":
                    if not api_key:
                        raise ValueError("OpenAI-compatible 모드에서는 API Key가 필요합니다.")
                    pairs = call_n_times(
                        prompt, free_system_prompt, cloud_n,
                        provider, api_url, api_key, model,
                        cloud_temperature, max_tokens,
                    )
                    cloud_responses = [p[0] for p in pairs]
                    raw_free = {"variants": [p[1] for p in pairs]}
                else:
                    pairs = call_n_times(
                        prompt, free_system_prompt, cloud_n,
                        provider, api_url, api_key, model,
                        cloud_temperature, max_tokens,
                    )
                    cloud_responses = [p[0] for p in pairs]
                    raw_free = {"variants": [p[1] for p in pairs]}

                # 표시용 free/steered는 첫·마지막 응답으로 (다른 두 변주)
                free_output = cloud_responses[0]
                steered_output = cloud_responses[-1]
                raw_steered = {"note": "클라우드 모드: 별도 steered 호출 없음"}

            else:
                # ----- 기본 모드 (기존 v2 그대로) -----
                if provider == "Mock":
                    free_output, raw_free = mock_answer(prompt, "free")
                    steered_output, raw_steered = mock_answer(prompt, "steered")
                elif provider == "OpenAI-compatible":
                    if not api_key:
                        raise ValueError("OpenAI-compatible 모드에서는 API Key가 필요합니다.")
                    free_output, raw_free = call_openai_compatible(
                        prompt, free_system_prompt, api_url, api_key, model, temperature, max_tokens
                    )
                    steered_output, raw_steered = call_openai_compatible(
                        prompt, steered_system_prompt, api_url, api_key, model, temperature, max_tokens
                    )
                else:
                    free_output, raw_free = call_custom_json_api(
                        prompt, free_system_prompt, api_url, api_key, model, temperature, max_tokens
                    )
                    steered_output, raw_steered = call_custom_json_api(
                        prompt, steered_system_prompt, api_url, api_key, model, temperature, max_tokens
                    )

            # 불일치 계산: 클라우드 모드면 클라우드 위상 점수, 아니면 v2 그대로
            mismatch_method = "embedding"
            mismatch_warning: Optional[str] = None
            effective_emb_key = embedding_key or api_key

            if using_cloud:
                # 클라우드 mismatch
                try:
                    if not effective_emb_key:
                        raise RuntimeError("임베딩 키 필요")
                    mismatch_rate, cloud_features_dict = cloud_mismatch_from_texts(
                        cloud_responses,
                        api_key=effective_emb_key,
                        emb_model=embedding_model,
                        emb_url=embedding_url,
                    )
                    mismatch_method = f"cloud_embedding(N={cloud_n})"
                except Exception as cloud_err:
                    # 클라우드 실패 시 기존 두 점 폴백
                    mismatch_rate = calculate_mismatch_fallback(free_output, steered_output)
                    mismatch_method = "fallback"
                    mismatch_warning = (
                        f"클라우드 모드 계산 실패로 폴백 사용: {cloud_err}"
                    )
            else:
                # 기본 모드: 기존 v2 흐름
                try:
                    if provider == "Mock" or not effective_emb_key:
                        raise RuntimeError("fallback")
                    mismatch_rate = calculate_mismatch_embedding(
                        free_output, steered_output,
                        api_key=effective_emb_key,
                        model=embedding_model,
                        url=embedding_url,
                    )
                except Exception as emb_err:
                    mismatch_rate = calculate_mismatch_fallback(free_output, steered_output)
                    mismatch_method = "fallback"
                    if provider == "Mock":
                        mismatch_warning = "Mock 모드: 의미 기반이 아닌 글자 분포 폴백으로 계산되었습니다. 판정은 참고용입니다."
                    elif not effective_emb_key:
                        mismatch_warning = "임베딩 API Key가 없어 글자 분포 폴백으로 계산되었습니다. 의미 반전을 잡지 못하니 임베딩 키를 설정하세요."
                    else:
                        mismatch_warning = f"임베딩 호출 실패로 폴백을 사용했습니다: {emb_err}"

            threshold = float(st.session_state.threshold)

            if mismatch_rate <= threshold:
                status = "PASS_SAFE"
                final_output = free_output
            else:
                status = "MISMATCH_STEERED"
                final_output = steered_output

            st.session_state.last_result = {
                "prompt": prompt,
                "free_output": free_output,
                "steered_output": steered_output,
                "final_output": final_output,
                "mismatch_rate": mismatch_rate,
                "threshold": threshold,
                "status": status,
                "provider": provider,
                "model": model,
                "raw_response": {"free": raw_free, "steered": raw_steered},
                "mismatch_method": mismatch_method,
                "mismatch_warning": mismatch_warning,
                "cloud_features": cloud_features_dict,
                "cloud_responses": cloud_responses if using_cloud else None,
            }

            # ============== Safety 누적 기록 ==============
            if safety_mode and SAFETY_AVAILABLE:
                safety_event = safety_record(
                    st.session_state.safety_state,
                    status,
                    st.session_state.safety_cfg,
                )
                if safety_event.get("just_triggered"):
                    st.error(
                        "🚨 안전 누적 차단이 발동되었습니다. "
                        f"최근 {safety_event['window_size']}회 중 "
                        f"{safety_event['inconsistent_count']}회가 INCONSISTENT "
                        f"(비율 {safety_event['ratio']:.0%}). "
                        "다음 호출부터 차단됩니다. 관리자 승인으로 해제하세요."
                    )

            save_log(
                prompt=prompt,
                free_output=free_output,
                steered_output=steered_output,
                final_output=final_output,
                mismatch_rate=mismatch_rate,
                threshold=threshold,
                status=status,
                provider=provider,
                model=model,
                raw_response={"free": raw_free, "steered": raw_steered},
            )

        except Exception as e:
            st.error(f"API 호출 또는 검증 중 오류가 발생했습니다: {e}")

result = st.session_state.last_result

if result:
    st.write("---")
    left, right = st.columns([1, 1])

    with left:
        st.metric("분석된 불일치율", f"{result['mismatch_rate']:.2f}%")
        st.metric("현재 허용 임계값 τ", f"{result['threshold']:.1f}%")
        method = result.get("mismatch_method", "embedding")
        if method == "embedding":
            st.caption("측정: OpenAI 임베딩 cosine 거리 (0=동일, 100=정반대)")
        else:
            st.caption("측정: 글자 분포 폴백 (의미 반전 감지 불가, 참고용)")
        if result.get("mismatch_warning"):
            st.warning(result["mismatch_warning"])

        if result["status"] == "PASS_SAFE":
            st.success("✅ [판별: 정확] 자유 추론 방향이 허용 신뢰 구간 안에 있습니다.")
        else:
            st.warning("⚠️ [판별: 환각/이탈 위험] 자유 응답과 제어 응답의 차이가 커서 제어 응답을 최종 채택했습니다.")

        # Safety 누적 상태 표시
        if safety_mode and SAFETY_AVAILABLE:
            st_info = safety_status(st.session_state.safety_state, st.session_state.safety_cfg)
            pct = int(st_info["inconsistent_ratio"] * 100)
            thr_pct = int(st_info["threshold"] * 100)
            st.markdown("---")
            st.markdown("**🚨 안전 누적 상태**")
            if st_info["locked"]:
                st.error(f"🔒 잠김 ({st_info['inconsistent_count']}/{st_info['window_size']}, {pct}% ≥ {thr_pct}%)")
            else:
                st.caption(
                    f"INCONSISTENT 비율 {pct}% / 임계 {thr_pct}% "
                    f"(표본 {st_info['window_size']}/{st_info['window_capacity']})"
                )
                st.progress(min(pct / max(thr_pct, 1), 1.0))

    with right:
        st.info(f"**최종 답변**\n\n{result['final_output']}")

    # 클라우드 모드면 추가 탭, 아니면 기본 3탭
    if result.get("cloud_features") is not None:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Free Answer", "Steered Answer", "Raw API Response", "🧪 Cloud 특성"]
        )
    else:
        tab1, tab2, tab3 = st.tabs(["Free Answer", "Steered Answer", "Raw API Response"])
        tab4 = None

    with tab1:
        st.code(result["free_output"], language="markdown")
    with tab2:
        st.code(result["steered_output"], language="markdown")
    with tab3:
        st.json(result["raw_response"])

    if tab4 is not None:
        with tab4:
            st.caption(
                "클라우드 모드는 실험적입니다. 이 특성들의 의미는 README의 "
                "cloud_features 설명을 참고하세요. 임계값과의 직접 매핑은 검증되지 않았습니다."
            )
            cf = result["cloud_features"]
            # 특성 표
            import pandas as pd
            df_cf = pd.DataFrame(
                [{"feature": k, "value": round(v, 4)} for k, v in cf.items()]
            )
            st.dataframe(df_cf, use_container_width=True, hide_index=True)
            # N개 응답 미리보기
            if result.get("cloud_responses"):
                st.markdown("**N개 응답 미리보기**")
                for i, r in enumerate(result["cloud_responses"], 1):
                    st.text(f"[{i}] {r[:200]}{'...' if len(r) > 200 else ''}")

    st.write("---")
    st.write("### 📌 이 답변은 괜찮았나요?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 괜찮다 (허용치 유지)", use_container_width=True):
            save_log(
                prompt=result["prompt"],
                free_output=result["free_output"],
                steered_output=result["steered_output"],
                final_output=result["final_output"],
                mismatch_rate=result["mismatch_rate"],
                threshold=float(st.session_state.threshold),
                status="USER_APPROVED",
                provider=result["provider"],
                model=result["model"],
                raw_response=result["raw_response"],
            )
            st.success("피드백이 반영되었습니다. 임계값을 유지합니다.")

    with col2:
        if st.button("👎 아니다 (허용도 엄격하게 재조정)", use_container_width=True):
            st.session_state.threshold = max(5.0, float(st.session_state.threshold) - 5.0)
            save_log(
                prompt=result["prompt"],
                free_output=result["free_output"],
                steered_output=result["steered_output"],
                final_output=result["final_output"],
                mismatch_rate=result["mismatch_rate"],
                threshold=float(st.session_state.threshold),
                status="USER_REJECTED_THRESHOLD_DOWN",
                provider=result["provider"],
                model=result["model"],
                raw_response=result["raw_response"],
            )
            st.warning("시스템이 더 엄격해졌습니다. 다음 추론부터 임계값을 낮춰 판정합니다.")
            time.sleep(0.5)
            st.rerun()

# =========================================================
# 🚨 안전 잠금 관리 패널 (잠긴 상태에서만 표시)
# =========================================================
if safety_mode and SAFETY_AVAILABLE and st.session_state.safety_state.locked:
    st.write("---")
    st.markdown("## 🚨 안전 잠금 관리")
    cfg = st.session_state.safety_cfg
    s = st.session_state.safety_state
    st.error(f"잠김 사유: {s.locked_reason}")
    st.caption(
        f"승인 필요: **{cfg.required_signatures}명** "
        f"({', '.join(cfg.authorized_signers)} 중)"
    )
    st.caption(f"현재 서명: {s.pending_signatures or '없음'}")

    col_sig, col_reset = st.columns([2, 1])
    with col_sig:
        signer_input = st.text_input("관리자 ID 입력", key="safety_signer_input",
                                      placeholder="예: admin1")
        if st.button("✍️ 해제 서명 제출", use_container_width=True):
            if signer_input:
                result_sig = safety_request_release(s, signer_input.strip(), cfg)
                if result_sig.get("released"):
                    st.success(result_sig["msg"])
                    time.sleep(0.5)
                    st.rerun()
                elif result_sig.get("ok"):
                    st.info(result_sig["msg"])
                    st.rerun()
                else:
                    st.error(result_sig["msg"])

    with col_reset:
        st.caption("⚠️ 비상시")
        if st.button("강제 초기화 (모든 누적 삭제)", use_container_width=True):
            safety_reset(s)
            st.warning("안전 누적 상태가 강제 초기화되었습니다.")
            time.sleep(0.5)
            st.rerun()

    st.caption(
        "정직한 한계: 이 잠금 임계값(window/ratio/min_samples)도 휴리스틱입니다. "
        "도메인별로 보정하세요."
    )

with st.expander("최근 로그 보기"):
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT created_at, prompt, mismatch_rate, threshold, status, provider, model
        FROM trajectory_logs
        ORDER BY id DESC
        LIMIT 20
        """
    ).fetchall()
    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            column_config={
                0: "created_at",
                1: "prompt",
                2: "mismatch_rate",
                3: "threshold",
                4: "status",
                5: "provider",
                6: "model",
            },
        )
    else:
        st.caption("아직 저장된 로그가 없습니다.")
