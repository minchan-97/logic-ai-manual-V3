"""
safety.py — 누적 불일치 감시 및 자동 차단 (Logic AI v4)
==========================================================

Grav Prison 패키지에서 추출한 패턴:
  1) 최근 N회 호출의 결과를 누적 모니터
  2) INCONSISTENT 비율이 임계값을 넘으면 시스템을 LOCKED 상태로
  3) 관리자 코드 K개 이상 입력해야 해제 (다중 승인)

추출 과정에서 버린 것:
  - 물리/중력 비유 (gravitational_potential, horizon_factor)
  - 점진적 slowdown (LLM API에선 의미 없음)
  - hard_kill / SIGTERM (Streamlit 환경에 부적합)
  - 별도 모니터 스레드 (호출 시점마다 동기적으로 체크하면 충분)

설계 원칙:
  - 외부 의존성 없음 (sqlite, streamlit 모두 호출부에서 처리)
  - 상태는 명시적으로 dict로 주고받기 → session_state에 쉽게 매핑
  - "잠긴 시스템을 해제하는 일"은 신중해야 하므로 다중 승인이 기본
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


@dataclass
class SafetyConfig:
    window_size: int = 10            # 최근 N회 추적
    inconsistent_threshold: float = 0.5   # 비율 임계값 (0.5 = 절반 이상)
    min_samples: int = 5             # 최소 N회 이후부터 판정 (작은 표본 보호)
    required_signatures: int = 2     # 해제 필요 승인 수
    authorized_signers: List[str] = field(default_factory=lambda: ["admin1", "admin2", "admin3"])


@dataclass
class SafetyState:
    """순수 데이터. Streamlit session_state에 그대로 매핑 가능."""
    history: List[str] = field(default_factory=list)  # "PASS" / "INCONSISTENT"
    locked: bool = False
    locked_reason: Optional[str] = None
    locked_at_ratio: Optional[float] = None
    pending_signatures: List[str] = field(default_factory=list)


def record_verdict(state: SafetyState, verdict: str, cfg: SafetyConfig) -> dict:
    """판정 1건 기록. 잠금 트리거 시 lock=True로 변경.
    return: {'locked': bool, 'just_triggered': bool, 'ratio': float, 'window': list}"""
    if verdict not in ("PASS_SAFE", "MISMATCH_STEERED", "PASS", "INCONSISTENT"):
        # 호출부 명칭 차이 흡수
        verdict = "INCONSISTENT" if "MISMATCH" in verdict or "INCONS" in verdict else "PASS"
    # 정규화
    v = "INCONSISTENT" if "INCONS" in verdict or "MISMATCH" in verdict else "PASS"

    state.history.append(v)
    # window 유지
    if len(state.history) > cfg.window_size:
        state.history = state.history[-cfg.window_size:]

    just_triggered = False
    window = state.history
    n = len(window)
    inconsistent = sum(1 for x in window if x == "INCONSISTENT")
    ratio = inconsistent / max(n, 1)

    if (not state.locked) and n >= cfg.min_samples and ratio >= cfg.inconsistent_threshold:
        state.locked = True
        state.locked_at_ratio = ratio
        state.locked_reason = (
            f"최근 {n}회 호출 중 {inconsistent}회가 INCONSISTENT "
            f"(비율 {ratio:.0%}, 임계값 {cfg.inconsistent_threshold:.0%})."
        )
        just_triggered = True

    return {
        "locked": state.locked,
        "just_triggered": just_triggered,
        "ratio": ratio,
        "window_size": n,
        "inconsistent_count": inconsistent,
    }


def request_release(state: SafetyState, signer_id: str, cfg: SafetyConfig) -> dict:
    """해제 요청 1건 추가. 충분히 모이면 자동 해제."""
    if not state.locked:
        return {"ok": False, "msg": "잠긴 상태가 아닙니다."}
    if signer_id not in cfg.authorized_signers:
        return {"ok": False, "msg": f"인증되지 않은 ID: {signer_id}"}
    if signer_id in state.pending_signatures:
        return {"ok": False, "msg": f"이미 서명한 ID: {signer_id}"}

    state.pending_signatures.append(signer_id)
    have = len(state.pending_signatures)
    need = cfg.required_signatures

    if have >= need:
        # 해제
        state.locked = False
        state.locked_reason = None
        state.locked_at_ratio = None
        state.pending_signatures = []
        # history는 유지 (다시 누적 시작 시 천천히 회복하게)
        # 다만 너무 빨리 재트리거 막기 위해 window 절반은 비움
        keep = len(state.history) // 2
        state.history = state.history[:keep]
        return {"ok": True, "msg": "해제 완료. 누적 기록 일부 초기화됨.", "released": True}
    return {
        "ok": True,
        "msg": f"서명 추가됨 ({have}/{need}). 추가 승인 대기 중.",
        "released": False,
        "have": have, "need": need,
    }


def reset_state(state: SafetyState) -> None:
    """관리자가 수동으로 전체 상태를 비울 때."""
    state.history = []
    state.locked = False
    state.locked_reason = None
    state.locked_at_ratio = None
    state.pending_signatures = []


def status(state: SafetyState, cfg: SafetyConfig) -> dict:
    n = len(state.history)
    inc = sum(1 for x in state.history if x == "INCONSISTENT")
    ratio = inc / max(n, 1)
    return {
        "locked": state.locked,
        "locked_reason": state.locked_reason,
        "window_size": n,
        "window_capacity": cfg.window_size,
        "inconsistent_count": inc,
        "inconsistent_ratio": ratio,
        "threshold": cfg.inconsistent_threshold,
        "min_samples": cfg.min_samples,
        "pending_signatures": list(state.pending_signatures),
        "signatures_needed": cfg.required_signatures,
        "history_tail": state.history[-10:],
    }
