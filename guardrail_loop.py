"""
guardrail_loop.py — CoreAI 재생성 루프
=======================================
LLM 답변 생성 → 가드레일 평가 → 이탈 시 재생성
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class AttemptResult:
    attempt: int
    answer: str
    status: str
    avg_logp: float
    mismatch: float
    elapsed_ms: float


@dataclass
class CoreAIResponse:
    answer: str
    status: str
    attempts: int
    history: list = field(default_factory=list)
    total_ms: float = 0.0
    final_logp: float = 0.0


def _build_retry_prompt(original_question: str, bad_answer: str,
                         status: str, guideline_hint: str = "") -> str:
    """이탈 답변에 대한 재생성 프롬프트 생성."""
    reason = {
        "WARNING": "일부 표현이 가이드라인 범위를 벗어났습니다",
        "CRITICAL": "답변의 맥락이 가이드라인과 맞지 않습니다",
        "FATAL":    "답변이 가이드라인 도메인을 완전히 벗어났습니다",
    }.get(status, "가이드라인 범위를 벗어났습니다")

    prompt = f"""이전 답변이 가이드라인 검증에서 실패했습니다.
사유: {reason}

{f'가이드라인 핵심 내용: {guideline_hint}' if guideline_hint else ''}

질문: {original_question}

위 질문에 대해 가이드라인 범위 안에서만 답변해주세요.
가이드라인에 없는 내용은 "가이드라인에 포함되지 않은 내용입니다"라고 답하세요."""
    return prompt


def run_guardrail_loop(
    question: str,
    llm_fn,           # fn(prompt: str) -> str
    engine,           # NeuralMarkovEngine
    max_attempts: int = 3,
    logp_thr: float = -11.5,
    mis_thr: float = 0.55,
    guideline_hint: str = "",
) -> CoreAIResponse:
    """
    가드레일 루프 실행.

    1. LLM 답변 생성
    2. 가드레일 평가
    3. PASS → 즉시 반환
    4. WARNING/FATAL → 재생성 프롬프트로 재시도
    5. max_attempts 초과 → 마지막 답변 반환
    """
    t_total = time.perf_counter()
    history = []
    current_prompt = question

    for attempt in range(1, max_attempts + 1):
        # LLM 호출
        answer = llm_fn(current_prompt)

        # 가드레일 평가
        if engine.is_trained:
            result = engine.evaluate(answer, logp_thr=logp_thr, mis_thr=mis_thr)
            status    = result["status"]
            avg_logp  = result["avg_logp"]
            mismatch  = result["mismatch"]
            elapsed   = result["elapsed_ms"]
        else:
            status = "PASS"; avg_logp = 0.0; mismatch = 0.0; elapsed = 0.0

        history.append(AttemptResult(
            attempt=attempt, answer=answer, status=status,
            avg_logp=avg_logp, mismatch=mismatch, elapsed_ms=elapsed,
        ))

        # PASS면 즉시 반환
        if status == "PASS":
            return CoreAIResponse(
                answer=answer, status=status, attempts=attempt,
                history=history,
                total_ms=(time.perf_counter() - t_total) * 1000,
                final_logp=avg_logp,
            )

        # 마지막 시도면 그대로 반환
        if attempt == max_attempts:
            break

        # 재생성 프롬프트 구성
        current_prompt = _build_retry_prompt(
            question, answer, status, guideline_hint
        )

    last = history[-1]
    return CoreAIResponse(
        answer=last.answer, status=last.status, attempts=max_attempts,
        history=history,
        total_ms=(time.perf_counter() - t_total) * 1000,
        final_logp=last.avg_logp,
    )
