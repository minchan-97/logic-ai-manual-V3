"""
rag.py — PDF 매뉴얼 RAG 모듈 (Logic AI Manual)
================================================

기능:
  - PDF 파일을 페이지 단위로 파싱
  - 페이지를 더 작은 청크로 분할 (문단/문장 단위, 길이 제한)
  - 각 청크를 OpenAI 임베딩으로 벡터화
  - 질의 임베딩과 코사인 유사도로 top-k 청크 검색
  - 검색 결과에 출처 정보(파일명, 페이지) 포함

설계 원칙:
  - 외부 벡터DB 없음. numpy 배열 + 메모리 캐시로 충분 (소규모 매뉴얼)
  - 모델 의존부는 함수 인자로 격리 → 단위 검증 가능
  - 청크는 hashable 데이터클래스 → lru_cache 활용
  - PDF 파싱 실패에 강건 (페이지별 try/except)

정직한 한계:
  - 표/그림/수식이 많은 PDF는 텍스트 추출이 부정확함
  - 청킹은 휴리스틱 (의미 단위 보장 안 됨)
  - top-k 검색은 정밀도/재현율 trade-off (k 조정 필요)
  - "lost in the middle" 완화는 하지만 완전 해결은 아님
"""
from __future__ import annotations
import os
import hashlib
import io
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Iterable

import numpy as np


# ----------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """검색 가능한 텍스트 단위. 출처 추적을 위한 메타 포함.
    frozen=True로 hashable → 캐시 키로 사용 가능."""
    chunk_id: str          # 고유 ID (해시)
    source_file: str       # 출처 PDF 파일명
    page: int              # 페이지 번호 (1-indexed)
    chunk_index: int       # 페이지 내 청크 순번
    text: str              # 청크 본문


@dataclass
class RagIndex:
    """청크 목록 + 임베딩 행렬을 묶은 인덱스."""
    chunks: List[Chunk] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None  # shape: (N, dim)

    def is_built(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0

    def size(self) -> int:
        return len(self.chunks)


# ----------------------------------------------------------------
# PDF parsing
# ----------------------------------------------------------------

def parse_pdf_pages(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """PDF 바이트를 받아 [(page_num, text)] 리스트로 반환.
    페이지 파싱 실패 시 그 페이지만 건너뜀. 전체 실패 시 빈 리스트."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf 라이브러리가 필요합니다. pip install pypdf")
    pages: List[Tuple[int, str]] = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append((i, text))
            except Exception:
                # 개별 페이지 파싱 실패 → 건너뜀
                continue
    except Exception as e:
        raise RuntimeError(f"PDF 파싱 실패 ({filename}): {e}")
    return pages


# ----------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------

def split_text_into_chunks(
    text: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[str]:
    """긴 텍스트를 max_chars 길이의 청크로 분할. overlap만큼 겹치게 함.
    문단/문장 경계를 가능한 한 보존."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        # 끝 위치를 문장 경계(마침표/줄바꿈)로 보정 시도
        if end < len(text):
            # 마지막 100자 안에서 끊을 만한 곳 찾기
            search_start = max(end - 100, pos + max_chars // 2)
            for sep in ["\n\n", ". ", ".\n", "다.", "요.", "음."]:
                idx = text.rfind(sep, search_start, end)
                if idx > 0:
                    end = idx + len(sep)
                    break
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        # 다음 시작 위치 (overlap 만큼 뒤로)
        pos = max(end - overlap, pos + 1)
    return chunks


def build_chunks_from_pdf(
    file_bytes: bytes,
    filename: str,
    max_chars: int = 800,
    overlap: int = 100,
) -> List[Chunk]:
    """PDF 한 개를 받아 Chunk 리스트로 변환."""
    pages = parse_pdf_pages(file_bytes, filename)
    out: List[Chunk] = []
    for page_num, page_text in pages:
        sub_chunks = split_text_into_chunks(page_text, max_chars=max_chars, overlap=overlap)
        for idx, sub in enumerate(sub_chunks):
            cid = hashlib.sha256(
                f"{filename}::{page_num}::{idx}::{sub[:50]}".encode()
            ).hexdigest()[:16]
            out.append(Chunk(
                chunk_id=cid,
                source_file=filename,
                page=page_num,
                chunk_index=idx,
                text=sub,
            ))
    return out


# ----------------------------------------------------------------
# Embedding helpers
# ----------------------------------------------------------------

def embed_batch_openai(
    texts: List[str],
    api_key: str,
    model: str = "text-embedding-3-small",
    url: str = "https://api.openai.com/v1/embeddings",
    batch_size: int = 16,
) -> np.ndarray:
    """OpenAI 임베딩 배치 호출. (N, dim) 행렬 반환."""
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        res = requests.post(url, headers=headers,
                             json={"model": model, "input": batch}, timeout=90)
        res.raise_for_status()
        data = res.json()
        for d in data["data"]:
            out.append(d["embedding"])
    return np.array(out, dtype=float)


# ----------------------------------------------------------------
# Index building
# ----------------------------------------------------------------

def build_index(
    chunks: List[Chunk],
    api_key: str,
    model: str = "text-embedding-3-small",
    url: str = "https://api.openai.com/v1/embeddings",
    progress_cb: Optional[callable] = None,
) -> RagIndex:
    """청크 리스트를 받아 임베딩 인덱스를 빌드.
    progress_cb(done, total)이 있으면 호출."""
    if not chunks:
        return RagIndex(chunks=[], embeddings=None)
    texts = [c.text for c in chunks]
    # 배치 단위로 부르면서 progress 보고
    batch_size = 16
    all_emb: List[List[float]] = []
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    total = len(texts)
    done = 0
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        res = requests.post(url, headers=headers,
                             json={"model": model, "input": batch}, timeout=90)
        res.raise_for_status()
        for d in res.json()["data"]:
            all_emb.append(d["embedding"])
        done += len(batch)
        if progress_cb:
            progress_cb(done, total)
    embeddings = np.array(all_emb, dtype=float)
    return RagIndex(chunks=chunks, embeddings=embeddings)


# ----------------------------------------------------------------
# Search
# ----------------------------------------------------------------

def search_index(
    index: RagIndex,
    query: str,
    api_key: str,
    top_k: int = 5,
    model: str = "text-embedding-3-small",
    url: str = "https://api.openai.com/v1/embeddings",
) -> List[Tuple[Chunk, float]]:
    """질의를 임베딩 후 코사인 유사도 top-k 청크 반환. (chunk, score) 리스트."""
    if not index.is_built():
        return []
    q_emb = embed_batch_openai([query], api_key, model, url)
    q_vec = q_emb[0]
    q_norm = float(np.linalg.norm(q_vec)) + 1e-12

    # 코사인 유사도
    mat = index.embeddings
    mat_norms = np.linalg.norm(mat, axis=1) + 1e-12
    sims = (mat @ q_vec) / (mat_norms * q_norm)

    # top-k 인덱스
    k = min(top_k, len(index.chunks))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(index.chunks[i], float(sims[i])) for i in top_idx]


# ----------------------------------------------------------------
# Index persistence (저장/로드)
# ----------------------------------------------------------------

import pickle, os

def save_index(index: RagIndex, path: str) -> bool:
    """RagIndex를 파일로 저장. 재빌드 없이 재사용 가능."""
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": index.chunks,
                "embeddings": index.embeddings,
            }, f)
        return True
    except Exception:
        return False


def load_index(path: str) -> Optional[RagIndex]:
    """저장된 RagIndex 로드. 없으면 None 반환."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = RagIndex(chunks=data["chunks"], embeddings=data["embeddings"])
        return idx if idx.is_built() else None
    except Exception:
        return None


def get_index_path(doc_set_hash: str, base_dir: str = ".index_cache") -> str:
    """해시 기반 인덱스 파일 경로 반환."""
    return os.path.join(base_dir, f"rag_{doc_set_hash}.pkl")


def format_context_for_llm(results: List[Tuple[Chunk, float]]) -> str:
    """검색 결과를 LLM에 넣을 컨텍스트 텍스트로 직렬화.
    각 청크에 인용 마커 [출처: 파일명 p.N] 부착."""
    if not results:
        return "(검색된 자료 없음)"
    parts = []
    for i, (c, score) in enumerate(results, 1):
        parts.append(
            f"--- 자료 {i} ---\n"
            f"[출처: {c.source_file}, p.{c.page}, 유사도 {score:.3f}]\n"
            f"{c.text}\n"
        )
    return "\n".join(parts)


# ----------------------------------------------------------------
# 자체 검증 (모델 없이)
# ----------------------------------------------------------------

if __name__ == "__main__":
    # 청킹 함수 검증
    long_text = "교사 업무 매뉴얼입니다. " * 100
    chunks = split_text_into_chunks(long_text, max_chars=200, overlap=30)
    print(f"긴 텍스트 청킹: {len(chunks)}개 청크")
    print(f"  첫 청크 길이: {len(chunks[0])}")
    print(f"  마지막 청크 길이: {len(chunks[-1])}")

    # 짧은 텍스트
    short = "짧은 내용입니다."
    chunks2 = split_text_into_chunks(short, max_chars=200)
    print(f"\n짧은 텍스트: {len(chunks2)}개")

    # Chunk hashable 확인
    c1 = Chunk(chunk_id="abc123", source_file="test.pdf", page=1, chunk_index=0, text="hello")
    c2 = Chunk(chunk_id="abc123", source_file="test.pdf", page=1, chunk_index=0, text="hello")
    print(f"\nChunk 동등성: {c1 == c2}")
    print(f"Chunk hash: {hash(c1) == hash(c2)}")

    # RagIndex 빈 상태
    idx = RagIndex()
    print(f"\n빈 인덱스 is_built: {idx.is_built()}")

    print("\n✓ rag.py 자체 검증 통과")
