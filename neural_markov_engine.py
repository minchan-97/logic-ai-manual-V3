"""
neural_markov_engine.py — GasCode NeuralMarkov 엔진
====================================================
TinyTransformer(word-level) + JM Smoothing 마르코프 결합

핵심 발상:
  마르코프: "이 단어 다음에 저 단어가 코퍼스에 있었는가" — 정적
  TinyTransformer tok_emb: 의미적으로 비슷한 단어는 가까운 공간
  → 마르코프가 OOV로 잡아도, 임베딩이 가까우면 패널티 완화
  → 조사 변화/유사 표현 오탐 대폭 감소

GPU 없음, numpy만, CPU only
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

try:
    from korean_tokenizer import tokenize as ko_tokenize, tokenize_dual
    _KO = True
except Exception:
    _KO = False
    def ko_tokenize(text): return text.strip().split()
    def tokenize_dual(text): return [(w, w) for w in text.strip().split()]


# ── 유틸 ──────────────────────────────────────────────────────
def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def _layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── Word-level TinyTransformer ─────────────────────────────────
class WordTransformer:
    """
    단어 단위 Transformer (char 대신 word).
    tok_emb 행렬이 핵심 — 학습 후 의미 공간이 형성됨.
    GPU 없음, numpy만.
    """
    def __init__(self, vocab_size: int, dim: int = 32,
                 n_layers: int = 1, max_len: int = 128, seed: int = 42):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_len = max_len
        rng = np.random.default_rng(seed)
        sc = 0.1
        self.tok_emb = rng.normal(0, sc, (vocab_size, dim))
        self.pos_emb = rng.normal(0, sc, (max_len, dim))
        self.layers = [{
            "Wq": rng.normal(0, sc, (dim, dim)),
            "Wk": rng.normal(0, sc, (dim, dim)),
            "Wv": rng.normal(0, sc, (dim, dim)),
            "Wo": rng.normal(0, sc, (dim, dim)),
            "W1": rng.normal(0, sc, (dim, 2 * dim)),
            "W2": rng.normal(0, sc, (2 * dim, dim)),
        } for _ in range(n_layers)]
        self.W_out = rng.normal(0, sc, (dim, vocab_size))

    def forward(self, ids: List[int], cache: bool = False):
        T = min(len(ids), self.max_len)
        ids = ids[:T]
        x = self.tok_emb[ids] + self.pos_emb[:T]
        saved = {"ids": ids, "T": T, "layer_inputs": [], "layer_caches": []}
        for layer in self.layers:
            x_in = x.copy()
            x_norm = _layer_norm(x)
            Q = x_norm @ layer["Wq"]
            K = x_norm @ layer["Wk"]
            V = x_norm @ layer["Wv"]
            scores = Q @ K.T / np.sqrt(self.dim)
            mask = np.triu(np.ones((T, T)), k=1).astype(bool)
            scores[mask] = -1e10
            attn = _softmax(scores, axis=-1)
            h = attn @ V
            h_out = h @ layer["Wo"]
            x_after = x + h_out
            x_norm2 = _layer_norm(x_after)
            ff = np.maximum(0, x_norm2 @ layer["W1"]) @ layer["W2"]
            x = x_after + ff
            if cache:
                saved["layer_caches"].append({
                    "x_in": x_in, "x_norm": x_norm,
                    "Q": Q, "K": K, "V": V, "attn": attn, "h": h,
                    "x_after_attn": x_after, "x_norm2": x_norm2,
                    "ff_pre": x_norm2 @ layer["W1"],
                    "ff_act": np.maximum(0, x_norm2 @ layer["W1"]),
                })
        logits = x @ self.W_out
        probs = _softmax(logits, axis=-1)
        if cache:
            saved["x_final"] = x
            saved["probs"] = probs
            return probs, saved
        return probs

    def _ln_backward(self, x, g, eps=1e-5):
        std = np.sqrt(x.var(axis=-1, keepdims=True) + eps)
        x_hat = (x - x.mean(axis=-1, keepdims=True)) / std
        return (g - g.mean(axis=-1, keepdims=True)
                - x_hat * (g * x_hat).mean(axis=-1, keepdims=True)) / std

    def backward(self, saved):
        T = saved["T"]; probs = saved["probs"]; ids = saved["ids"]
        if T < 2: return None
        vT = T - 1
        tgt = np.array(ids[1:])
        g_logits = probs.copy()
        g_logits[:vT][np.arange(vT), tgt] -= 1.0
        g_logits[:vT] /= vT
        g_logits[vT:] = 0
        grads = {"W_out": saved["x_final"].T @ g_logits}
        g_x = g_logits @ self.W_out.T
        layer_grads = []
        for i in reversed(range(self.n_layers)):
            c = saved["layer_caches"][i]
            layer = self.layers[i]
            g_ff = g_x.copy(); g_xa = g_x.copy()
            g_W2 = c["ff_act"].T @ g_ff
            g_ffa = g_ff @ layer["W2"].T
            g_ffp = g_ffa * (c["ff_pre"] > 0)
            g_W1 = c["x_norm2"].T @ g_ffp
            g_xn2 = g_ffp @ layer["W1"].T
            g_xa += self._ln_backward(c["x_after_attn"], g_xn2)
            g_ho = g_xa.copy(); g_xi = g_xa.copy()
            g_Wo = c["h"].T @ g_ho; g_h = g_ho @ layer["Wo"].T
            g_attn = g_h @ c["V"].T; g_V = c["attn"].T @ g_h
            s = (g_attn * c["attn"]).sum(axis=-1, keepdims=True)
            g_sc = c["attn"] * (g_attn - s) / np.sqrt(self.dim)
            g_sc[np.triu(np.ones_like(g_sc), k=1).astype(bool)] = 0
            g_Q = g_sc @ c["K"]; g_K = g_sc.T @ c["Q"]
            g_Wq = c["x_norm"].T @ g_Q
            g_Wk = c["x_norm"].T @ g_K
            g_Wv = c["x_norm"].T @ g_V
            g_xn = (g_Q @ layer["Wq"].T + g_K @ layer["Wk"].T
                    + g_V @ layer["Wv"].T)
            g_x = g_xi + self._ln_backward(c["x_in"], g_xn)
            layer_grads.append({"Wq":g_Wq,"Wk":g_Wk,"Wv":g_Wv,"Wo":g_Wo,
                                 "W1":g_W1,"W2":g_W2})
        layer_grads.reverse()
        grads["layers"] = layer_grads
        g_te = np.zeros_like(self.tok_emb)
        for i, tid in enumerate(ids): g_te[tid] += g_x[i]
        grads["tok_emb"] = g_te
        g_pe = np.zeros_like(self.pos_emb); g_pe[:T] = g_x
        grads["pos_emb"] = g_pe
        return grads

    def _apply(self, grads, lr, clip=1.0):
        def cl(g):
            n = np.linalg.norm(g)
            return g * (clip / n) if n > clip else g
        self.W_out   -= lr * cl(grads["W_out"])
        self.tok_emb -= lr * cl(grads["tok_emb"])
        self.pos_emb -= lr * cl(grads["pos_emb"])
        for i, lg in enumerate(grads["layers"]):
            for k, g in lg.items():
                self.layers[i][k] -= lr * cl(g)

    def fit(self, ids: List[int], epochs: int = 20, lr: float = 0.05,
            seq_len: int = 32, seed: int = 0, on_epoch=None):
        rng = np.random.default_rng(seed)
        n = len(ids)
        history = []
        for ep in range(epochs):
            n_steps = max(1, n // seq_len)
            starts = rng.integers(0, max(1, n - seq_len), size=n_steps)
            ep_loss = 0.0; count = 0
            for s in starts:
                chunk = ids[int(s): int(s) + seq_len]
                if len(chunk) < 4: continue
                probs, saved = self.forward(chunk, cache=True)
                vT = len(chunk) - 1
                tgt = np.array(chunk[1:])
                loss = -np.mean(np.log(probs[np.arange(vT), tgt] + 1e-10))
                ep_loss += float(loss); count += 1
                g = self.backward(saved)
                if g: self._apply(g, lr)
            avg = ep_loss / max(count, 1)
            history.append(avg)
            if on_epoch: on_epoch(ep, avg)
        # 임베딩 정규화
        norms = np.linalg.norm(self.tok_emb, axis=1, keepdims=True) + 1e-12
        self.tok_emb = self.tok_emb / norms
        return history

    def get_vec(self, idx: int) -> np.ndarray:
        return self.tok_emb[idx]


# ── NeuralMarkovEngine ────────────────────────────────────────
class NeuralMarkovEngine:
    """
    TinyTransformer(word-level) + JM Smoothing 마르코프 결합 가드레일

    평가 흐름:
      1. 마르코프 logP 계산 (JM Smoothing)
      2. OOV/저확률 토큰: TinyTransformer 임베딩으로 의미 유사 토큰 확인
      3. 유사 토큰이 코퍼스 안에 있으면 → 패널티 완화
      4. 최종 점수 = 마르코프 logP + 의미 보정값
    """
    def __init__(self, lambda_1=0.6, lambda_2=0.3, lambda_3=0.1, alpha=0.001):
        self.l1 = lambda_1; self.l2 = lambda_2; self.l3 = lambda_3
        self.alpha = alpha
        # 마르코프
        self.uni = Counter(); self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter); self.total = 0
        # 신경망
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        self.model: Optional[WordTransformer] = None
        self.is_trained = False
        self.corpus_name = ""
        self.dim = 32

    def train(self, corpus_text: str, embedding_dim: int = 32,
              epochs: int = 20, on_epoch=None):
        self.dim = embedding_dim
        tokens = ko_tokenize(corpus_text)
        self.total = len(tokens)

        # 마르코프 학습
        for i, t in enumerate(tokens):
            self.uni[t] += 1
            if i >= 1: self.bi[tokens[i-1]][t] += 1
            if i >= 2: self.tri[(tokens[i-2], tokens[i-1])][t] += 1

        # 어휘 구축 (min_count=1)
        self.idx2word = list(self.uni.keys())
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        V = len(self.idx2word)

        # TinyTransformer 학습
        ids = [self.word2idx[t] for t in tokens if t in self.word2idx]
        self.model = WordTransformer(
            vocab_size=V, dim=embedding_dim,
            n_layers=1, max_len=128, seed=42,
        )
        self.model.fit(ids, epochs=epochs, lr=0.05, seq_len=32, on_epoch=on_epoch)
        self.is_trained = True

    def _get_vec(self, word: str) -> Optional[np.ndarray]:
        if word in self.word2idx and self.model:
            return self.model.get_vec(self.word2idx[word])
        return None

    # 한국어 기능어 목록 (질문어/접속어 등) — logP 계산 제외
    _FUNCTION_WORDS = {
        "어떻게", "무엇", "무엇인가요", "어디서", "언제", "누가", "왜", "어떤",
        "어디", "얼마나", "어느", "뭐", "몇", "무슨",
        "그리고", "그러나", "하지만", "그래서", "따라서", "또한", "그런데",
        "이", "그", "저", "이것", "그것", "저것",
        "있다", "없다", "이다", "되다", "하다",
        "입니다", "합니다", "됩니다", "있습니다", "없습니다",
        "나요", "가요", "해요", "되요", "네요",
    }

    def _semantic_bonus(self, token: str, logp: float) -> float:
        """
        마르코프가 낮게 잡은 토큰에 의미 보정 적용.
        TinyTransformer 임베딩으로 코퍼스 어휘와 유사도 측정.
        """
        if self.model is None: return 0.0
        # 기능어는 보정 없이 무시 (패널티도 없음)
        if token in self._FUNCTION_WORDS: return 3.0  # 통과
        vec = self._get_vec(token)
        if vec is None: return 0.0

        sample_words = [w for w, c in self.uni.most_common(300)]
        sims = []
        for w in sample_words:
            v2 = self._get_vec(w)
            if v2 is not None:
                sims.append(float(np.dot(vec, v2)))
        if not sims: return 0.0
        max_sim = max(sims)

        # 유사도 0.5 이상이면 보정 (임계값 낮춤)
        if max_sim > 0.5:
            bonus = (max_sim - 0.5) / 0.5 * 5.0  # 최대 5점 보정
            return bonus
        return 0.0

    def _score_jm(self, tokens: List[str]):
        V = len(self.uni)
        total_lp = 0.0
        per = []
        if len(tokens) < 3:
            return 0.0, []
        for t in range(2, len(tokens)):
            wc, wp, wpp = tokens[t], tokens[t-1], tokens[t-2]
            p1 = (self.uni[wc] + self.alpha) / (self.total + self.alpha * V)
            cp = self.uni[wp]
            p2 = (self.bi[wp][wc] / cp) if cp > 0 else 0.0
            cpp = self.bi[wpp][wp]
            p3 = (self.tri[(wpp, wp)][wc] / cpp) if cpp > 0 else 0.0
            p_jm = self.l1 * p3 + self.l2 * p2 + self.l3 * p1
            lp_raw = float(np.log(p_jm + 1e-12))
            in_graph = (p2 > 0 or p3 > 0)

            # 의미 보정 (마르코프가 낮게 잡은 경우 적용)
            bonus = 0.0
            if lp_raw < -5.0:
                bonus = self._semantic_bonus(wc, lp_raw)
            lp_corrected = lp_raw + bonus

            total_lp += lp_corrected
            is_outlier = lp_corrected < -12.0 and not in_graph
            per.append({
                "token": wc,
                "logp_raw": lp_raw,
                "logp": lp_corrected,
                "bonus": bonus,
                "in_graph": in_graph,
                "outlier": is_outlier,
            })
        return total_lp / max(len(tokens) - 2, 1), per

    def _score_mismatch(self, tokens: List[str]) -> float:
        if self.model is None: return 0.5
        vecs = [self._get_vec(w) for w in tokens if self._get_vec(w) is not None]
        if len(vecs) < 2: return 0.5
        sims = [float(np.dot(vecs[i], vecs[i+1])) for i in range(len(vecs)-1)]
        return float((1.0 - np.mean(sims)) / 2.0)

    def evaluate(self, text: str, logp_thr: float = -8.0,
                 mis_thr: float = 0.55) -> dict:
        import time
        t0 = time.perf_counter()
        pairs = tokenize_dual(text)
        tokens = [norm for _, norm in pairs]
        raw_tokens = [raw for raw, _ in pairs]
        if len(tokens) < 3:
            return {"status": "SKIP", "avg_logp": 0.0, "mismatch": 0.0,
                    "elapsed_ms": 0.0, "per_token": []}

        avg_logp, per = self._score_jm(tokens)
        for i, p in enumerate(per):
            p["raw_token"] = raw_tokens[i + 2] if i + 2 < len(raw_tokens) else p["token"]

        mismatch = self._score_mismatch(tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        mf = avg_logp < logp_thr; ef = mismatch > mis_thr

        # 3구간 판정
        # PASS:     logP > -10          → 도메인 안
        # WARNING:  -10 ~ -12           → 경계 (도메인 안이지만 이상한 것 포함)
        # FATAL:    logP < -14          → 도메인 완전 이탈
        if avg_logp >= -10.0 and not ef:
            status = "PASS"
        elif avg_logp >= -14.0 and not ef:
            status = "WARNING"
        elif avg_logp < -14.0 or (mf and ef):
            status = "FATAL"
        elif not mf and ef:
            status = "CRITICAL"
        else:
            status = "WARNING"

        return {
            "status": status,
            "avg_logp": avg_logp,
            "mismatch": mismatch,
            "elapsed_ms": elapsed_ms,
            "per_token": per,
            "neural_active": self.is_trained,
        }
