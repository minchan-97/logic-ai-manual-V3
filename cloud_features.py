"""
cloud_features.py — N개 임베딩 점 클라우드에서 일관성 특성 추출
=================================================================

같은 질문을 N번 호출 → N개 응답 → N개 임베딩 → 점 클라우드.
이 클라우드의 모양이 일관성/환각의 신호를 담는다는 가설.

특성들 (의미 있는 것만, 휴리스틱 임의 가중치 없이):
  basic_:    평균/최대/표준편차 pairwise distance (단순 분산)
  cluster_:  k=2 계층 클러스터링했을 때 두 그룹 분리도 (응답이 양분되는가)
  spread_:   PCA 1차원/2차원 설명 분산 (응답이 1차원 vs 다차원 분포인가)
  topology_: persistent homology 0차원 (연결 컴포넌트 수, gudhi 있을 때만)

각 특성은 '단순 평균 거리' 베이스라인과 함께 평가 가능.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """N개 임베딩의 N×N cosine distance 행렬. distance = (1 - cos_sim) / 2 ∈ [0, 1]."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    cs = normed @ normed.T
    cs = np.clip(cs, -1.0, 1.0)
    return (1.0 - cs) / 2.0


def basic_features(D: np.ndarray) -> Dict[str, float]:
    """가장 기본적인 분산 특성."""
    n = D.shape[0]
    if n < 2:
        return {"basic_mean": 0.0, "basic_max": 0.0, "basic_std": 0.0}
    iu = np.triu_indices(n, k=1)
    dists = D[iu]
    return {
        "basic_mean": float(dists.mean()),
        "basic_max": float(dists.max()),
        "basic_std": float(dists.std()),
    }


def cluster_features(D: np.ndarray) -> Dict[str, float]:
    """k=2 계층 클러스터링 후 그룹 내 vs 그룹 간 거리 비교.
    응답이 두 진영으로 갈리면 (예: 환각 vs 정답) 이 값이 크게 나옴."""
    n = D.shape[0]
    if n < 3:
        return {"cluster_separation": 0.0, "cluster_balance": 0.0}
    # condensed form (linkage 입력)
    iu = np.triu_indices(n, k=1)
    condensed = D[iu]
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=2, criterion="maxclust")
    g1 = np.where(labels == 1)[0]
    g2 = np.where(labels == 2)[0]
    if len(g1) == 0 or len(g2) == 0:
        return {"cluster_separation": 0.0, "cluster_balance": 0.0}
    # 그룹 내 평균 거리
    def intra(g):
        if len(g) < 2: return 0.0
        return float(np.mean([D[i, j] for i in g for j in g if i < j]))
    # 그룹 간 평균 거리
    inter = float(np.mean([D[i, j] for i in g1 for j in g2]))
    intra_mean = (intra(g1) + intra(g2)) / 2
    separation = inter - intra_mean  # 클수록 두 그룹이 잘 갈림
    balance = min(len(g1), len(g2)) / max(len(g1), len(g2))  # 1에 가까우면 균형 분할
    return {
        "cluster_separation": separation,
        "cluster_balance": balance,
    }


def spread_features(embeddings: np.ndarray) -> Dict[str, float]:
    """PCA로 분산이 몇 차원에 퍼져 있는지.
    1차원 비율이 높으면 응답들이 한 축 위에 있음 (단순 다양성).
    1차원 비율이 낮으면 응답들이 여러 방향으로 흩어짐 (혼란/모순)."""
    n = embeddings.shape[0]
    if n < 3:
        return {"spread_pc1_ratio": 1.0, "spread_pc2_ratio": 0.0, "spread_effective_dim": 1.0}
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    # SVD로 분산 분해
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var = s ** 2
        total = var.sum() + 1e-12
        ratios = var / total
        pc1 = float(ratios[0])
        pc2 = float(ratios[1]) if len(ratios) > 1 else 0.0
        # 유효 차원: entropy-based
        ratios_clean = ratios[ratios > 1e-12]
        entropy = -np.sum(ratios_clean * np.log(ratios_clean))
        eff_dim = float(np.exp(entropy))
        return {
            "spread_pc1_ratio": pc1,
            "spread_pc2_ratio": pc2,
            "spread_effective_dim": eff_dim,
        }
    except Exception:
        return {"spread_pc1_ratio": 1.0, "spread_pc2_ratio": 0.0, "spread_effective_dim": 1.0}


def topology_features(embeddings: np.ndarray, max_edge: float = 1.0) -> Dict[str, float]:
    """Persistent homology 0차원 특성. gudhi 있을 때만 계산.
    H0 birth-death 길이 = 연결 컴포넌트의 지속성. 길수록 분리된 군집이 오래 유지됨."""
    try:
        import gudhi
    except ImportError:
        return {"topo_h0_count": float("nan"), "topo_h0_max_persistence": float("nan")}
    try:
        rc = gudhi.RipsComplex(points=embeddings.tolist(), max_edge_length=max_edge)
        st = rc.create_simplex_tree(max_dimension=1)
        persistence = st.persistence()
        h0 = [(b, d) for dim, (b, d) in persistence if dim == 0]
        # 가장 오래 산 H0 (무한대 제외)
        finite_h0 = [(b, d) for b, d in h0 if d != float("inf")]
        max_pers = max((d - b for b, d in finite_h0), default=0.0)
        return {
            "topo_h0_count": float(len(h0)),
            "topo_h0_max_persistence": float(max_pers),
        }
    except Exception:
        return {"topo_h0_count": float("nan"), "topo_h0_max_persistence": float("nan")}


def all_features(embeddings: np.ndarray, include_topology: bool = True) -> Dict[str, float]:
    """N개 임베딩 행렬을 받아 모든 특성을 한 번에 반환."""
    D = cosine_distance_matrix(embeddings)
    feats = {}
    feats.update(basic_features(D))
    feats.update(cluster_features(D))
    feats.update(spread_features(embeddings))
    if include_topology:
        feats.update(topology_features(embeddings))
    return feats


FEATURE_ORDER = [
    "basic_mean", "basic_max", "basic_std",
    "cluster_separation", "cluster_balance",
    "spread_pc1_ratio", "spread_pc2_ratio", "spread_effective_dim",
    "topo_h0_count", "topo_h0_max_persistence",
]


def feature_vector(embeddings: np.ndarray, include_topology: bool = True) -> np.ndarray:
    f = all_features(embeddings, include_topology)
    return np.array([f.get(k, 0.0) for k in FEATURE_ORDER], dtype=float)


if __name__ == "__main__":
    # 자체 테스트: 일관된 클라우드 vs 분산된 클라우드
    np.random.seed(0)
    rng = np.random.default_rng(0)
    print("=== 자체 sanity check ===")
    print()
    # 1) 일관: 같은 점 근처에 모인 5개
    consistent = np.array([[1, 0, 0]] * 5, dtype=float) + rng.normal(0, 0.01, (5, 3))
    f = all_features(consistent)
    print("일관된 임베딩 5개:")
    for k, v in f.items(): print(f"  {k:30s}: {v:+.4f}")
    print()
    # 2) 분산: 무작위 5개
    diverse = rng.normal(0, 1, (5, 3))
    f = all_features(diverse)
    print("분산된 임베딩 5개:")
    for k, v in f.items(): print(f"  {k:30s}: {v:+.4f}")
    print()
    # 3) 양분: 두 군집으로 갈린 5개
    split = np.vstack([
        rng.normal(0, 0.05, (3, 3)) + np.array([1, 0, 0]),
        rng.normal(0, 0.05, (2, 3)) + np.array([-1, 0, 0]),
    ])
    f = all_features(split)
    print("두 군집으로 갈린 5개 (3+2):")
    for k, v in f.items(): print(f"  {k:30s}: {v:+.4f}")
