"""1차 결과(resolution=1.0)에서 대상 4개동의 응집도가 무작위보다 오히려 낮게 나온 것이
Louvain 해상도(커뮤니티 크기)를 너무 잘게 잡았기 때문인지 확인하기 위해 resolution을
스윕하고, 동시에 7장⑩과 동일한 표준지표(E-I index·conductance)로도 같은 질문을
재확인한다 — "대상 4개동이 성북구 도로망에서 응집된 하나의 그룹을 이루는가."
"""
import pickle
import random
import time
from collections import Counter

import numpy as np
import networkx as nx

TARGET_DONGS = ["정릉2동", "정릉3동", "성북동", "돈암2동"]


def ei_index(G, in_set):
    internal, external = 0, 0
    for u, v in G.edges():
        u_in = u in in_set
        v_in = v in in_set
        if u_in and v_in:
            internal += 1
        elif u_in or v_in:
            external += 1
    total = internal + external
    if total == 0:
        return None
    return (external - internal) / total


def main():
    t0 = time.time()
    with open('/tmp/seongbuk_community.pkl', 'rb') as f:
        d = pickle.load(f)
    Gg = nx.Graph()
    Gg.add_nodes_from(d['nodes'])
    Gg.add_edges_from(d['edges'])
    print(f"그래프 로드: N={Gg.number_of_nodes():,} M={Gg.number_of_edges():,}", flush=True)

    all_dongs = sorted(set(nx.get_node_attributes(Gg, 'dong').values()))
    dong_node_lists = {dn: [n for n in Gg.nodes() if Gg.nodes[n]['dong'] == dn] for dn in all_dongs}

    # ---- 1) Louvain resolution 스윕 ----
    print("\n=== Louvain resolution 스윕 ===", flush=True)
    for res in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        t1 = time.time()
        comms = nx.algorithms.community.louvain_communities(Gg, seed=42, resolution=res)
        Q = nx.algorithms.community.modularity(Gg, comms)
        avg_size = Gg.number_of_nodes() / len(comms)
        node_comm = {}
        for cid, comm in enumerate(comms):
            for n in comm:
                node_comm[n] = cid
        target_nodes = [n for dn in TARGET_DONGS for n in dong_node_lists[dn]]
        c = Counter(node_comm[n] for n in target_nodes)
        top2_frac = sum(cnt for _, cnt in c.most_common(2)) / len(target_nodes)
        print(f"  resolution={res:>4}: 커뮤니티 {len(comms):>3}개(평균크기{avg_size:>6.0f}) Q={Q:.3f} "
              f"대상4동 top-2커뮤니티 점유율={top2_frac:.3f} ({time.time()-t1:.1f}s)", flush=True)

    # ---- 2) E-I index / conductance: 대상4동 vs 무작위 4개동(순열검정) ----
    print("\n=== E-I index / Conductance: 대상4동 그룹 vs 무작위 4개동 (7장⑩과 동일 지표) ===", flush=True)
    target_set = set(n for dn in TARGET_DONGS for n in dong_node_lists[dn])
    obs_ei = ei_index(Gg, target_set)
    obs_cond = nx.conductance(Gg, target_set, weight=None)
    print(f"관측값(대상4동): E-I index={obs_ei:.4f}  Conductance={obs_cond:.4f}  (N={len(target_set)})", flush=True)

    candidates = [dn for dn in all_dongs if len(dong_node_lists[dn]) >= 20]
    rng = random.Random(0)
    null_ei, null_cond = [], []
    t2 = time.time()
    for _ in range(500):
        sample = rng.sample(candidates, 4)
        s = set(n for dn in sample for n in dong_node_lists[dn])
        e = ei_index(Gg, s)
        if e is not None:
            null_ei.append(e)
        try:
            cnd = nx.conductance(Gg, s, weight=None)
            null_cond.append(cnd)
        except Exception:
            pass
    null_ei = np.array(null_ei); null_cond = np.array(null_cond)
    print(f"무작위 4개동 500회 ({time.time()-t2:.1f}s): "
          f"E-I 평균={null_ei.mean():.4f}(sd={null_ei.std():.4f}), "
          f"conductance 평균={null_cond.mean():.4f}(sd={null_cond.std():.4f})", flush=True)
    z_ei = (obs_ei - null_ei.mean()) / null_ei.std()
    z_cond = (obs_cond - null_cond.mean()) / null_cond.std()
    pct_ei = (null_ei < obs_ei).mean() * 100
    pct_cond = (null_cond < obs_cond).mean() * 100
    print(f"대상4동 E-I z={z_ei:.2f} (백분위 {pct_ei:.1f}%ile, 더 음수=더 응집이면 z<0이 응집)", flush=True)
    print(f"대상4동 conductance z={z_cond:.2f} (백분위 {pct_cond:.1f}%ile, 더 작으면 더 응집)", flush=True)

    # ---- 3) 지리적 인접성만으로 본 4개동: 서로 실제로 맞닿아 있는가? ----
    print("\n=== 참고: 대상4동 쌍별 경계공유 엣지 수 (지리적 인접성) ===", flush=True)
    from itertools import combinations
    for a, b in combinations(TARGET_DONGS, 2):
        na, nb = set(dong_node_lists[a]), set(dong_node_lists[b])
        cross = sum(1 for u, v in Gg.edges() if (u in na and v in nb) or (u in nb and v in na))
        print(f"  {a} - {b}: 경계 연결 엣지 {cross}개", flush=True)


if __name__ == "__main__":
    main()
