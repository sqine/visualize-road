"""Motter-Lai 부하재분배 모형으로 Synergy_ij를 직접 산출 [2026-09-23].

배경: 8장의 Bonus_ij는 current-flow(전파) 모형으로 "동시 결빙 시 공유 이웃에 걸리는
부담의 초과분"을 계산한다. 같은 질문을 전혀 다른 모형(Motter-Lai cascading failure)으로
다시 풀어서, 두 모형이 같은 쌍을 지목하는지 교차검증한다.

Motter-Lai 정식화:
  - 초기부하 L_i = betweenness centrality
  - 용량      C_i = (1+alpha) * L_i          (alpha = 여유율)
  - 노드 제거 -> betweenness 재계산 -> L_i > C_i 인 노드 과부하 -> 제거 -> 반복
  - cascade 크기 = 최종적으로 기능을 잃은 노드 수 (초기 제거분 포함)

Synergy_ML(i,j) = cascade({i,j}) - cascade({i}) - cascade({j})
  > 0 이면 "두 구간이 동시에 막힐 때 피해가 개별 합보다 크다" = 함께 보호할 근거.
"""
import json
import time

import numpy as np
import pandas as pd
import networkx as nx

DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
ALPHA = 0.20        # 용량 여유율
MAX_ROUNDS = 5      # cascade 반복 상한
BTW_K = 60          # betweenness 샘플링 노드 수(속도)
SEED = 42
CKPT = "/tmp/synergy_ml_ckpt.pkl"


def cascade_size(G, L0, C, removed, k=BTW_K, seed=SEED):
    """removed 집합을 제거한 뒤 연쇄 과부하로 추가 제거되는 노드까지 센다."""
    dead = set(removed)
    for _ in range(MAX_ROUNDS):
        alive = [n for n in G.nodes() if n not in dead]
        if len(alive) < 3:
            break
        H = G.subgraph(alive)
        kk = min(k, H.number_of_nodes())
        L = nx.betweenness_centrality(H, k=kk, seed=seed, normalized=True)
        newly = [n for n in alive if L.get(n, 0) > C.get(n, np.inf)]
        if not newly:
            break
        dead.update(newly)
    return len(dead)


def main():
    t0 = time.time()
    with open(f"{DATA}/road_graph.json") as f:
        G_full = nx.node_link_graph(json.load(f), edges="links")
    comps = sorted(nx.connected_components(G_full), key=len, reverse=True)
    G = G_full.subgraph(comps[0]).copy()
    print(f"거대성분: N={G.number_of_nodes()} M={G.number_of_edges()}", flush=True)

    L0 = nx.betweenness_centrality(G, k=min(BTW_K, G.number_of_nodes()), seed=SEED, normalized=True)
    C = {n: (1 + ALPHA) * L0[n] for n in G.nodes()}
    print(f"초기부하/용량 계산 완료 ({time.time()-t0:.1f}s)", flush=True)

    gap_nodes = [n for n in G.nodes()
                 if (G.nodes[n].get("slope_pct") or -1) >= 6.0
                 and not G.nodes[n].get("heating_installed_precise")
                 and G.nodes[n].get("road_type") == "골목길"]
    gap_set = set(gap_nodes)
    pairs = [(u, v) for u, v in G.edges() if u in gap_set and v in gap_set]
    print(f"거대성분 내 후보 {len(gap_nodes)}개, 후보-후보 인접쌍 {len(pairs)}개", flush=True)

    involved = sorted({n for p in pairs for n in p})
    print(f"쌍에 등장하는 고유 노드: {len(involved)}개 → 단독 cascade 계산", flush=True)

    import os
    import pickle
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            ck = pickle.load(f)
        single = ck.get("single", {}); joint_map = ck.get("joint", {})
    else:
        single, joint_map = {}, {}

    def save():
        with open(CKPT, "wb") as f:
            pickle.dump({"single": single, "joint": joint_map}, f)

    BUDGET = 150.0  # 초 — 넘으면 저장하고 종료(다음 호출에서 이어감)
    for i, n in enumerate(involved):
        if n in single:
            continue
        single[n] = cascade_size(G, L0, C, {n})
        if (i + 1) % 20 == 0:
            print(f"  단독 {i+1}/{len(involved)} ({time.time()-t0:.1f}s)", flush=True)
        if time.time() - t0 > BUDGET:
            save()
            print(f"시간초과 — 단독 {len(single)}/{len(involved)} 저장 후 종료. 다시 실행하세요.", flush=True)
            return
    save()

    rows = []
    for i, (u, v) in enumerate(pairs):
        key = (u, v)
        if key not in joint_map:
            joint_map[key] = cascade_size(G, L0, C, {u, v})
            if time.time() - t0 > BUDGET:
                save()
                print(f"시간초과 — 쌍 {len(joint_map)}/{len(pairs)} 저장 후 종료. 다시 실행하세요.", flush=True)
                return
        joint = joint_map[key]
        syn = joint - single[u] - single[v]
        rows.append(dict(i=u, j=v, cascade_i=single[u], cascade_j=single[v],
                         cascade_joint=joint, synergy_ML=syn,
                         slope_i=G.nodes[u].get("slope_pct"), slope_j=G.nodes[v].get("slope_pct"),
                         flow_i=G.nodes[u].get("flow_sum", 0) or 0,
                         flow_j=G.nodes[v].get("flow_sum", 0) or 0,
                         dong=G.nodes[u].get("dong")))
        if (i + 1) % 20 == 0:
            print(f"  쌍 {i+1}/{len(pairs)} ({time.time()-t0:.1f}s)", flush=True)
    save()

    df = pd.DataFrame(rows)
    df.to_csv("/tmp/synergy_motter_lai.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: /tmp/synergy_motter_lai.csv ({time.time()-t0:.1f}s)")
    print(f"synergy_ML>0 인 쌍: {(df.synergy_ML>0).sum()}/{len(df)} "
          f"({(df.synergy_ML>0).mean()*100:.1f}%)")
    print(f"평균 synergy={df.synergy_ML.mean():.2f}, 최대={df.synergy_ML.max():.0f}")
    print("\n상위 10쌍:")
    print(df.nlargest(10, "synergy_ML")[["i", "j", "cascade_i", "cascade_j",
                                          "cascade_joint", "synergy_ML", "dong"]].to_string(index=False))


if __name__ == "__main__":
    main()
