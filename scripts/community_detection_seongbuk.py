"""'왜 하필 이 4개동(정릉2동·정릉3동·성북동·돈암2동)을 묶었는가'를 네트워크 커뮤니티
탐지로 검증. 성북구 전체 20개 행정동의 도로망 위상 그래프를 만들고 Louvain
community detection을 돌려서, 이 4개동 노드가 실제로 하나(또는 소수)의 응집된
커뮤니티로 뭉치는지, 아니면 그냥 행정구역상 인접하다는 이유만으로 임의로 묶인
것인지 확인한다.
"""
import json
import pickle
import random
import time
from collections import Counter, defaultdict

import numpy as np
import networkx as nx
import shapely
from shapely.geometry import shape

HERE_DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
TARGET_DONGS = ["정릉2동", "정릉3동", "성북동", "돈암2동"]
CELL_M = 5.0
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0


def main():
    t0 = time.time()
    with open(f"{HERE_DATA}/road_links.geojson") as f:
        rl = json.load(f)
    feats = rl["features"]
    n = len(feats)

    lon_s = np.zeros(n); lat_s = np.zeros(n)
    lon_e = np.zeros(n); lat_e = np.zeros(n)
    mid_lon = np.zeros(n); mid_lat = np.zeros(n)
    for i, feat in enumerate(feats):
        c = feat["geometry"]["coordinates"]
        lon_s[i], lat_s[i] = c[0]
        lon_e[i], lat_e[i] = c[-1]
        mid = c[len(c) // 2]
        mid_lon[i], mid_lat[i] = mid

    with open(f"{HERE_DATA}/seongbuk_dong.geojson", encoding="utf-8") as f:
        dong_gj = json.load(f)
    dong_polys = [shape(ft["geometry"]) for ft in dong_gj["features"]]
    dong_names = [ft["properties"]["adm_nm"].split()[-1] for ft in dong_gj["features"]]

    pts = shapely.points(mid_lon, mid_lat)
    dong_arr = np.array([None] * n, dtype=object)
    for name, poly in zip(dong_names, dong_polys):
        mask = shapely.contains(poly, pts)
        dong_arr[mask] = name
    idxs = np.where(dong_arr != None)[0]  # noqa: E711
    print(f"성북구 20개 행정동 내 링크: {len(idxs):,}개 ({time.time()-t0:.1f}s)", flush=True)
    print(Counter(dong_arr[idxs]), flush=True)

    # ---- 위상 그래프(끝점 5m 격자스냅) ----
    t1 = time.time()
    cell_lat = CELL_M / _M_PER_DEG_LAT
    cell_lon = CELL_M / _M_PER_DEG_LON

    def qkey(lon, lat):
        return (np.round(lon / cell_lon).astype(np.int64), np.round(lat / cell_lat).astype(np.int64))

    ks_lon, ks_lat = qkey(lon_s[idxs], lat_s[idxs])
    ke_lon, ke_lat = qkey(lon_e[idxs], lat_e[idxs])
    junction_links = defaultdict(list)
    for local_i, gi in enumerate(idxs):
        junction_links[(ks_lon[local_i], ks_lat[local_i])].append(gi)
        junction_links[(ke_lon[local_i], ke_lat[local_i])].append(gi)

    G = nx.Graph()
    for gi in idxs:
        G.add_node(int(gi), dong=dong_arr[gi])

    edge_set = set()
    MAX_JUNCTION = 40
    for jid, lids in junction_links.items():
        uniq = sorted(set(lids))
        if len(uniq) < 2:
            continue
        if len(uniq) > MAX_JUNCTION:
            hub = uniq[0]
            for other in uniq[1:]:
                a, b = (hub, other) if hub < other else (other, hub)
                edge_set.add((a, b))
            continue
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                edge_set.add((uniq[a], uniq[b]))
    G.add_edges_from(edge_set)
    print(f"위상 그래프: N={G.number_of_nodes():,} M={G.number_of_edges():,} ({time.time()-t1:.1f}s)", flush=True)

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    Gg = G.subgraph(comps[0]).copy()
    print(f"거대성분: N={Gg.number_of_nodes():,} ({Gg.number_of_nodes()/G.number_of_nodes()*100:.1f}%)", flush=True)

    # ---- Louvain community detection ----
    t2 = time.time()
    random.seed(42)
    communities = nx.algorithms.community.louvain_communities(Gg, seed=42, resolution=1.0)
    Q = nx.algorithms.community.modularity(Gg, communities)
    print(f"\nLouvain: 커뮤니티 {len(communities)}개, modularity Q={Q:.4f} ({time.time()-t2:.1f}s)", flush=True)

    # 노드 -> 커뮤니티 id
    node_comm = {}
    for cid, comm in enumerate(communities):
        for n_ in comm:
            node_comm[n_] = cid

    # 커뮤니티별 동 구성
    comm_dong_counts = defaultdict(Counter)
    for n_ in Gg.nodes():
        d = Gg.nodes[n_]['dong']
        comm_dong_counts[node_comm[n_]][d] += 1

    print("\n=== 커뮤니티별 행정동 구성 (상위 15개 커뮤니티, 크기순) ===", flush=True)
    comm_sizes = sorted(((cid, len(comm)) for cid, comm in enumerate(communities)), key=lambda x: -x[1])
    for cid, size in comm_sizes[:15]:
        counts = comm_dong_counts[cid]
        top = counts.most_common(5)
        top_str = ", ".join(f"{d}({c})" for d, c in top)
        target_n = sum(c for d, c in counts.items() if d in TARGET_DONGS)
        print(f"  커뮤니티 {cid} (N={size}): {top_str}  [대상4동 비율={target_n/size*100:.1f}%]", flush=True)

    # ---- 대상 4개동 노드가 어느 커뮤니티들에 분포하는지 ----
    print("\n=== 대상 4개동 각각의 커뮤니티 분포 ===", flush=True)
    for d in TARGET_DONGS:
        d_nodes = [n_ for n_ in Gg.nodes() if Gg.nodes[n_]['dong'] == d]
        d_comms = Counter(node_comm[n_] for n_ in d_nodes)
        total = len(d_nodes)
        top = d_comms.most_common(3)
        top_str = ", ".join(f"커뮤니티{cid}({c}개,{c/total*100:.0f}%)" for cid, c in top)
        print(f"  {d} (거대성분 내 N={total}): {top_str}", flush=True)

    # ---- 대상 4개동이 같은 커뮤니티에 뭉치는 정도를 무작위 4개동 선택과 비교 ----
    all_dongs = sorted(set(dong_names))
    dong_node_lists = {d: [n_ for n_ in Gg.nodes() if Gg.nodes[n_]['dong'] == d] for d in all_dongs}

    def cohesion_score(dong_subset):
        """이 동 집합 노드들이 소수의 커뮤니티에 얼마나 몰려있는지: 최빈 top-2 커뮤니티 비율 합"""
        nodes = [n_ for d in dong_subset for n_ in dong_node_lists.get(d, [])]
        if not nodes:
            return None
        c = Counter(node_comm[n_] for n_ in nodes)
        total = len(nodes)
        top2 = sum(cnt for _, cnt in c.most_common(2))
        return top2 / total

    obs_score = cohesion_score(TARGET_DONGS)
    print(f"\n대상 4개동 응집도(상위2개 커뮤니티 점유율): {obs_score:.3f}", flush=True)

    rng = random.Random(0)
    candidates = [d for d in all_dongs if len(dong_node_lists.get(d, [])) >= 20]
    null_scores = []
    for _ in range(2000):
        sample = rng.sample(candidates, min(4, len(candidates)))
        s = cohesion_score(sample)
        if s is not None:
            null_scores.append(s)
    null_scores = np.array(null_scores)
    pct = (null_scores < obs_score).mean() * 100
    print(f"무작위 4개동 선택 2000회 응집도: 평균={null_scores.mean():.3f}, "
          f"표준편차={null_scores.std():.3f}, 관측값 백분위={pct:.1f}%ile "
          f"(z={(obs_score-null_scores.mean())/null_scores.std():.2f})", flush=True)

    with open('/tmp/seongbuk_community.pkl', 'wb') as f:
        pickle.dump({
            'nodes': list(Gg.nodes(data=True)), 'edges': list(Gg.edges()),
            'node_comm': node_comm, 'communities': communities, 'Q': Q,
            'obs_score': obs_score, 'null_scores': null_scores,
        }, f)
    print(f"\n체크포인트 저장 완료 (총 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
