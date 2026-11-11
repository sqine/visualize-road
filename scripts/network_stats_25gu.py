"""자치구별 네트워크 지표 — 25개구 전체 확대판 [2026-09-23].

배경: 기존 network_stats_by_tier_gu.py는 1:1,000 정밀 데이터가 9개구에만 있어
'정밀/비정밀 계층'을 분리해 비교했고, 그 계층 구분이 '구릉지 자치구 vs 평지 자치구'라는
도시형태 차이와 혼입(confound)돼 있다는 한계를 명시했었다. 이번에 서울 전역 2,489개
도엽을 확보해 25개구 전체가 83.7~100% 정밀도로 균질화됐으므로, 계층 분리 없이
25개구를 동일 기준으로 비교할 수 있게 됐다.

산출:
  (1) 자치구별 기본 위상통계: N, M, <k>, 밀도, 평균 클러스터링, 거대성분 비율
  (2) 자치구별 위험군(급경사 골목길) 비중
  (3) 결빙구역 / 열선설치 그룹의 응집도: E-I index, conductance + 랜덤 null 대비 z
"""
import json
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import shapely
from shapely.geometry import shape

DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
CELL_M = 5.0
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0
N_RANDOM = 2000
SEED = 42


def vec_ei_conductance(eu, ev, degree, in_mask):
    u_in = in_mask[eu]; v_in = in_mask[ev]
    internal = np.sum(u_in & v_in)
    external = np.sum(u_in ^ v_in)
    total = internal + external
    ei = (external - internal) / total if total else np.nan
    vol_s = degree[in_mask].sum()
    vol_t = degree[~in_mask].sum()
    cond = external / min(vol_s, vol_t) if min(vol_s, vol_t) > 0 else np.nan
    return ei, cond


def build_or_load_graph():
    cache = "/tmp/citywide_topo_graph_v2.pkl"
    import os
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            d = pickle.load(f)
        print(f"그래프 캐시 로드: N={len(d['nodes']):,} M={len(d['edges']):,}", flush=True)
        return d

    t0 = time.time()
    with open(f"{DATA}/road_links.geojson") as f:
        rl = json.load(f)
    feats = rl["features"]
    n = len(feats)
    lon_s = np.zeros(n); lat_s = np.zeros(n)
    lon_e = np.zeros(n); lat_e = np.zeros(n)
    mid_lon = np.zeros(n); mid_lat = np.zeros(n)
    slope = np.full(n, np.nan)
    rtype = [None] * n
    src1000 = np.zeros(n, dtype=bool)
    for i, feat in enumerate(feats):
        c = feat["geometry"]["coordinates"]
        lon_s[i], lat_s[i] = c[0]
        lon_e[i], lat_e[i] = c[-1]
        mid = c[len(c) // 2]
        mid_lon[i], mid_lat[i] = mid
        p = feat["properties"]
        sp = p.get("slope_pct"); slope[i] = sp if sp is not None else np.nan
        rtype[i] = p.get("road_type")
        src1000[i] = bool(p.get("source_1000"))

    with open(f"{DATA}/seoul_gu.geojson") as f:
        gu_gj = json.load(f)
    gu_polys = [shape(ft["geometry"]) for ft in gu_gj["features"]]
    gu_names = [ft["properties"]["gu"] for ft in gu_gj["features"]]
    pts = shapely.points(mid_lon, mid_lat)
    gu_arr = np.array([None] * n, dtype=object)
    for name, poly in zip(gu_names, gu_polys):
        gu_arr[shapely.contains(poly, pts)] = name
    print(f"gu 공간조인 ({time.time()-t0:.1f}s)", flush=True)

    idxs = np.where(gu_arr != None)[0]  # noqa: E711
    cell_lat = CELL_M / _M_PER_DEG_LAT
    cell_lon = CELL_M / _M_PER_DEG_LON
    ks_lon = np.round(lon_s[idxs] / cell_lon).astype(np.int64)
    ks_lat = np.round(lat_s[idxs] / cell_lat).astype(np.int64)
    ke_lon = np.round(lon_e[idxs] / cell_lon).astype(np.int64)
    ke_lat = np.round(lat_e[idxs] / cell_lat).astype(np.int64)

    junction_links = defaultdict(list)
    for li, gi in enumerate(idxs):
        junction_links[(ks_lon[li], ks_lat[li])].append(int(gi))
        junction_links[(ke_lon[li], ke_lat[li])].append(int(gi))

    edge_set = set()
    MAX_J = 40
    for lids in junction_links.values():
        uniq = sorted(set(lids))
        if len(uniq) < 2:
            continue
        if len(uniq) > MAX_J:
            hub = uniq[0]
            for other in uniq[1:]:
                edge_set.add((hub, other) if hub < other else (other, hub))
            continue
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                edge_set.add((uniq[a], uniq[b]))

    nodes = [(int(gi), dict(gu=gu_arr[gi], slope=float(slope[gi]) if slope[gi] == slope[gi] else None,
                            road_type=rtype[gi], src1000=bool(src1000[gi]))) for gi in idxs]
    d = {"nodes": nodes, "edges": sorted(edge_set)}
    with open(cache, "wb") as f:
        pickle.dump(d, f)
    print(f"그래프 구축 완료 ({time.time()-t0:.1f}s): N={len(nodes):,} M={len(edge_set):,}", flush=True)
    return d


def main():
    t0 = time.time()
    d = build_or_load_graph()
    G = nx.Graph()
    G.add_nodes_from(d["nodes"])
    G.add_edges_from(d["edges"])

    icing_mask_full = np.load("/tmp/icing_mask_citywide.npy")
    # 열선 마스크: v3 캐시 기반으로 재구성
    heating_mask_full = np.load("/tmp/heating_mask_citywide_v3.npy") \
        if __import__("os").path.exists("/tmp/heating_mask_citywide_v3.npy") else None

    rng = np.random.default_rng(SEED)
    rows = []
    gus = sorted(set(nd.get("gu") for _, nd in G.nodes(data=True) if nd.get("gu")))
    print(f"자치구 {len(gus)}개 처리 시작 ({time.time()-t0:.1f}s)", flush=True)

    for gu in gus:
        sub_nodes = [nid for nid, nd in G.nodes(data=True) if nd.get("gu") == gu]
        Gs = G.subgraph(sub_nodes)
        N = Gs.number_of_nodes(); M = Gs.number_of_edges()
        if N == 0:
            continue
        nodes_list = list(Gs.nodes())
        idx_of = {nid: i for i, nid in enumerate(nodes_list)}
        eu = np.array([idx_of[u] for u, v in Gs.edges()], dtype=np.int64)
        ev = np.array([idx_of[v] for u, v in Gs.edges()], dtype=np.int64)
        degree = np.zeros(N, dtype=np.int64)
        if len(eu):
            np.add.at(degree, eu, 1); np.add.at(degree, ev, 1)

        slopes = np.array([Gs.nodes[nid].get("slope") if Gs.nodes[nid].get("slope") is not None else np.nan
                           for nid in nodes_list])
        rtypes = np.array([Gs.nodes[nid].get("road_type") for nid in nodes_list], dtype=object)
        risk = (rtypes == "골목길") & (slopes >= 6)

        comps = list(nx.connected_components(Gs))
        giant = max((len(c) for c in comps), default=0)
        avg_k = 2 * M / N if N else 0
        dens = nx.density(Gs) if N > 1 else 0
        clust = nx.average_clustering(Gs) if N > 2 else 0

        row = dict(gu=gu, N=N, M=M, avg_k=round(avg_k, 3), density=round(dens, 6),
                   clustering=round(clust, 4), giant_ratio=round(giant / N, 4),
                   n_components=len(comps),
                   risk_n=int(risk.sum()), risk_ratio=round(float(risk.mean()), 4))

        # 결빙구역 그룹 응집도
        for mask_full, label in [(icing_mask_full, "icing"), (heating_mask_full, "heating")]:
            if mask_full is None:
                continue
            grp = np.array([bool(mask_full[nid]) for nid in nodes_list])
            k = int(grp.sum())
            row[f"{label}_n"] = k
            if k >= 3 and len(eu):
                ei, cond = vec_ei_conductance(eu, ev, degree, grp)
                rand_ei = np.empty(N_RANDOM); rand_cond = np.empty(N_RANDOM)
                all_idx = np.arange(N)
                for i in range(N_RANDOM):
                    sel = rng.choice(all_idx, size=k, replace=False)
                    m = np.zeros(N, dtype=bool); m[sel] = True
                    rand_ei[i], rand_cond[i] = vec_ei_conductance(eu, ev, degree, m)
                z_ei = (ei - np.nanmean(rand_ei)) / np.nanstd(rand_ei) if np.nanstd(rand_ei) > 0 else np.nan
                row[f"{label}_EI"] = round(float(ei), 4)
                row[f"{label}_EI_z"] = round(float(z_ei), 2) if z_ei == z_ei else None
                row[f"{label}_precision"] = round(float(risk[grp].mean()), 4)
            else:
                row[f"{label}_EI"] = None; row[f"{label}_EI_z"] = None; row[f"{label}_precision"] = None
        rows.append(row)
        print(f"  [{gu}] N={N:,} M={M:,} <k>={avg_k:.2f} 위험군={risk.sum():,}({risk.mean()*100:.1f}%) "
              f"({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("/tmp/network_stats_25gu.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: /tmp/network_stats_25gu.csv ({time.time()-t0:.1f}s)")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
