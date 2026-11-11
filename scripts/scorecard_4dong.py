"""4개동 road_graph.json(1,190노드) 위에서: 결빙구역·열선설치(정밀라우팅)·시급도(top-P_i
naive)·Greedy-submodular·MILP 다섯 그룹을 동일 기준(precision=급경사골목길 적중률,
E-I/conductance=응집도, 랜덤 10,000회 대비 z)으로 비교. "정확도 손실 없이 응집도를
회복하는가"가 이 스크립트의 핵심 질문.
"""
import json
import math
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0
STEEP_THRESH = 6.0
N_RANDOM = 10000
SEED = 42
ICING_MAX_DIST_M = 50.0


def _euclid_m(lon1, lat1, lon2, lat2):
    return ((lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2 + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2) ** 0.5


def point_seg_dist_m(px, py, ax, ay, bx, by):
    def to_m(x, y):
        return (x * _M_PER_DEG_LON, y * _M_PER_DEG_LAT)
    pxm, pym = to_m(px, py); axm, aym = to_m(ax, ay); bxm, bym = to_m(bx, by)
    dx, dy = bxm - axm, bym - aym
    if dx == 0 and dy == 0:
        return math.hypot(pxm - axm, pym - aym)
    t = ((pxm - axm) * dx + (pym - aym) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = axm + t * dx, aym + t * dy
    return math.hypot(pxm - cx, pym - cy)


def split_sides(loc):
    parts = re.split(r"[~∼]", str(loc))
    return [p.strip() for p in parts if p.strip()]


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


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ---- road_graph.json 로드 (노드=도로링크, 엣지=실제 위상연결) ----
    with open(f"{DATA}/road_graph.json") as f:
        rg = json.load(f)
    G = nx.node_link_graph(rg, edges="links")
    nodes = list(G.nodes())
    idx = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)
    print(f"road_graph.json: N={n} M={G.number_of_edges()}", flush=True)

    slope_pct = np.full(n, np.nan)
    road_type = np.empty(n, dtype=object)
    lon = np.zeros(n); lat = np.zeros(n)
    for nid, attrs in G.nodes(data=True):
        i = idx[nid]
        sp = attrs.get("slope_pct")
        slope_pct[i] = sp if sp is not None else np.nan
        road_type[i] = attrs.get("road_type")
        lon[i] = attrs.get("lon", np.nan); lat[i] = attrs.get("lat", np.nan)

    is_golmok = road_type == "골목길"
    is_steep = slope_pct >= STEEP_THRESH
    risk_mask = is_golmok & is_steep
    n_risk = int(risk_mask.sum())
    print(f"위험군(급경사골목길)={n_risk}/{n} ({n_risk/n*100:.1f}%)", flush=True)

    eu_l = []; ev_l = []
    for u, v in G.edges():
        eu_l.append(idx[u]); ev_l.append(idx[v])
    eu = np.array(eu_l); ev = np.array(ev_l)
    degree = np.zeros(n, dtype=np.int64)
    np.add.at(degree, eu, 1); np.add.at(degree, ev, 1)

    # ---- 그룹 1: 결빙구역(공식지정, 거리매칭) ----
    icing_mask = np.array([bool(G.nodes[nid].get("icing")) for nid in nodes])

    # ---- 그룹 2: 열선설치(정밀라우팅, /tmp 캐시 재사용) ----
    import pickle
    with open("/tmp/heating_geocode_cache.pkl", "rb") as f:
        geo_cache = pickle.load(f)
    heat_df = pd.read_csv(f"{DATA}/heating_2026.csv", encoding="utf-8-sig")
    sub_heat = heat_df[heat_df["관리기관"] == "성북구"].copy()
    sub_heat["sides"] = sub_heat["설치위치 (시점 ~ 종점)"].apply(split_sides)

    SNAP_MAX_M = 80.0; MAX_ROUTE_M = 1200.0
    SNAP_CELL = 40.0
    scell_lat = SNAP_CELL / _M_PER_DEG_LAT; scell_lon = SNAP_CELL / _M_PER_DEG_LON
    snap_bins = defaultdict(list)
    for i in range(n):
        k = (int(lon[i] / scell_lon), int(lat[i] / scell_lat))
        snap_bins[k].append(i)

    def snap_nearest(lo, la, max_m=SNAP_MAX_M):
        cx, cy = int(lo / scell_lon), int(la / scell_lat)
        best_d, best_i = None, None
        rad = 1
        while best_i is None and rad <= 4:
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    for i in snap_bins.get((cx + dx, cy + dy), []):
                        d = _euclid_m(lo, la, lon[i], lat[i])
                        if best_d is None or d < best_d:
                            best_d, best_i = d, i
            rad += 1
        if best_d is not None and best_d <= max_m:
            return best_i, best_d
        return None, None

    Gr = nx.Graph()
    Gr.add_nodes_from(range(n))
    for u, v in G.edges():
        a, b = idx[u], idx[v]
        w = _euclid_m(lon[a], lat[a], lon[b], lat[b])
        Gr.add_edge(a, b, weight=max(w, 1.0))
    comps = sorted(nx.connected_components(Gr), key=len, reverse=True)
    giant_nodes = set(comps[0])

    heating_mask = np.zeros(n, dtype=bool)
    n_routed = 0
    for _, row in sub_heat.iterrows():
        pts = []
        for side in row["sides"]:
            full = f"서울특별시 성북구 {side}"
            pts.append(geo_cache.get(full))
        if len(pts) < 2 or pts[0] is None or pts[1] is None:
            continue
        (lon1, lat1), (lon2, lat2) = pts[0], pts[1]
        i1, _ = snap_nearest(lon1, lat1); i2, _ = snap_nearest(lon2, lat2)
        if i1 is None or i2 is None or i1 not in giant_nodes or i2 not in giant_nodes:
            continue
        try:
            plen = nx.shortest_path_length(Gr, i1, i2, weight="weight")
            if plen > MAX_ROUTE_M:
                continue
            path = nx.shortest_path(Gr, i1, i2, weight="weight")
        except nx.NetworkXNoPath:
            continue
        for node in path:
            heating_mask[node] = True
        n_routed += 1
    print(f"열선설치(정밀라우팅): {n_routed}/{len(sub_heat)}건 매칭, 링크수={heating_mask.sum()}", flush=True)

    # ---- 그룹 3~5: MILP / Greedy-submod / Greedy-naive(top-P_i) ----
    comp = pd.read_csv("/tmp/greedy_vs_milp_comparison.csv")
    comp["node"] = comp["node"].astype(str)
    node_str_to_idx = {str(nid): idx[nid] for nid in nodes}
    milp_mask = np.zeros(n, dtype=bool)
    submod_mask = np.zeros(n, dtype=bool)
    naive_mask = np.zeros(n, dtype=bool)
    n_missing = 0
    for _, row in comp.iterrows():
        ni = node_str_to_idx.get(row["node"])
        if ni is None:
            n_missing += 1
            continue
        if row["in_milp"]:
            milp_mask[ni] = True
        if row["in_greedy_submod"]:
            submod_mask[ni] = True
        if row["in_greedy_naive"]:
            naive_mask[ni] = True
    print(f"MILP={milp_mask.sum()} Greedy-submod={submod_mask.sum()} top-P_i(naive)={naive_mask.sum()} "
          f"(그래프에서 못찾은 후보 {n_missing}개)", flush=True)

    # ---- 랜덤 10,000회 (heating_mask 크기 기준 — 그룹간 크기 다르므로 그룹별로 각각) ----
    def random_null(k):
        rand_precision = np.empty(N_RANDOM)
        rand_ei = np.empty(N_RANDOM)
        rand_cond = np.empty(N_RANDOM)
        all_idx = np.arange(n)
        for i in range(N_RANDOM):
            sel = rng.choice(all_idx, size=k, replace=False)
            mask = np.zeros(n, dtype=bool); mask[sel] = True
            rand_precision[i] = risk_mask[mask].mean()
            ei, cond = vec_ei_conductance(eu, ev, degree, mask)
            rand_ei[i] = ei; rand_cond[i] = cond
        return rand_precision, rand_ei, rand_cond

    def group_stats(mask, label):
        k = int(mask.sum())
        if k == 0:
            return None
        rp, rei, rcond = random_null(k)
        precision = risk_mask[mask].mean()
        recall = mask[risk_mask].mean() if n_risk else np.nan
        ei, cond = vec_ei_conductance(eu, ev, degree, mask)
        z_prec = (precision - np.nanmean(rp)) / np.nanstd(rp) if np.nanstd(rp) > 0 else np.nan
        pct_prec = (rp < precision).mean() * 100
        z_ei = (ei - np.nanmean(rei)) / np.nanstd(rei) if np.nanstd(rei) > 0 else np.nan
        z_cond = (cond - np.nanmean(rcond)) / np.nanstd(rcond) if np.nanstd(rcond) > 0 else np.nan
        return dict(group=label, N=k, precision=round(precision, 4),
                    recall=round(recall, 4) if recall == recall else None,
                    precision_z=round(z_prec, 2), precision_pct=round(pct_prec, 1),
                    E_I=round(ei, 4), EI_z=round(z_ei, 2),
                    conductance=round(cond, 4), cond_z=round(z_cond, 2))

    results = []
    for mask, label in [
        (icing_mask, "결빙구역(공식지정)"),
        (heating_mask, "열선설치(정밀라우팅)"),
        (naive_mask, "시급도(top-P_i)"),
        (submod_mask, "Greedy-submodular"),
        (milp_mask, "MILP+Bonus_ij"),
    ]:
        r = group_stats(mask, label)
        if r:
            results.append(r)
            print(f"  [{label:>18}] N={r['N']:>4} precision={r['precision']:.3f}(z={r['precision_z']}, {r['precision_pct']}%ile) "
                  f"recall={r['recall']} E-I={r['E_I']}(z={r['EI_z']}) cond={r['conductance']}(z={r['cond_z']})", flush=True)

    df = pd.DataFrame(results)
    df.to_csv("/tmp/scorecard_4dong.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 (총 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
