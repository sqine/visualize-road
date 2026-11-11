"""v2: 열선설치 그룹을 도로명 매칭이 아니라 VWorld 지오코딩(시점/종점) + 도로망
라우팅 최단경로 클리핑으로 재구성 — 나머지(결빙구역/시급도/랜덤 10,000회)는 v1과
동일 파이프라인 유지, 정확히 비교 가능하게.
"""
import json
import math
import pickle
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import shapely
from shapely.geometry import shape

HERE_DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0
CELL_M = 5.0
STEEP_THRESH = 6.0
N_RANDOM = 10000
SEED = 42
SNAP_MAX_M = 80.0       # 지오코딩된 점 -> 가장 가까운 링크로 스냅 허용 거리
MAX_ROUTE_M = 1200.0    # 시점~종점 최단경로가 이보다 길면 매칭 실패로 간주(엉뚱한 경로 방지)

TARGET8 = ["성북구", "중랑구", "광진구", "서대문구", "종로구", "성동구", "동대문구", "중구"]


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

    with open("/tmp/heating_geocode_cache.pkl", "rb") as f:
        geo_cache = pickle.load(f)

    heat_df = pd.read_csv(f"{HERE_DATA}/heating_2026.csv", encoding="utf-8-sig")
    icing_df = pd.read_csv(f"{HERE_DATA}/icing_zones.csv", encoding="utf-8-sig")
    icing_df.columns = [c.strip() for c in icing_df.columns]
    ICING_MAX_DIST_M = 50.0

    with open(f"{HERE_DATA}/seoul_gu.geojson") as f:
        gu_gj = json.load(f)
    gu_polys = {ft["properties"]["gu"]: shape(ft["geometry"]) for ft in gu_gj["features"]}

    with open(f"{HERE_DATA}/road_links.geojson") as f:
        rl = json.load(f)
    feats = rl["features"]
    print(f"[준비] road_links.geojson 전체 {len(feats):,}건 로드 ({time.time()-t0:.1f}s)", flush=True)

    all_results = []
    match_diag = []

    for gu in TARGET8:
        tg = time.time()
        print(f"\n================ {gu} ================", flush=True)
        poly = gu_polys[gu]

        gu_feats = []
        for feat in feats:
            if not feat["properties"].get("source_1000"):
                continue
            c = feat["geometry"]["coordinates"]
            mid = c[len(c) // 2]
            if poly.contains(shapely.points(mid[0], mid[1])):
                gu_feats.append(feat)
        n = len(gu_feats)
        if n < 200:
            print("  [건너뜀] 표본 부족", flush=True)
            continue

        lon_s = np.zeros(n); lat_s = np.zeros(n); lon_e = np.zeros(n); lat_e = np.zeros(n)
        mid_lon = np.zeros(n); mid_lat = np.zeros(n)
        slope_pct = np.full(n, np.nan)
        road_type = np.empty(n, dtype=object)
        for i, feat in enumerate(gu_feats):
            c = feat["geometry"]["coordinates"]
            lon_s[i], lat_s[i] = c[0]; lon_e[i], lat_e[i] = c[-1]
            mid = c[len(c) // 2]
            mid_lon[i], mid_lat[i] = mid
            p = feat["properties"]
            sp = p.get("slope_pct")
            slope_pct[i] = sp if sp is not None else np.nan
            road_type[i] = p.get("road_type")

        is_golmok = road_type == "골목길"
        is_steep = slope_pct >= STEEP_THRESH
        risk_mask = is_golmok & is_steep
        n_risk = int(risk_mask.sum())
        print(f"  정밀 링크 N={n:,}  위험군(급경사골목길)={n_risk}개 ({time.time()-tg:.1f}s)", flush=True)

        # ---- 위상그래프(5m 격자스냅) + 라우팅용 가중치(링크간 직선거리) ----
        cell_lat = CELL_M / _M_PER_DEG_LAT
        cell_lon = CELL_M / _M_PER_DEG_LON

        def qkey(lon, lat):
            return (round(lon / cell_lon), round(lat / cell_lat))

        junction_links = defaultdict(list)
        for i in range(n):
            junction_links[qkey(lon_s[i], lat_s[i])].append(i)
            junction_links[qkey(lon_e[i], lat_e[i])].append(i)
        edge_set = set()
        MAX_J = 40
        for jid, lids in junction_links.items():
            uniq = sorted(set(lids))
            if len(uniq) < 2:
                continue
            if len(uniq) > MAX_J:
                hub = uniq[0]
                for other in uniq[1:]:
                    a, b = (hub, other) if hub < other else (other, hub)
                    edge_set.add((a, b))
                continue
            for a in range(len(uniq)):
                for b in range(a + 1, len(uniq)):
                    edge_set.add((uniq[a], uniq[b]))

        edges = np.array(sorted(edge_set), dtype=np.int64)
        eu, ev = edges[:, 0], edges[:, 1]
        degree = np.zeros(n, dtype=np.int64)
        np.add.at(degree, eu, 1); np.add.at(degree, ev, 1)

        Gr = nx.Graph()
        Gr.add_nodes_from(range(n))
        for a, b in edge_set:
            w = _euclid_m(mid_lon[a], mid_lat[a], mid_lon[b], mid_lat[b])
            Gr.add_edge(a, b, weight=max(w, 1.0))
        comps = sorted(nx.connected_components(Gr), key=len, reverse=True)
        giant_nodes = set(comps[0])

        # snap 용 그리드 인덱스 (링크 중점)
        SNAP_CELL = 40.0
        scell_lat = SNAP_CELL / _M_PER_DEG_LAT; scell_lon = SNAP_CELL / _M_PER_DEG_LON
        snap_bins = defaultdict(list)
        for i in range(n):
            k = (int(mid_lon[i] / scell_lon), int(mid_lat[i] / scell_lat))
            snap_bins[k].append(i)

        def snap_nearest(lon, lat, max_m=SNAP_MAX_M):
            cx, cy = int(lon / scell_lon), int(lat / scell_lat)
            best_d, best_i = None, None
            rad = 1
            while best_i is None and rad <= 4:
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        for i in snap_bins.get((cx + dx, cy + dy), []):
                            d = _euclid_m(lon, lat, mid_lon[i], mid_lat[i])
                            if best_d is None or d < best_d:
                                best_d, best_i = d, i
                rad += 1
            if best_d is not None and best_d <= max_m:
                return best_i, best_d
            return None, None

        # ---- 열선설치: 지오코딩된 시점/종점 -> 스냅 -> 최단경로 라우팅 ----
        sub_heat = heat_df[heat_df["관리기관"] == gu].copy()
        sub_heat["sides"] = sub_heat["설치위치 (시점 ~ 종점)"].apply(split_sides)

        heating_mask = np.zeros(n, dtype=bool)
        n_recs = len(sub_heat)
        n_routed = 0
        n_geo_fail = 0
        n_snap_fail = 0
        n_route_fail = 0
        n_toolong = 0
        route_lengths = []
        for _, row in sub_heat.iterrows():
            sides = row["sides"]
            pts = []
            for side in sides:
                full = f"서울특별시 {gu} {side}"
                pt = geo_cache.get(full)
                pts.append(pt)
            if len(pts) < 2 or pts[0] is None or pts[1] is None:
                n_geo_fail += 1
                continue
            (lon1, lat1), (lon2, lat2) = pts[0], pts[1]
            i1, d1 = snap_nearest(lon1, lat1)
            i2, d2 = snap_nearest(lon2, lat2)
            if i1 is None or i2 is None:
                n_snap_fail += 1
                continue
            if i1 not in giant_nodes or i2 not in giant_nodes:
                n_route_fail += 1
                continue
            try:
                path = nx.shortest_path(Gr, i1, i2, weight="weight")
                plen = nx.shortest_path_length(Gr, i1, i2, weight="weight")
            except nx.NetworkXNoPath:
                n_route_fail += 1
                continue
            if plen > MAX_ROUTE_M:
                n_toolong += 1
                continue
            for node in path:
                heating_mask[node] = True
            n_routed += 1
            route_lengths.append(plen)

        print(f"  열선설치 매칭: 전체 {n_recs}건 중 경로매칭 성공 {n_routed}건 "
              f"(지오코딩실패 {n_geo_fail}, 스냅실패 {n_snap_fail}, 경로없음/성분분리 {n_route_fail}, "
              f"경로과다({MAX_ROUTE_M:.0f}m초과) {n_toolong})  "
              f"경로길이 평균={np.mean(route_lengths):.0f}m (n={len(route_lengths)})", flush=True)
        match_diag.append(dict(gu=gu, 전체건수=n_recs, 라우팅성공=n_routed, 지오코딩실패=n_geo_fail,
                                스냅실패=n_snap_fail, 경로실패=n_route_fail, 경로과다=n_toolong,
                                heating_링크수=int(heating_mask.sum())))

        # ---- 결빙구역 (v1과 동일: 거리기반) ----
        SEOUL_BBOX = (126.76, 37.42, 127.19, 37.71)
        idf = icing_df.dropna(subset=["기점 위도(WGS84(4326))", "기점 경도(WGS84(4326))",
                                       "종점 위도(WGS84(4326))", "종점 경도(WGS84(4326))"]).copy()
        idf = idf[idf["기점 경도(WGS84(4326))"].astype(float).between(*SEOUL_BBOX[0:3:2]) &
                  idf["기점 위도(WGS84(4326))"].astype(float).between(*SEOUL_BBOX[1:4:2])]
        segs = idf[["기점 경도(WGS84(4326))", "기점 위도(WGS84(4326))",
                    "종점 경도(WGS84(4326))", "종점 위도(WGS84(4326))"]].astype(float).values
        icing_mask = np.zeros(n, dtype=bool)
        if len(segs) > 0:
            for i in range(n):
                best = min(point_seg_dist_m(mid_lon[i], mid_lat[i], s[0], s[1], s[2], s[3]) for s in segs)
                if best <= ICING_MAX_DIST_M:
                    icing_mask[i] = True

        # ---- 시급도 우선배치 (N=열선 링크수와 동일) ----
        n_target = int(heating_mask.sum())
        urgency_mask = np.zeros(n, dtype=bool)
        golmok_idx = np.where(is_golmok & ~np.isnan(slope_pct))[0]
        if n_target > 0 and len(golmok_idx) > 0:
            order = golmok_idx[np.argsort(-slope_pct[golmok_idx])]
            urgency_mask[order[:n_target]] = True

        print(f"  그룹 크기: 열선≈{heating_mask.sum()} 결빙≈{icing_mask.sum()} 시급도={urgency_mask.sum()}", flush=True)

        # ---- 랜덤 10,000회 ----
        tR = time.time()
        if n_target > 0:
            rand_precision = np.empty(N_RANDOM)
            rand_ei = np.empty(N_RANDOM)
            rand_cond = np.empty(N_RANDOM)
            all_idx = np.arange(n)
            for k in range(N_RANDOM):
                sel = rng.choice(all_idx, size=n_target, replace=False)
                mask = np.zeros(n, dtype=bool); mask[sel] = True
                rand_precision[k] = risk_mask[mask].mean()
                ei, cond = vec_ei_conductance(eu, ev, degree, mask)
                rand_ei[k] = ei; rand_cond[k] = cond
        else:
            rand_precision = rand_ei = rand_cond = np.array([np.nan])
        print(f"  랜덤 {N_RANDOM}회 완료 ({time.time()-tR:.1f}s)", flush=True)

        def group_stats(mask, label):
            k = int(mask.sum())
            if k == 0:
                return None
            precision = risk_mask[mask].mean()
            recall = mask[risk_mask].mean() if n_risk else np.nan
            ei, cond = vec_ei_conductance(eu, ev, degree, mask)
            z_prec = (precision - np.nanmean(rand_precision)) / np.nanstd(rand_precision) if np.nanstd(rand_precision) > 0 else np.nan
            pct_prec = (rand_precision < precision).mean() * 100
            z_ei = (ei - np.nanmean(rand_ei)) / np.nanstd(rand_ei) if np.nanstd(rand_ei) > 0 else np.nan
            z_cond = (cond - np.nanmean(rand_cond)) / np.nanstd(rand_cond) if np.nanstd(rand_cond) > 0 else np.nan
            return dict(gu=gu, group=label, N=k, precision=round(precision, 4),
                        recall=round(recall, 4) if recall == recall else None,
                        precision_z=round(z_prec, 2) if z_prec == z_prec else None,
                        precision_pct=round(pct_prec, 1),
                        E_I=round(ei, 4) if ei == ei else None, EI_z=round(z_ei, 2) if z_ei == z_ei else None,
                        conductance=round(cond, 4) if cond == cond else None,
                        cond_z=round(z_cond, 2) if z_cond == z_cond else None)

        for mask, label in [(heating_mask, "열선설치(정밀라우팅)"), (icing_mask, "결빙구역"), (urgency_mask, "시급도우선배치")]:
            r = group_stats(mask, label)
            if r:
                all_results.append(r)
                print(f"    [{label:>12}] N={r['N']:>4} precision={r['precision']:.3f}(z={r['precision_z']}, {r['precision_pct']}%ile) "
                      f"recall={r['recall']} E-I={r['E_I']}(z={r['EI_z']}) cond={r['conductance']}(z={r['cond_z']})", flush=True)
        all_results.append(dict(gu=gu, group="랜덤(평균,n=10000)", N=n_target,
                                 precision=round(float(np.nanmean(rand_precision)), 4), recall=None,
                                 precision_z=0.0, precision_pct=50.0,
                                 E_I=round(float(np.nanmean(rand_ei)), 4), EI_z=0.0,
                                 conductance=round(float(np.nanmean(rand_cond)), 4), cond_z=0.0))
        print(f"  {gu} 완료 (총 {time.time()-tg:.1f}s)", flush=True)

    df = pd.DataFrame(all_results)
    df.to_csv("/tmp/unified_district_eval_v2.csv", index=False, encoding="utf-8-sig")
    ddf = pd.DataFrame(match_diag)
    ddf.to_csv("/tmp/heating_routing_diag.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 (총 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
