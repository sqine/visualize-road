"""8개 정밀구에 대해 동일 기준으로:
  - 그룹 4종: 열선설치(도로명매칭) / 결빙구역(거리매칭) / 시급도우선배치(경사도 top-N,
    골목길 한정) / 랜덤(n=10,000, 벡터화)
  - 위험 기준: 골목길×급경사(>=6%) 2x2 층화, 각 그룹의 정밀도(precision)·재현율(recall)
  - 네트워크 지표: E-I index, Conductance (그룹 vs 무작위 10,000회 z-score)
를 동일 파이프라인으로 계산. 매칭 방법은 주석에 명시.
"""
import json
import math
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

GU_DONG_FILES = {
    "성북구": "seongbuk_dong.geojson",
    "서대문구": "seodaemun_dong.geojson",
}
TARGET8 = ["성북구", "중랑구", "광진구", "서대문구", "종로구", "성동구", "동대문구", "중구"]


def _euclid_m(lon1, lat1, lon2, lat2):
    return ((lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2 + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2) ** 0.5


def extract_names(loc):
    """heating_2026.csv '설치위치' 텍스트에서 도로명(로/길/가 접미사) 추출.
    예: '정릉로44길 37 ~ 동소문로34길 24' -> ['정릉로44길','동소문로34길']
    (첫 정규식은 숫자에서 바로 끊어 '정릉로44길'->'정릉로'로 뭉개지는 버그가 있었음 — 수정됨)
    """
    parts = re.split(r"[~∼]", str(loc))
    names = []
    for p in parts:
        p = p.strip()
        m = re.match(r"^([가-힣0-9]+(?:로|길|가))(?=\s|$)", p)
        if m:
            names.append(m.group(1))
    return names


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


class NameGridIndex:
    """OSM road_name 최근접 매칭용 그리드(classify_road_type.py와 동일 원리, 30m 반경)."""
    def __init__(self, cell_m=30.0):
        self.cell_lat = cell_m / _M_PER_DEG_LAT
        self.cell_lon = cell_m / _M_PER_DEG_LON
        self.bins = {}
        self.lon = []; self.lat = []; self.name = []

    def _key(self, lon, lat):
        return (int(lon / self.cell_lon), int(lat / self.cell_lat))

    def add(self, lon, lat, name):
        idx = len(self.lon)
        self.lon.append(lon); self.lat.append(lat); self.name.append(name)
        self.bins.setdefault(self._key(lon, lat), []).append(idx)

    def nearest(self, lon, lat, max_m=30.0):
        cx, cy = self._key(lon, lat)
        cl = int(math.ceil(max_m / _M_PER_DEG_LON / self.cell_lon))
        ca = int(math.ceil(max_m / _M_PER_DEG_LAT / self.cell_lat))
        best_d, best_name = None, None
        for dx in range(-cl, cl + 1):
            for dy in range(-ca, ca + 1):
                for i in self.bins.get((cx + dx, cy + dy), []):
                    d = _euclid_m(lon, lat, self.lon[i], self.lat[i])
                    if best_d is None or d < best_d:
                        best_d, best_name = d, self.name[i]
        if best_d is not None and best_d <= max_m:
            return best_name
        return None


def resample_line(coords, interval_m=25.0):
    if not coords:
        return []
    pts = [(coords[0][0], coords[0][1])]
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]; lon2, lat2 = coords[i]
        seg_len = _euclid_m(lon1, lat1, lon2, lat2)
        if seg_len <= 0:
            continue
        d = 0.0
        while d + interval_m <= seg_len:
            d += interval_m
            t = d / seg_len
            pts.append((lon1 + (lon2 - lon1) * t, lat1 + (lat2 - lat1) * t))
    pts.append((coords[-1][0], coords[-1][1]))
    return pts


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

    print("[준비] OSM road_name 그리드 구축 (열선 도로명 매칭용)", flush=True)
    with open(f"{HERE_DATA}/seoul_roads.geojson") as f:
        osm_gj = json.load(f)
    name_idx = NameGridIndex(cell_m=30.0)
    for feat in osm_gj["features"]:
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        name = feat.get("properties", {}).get("road_name")
        if not name:
            continue
        for lon, lat in resample_line(geom["coordinates"], interval_m=25.0):
            name_idx.add(lon, lat, name)
    print(f"  완료 ({time.time()-t0:.1f}s, {len(name_idx.lon):,}점)", flush=True)

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
    strata_results = []

    for gu in TARGET8:
        tg = time.time()
        print(f"\n================ {gu} ================", flush=True)
        poly = gu_polys[gu]

        # ---- 1) gu 내 정밀(source_1000=True) 링크만 추출 ----
        gu_feats = []
        for feat in feats:
            if not feat["properties"].get("source_1000"):
                continue
            c = feat["geometry"]["coordinates"]
            mid = c[len(c) // 2]
            if poly.contains(shapely.points(mid[0], mid[1])):
                gu_feats.append(feat)
        n = len(gu_feats)
        print(f"  정밀 링크: {n:,}개 ({time.time()-tg:.1f}s)", flush=True)
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
        risk_mask = is_golmok & is_steep  # "급경사 골목길" = 사각지대 위험군(ground truth)
        n_risk = int(risk_mask.sum())

        # 2x2 층화 기초 통계 (건수 기준)
        strata = {
            "골목길×급경사": int((is_golmok & is_steep).sum()),
            "골목길×비급경사": int((is_golmok & ~is_steep).sum()),
            "대형도로×급경사": int((~is_golmok & is_steep).sum()),
            "대형도로×비급경사": int((~is_golmok & ~is_steep).sum()),
        }
        strata_results.append({"gu": gu, "N_전체": n, **strata})
        print(f"  2x2 층화: {strata}  (위험군 급경사골목길={n_risk}개, {n_risk/n*100:.1f}%)", flush=True)

        # ---- 2) 위상 그래프(5m 격자스냅) ----
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

        # 거대성분만 사용 (networkx로 성분탐색, 이후 지표계산은 벡터화)
        G_topo = nx.Graph()
        G_topo.add_nodes_from(range(n))
        G_topo.add_edges_from(edge_set)
        comps = sorted(nx.connected_components(G_topo), key=len, reverse=True)
        giant = np.zeros(n, dtype=bool)
        giant[list(comps[0])] = True
        print(f"  위상그래프: N={n:,} M={len(edge_set):,} 거대성분={giant.sum():,} ({time.time()-tg:.1f}s)", flush=True)

        # 거대성분 내부 엣지만 남김(간단화: 모든 엣지가 이미 같은 성분 내부일 수밖에 없음 - giant는 정보용)

        # ---- 3) 그룹 정의 ----
        # (a) 열선설치: 도로명 매칭 (OSM road_name 최근접(<=30m) == heating_2026.csv 도로명)
        sub_heat = heat_df[heat_df["관리기관"] == gu].copy()
        sub_heat["names"] = sub_heat["설치위치 (시점 ~ 종점)"].apply(extract_names)
        heating_names = set(nm for ns in sub_heat["names"] for nm in ns)
        heating_mask = np.zeros(n, dtype=bool)
        matched_names = np.empty(n, dtype=object)
        for i in range(n):
            nm = name_idx.nearest(mid_lon[i], mid_lat[i], max_m=30.0)
            matched_names[i] = nm
            if nm and nm in heating_names:
                heating_mask[i] = True
        n_heat_recs = len(sub_heat)
        n_heat_matched_recs = sum(1 for ns in sub_heat["names"] if any(nm in heating_names for nm in ns))

        # (b) 결빙구역: icing_zones.csv 각 레코드(기점~종점 직선)까지 최근접거리 <=50m
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
        n_icing_recs = len(segs)

        # (c) 시급도 우선배치: 골목길 중 경사도 상위 N개 (N = heating_mask 개수와 동일, 대형도로 제외)
        n_target = int(heating_mask.sum())
        urgency_mask = np.zeros(n, dtype=bool)
        golmok_idx = np.where(is_golmok & ~np.isnan(slope_pct))[0]
        if n_target > 0 and len(golmok_idx) > 0:
            order = golmok_idx[np.argsort(-slope_pct[golmok_idx])]
            top_idx = order[:n_target]
            urgency_mask[top_idx] = True

        print(f"  그룹 크기: 열선≈{heating_mask.sum()}(레코드 {n_heat_matched_recs}/{n_heat_recs} 매칭) "
              f"결빙≈{icing_mask.sum()}(레코드 {n_icing_recs}) 시급도={urgency_mask.sum()} "
              f"(N_target={n_target})", flush=True)

        # ---- 4) 랜덤 널분포 (n=10,000, 벡터화) ----
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
            return dict(gu=gu, group=label, N=k, precision=round(precision, 4), recall=round(recall, 4) if recall == recall else None,
                        precision_z=round(z_prec, 2) if z_prec == z_prec else None,
                        precision_pct=round(pct_prec, 1),
                        E_I=round(ei, 4) if ei == ei else None, EI_z=round(z_ei, 2) if z_ei == z_ei else None,
                        conductance=round(cond, 4) if cond == cond else None, cond_z=round(z_cond, 2) if z_cond == z_cond else None)

        for mask, label in [(heating_mask, "열선설치"), (icing_mask, "결빙구역"), (urgency_mask, "시급도우선배치")]:
            r = group_stats(mask, label)
            if r:
                all_results.append(r)
                print(f"    [{label:>8}] N={r['N']:>4} precision={r['precision']:.3f}(z={r['precision_z']}, {r['precision_pct']}%ile) "
                      f"recall={r['recall']} E-I={r['E_I']}(z={r['EI_z']}) cond={r['conductance']}(z={r['cond_z']})", flush=True)
        all_results.append(dict(gu=gu, group="랜덤(평균,n=10000)", N=n_target,
                                 precision=round(float(np.nanmean(rand_precision)), 4), recall=None,
                                 precision_z=0.0, precision_pct=50.0,
                                 E_I=round(float(np.nanmean(rand_ei)), 4), EI_z=0.0,
                                 conductance=round(float(np.nanmean(rand_cond)), 4), cond_z=0.0))
        print(f"  {gu} 처리 완료 (총 {time.time()-tg:.1f}s)", flush=True)

    df = pd.DataFrame(all_results)
    df.to_csv("/tmp/unified_district_eval.csv", index=False, encoding="utf-8-sig")
    sdf = pd.DataFrame(strata_results)
    sdf.to_csv("/tmp/unified_district_strata.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: /tmp/unified_district_eval.csv, /tmp/unified_district_strata.csv (총 {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
