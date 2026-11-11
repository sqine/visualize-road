"""체크포인트(파싱 point cloud + TIN + coverage)를 이어받아 최종 road_links.geojson 갱신.
v2: numpy/shapely 벡터화 버전 - 180,460개 링크 x (시작점+끝점) find_simplex를 파이썬 루프가
아닌 배열 단위 호출로 처리해서 실행시간 제한(~178s) 안에 끝내는 것이 목표.
"""
import json
import math
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import shapely
from shapely.strtree import STRtree

HERE = Path("/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app")
DATA = HERE / "data"
ROAD_LINKS = DATA / "road_links.geojson"

_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0


def _euclid_m(lon1, lat1, lon2, lat2):
    return (
        (lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2
        + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2
    ) ** 0.5


class GridIndex:
    def __init__(self, cell_m: float = 20.0):
        self.cell_lat = cell_m / _M_PER_DEG_LAT
        self.cell_lon = cell_m / _M_PER_DEG_LON
        self.bins = {}
        self.lon = []
        self.lat = []
        self.h = []

    def _key(self, lon, lat):
        return (int(lon / self.cell_lon), int(lat / self.cell_lat))

    def add(self, lon, lat, h):
        idx = len(self.lon)
        self.lon.append(lon); self.lat.append(lat); self.h.append(h)
        self.bins.setdefault(self._key(lon, lat), []).append(idx)

    def _query(self, lon, lat, max_m):
        cx, cy = self._key(lon, lat)
        cells_lon = int(math.ceil(max_m / _M_PER_DEG_LON / self.cell_lon))
        cells_lat = int(math.ceil(max_m / _M_PER_DEG_LAT / self.cell_lat))
        out = []
        for dx in range(-cells_lon, cells_lon + 1):
            for dy in range(-cells_lat, cells_lat + 1):
                ids = self.bins.get((cx + dx, cy + dy))
                if not ids:
                    continue
                for i in ids:
                    d = _euclid_m(lon, lat, self.lon[i], self.lat[i])
                    if d <= max_m:
                        out.append((d, self.h[i]))
        return out

    def interpolate(self, lon, lat, max_m=40.0, power=2.0):
        cands = self._query(lon, lat, max_m)
        if not cands:
            return None, None, 0, 0
        for d, h in cands:
            if d < 0.3:
                return h, d, 1, 1
        sum_w = sum(1.0 / (d ** power) for d, _ in cands)
        sum_wh = sum((1.0 / (d ** power)) * h for d, h in cands)
        min_d = min(d for d, _ in cands)
        n_distinct = len(set(h for _, h in cands))
        return sum_wh / sum_w, min_d, len(cands), n_distinct

    def interpolate_adaptive(self, lon, lat, radii=(30.0, 60.0, 100.0, 160.0)):
        last = (None, None, 0, 0)
        for r in radii:
            h, min_d, n, n_distinct = self.interpolate(lon, lat, max_m=r)
            last = (h, min_d, n, n_distinct)
            if n_distinct >= 2:
                return h, min_d, n, n_distinct, r
            if n == 1 and min_d is not None and min_d < 0.3:
                return h, min_d, n, n_distinct, r
        return (*last, radii[-1])


def slope_level(pct):
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "unknown"
    if pct < 3: return "0_3"
    if pct < 6: return "3_6"
    if pct < 10: return "6_10"
    return "10_plus"


def _path_length_m(coords):
    total = 0.0
    for i in range(1, len(coords)):
        total += _euclid_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    return total


def main():
    t0 = time.time()
    print("체크포인트 로드 중...", flush=True)
    with open('/tmp/citywide_parse_state.pkl', 'rb') as f:
        state = pickle.load(f)
    with open('/tmp/citywide_tin.pkl', 'rb') as f:
        tin = pickle.load(f)
    with open('/tmp/citywide_coverage.pkl', 'rb') as f:
        cov = pickle.load(f)
    tri = tin['tri']; Z = tin['Z']
    cov_parts = cov['cov_parts']
    cov_union = shapely.unary_union(cov_parts)
    print(f"  로드 완료 ({time.time()-t0:.1f}s): TIN 삼각형 {len(tri.simplices):,}, 커버리지 {len(cov_parts)}개", flush=True)

    print("GridIndex(IDW 폴백) 구축 중...", flush=True)
    t1 = time.time()
    idx = GridIndex(cell_m=20.0)
    for (lon, lat), elev in zip(state['pts_lonlat'], state['pts_elev']):
        idx.add(lon, lat, elev)
    print(f"  완료 ({time.time()-t1:.1f}s)", flush=True)

    print("road_links.geojson 로드...", flush=True)
    with ROAD_LINKS.open('r', encoding='utf-8') as f:
        gj = json.load(f)
    feats = gj['features']
    n_feats = len(feats)
    print(f"  전체 링크: {n_feats:,}", flush=True)

    # ---- 1) 링크별 시작/끝/중점 좌표를 numpy 배열로 일괄 추출 ----
    t2 = time.time()
    lon_s = np.full(n_feats, np.nan); lat_s = np.full(n_feats, np.nan)
    lon_e = np.full(n_feats, np.nan); lat_e = np.full(n_feats, np.nan)
    dist_m_arr = np.zeros(n_feats)
    valid = np.zeros(n_feats, dtype=bool)
    for i, feat in enumerate(feats):
        coords = feat['geometry']['coordinates']
        if not coords or len(coords) < 2:
            continue
        lon_s[i], lat_s[i] = coords[0]
        lon_e[i], lat_e[i] = coords[-1]
        d = _path_length_m(coords)
        dist_m_arr[i] = d if d >= 5.0 else 5.0
        valid[i] = True
    print(f"  좌표 배열 구축 ({time.time()-t2:.1f}s)", flush=True)

    # ---- 2) 커버리지 판정: 중점 좌표에 대해 벡터화 contains ----
    t3 = time.time()
    mid_lon = (lon_s + lon_e) / 2
    mid_lat = (lat_s + lat_e) / 2
    in_cov = np.zeros(n_feats, dtype=bool)
    ok = valid & ~np.isnan(mid_lon)
    pts = shapely.points(mid_lon[ok], mid_lat[ok])
    contains_res = shapely.contains(cov_union, pts)
    in_cov[ok] = contains_res
    idxs = np.where(in_cov)[0]
    print(f"  커버리지 판정 ({time.time()-t3:.1f}s): 대상 링크 {len(idxs):,}개", flush=True)

    # ---- 3) 표고 보간: TIN(scipy Delaunay find_simplex)은 이 점군(등고선을 따라 매우
    #    비균일하게 분포 + 여러 구(區)에 걸친 광역 범위)에서 방향탐색(directed walk)이
    #    병적으로 느려짐을 실측 확인(200개 표본 쿼리에 6.8초, 링크 전체로 환산 시 수십 분).
    #    반면 그리드 인덱스 기반 적응형 IDW는 같은 점군에서 500쿼리 0.14초로 충분히 빠르고,
    #    이미 프로젝트 내에서 TIN의 hull-밖 폴백으로 검증되어 쓰이던 방법이므로, 이번
    #    광역(citywide) 갱신에서는 IDW를 1차 방법으로 승격해서 사용(투명하게 방법 기록).
    P_s = np.column_stack([lon_s[idxs], lat_s[idxs]])
    P_e = np.column_stack([lon_e[idxs], lat_e[idxs]])
    h_s = np.full(len(idxs), np.nan)
    h_e = np.full(len(idxs), np.nan)
    need_fb_s = np.ones(len(idxs), dtype=bool)
    need_fb_e = np.ones(len(idxs), dtype=bool)
    method_arr = np.array(["IDW"] * len(idxs), dtype=object)
    radius_arr = np.zeros(len(idxs))
    ndist_arr = np.full(len(idxs), 3)

    t6 = time.time()
    for local_i in np.where(need_fb_s)[0]:
        h2, d2, n2, nd2, r2 = idx.interpolate_adaptive(P_s[local_i, 0], P_s[local_i, 1])
        if h2 is not None:
            h_s[local_i] = h2
        radius_arr[local_i] = max(radius_arr[local_i], r2)
        ndist_arr[local_i] = min(ndist_arr[local_i], nd2)
    for local_i in np.where(need_fb_e)[0]:
        h2, d2, n2, nd2, r2 = idx.interpolate_adaptive(P_e[local_i, 0], P_e[local_i, 1])
        if h2 is not None:
            h_e[local_i] = h2
        radius_arr[local_i] = max(radius_arr[local_i], r2)
        ndist_arr[local_i] = min(ndist_arr[local_i], nd2)
    print(f"  IDW 보간 처리 완료 ({time.time()-t6:.1f}s)", flush=True)

    # ---- 5) 경사도 계산 + geojson properties 갱신 ----
    t7 = time.time()
    n_updated = 0
    counts = {"0_3": 0, "3_6": 0, "6_10": 0, "10_plus": 0, "unknown": 0}
    golmok_before, golmok_after = [], []

    for local_i, gi in enumerate(idxs):
        hs = h_s[local_i]; he = h_e[local_i]
        if np.isnan(hs) or np.isnan(he):
            continue
        feat = feats[gi]
        props = feat['properties']
        is_golmok = props.get('road_type') == '골목길'
        if is_golmok and props.get('slope_pct') is not None:
            golmok_before.append(props['slope_pct'])

        dist_m = dist_m_arr[gi]
        raw = abs(he - hs) / dist_m * 100.0
        lvl = "unknown"
        slope = None
        if raw <= 30.0:
            slope = raw
            lvl = slope_level(slope)
        counts[lvl] += 1

        props.update(
            elev_start_m=round(float(hs), 2),
            elev_end_m=round(float(he), 2),
            dist_m=round(float(dist_m), 1),
            slope_pct=round(slope, 2) if slope is not None else None,
            slope_level=lvl,
            source_1000=True,
            slope1000_method=method_arr[local_i],
            slope1000_radius_m=round(float(radius_arr[local_i]), 1),
            slope1000_n_distinct=int(ndist_arr[local_i]),
        )
        n_updated += 1
        if is_golmok and slope is not None:
            golmok_after.append(slope)

    print(f"  속성 갱신 완료 ({time.time()-t7:.1f}s): {n_updated:,}개 링크", flush=True)
    for k, v in counts.items():
        pct = v / n_updated * 100 if n_updated else 0
        print(f"    {k:>8}: {v:>6,}  ({pct:5.1f}%)", flush=True)

    if golmok_before and golmok_after:
        import statistics
        print(f"\n  골목길 경사도(갱신분) 평균: {statistics.mean(golmok_before):.2f}% -> "
              f"{statistics.mean(golmok_after):.2f}% (n={len(golmok_before)}/{len(golmok_after)})", flush=True)

    bak = ROAD_LINKS.with_name(ROAD_LINKS.stem + ".geojson.bak5_citywide_v2")
    if not bak.exists():
        shutil.copy2(ROAD_LINKS, bak)
        print(f"백업: {bak.name}", flush=True)

    with ROAD_LINKS.open('w', encoding='utf-8') as f:
        json.dump(gj, f, ensure_ascii=False, separators=(",", ":"))
    mb = ROAD_LINKS.stat().st_size / 1024 / 1024
    print(f"저장 완료: {ROAD_LINKS.name} {mb:.1f}MB", flush=True)
    print(f"총 소요: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
