"""서울 전역(2,489개 도엽, 25개 자치구 전량) 1:1,000 수치지형도 반영 - v2.
parse_citywide_chunk.py(SRC_DIR를 서울시 수치지형도 파일/수치지형도/1_1000 으로 변경, state를
citywide_parse_state_v2.pkl 로 분리)의 체크포인트를 이어받아 road_links.geojson 전체를 갱신한다.

방법론: TIN(Delaunay) find_simplex는 이 점군(등고선을 따라 매우 비균일하게 분포 + 25개구
전역 범위)에서 방향탐색이 병적으로 느려짐을 이전 8~9구 확장 때 이미 실측 확인했으므로,
이번 전역(25개구) 갱신에서도 처음부터 그리드 인덱스 기반 적응형 IDW를 1차(유일) 방법으로 사용한다.

체크포인트 방식: 링크별 갱신 결과를 /tmp/citywide_updates_v2.pkl 에 {link_index: propertiesدict}
로 누적 저장(매 호출마다 로드+병합+저장). 모든 대상 링크 처리가 끝나면 road_links.geojson을
1회만 다시 읽어 누적된 갱신을 전부 적용하고 저장한다.
"""
import json
import math
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import shapely

HERE = Path("/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app")
DATA = HERE / "data"
ROAD_LINKS = DATA / "road_links.geojson"
STATE_FILE = Path("/tmp/citywide_parse_state_v2.pkl")
GRIDIDX_CACHE = Path("/tmp/citywide_gridindex_v2.pkl")
TARGET_IDXS_CACHE = Path("/tmp/citywide_target_idxs_v2.pkl")
UPDATES_CACHE = Path("/tmp/citywide_updates_v2.pkl")

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


def build_gridindex_and_targets():
    t0 = time.time()
    print("체크포인트 로드...", flush=True)
    with open(STATE_FILE, "rb") as f:
        state = pickle.load(f)
    print(f"  point cloud {len(state['pts_lonlat']):,}점, 도엽 {len(state['tile_polys'])}개 "
          f"({time.time()-t0:.1f}s)", flush=True)

    print("커버리지 마스크(도엽 합집합) 구축...", flush=True)
    t1 = time.time()
    coverage = shapely.unary_union(state["tile_polys"]).buffer(0.00003)
    cov_parts = list(coverage.geoms) if coverage.geom_type == "MultiPolygon" else [coverage]
    cov_union = shapely.unary_union(cov_parts)
    print(f"  완료 ({time.time()-t1:.1f}s): 분리영역 {len(cov_parts)}개", flush=True)

    print("GridIndex 구축...", flush=True)
    t2 = time.time()
    idx = GridIndex(cell_m=20.0)
    for (lon, lat), elev in zip(state["pts_lonlat"], state["pts_elev"]):
        idx.add(lon, lat, elev)
    print(f"  완료 ({time.time()-t2:.1f}s)", flush=True)

    with open(GRIDIDX_CACHE, "wb") as f:
        pickle.dump(idx, f)

    print("road_links.geojson 로드 + 좌표 배열 구축...", flush=True)
    with ROAD_LINKS.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    n_feats = len(feats)
    lon_s = np.full(n_feats, np.nan); lat_s = np.full(n_feats, np.nan)
    lon_e = np.full(n_feats, np.nan); lat_e = np.full(n_feats, np.nan)
    dist_m_arr = np.zeros(n_feats)
    valid = np.zeros(n_feats, dtype=bool)
    for i, feat in enumerate(feats):
        coords = feat["geometry"]["coordinates"]
        if not coords or len(coords) < 2:
            continue
        lon_s[i], lat_s[i] = coords[0]
        lon_e[i], lat_e[i] = coords[-1]
        d = _path_length_m(coords)
        dist_m_arr[i] = d if d >= 5.0 else 5.0
        valid[i] = True

    mid_lon = (lon_s + lon_e) / 2
    mid_lat = (lat_s + lat_e) / 2
    in_cov = np.zeros(n_feats, dtype=bool)
    ok = valid & ~np.isnan(mid_lon)
    pts = shapely.points(mid_lon[ok], mid_lat[ok])
    contains_res = shapely.contains(cov_union, pts)
    in_cov[ok] = contains_res
    idxs = np.where(in_cov)[0]
    print(f"  커버리지 판정 완료: 대상 링크 {len(idxs):,}개 / 전체 {n_feats:,}개", flush=True)

    with open(TARGET_IDXS_CACHE, "wb") as f:
        pickle.dump(dict(idxs=idxs, lon_s=lon_s, lat_s=lat_s, lon_e=lon_e, lat_e=lat_e,
                          dist_m_arr=dist_m_arr, n_feats=n_feats), f)
    print(f"전처리 총 소요: {time.time()-t0:.1f}s", flush=True)


def main():
    t0 = time.time()
    if not GRIDIDX_CACHE.exists() or not TARGET_IDXS_CACHE.exists():
        build_gridindex_and_targets()

    print("캐시 로드...", flush=True)
    with open(GRIDIDX_CACHE, "rb") as f:
        idx = pickle.load(f)
    with open(TARGET_IDXS_CACHE, "rb") as f:
        tgt = pickle.load(f)
    idxs = tgt["idxs"]; lon_s = tgt["lon_s"]; lat_s = tgt["lat_s"]
    lon_e = tgt["lon_e"]; lat_e = tgt["lat_e"]; dist_m_arr = tgt["dist_m_arr"]
    print(f"  로드 완료 ({time.time()-t0:.1f}s): 대상 링크 {len(idxs):,}개", flush=True)

    if UPDATES_CACHE.exists():
        with open(UPDATES_CACHE, "rb") as f:
            updates = pickle.load(f)
    else:
        updates = {}

    remaining = [gi for gi in idxs if gi not in updates]
    print(f"  이미 처리: {len(updates):,}  남음: {len(remaining):,}", flush=True)

    BATCH = 50000
    batch = remaining[:BATCH]
    t1 = time.time()
    counts = {"0_3": 0, "3_6": 0, "6_10": 0, "10_plus": 0, "unknown": 0}
    for gi in batch:
        h_s, d_s, n_s, nd_s, r_s = idx.interpolate_adaptive(lon_s[gi], lat_s[gi])
        h_e, d_e, n_e, nd_e, r_e = idx.interpolate_adaptive(lon_e[gi], lat_e[gi])
        if h_s is None or h_e is None:
            updates[gi] = None  # 보간 실패, 갱신 스킵 표시(재시도 안 하게)
            continue
        dist_m = dist_m_arr[gi]
        raw = abs(h_e - h_s) / dist_m * 100.0
        lvl = "unknown"
        slope = None
        if raw <= 30.0:
            slope = raw
            lvl = slope_level(slope)
        counts[lvl] += 1
        updates[gi] = dict(
            elev_start_m=round(float(h_s), 2),
            elev_end_m=round(float(h_e), 2),
            dist_m=round(float(dist_m), 1),
            slope_pct=round(slope, 2) if slope is not None else None,
            slope_level=lvl,
            source_1000=True,
            slope1000_method="IDW_v2_citywide",
            slope1000_radius_m=round(float(max(r_s, r_e)), 1),
            slope1000_n_distinct=int(min(nd_s, nd_e)),
        )
    print(f"  배치 처리 완료 ({time.time()-t1:.1f}s): 이번 배치 {len(batch):,}개, "
          f"누적 {len(updates):,}/{len(idxs):,}", flush=True)
    for k, v in counts.items():
        print(f"    {k:>8}: {v:>7,}", flush=True)

    with open(UPDATES_CACHE, "wb") as f:
        pickle.dump(updates, f)

    if len(updates) < len(idxs):
        print(f"남은 링크: {len(idxs)-len(updates):,}개 -- 다시 실행하세요 (총 소요 {time.time()-t0:.1f}s)", flush=True)
        return

    print("모든 링크 처리 완료. road_links.geojson에 최종 반영...", flush=True)
    with ROAD_LINKS.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    n_applied = 0
    for gi, props_update in updates.items():
        if props_update is None:
            continue
        feats[gi]["properties"].update(props_update)
        n_applied += 1
    print(f"  {n_applied:,}개 링크에 반영", flush=True)

    bak = ROAD_LINKS.with_name(ROAD_LINKS.stem + ".geojson.bak_pre_citywide_v2_full")
    if not bak.exists():
        shutil.copy2(ROAD_LINKS, bak)
        print(f"백업: {bak.name}", flush=True)
    with ROAD_LINKS.open("w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, separators=(",", ":"))
    mb = ROAD_LINKS.stat().st_size / 1024 / 1024
    print(f"저장 완료: {ROAD_LINKS.name} {mb:.1f}MB", flush=True)
    print(f"총 소요(이번 호출): {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
