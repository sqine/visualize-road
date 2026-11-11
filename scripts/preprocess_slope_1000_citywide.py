"""1:1,000 수치지형도(신규 589개 도엽, 16개 자치구 부분 커버) → 경사도 재계산 (2차 확장).

배경:
    preprocess_slope_1000.py 가 정릉2동/정릉3동/성북동/돈암2동 4개동(24개 도엽)에 대해
    수행했던 것과 동일한 방식을, 이번에 새로 받은 589개 도엽(서울시 수치지형도 파일/
    (B010)수치지도_*.zip)에 대해 적용한다. 단, 이번 도엽들은 한 덩어리가 아니라
    16개 자치구에 걸쳐 흩어져 있음(성북구 89매·종로구 85매·광진구 73매 등, 사이사이
    빈 공간 많음) — 그래서 기존처럼 "전체 bbox 사각형 안이면 갱신" 방식을 쓰면 실제로는
    데이터가 없는 자치구 사이 빈 공간의 링크까지 TIN 볼록껍질에 걸려 엉뚱한 값(서로 먼
    두 클러스터를 잇는 긴 삼각형으로 보간)이 나올 위험이 있다.
    이를 막기 위해 "589개 도엽 인덱스 폴리곤(N1A_H0010000)의 합집합"을 실제 커버리지
    마스크로 만들고, 그 마스크 안에 있는 링크만 갱신한다(단순 bbox 대신 폴리곤 포함 판정).

입력:
    ../../서울시 수치지형도 파일/(B010)수치지도_*.zip 안의
        N1A_H0010000.shp/.dbf/.shx (도엽 인덱스 폴리곤, 커버리지 마스크용)
        N1L_F0010000.shp/.dbf/.shx (등고선/계곡선 + 등고수치, point cloud용)
    data/road_links.geojson (갱신 대상: 커버리지 마스크 내부 링크만)

좌표계: EPSG:5186 (Korea 2000 / Central Belt 2010) → WGS84

처리:
    1) 589개 zip에서 인덱스 폴리곤 + 등고선 point cloud를 메모리에서 직접 파싱(디스크 압축해제 없음)
    2) 인덱스 폴리곤 합집합(3m 버퍼)을 커버리지 마스크로 사용, STRtree로 링크 필터링
    3) TIN(Delaunay) 선형보간 1순위, 볼록껍질 밖(도엽 경계 부근)은 적응형 IDW로 대체
       (기존 preprocess_slope_1000.py 와 동일 알고리즘)
    4) 커버리지 마스크 내부 링크만 elev_start_m/elev_end_m/dist_m/slope_pct/slope_level 갱신,
       source_1000=True 플래그 갱신(기존에 이미 1:1,000이었던 4개동 링크도 새 데이터로 재계산됨)
    5) 원본은 .bak3_pre_citywide1000 으로 백업 후 덮어쓰기
"""

from __future__ import annotations

import glob
import io
import json
import math
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import shapefile
from pyproj import Transformer
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon, shape as shp_shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent.parent  # streamlit_app
ROOT = HERE.parent  # road-analysis
SRC_DIR = ROOT / "서울시 수치지형도 파일"
DATA = HERE / "data"
ROAD_LINKS = DATA / "road_links.geojson"

_TRANSFORMER = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
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
        self.bins: dict[tuple[int, int], list[int]] = {}
        self.lon: list[float] = []
        self.lat: list[float] = []
        self.h: list[float] = []

    def _key(self, lon, lat):
        return (int(lon / self.cell_lon), int(lat / self.cell_lat))

    def add(self, lon, lat, h):
        idx = len(self.lon)
        self.lon.append(lon)
        self.lat.append(lat)
        self.h.append(h)
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
    if pct < 3:
        return "0_3"
    if pct < 6:
        return "3_6"
    if pct < 10:
        return "6_10"
    return "10_plus"


def _path_length_m(coords):
    total = 0.0
    for i in range(1, len(coords)):
        total += _euclid_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    return total


def main():
    t0 = time.time()

    zips = sorted(glob.glob(str(SRC_DIR / "(B010)수치지도_*.zip")))
    print(f"[1/7] 대상 zip: {len(zips)}개")

    tile_polys = []
    idx = GridIndex(cell_m=20.0)
    pts_lonlat: list[tuple[float, float]] = []
    pts_elev: list[float] = []
    n_lines_total = 0
    n_zip_err = 0

    print("[2/7] 도엽 인덱스 폴리곤 + 등고선 point cloud 파싱 (메모리, 압축해제 없음)")
    for zi, zpath in enumerate(zips):
        try:
            with zipfile.ZipFile(zpath) as z:
                # --- 인덱스 폴리곤 (커버리지 마스크) ---
                shp = io.BytesIO(z.read("N1A_H0010000.shp"))
                dbf = io.BytesIO(z.read("N1A_H0010000.dbf"))
                shx = io.BytesIO(z.read("N1A_H0010000.shx"))
                r = shapefile.Reader(shp=shp, dbf=dbf, shx=shx, encoding="cp949")
                s0 = r.shape(0)
                ring = [_TRANSFORMER.transform(x, y) for x, y in s0.points]
                if len(ring) >= 3:
                    tile_polys.append(Polygon(ring))

                # --- 등고선(계곡선) + 등고수치 ---
                shp2 = io.BytesIO(z.read("N1L_F0010000.shp"))
                dbf2 = io.BytesIO(z.read("N1L_F0010000.dbf"))
                shx2 = io.BytesIO(z.read("N1L_F0010000.shx"))
                r2 = shapefile.Reader(shp=shp2, dbf=dbf2, shx=shx2, encoding="cp949")
                field_names = [f[0] for f in r2.fields[1:]]
                elev_idx = field_names.index("등고수치")
                for rec, s in zip(r2.records(), r2.shapes()):
                    try:
                        elev = float(rec[elev_idx])
                    except (ValueError, TypeError):
                        continue
                    n_lines_total += 1
                    for x, y in s.points:
                        lon, lat = _TRANSFORMER.transform(x, y)
                        idx.add(lon, lat, elev)
                        pts_lonlat.append((lon, lat))
                        pts_elev.append(elev)
        except Exception as e:
            n_zip_err += 1
        if (zi + 1) % 100 == 0:
            print(f"  {zi+1}/{len(zips)} 처리...")

    print(f"  파싱 완료: 도엽 {len(tile_polys)}개, 등고선 {n_lines_total:,}개, "
          f"point cloud {len(pts_lonlat):,}점, 에러 {n_zip_err}개")

    print("[3/7] 커버리지 마스크(도엽 합집합) 구축")
    coverage = unary_union(tile_polys).buffer(0.00003)  # ~3m 버퍼 (타일 경계 수치오차 보정)
    cov_parts = list(coverage.geoms) if coverage.geom_type == "MultiPolygon" else [coverage]
    cov_tree = STRtree(cov_parts)
    print(f"  커버리지 폴리곤(분리영역) {len(cov_parts)}개")

    print("[4/7] TIN(Delaunay) 구축")
    P = np.array(pts_lonlat)
    Z = np.array(pts_elev)
    tri = Delaunay(P)
    print(f"  TIN 삼각형 {len(tri.simplices):,}개")

    def query_tin(lon, lat):
        p = np.array([lon, lat])
        simplex = tri.find_simplex(p)
        if simplex < 0:
            return None
        verts = tri.simplices[simplex]
        b = tri.transform[simplex, :2].dot(p - tri.transform[simplex, 2])
        bary = np.array([b[0], b[1], 1 - b[0] - b[1]])
        return float(np.dot(bary, Z[verts]))

    def in_coverage(lon, lat):
        pt = Point(lon, lat)
        for i in cov_tree.query(pt):
            if cov_parts[i].contains(pt):
                return True
        return False

    print("[5/7] road_links.geojson 로드 + 커버리지 내부 링크 필터링")
    with ROAD_LINKS.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    print(f"  전체 링크: {len(feats):,}")

    t1 = time.time()
    n_in_range = 0
    n_updated = 0
    n_prev_1000 = 0
    counts = {"0_3": 0, "3_6": 0, "6_10": 0, "10_plus": 0, "unknown": 0}
    golmok_before, golmok_after = [], []

    for feat in feats:
        coords = feat["geometry"]["coordinates"]
        if not coords or len(coords) < 2:
            continue
        lon_s, lat_s = coords[0]
        lon_e, lat_e = coords[-1]
        mid_lon = (lon_s + lon_e) / 2
        mid_lat = (lat_s + lat_e) / 2
        if not in_coverage(mid_lon, mid_lat):
            continue
        n_in_range += 1
        props = feat["properties"]
        was_1000 = bool(props.get("source_1000"))
        if was_1000:
            n_prev_1000 += 1
        is_golmok = props.get("road_type") == "골목길"
        if is_golmok and props.get("slope_pct") is not None:
            golmok_before.append(props["slope_pct"])

        dist_m = _path_length_m(coords)
        if dist_m < 5.0:
            dist_m = 5.0

        h_s = query_tin(lon_s, lat_s)
        h_e = query_tin(lon_e, lat_e)
        method = "TIN"
        r_s = r_e = 0.0
        nd_s = nd_e = 3

        if h_s is None or h_e is None:
            method = "IDW_fallback"
            h_s2, d_s, n_s, nd_s, r_s = idx.interpolate_adaptive(lon_s, lat_s)
            h_e2, d_e, n_e, nd_e, r_e = idx.interpolate_adaptive(lon_e, lat_e)
            h_s = h_s if h_s is not None else h_s2
            h_e = h_e if h_e is not None else h_e2

        if h_s is None or h_e is None:
            continue

        raw = abs(h_e - h_s) / dist_m * 100.0
        lvl = "unknown"
        slope = None
        if raw <= 30.0:
            slope = raw
            lvl = slope_level(slope)
        counts[lvl] += 1

        props.update(
            elev_start_m=round(h_s, 2),
            elev_end_m=round(h_e, 2),
            dist_m=round(dist_m, 1),
            slope_pct=round(slope, 2) if slope is not None else None,
            slope_level=lvl,
            source_1000=True,
            slope1000_method=method,
            slope1000_radius_m=round(max(r_s, r_e), 1),
            slope1000_n_distinct=min(nd_s, nd_e),
        )
        n_updated += 1
        if is_golmok and slope is not None:
            golmok_after.append(slope)

    print(f"  커버리지 내부 링크: {n_in_range:,} (기존 1:1,000이었던 링크 {n_prev_1000:,}개 포함) "
          f"실제 갱신: {n_updated:,}  ({time.time()-t1:.1f}s)")
    print("  갱신된 링크의 경사도 등급 분포:")
    for k, v in counts.items():
        pct = v / n_updated * 100 if n_updated else 0
        print(f"    {k:>8}: {v:>5,}  ({pct:5.1f}%)")

    if golmok_before and golmok_after:
        import statistics
        print(f"\n  골목길 경사도(갱신분, 갱신 전 값 있던 것만) 평균: "
              f"{statistics.mean(golmok_before):.2f}% → {statistics.mean(golmok_after):.2f}%  "
              f"(n={len(golmok_before)}/{len(golmok_after)})")

    print("[6/7] 백업")
    bak = ROAD_LINKS.with_name(ROAD_LINKS.stem + ".geojson.bak3_pre_citywide1000")
    if not bak.exists():
        shutil.copy2(ROAD_LINKS, bak)
        print(f"  백업: {bak.name}")
    else:
        print(f"  백업 이미 존재, 스킵: {bak.name}")

    print("[7/7] 저장")
    with ROAD_LINKS.open("w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, separators=(",", ":"))
    mb = ROAD_LINKS.stat().st_size / 1024 / 1024
    print(f"  {ROAD_LINKS.name} {mb:.1f} MB")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
