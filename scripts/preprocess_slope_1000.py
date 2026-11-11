"""1:1,000 수치지형도 등고선(계곡선) → 골목 단위 고정밀 경사도 재계산.

배경:
    기존 preprocess_slope.py 는 서울시 1:5,000 표고점(76,580개, 점간 평균 ~89m)을
    100m 격자로 IDW 보간해 사용했다. 이번에 국토정보플랫폼에서 정릉2동·정릉3동·
    성북동·돈암2동 4개 행정동을 커버하는 1:1,000 수치지형도 24개 도엽을 받았는데,
    그 안의 "표고점(N1P_F0020000)" 레이어는 타일당 3개뿐이라 오히려 더 희박하다.
    대신 "등고선/계곡선(N1L_F0010000)" 레이어가 타일당 100~200개 라인, 정점 간격
    평균 1.84m 로 훨씬 촘촘하다 — 그래서 표고점이 아니라 등고선 정점을
    "그 등고선의 표고값을 가진 점들의 집합"으로 변환해 point cloud 로 재구성하고,
    기존과 동일한 IDW 방식을 그대로 적용한다.

입력:
    data/수치지형도/정릉성북_1000_1600/*.zip 안의 N1L_F0010000.shp/.dbf/.prj
    data/road_links.geojson (대상: road_type 무관, 좌표가 새 데이터 커버 범위 내인 링크만 갱신)

좌표계:
    원본 EPSG:5186 (Korea 2000 / Central Belt 2010) → WGS84. pyproj 사용
    (전처리 스크립트 전용 의존성 — 배포되는 Streamlit 앱 런타임에는 불필요).

처리:
    1) 24개 도엽 zip 압축 해제 (임시 디렉토리)
    2) 각 도엽의 N1L_F0010000.shp(라인)+dbf(등고수치) 파싱
    3) 각 라인의 정점(vertex) 전체를 (lon, lat, elevation) 점으로 변환
       → 등고선 하나당 평균 14개 정점이므로, 199개 라인/도엽이면 약 2,800점/도엽
    4) TIN(Delaunay 삼각분할) 선형보간을 1순위로 사용 — 단순 IDW는 두 질의점이
       같은 등고선 밴드 근방에 몰리면 표고차가 0으로 뭉개지는 문제가 실측으로
       확인되어(예: 32731 IDW 결과 slope 0.67%, 인접밴드 미검출 아님에도 발생),
       실제 지형을 삼각망으로 근사하는 TIN을 기본으로 채택함. TIN 볼록껍질
       바깥(도엽 경계 부근)에 걸리는 점만 GridIndex+적응형 반경 IDW로 대체.
    5) road_links.geojson 중 새 데이터 커버 범위(24개 도엽 합집합 bbox) 내에 있는
       링크만 elev_start_m/elev_end_m/dist_m/slope_pct/slope_level 갱신,
       source_1000=True 플래그 추가 (범위 밖은 기존 1:5,000 값 유지)
    6) 원본은 .bak2 로 백업 후 덮어쓰기

출력 컬럼 갱신/추가:
    elev_start_m, elev_end_m, dist_m, slope_pct, slope_level (덮어씀)
    source_1000 (True/False — 이번 1:1,000 데이터로 갱신됐는지 여부)
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
TOPO_DIR = DATA / "수치지형도" / "정릉성북_1000_1600"
ROAD_LINKS = DATA / "road_links.geojson"
TMP_EXTRACT = Path("/tmp/preprocess_1000_extract")

_TRANSFORMER = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)

_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0


def _euclid_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return (
        (lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2
        + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2
    ) ** 0.5


# ─────────────────────────────────────────────────────────────────────
# SHP/DBF 파싱 (기존 스크립트와 동일한 순수 struct 파싱 방식)
# ─────────────────────────────────────────────────────────────────────

def parse_shp_polylines(shp_path: Path):
    with shp_path.open("rb") as f:
        f.seek(100)
        while True:
            rec_hdr = f.read(8)
            if len(rec_hdr) < 8:
                return
            content_len = struct.unpack(">i", rec_hdr[4:8])[0] * 2
            content = f.read(content_len)
            st = struct.unpack("<i", content[0:4])[0]
            if st not in (3, 13):
                continue
            n_parts = struct.unpack("<i", content[36:40])[0]
            n_points = struct.unpack("<i", content[40:44])[0]
            pts_offset = 44 + 4 * n_parts
            pts = []
            for i in range(n_points):
                off = pts_offset + i * 16
                x, y = struct.unpack("<2d", content[off:off + 16])
                pts.append((x, y))
            yield pts


def parse_dbf_field(dbf_path: Path, field_name: str) -> list[str]:
    out: list[str] = []
    with dbf_path.open("rb") as f:
        hdr = f.read(32)
        n_records = struct.unpack("<i", hdr[4:8])[0]
        hdr_len = struct.unpack("<h", hdr[8:10])[0]
        rec_len = struct.unpack("<h", hdr[10:12])[0]
        f.seek(32)
        fields = []
        offset = 1
        target_offset = None
        target_len = None
        while True:
            b = f.read(32)
            if not b or b[0] == 0x0D:
                break
            name = b[0:11].split(b"\x00", 1)[0].decode("cp949", errors="replace")
            flen = b[16]
            if name == field_name:
                target_offset, target_len = offset, flen
            offset += flen
            fields.append((name, flen))
        if target_offset is None:
            raise ValueError(f"field {field_name} not found")
        f.seek(hdr_len)
        for _ in range(n_records):
            rec = f.read(rec_len)
            if len(rec) < rec_len:
                break
            v = rec[target_offset:target_offset + target_len].decode(
                "cp949", errors="replace"
            ).strip()
            out.append(v)
    return out


# ─────────────────────────────────────────────────────────────────────
# GridIndex + IDW (기존 preprocess_slope.py 와 동일 로직)
# ─────────────────────────────────────────────────────────────────────

class GridIndex:
    def __init__(self, cell_m: float = 20.0):
        self.cell_lat = cell_m / _M_PER_DEG_LAT
        self.cell_lon = cell_m / _M_PER_DEG_LON
        self.bins: dict[tuple[int, int], list[int]] = {}
        self.lon: list[float] = []
        self.lat: list[float] = []
        self.h: list[float] = []

    def _key(self, lon: float, lat: float) -> tuple[int, int]:
        return (int(lon / self.cell_lon), int(lat / self.cell_lat))

    def add(self, lon: float, lat: float, h: float) -> None:
        idx = len(self.lon)
        self.lon.append(lon)
        self.lat.append(lat)
        self.h.append(h)
        self.bins.setdefault(self._key(lon, lat), []).append(idx)

    def _query(self, lon: float, lat: float, max_m: float):
        """반경 내 (거리, 표고) 목록 그대로 반환 (보간 전 원시 후보)."""
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

    def interpolate(self, lon: float, lat: float, max_m: float = 40.0, power: float = 2.0):
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

    def interpolate_adaptive(
        self, lon: float, lat: float, radii: tuple[float, ...] = (30.0, 60.0, 100.0, 160.0)
    ):
        """서로 다른 등고선(표고값)이 최소 2개 이상 잡힐 때까지 반경을 넓혀가며 보간.

        단일 등고선 밴드 안에만 점이 몰려있으면 IDW 결과가 그 밴드 값으로 수렴해
        버려(=경사가 0으로 뭉개짐) 근본적으로 틀린 결과가 나온다. 이를 막기 위해
        최소 2개 서로 다른 표고값이 잡히는 첫 반경을 채택한다.
        """
        last = (None, None, 0, 0)
        for r in radii:
            h, min_d, n, n_distinct = self.interpolate(lon, lat, max_m=r)
            last = (h, min_d, n, n_distinct)
            if n_distinct >= 2:
                return h, min_d, n, n_distinct, r
            if n == 1 and min_d is not None and min_d < 0.3:
                # 거의 정확히 일치하는 점이 있으면 그대로 신뢰
                return h, min_d, n, n_distinct, r
        return (*last, radii[-1])


def slope_level(pct: float | None) -> str:
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "unknown"
    if pct < 3:
        return "0_3"
    if pct < 6:
        return "3_6"
    if pct < 10:
        return "6_10"
    return "10_plus"


def _path_length_m(coords: list[list[float]]) -> float:
    total = 0.0
    for i in range(1, len(coords)):
        total += _euclid_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    return total


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("[1/6] 도엽 zip 압축 해제")
    if TMP_EXTRACT.exists():
        shutil.rmtree(TMP_EXTRACT)
    TMP_EXTRACT.mkdir(parents=True)
    zips = sorted(TOPO_DIR.glob("*.zip"))
    for z in zips:
        code = None
        for token in z.stem.split("_"):
            if token.isdigit() and len(token) == 9:
                code = token
                break
        if code is None:
            code = z.stem
        out_dir = TMP_EXTRACT / code
        out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
    print(f"  {len(zips)}개 도엽 압축 해제 완료")

    print("[2/6] 등고선(계곡선) 정점 → point cloud 변환 + TIN 구축")
    idx = GridIndex(cell_m=20.0)
    pts_lonlat: list[tuple[float, float]] = []
    pts_elev: list[float] = []
    n_lines_total = 0
    bbox_min_lon, bbox_min_lat = 999.0, 999.0
    bbox_max_lon, bbox_max_lat = -999.0, -999.0
    for tile_dir in sorted(TMP_EXTRACT.iterdir()):
        shp = tile_dir / "N1L_F0010000.shp"
        dbf = tile_dir / "N1L_F0010000.dbf"
        if not shp.exists() or not dbf.exists():
            continue
        elevs = parse_dbf_field(dbf, "등고수치")
        for line_idx, pts_tm in enumerate(parse_shp_polylines(shp)):
            if line_idx >= len(elevs):
                break
            try:
                elev = float(elevs[line_idx])
            except ValueError:
                continue
            n_lines_total += 1
            for x, y in pts_tm:
                lon, lat = _TRANSFORMER.transform(x, y)
                idx.add(lon, lat, elev)
                pts_lonlat.append((lon, lat))
                pts_elev.append(elev)
                bbox_min_lon = min(bbox_min_lon, lon)
                bbox_max_lon = max(bbox_max_lon, lon)
                bbox_min_lat = min(bbox_min_lat, lat)
                bbox_max_lat = max(bbox_max_lat, lat)
    n_pts_total = len(pts_lonlat)
    print(f"  등고선 {n_lines_total:,}개  →  point cloud {n_pts_total:,}점")
    print(
        f"  커버 범위: lon {bbox_min_lon:.5f}~{bbox_max_lon:.5f}, "
        f"lat {bbox_min_lat:.5f}~{bbox_max_lat:.5f}"
    )

    P = np.array(pts_lonlat)
    Z = np.array(pts_elev)
    tri = Delaunay(P)
    print(f"  TIN 삼각형 {len(tri.simplices):,}개")

    def query_tin(lon: float, lat: float) -> float | None:
        p = np.array([lon, lat])
        simplex = tri.find_simplex(p)
        if simplex < 0:
            return None
        verts = tri.simplices[simplex]
        b = tri.transform[simplex, :2].dot(p - tri.transform[simplex, 2])
        bary = np.array([b[0], b[1], 1 - b[0] - b[1]])
        return float(np.dot(bary, Z[verts]))

    print("[3/6] road_links.geojson 로드")
    with ROAD_LINKS.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    print(f"  전체 링크: {len(feats):,}")

    print("[4/6] 커버 범위 내 링크만 골라 재계산")
    t1 = time.time()
    n_in_range = 0
    n_updated = 0
    counts = {"0_3": 0, "3_6": 0, "6_10": 0, "10_plus": 0, "unknown": 0}
    for feat in feats:
        coords = feat["geometry"]["coordinates"]
        if not coords or len(coords) < 2:
            continue
        lon_s, lat_s = coords[0]
        lon_e, lat_e = coords[-1]
        mid_lon = (lon_s + lon_e) / 2
        mid_lat = (lat_s + lat_e) / 2
        if not (
            bbox_min_lon <= mid_lon <= bbox_max_lon
            and bbox_min_lat <= mid_lat <= bbox_max_lat
        ):
            continue
        n_in_range += 1

        dist_m = _path_length_m(coords)
        if dist_m < 5.0:
            dist_m = 5.0

        h_s = query_tin(lon_s, lat_s)
        h_e = query_tin(lon_e, lat_e)
        method = "TIN"
        r_s = r_e = 0.0
        nd_s = nd_e = 3  # TIN은 항상 서로 다른 3개 정점으로 보간

        if h_s is None or h_e is None:
            # TIN 볼록껍질 바깥(도엽 경계 부근) → 적응형 IDW로 대체
            method = "IDW_fallback"
            h_s2, d_s, n_s, nd_s, r_s = idx.interpolate_adaptive(lon_s, lat_s)
            h_e2, d_e, n_e, nd_e, r_e = idx.interpolate_adaptive(lon_e, lat_e)
            h_s = h_s if h_s is not None else h_s2
            h_e = h_e if h_e is not None else h_e2

        if h_s is None or h_e is None:
            continue  # 범위 안이지만 160m 이내에도 등고선이 없는 예외 케이스 → 기존값 유지

        raw = abs(h_e - h_s) / dist_m * 100.0
        lvl = "unknown"
        slope = None
        if raw <= 30.0:
            slope = raw
            lvl = slope_level(slope)
        counts[lvl] += 1

        feat["properties"].update(
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

    print(f"  커버 범위 내 링크: {n_in_range:,}  실제 갱신: {n_updated:,}  ({time.time()-t1:.1f}s)")
    print("  갱신된 링크의 경사도 등급 분포:")
    for k, v in counts.items():
        pct = v / n_updated * 100 if n_updated else 0
        print(f"    {k:>8}: {v:>5,}  ({pct:5.1f}%)")

    print("[5/6] 백업")
    bak = ROAD_LINKS.with_suffix(".geojson.bak2")
    if not bak.exists():
        shutil.copy2(ROAD_LINKS, bak)
        print(f"  백업: {bak.name}")
    else:
        print(f"  백업 이미 존재, 스킵: {bak.name}")

    print("[6/6] 저장")
    with ROAD_LINKS.open("w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, separators=(",", ":"))
    mb = ROAD_LINKS.stat().st_size / 1024 / 1024
    print(f"  {ROAD_LINKS.name} {mb:.1f} MB")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
