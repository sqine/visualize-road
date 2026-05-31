"""표고 SHP/DBF → 도로별 경사도 계산 → seoul_roads.geojson 갱신.

입력:
    data/slope_raw/표고 5000/N3P_F002.shp   - 표고 POINT (EPSG:5174, X/Y만)
    data/slope_raw/표고 5000/N3P_F002.dbf   - HEIGHT 필드(m)
    data/seoul_roads.geojson                - 도로 라인 (이미 WGS84)

처리:
    1) SHP POINT 좌표 76,580 + DBF HEIGHT 추출
    2) EPSG:5174 → WGS84 변환
    3) 100m × 100m 그리드 인덱스 구축
    4) 도로별 시작/끝점에서 가장 가까운 표고 점 lookup (반경 200m 이내)
    5) slope_pct = |H_end - H_start| / horizontal_dist_m × 100
    6) slope_level 컬럼 추가 (0~3 / 3~6 / 6~10 / ≥10 / 미상)
    7) data/seoul_roads.geojson 덮어쓰기

출력 컬럼 추가:
    elev_start_m, elev_end_m, dist_m, slope_pct, slope_level
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from pathlib import Path

# proj 모듈 import 위해 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.proj import tm5174_to_wgs84  # noqa: E402


HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
SHP = DATA / "slope_raw" / "표고 5000" / "N3P_F002.shp"
DBF = DATA / "slope_raw" / "표고 5000" / "N3P_F002.dbf"
ROADS = DATA / "seoul_roads.geojson"


# ─────────────────────────────────────────────────────────────────────
# SHP/DBF 파싱
# ─────────────────────────────────────────────────────────────────────

def parse_dbf_height_field(dbf_path: Path) -> list[float]:
    """DBF에서 HEIGHT 필드만 추출."""
    heights: list[float] = []
    with dbf_path.open("rb") as f:
        hdr = f.read(32)
        n_records = struct.unpack("<i", hdr[4:8])[0]
        hdr_len = struct.unpack("<h", hdr[8:10])[0]
        rec_len = struct.unpack("<h", hdr[10:12])[0]

        # field descriptors
        f.seek(32)
        fields = []
        while True:
            b = f.read(32)
            if not b or b[0] == 0x0D:
                break
            name = b[0:11].split(b"\x00", 1)[0].decode("cp949", errors="replace")
            ftype = chr(b[11])
            flen = b[16]
            fdec = b[17]
            fields.append((name, ftype, flen, fdec))

        # HEIGHT 컬럼 인덱스/오프셋
        height_offset = 1  # 첫 1바이트는 삭제 플래그
        height_len = None
        for name, _ftype, flen, _fdec in fields:
            if name == "HEIGHT":
                height_len = flen
                break
            height_offset += flen
        if height_len is None:
            raise ValueError("HEIGHT field not found in DBF")

        f.seek(hdr_len)
        for _ in range(n_records):
            rec = f.read(rec_len)
            if len(rec) < rec_len:
                break
            v = rec[height_offset:height_offset + height_len].decode(
                "cp949", errors="replace"
            ).strip()
            try:
                heights.append(float(v))
            except ValueError:
                heights.append(float("nan"))
    return heights


def parse_shp_points(shp_path: Path) -> list[tuple[float, float]]:
    """SHP에서 POINT (X, Y) 추출."""
    pts: list[tuple[float, float]] = []
    with shp_path.open("rb") as f:
        f.seek(100)  # 헤더 스킵
        while True:
            rec_hdr = f.read(8)
            if len(rec_hdr) < 8:
                break
            content_len = struct.unpack(">i", rec_hdr[4:8])[0] * 2
            content = f.read(content_len)
            st = struct.unpack("<i", content[0:4])[0]
            if st == 1:  # POINT
                x, y = struct.unpack("<2d", content[4:20])
                pts.append((x, y))
            elif st == 11:  # POINT-Z
                x, y, _z = struct.unpack("<3d", content[4:28])
                pts.append((x, y))
            else:
                # null/unsupported: append NaN to keep index aligned
                pts.append((float("nan"), float("nan")))
    return pts


# ─────────────────────────────────────────────────────────────────────
# 그리드 인덱스 + 최근접 lookup
# ─────────────────────────────────────────────────────────────────────

# Seoul lat ≈ 37.5°
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0


def _euclid_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return (
        (lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2
        + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2
    ) ** 0.5


class GridIndex:
    """경위도 그리드 인덱스. 100m 셀 (위도 0.0009°, 경도 0.00113°)."""

    def __init__(self, cell_m: float = 100.0):
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

    def nearest(self, lon: float, lat: float, max_m: float = 200.0):
        """가장 가까운 점의 (height, dist_m). 없으면 (None, None)."""
        cx, cy = self._key(lon, lat)
        cells_lon = int(math.ceil(max_m / _M_PER_DEG_LON / self.cell_lon))
        cells_lat = int(math.ceil(max_m / _M_PER_DEG_LAT / self.cell_lat))
        best_d = float("inf")
        best_h: float | None = None
        for dx in range(-cells_lon, cells_lon + 1):
            for dy in range(-cells_lat, cells_lat + 1):
                ids = self.bins.get((cx + dx, cy + dy))
                if not ids:
                    continue
                for i in ids:
                    d = _euclid_m(lon, lat, self.lon[i], self.lat[i])
                    if d < best_d:
                        best_d = d
                        best_h = self.h[i]
        if best_h is None or best_d > max_m:
            return None, None
        return best_h, best_d

    def interpolate(self, lon: float, lat: float, max_m: float = 60.0, power: float = 2.0):
        """반경 max_m 내 표고 점들의 IDW 보간 (역거리 가중 평균).

        도로 양옆의 능선·계곡 표고가 섞여 노이즈가 클 때 단일 최근접보다 안정적.
        반환: (height, min_dist, n_points). 후보 없으면 (None, None, 0).
        """
        cx, cy = self._key(lon, lat)
        cells_lon = int(math.ceil(max_m / _M_PER_DEG_LON / self.cell_lon))
        cells_lat = int(math.ceil(max_m / _M_PER_DEG_LAT / self.cell_lat))
        sum_w = 0.0
        sum_wh = 0.0
        min_d = float("inf")
        n = 0
        for dx in range(-cells_lon, cells_lon + 1):
            for dy in range(-cells_lat, cells_lat + 1):
                ids = self.bins.get((cx + dx, cy + dy))
                if not ids:
                    continue
                for i in ids:
                    d = _euclid_m(lon, lat, self.lon[i], self.lat[i])
                    if d > max_m:
                        continue
                    if d < 0.5:
                        # 거의 같은 지점 — 그대로 반환
                        return self.h[i], d, 1
                    w = 1.0 / (d ** power)
                    sum_w += w
                    sum_wh += w * self.h[i]
                    if d < min_d:
                        min_d = d
                    n += 1
        if n == 0 or sum_w == 0:
            return None, None, 0
        return sum_wh / sum_w, min_d, n


# ─────────────────────────────────────────────────────────────────────
# 경사도 레벨화
# ─────────────────────────────────────────────────────────────────────

def slope_level(pct: float | None) -> str:
    """경사도 % → 등급 라벨."""
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "unknown"
    if pct < 3:
        return "0_3"
    if pct < 6:
        return "3_6"
    if pct < 10:
        return "6_10"
    return "10_plus"


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print(f"[1/5] 표고 SHP/DBF 파싱")
    pts5174 = parse_shp_points(SHP)
    heights = parse_dbf_height_field(DBF)
    print(f"  shp points: {len(pts5174):,}  dbf rows: {len(heights):,}")
    n = min(len(pts5174), len(heights))

    print(f"[2/5] EPSG:5174 → WGS84 변환 + 그리드 인덱스 구축")
    idx = GridIndex(cell_m=100.0)
    skip = 0
    for i in range(n):
        x, y = pts5174[i]
        h = heights[i]
        if math.isnan(x) or math.isnan(y) or math.isnan(h):
            skip += 1
            continue
        lat, lon = tm5174_to_wgs84(x, y)
        idx.add(lon, lat, h)
    print(f"  변환 완료: {len(idx.h):,}  (스킵: {skip})  {time.time()-t0:.1f}s")
    if idx.lat:
        print(
            f"  lat 범위: {min(idx.lat):.4f} ~ {max(idx.lat):.4f}  "
            f"lon 범위: {min(idx.lon):.4f} ~ {max(idx.lon):.4f}"
        )

    print(f"[3/5] 도로 GeoJSON 로드")
    with ROADS.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    print(f"  features: {len(feats):,}")

    print(f"[4/5] 도로별 시작/끝점 elevation 매칭 + slope 계산")
    counts = {"0_3": 0, "3_6": 0, "6_10": 0, "10_plus": 0, "unknown": 0}
    t1 = time.time()
    for i, feat in enumerate(feats):
        if i and i % 5000 == 0:
            print(f"  ...{i:,} 처리 ({time.time()-t1:.1f}s)")
        coords = feat["geometry"]["coordinates"]
        if not coords or len(coords) < 2:
            feat["properties"].update(
                elev_start_m=None,
                elev_end_m=None,
                dist_m=None,
                slope_pct=None,
                slope_level="unknown",
            )
            counts["unknown"] += 1
            continue

        lon_s, lat_s = coords[0]
        lon_e, lat_e = coords[-1]

        # 실제 path 길이 (굽힘 반영)
        dist_m = 0.0
        for i in range(1, len(coords)):
            dist_m += _euclid_m(
                coords[i-1][0], coords[i-1][1],
                coords[i][0], coords[i][1],
            )
        if dist_m < 5.0:
            dist_m = 5.0

        # IDW 보간 (도로 길이 1.5배 반경, 30~150m 범위)
        search_r = max(30.0, min(150.0, dist_m * 1.5))
        h_s, min_d_s, _ = idx.interpolate(lon_s, lat_s, max_m=search_r, power=2.0)
        h_e, min_d_e, _ = idx.interpolate(lon_e, lat_e, max_m=search_r, power=2.0)

        slope = None
        lvl = "unknown"
        if (
            h_s is not None and h_e is not None
            and min_d_s <= max(search_r, 80.0)
            and min_d_e <= max(search_r, 80.0)
        ):
            raw = abs(h_e - h_s) / dist_m * 100.0
            if raw <= 30.0:  # 한국 도로 현실 cap
                slope = raw
                lvl = slope_level(slope)
            # 30% 초과는 매칭 노이즈로 간주 → unknown 유지
        counts[lvl] += 1

        feat["properties"].update(
            elev_start_m=round(h_s, 2) if h_s is not None else None,
            elev_end_m=round(h_e, 2) if h_e is not None else None,
            dist_m=round(dist_m, 1) if dist_m else None,
            slope_pct=round(slope, 2) if slope is not None else None,
            slope_level=lvl,
        )

    total = len(feats)
    print(f"\n  경사도 등급 분포 ({total:,}):")
    for k, v in counts.items():
        pct = v / total * 100 if total else 0
        print(f"    {k:>8}: {v:>7,}  ({pct:5.1f}%)")

    print(f"[5/5] 저장")
    with ROADS.open("w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, separators=(",", ":"))
    mb = ROADS.stat().st_size / 1024 / 1024
    print(f"  {ROADS.name} {mb:.1f} MB")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
