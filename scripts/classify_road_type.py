"""OSM highway 태그 → TBGIS 도로링크(road_links.geojson) 공간매칭으로 도로유형 부여.

배경:
    최적화 후보(TOP10 등)는 TBGIS 도로링크(road_links.geojson, ROAD_LID 기준)
    단위로 산출되는데, TBGIS DBF에는 ROAD_LID/ROAD_CD/STDR_YM_CD 세 필드뿐이라
    도로 등급(대형도로 vs 골목길) 정보가 아예 없다.
    반면 OSM 도로(seoul_roads.geojson)에는 highway 태그가 있지만 TBGIS와
    ID 체계가 달라 직접 join이 안 된다. → 공간(위치) 기반 최근접 매칭으로 연결.

입력:
    data/seoul_roads.geojson   - OSM 도로 (highway 태그 보유, WGS84)
    data/road_links.geojson    - TBGIS 도로링크 (WGS84, highway 태그 없음, 180,460건)

처리:
    1) OSM 도로 각 라인을 ~25m 간격으로 재샘플링 → (lon, lat, category, highway) 점 그리드 인덱스 구축
    2) TBGIS 링크별로 최대 5개 지점(시작/중간들/끝) 샘플 → 반경 30m 내 최근접 OSM 샘플들의 category 다수결
    3) road_type 컬럼 추가:
         '골목길'   : residential / living_street / unclassified
         '대형도로' : primary/secondary/trunk/motorway(+_link) / busway / corridor / platform / road
         '애매'     : tertiary / tertiary_link
         'unknown'  : 매칭 실패 또는 그 외 태그
    4) 원본을 .bak 로 백업 후 road_links.geojson 덮어쓰기

출력 컬럼 추가:
    road_type, road_type_highway (매칭된 최빈 OSM highway 원본값), road_type_match_dist_m, road_type_votes
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OSM_ROADS = DATA / "seoul_roads.geojson"
ROAD_LINKS = DATA / "road_links.geojson"

# ─────────────────────────────────────────────────────────────────────
# 도로유형 분류 매핑
# ─────────────────────────────────────────────────────────────────────

GOLMOK = {"residential", "living_street", "unclassified"}
LARGE = {
    "primary", "secondary", "trunk", "motorway",
    "primary_link", "secondary_link", "trunk_link", "motorway_link",
    "busway", "corridor", "platform", "road",
}
AMBIGUOUS = {"tertiary", "tertiary_link"}


def to_category(highway: str | None) -> str:
    if not highway:
        return "unknown"
    if highway in GOLMOK:
        return "골목길"
    if highway in LARGE:
        return "대형도로"
    if highway in AMBIGUOUS:
        return "애매"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────
# 거리 계산 (서울 위도 기준 등장방형 근사, 기존 스크립트와 동일 상수)
# ─────────────────────────────────────────────────────────────────────

_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0


def _euclid_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return (
        (lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2
        + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2
    ) ** 0.5


def _resample_line(coords: list[list[float]], interval_m: float = 25.0) -> list[tuple[float, float]]:
    """라인을 interval_m 간격으로 재샘플링(원본 정점 + 보간점). 최소 시작/끝점은 항상 포함."""
    if not coords:
        return []
    if len(coords) == 1:
        return [(coords[0][0], coords[0][1])]

    pts: list[tuple[float, float]] = [(coords[0][0], coords[0][1])]
    carry = 0.0
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]
        lon2, lat2 = coords[i]
        seg_len = _euclid_m(lon1, lat1, lon2, lat2)
        if seg_len <= 0:
            continue
        d = carry
        while d + interval_m <= seg_len:
            d += interval_m
            t = d / seg_len
            pts.append((lon1 + (lon2 - lon1) * t, lat1 + (lat2 - lat1) * t))
        carry = (d + interval_m) - seg_len if d + interval_m > seg_len else 0.0
        carry = max(0.0, interval_m - (seg_len - d)) if seg_len > d else carry
    pts.append((coords[-1][0], coords[-1][1]))
    return pts


def _line_sample_points(coords: list[list[float]], max_n: int = 5) -> list[tuple[float, float]]:
    """TBGIS 링크 매칭용: 최대 max_n개 균등 간격 지점(시작~끝)."""
    if not coords:
        return []
    if len(coords) == 1:
        return [(coords[0][0], coords[0][1])]
    n = len(coords)
    if n <= max_n:
        return [(c[0], c[1]) for c in coords]
    idxs = [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    return [(coords[i][0], coords[i][1]) for i in idxs]


# ─────────────────────────────────────────────────────────────────────
# 그리드 인덱스 (카테고리 점 저장)
# ─────────────────────────────────────────────────────────────────────

class CategoryGridIndex:
    def __init__(self, cell_m: float = 30.0):
        self.cell_lat = cell_m / _M_PER_DEG_LAT
        self.cell_lon = cell_m / _M_PER_DEG_LON
        self.bins: dict[tuple[int, int], list[int]] = {}
        self.lon: list[float] = []
        self.lat: list[float] = []
        self.cat: list[str] = []
        self.hw: list[str] = []

    def _key(self, lon: float, lat: float) -> tuple[int, int]:
        return (int(lon / self.cell_lon), int(lat / self.cell_lat))

    def add(self, lon: float, lat: float, cat: str, hw: str) -> None:
        idx = len(self.lon)
        self.lon.append(lon)
        self.lat.append(lat)
        self.cat.append(cat)
        self.hw.append(hw)
        self.bins.setdefault(self._key(lon, lat), []).append(idx)

    def nearby(self, lon: float, lat: float, max_m: float):
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
                        out.append((d, self.cat[i], self.hw[i]))
        return out


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("[1/5] OSM 도로 로드 + 샘플 그리드 인덱스 구축")
    with OSM_ROADS.open("r", encoding="utf-8") as f:
        osm_gj = json.load(f)
    idx = CategoryGridIndex(cell_m=30.0)
    n_osm_pts = 0
    for feat in osm_gj["features"]:
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        hw = feat.get("properties", {}).get("highway", "")
        cat = to_category(hw)
        for lon, lat in _resample_line(geom["coordinates"], interval_m=25.0):
            idx.add(lon, lat, cat, hw)
            n_osm_pts += 1
    print(f"  OSM 라인 {len(osm_gj['features']):,}개 → 샘플점 {n_osm_pts:,}개  ({time.time()-t0:.1f}s)")

    print("[2/5] TBGIS road_links.geojson 로드")
    with ROAD_LINKS.open("r", encoding="utf-8") as f:
        rl_gj = json.load(f)
    feats = rl_gj["features"]
    print(f"  링크 {len(feats):,}건")

    print("[3/5] 링크별 최근접 OSM 카테고리 매칭 (반경 30m, 최대 5샘플)")
    t1 = time.time()
    counts = Counter()
    for i, feat in enumerate(feats):
        if i and i % 20000 == 0:
            print(f"  ...{i:,} 처리 ({time.time()-t1:.1f}s)")
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates") if geom.get("type") == "LineString" else None
        if not coords:
            feat["properties"].update(
                road_type="unknown", road_type_highway="", road_type_match_dist_m=None, road_type_votes=0,
            )
            counts["unknown"] += 1
            continue

        samples = _line_sample_points(coords, max_n=5)
        votes = Counter()
        hw_votes = Counter()
        dists = []
        for lon, lat in samples:
            near = idx.nearby(lon, lat, max_m=30.0)
            if not near:
                continue
            near.sort(key=lambda t: t[0])
            best_d, best_cat, best_hw = near[0]
            votes[best_cat] += 1
            hw_votes[best_hw] += 1
            dists.append(best_d)

        if not votes:
            feat["properties"].update(
                road_type="unknown", road_type_highway="", road_type_match_dist_m=None, road_type_votes=0,
            )
            counts["unknown"] += 1
            continue

        road_type, n_votes = votes.most_common(1)[0]
        hw_top = hw_votes.most_common(1)[0][0] if hw_votes else ""
        avg_d = round(sum(dists) / len(dists), 1) if dists else None
        feat["properties"].update(
            road_type=road_type,
            road_type_highway=hw_top,
            road_type_match_dist_m=avg_d,
            road_type_votes=n_votes,
        )
        counts[road_type] += 1

    total = len(feats)
    print(f"\n  분류 결과 ({total:,}):")
    for k, v in counts.most_common():
        pct = v / total * 100 if total else 0
        print(f"    {k:>6}: {v:>7,}  ({pct:5.1f}%)")

    print("[4/5] 백업 생성")
    bak = ROAD_LINKS.with_suffix(".geojson.bak")
    if not bak.exists():
        shutil.copy2(ROAD_LINKS, bak)
        print(f"  백업: {bak.name}")
    else:
        print(f"  백업 이미 존재, 스킵: {bak.name}")

    print("[5/5] 저장")
    with ROAD_LINKS.open("w", encoding="utf-8") as f:
        json.dump(rl_gj, f, ensure_ascii=False, separators=(",", ":"))
    mb = ROAD_LINKS.stat().st_size / 1024 / 1024
    print(f"  {ROAD_LINKS.name} {mb:.1f} MB")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
