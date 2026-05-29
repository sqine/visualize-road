"""OSM Overpass raw GeoJSON → 앱이 사용하는 정제된 도로 라인 GeoJSON 변환.

입력: data/seoul_roads_raw.geojson   (Overpass Turbo 결과)
입력: data/seoul_gu.geojson           (자치구 경계, 공간 조인용)
출력: data/seoul_roads.geojson        (앱이 기대하는 포맷)

주요 작업
1. Polygon geometry 제외 (휴게소 등)
2. 각 도로의 centroid가 어느 자치구에 속하는지 ray-casting으로 판정
3. 도로명 정규화 (name:ko > name > '')
4. 좌표 5자리 정밀도로 압축
5. 자치구별로 그룹핑 정보 추가
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import time

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "seoul_roads_raw.geojson"
GU = DATA / "seoul_gu.geojson"
OUT = DATA / "seoul_roads.geojson"


# ─────────────────────────────────────────────────────────────────────
# Point-in-polygon (ray casting)
# ─────────────────────────────────────────────────────────────────────

def point_in_ring(x: float, y: float, ring: list) -> bool:
    """ring: [[x,y], ...] 닫힌 다각형 외곽선."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_polygon(x: float, y: float, poly: list) -> bool:
    """poly: [outer_ring, hole1, hole2, ...]"""
    if not poly:
        return False
    if not point_in_ring(x, y, poly[0]):
        return False
    for hole in poly[1:]:
        if point_in_ring(x, y, hole):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────
# 자치구 인덱스 (bbox + 폴리곤)
# ─────────────────────────────────────────────────────────────────────

def build_gu_index(gu_gj: dict) -> list:
    """[{gu, bbox(minx,miny,maxx,maxy), polygons:[[ring,...], ...]}, ...]"""
    idx = []
    for feat in gu_gj["features"]:
        gu = feat["properties"]["gu"]
        geom = feat["geometry"]
        gtype = geom["type"]
        if gtype == "Polygon":
            polys = [geom["coordinates"]]
        elif gtype == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue

        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        for poly in polys:
            for x, y in poly[0]:  # 외곽선만 bbox 산출
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y

        idx.append(
            {
                "gu": gu,
                "bbox": (minx, miny, maxx, maxy),
                "polygons": polys,
            }
        )
    return idx


def find_gu(x: float, y: float, gu_idx: list) -> str:
    """좌표 (x=lon, y=lat)를 포함하는 자치구명 반환."""
    for entry in gu_idx:
        minx, miny, maxx, maxy = entry["bbox"]
        if x < minx or x > maxx or y < miny or y > maxy:
            continue
        for poly in entry["polygons"]:
            if point_in_polygon(x, y, poly):
                return entry["gu"]
    return ""


# ─────────────────────────────────────────────────────────────────────
# 좌표 정밀도 축소
# ─────────────────────────────────────────────────────────────────────

def round_coords(coords: list, ndigits: int = 5) -> list:
    return [[round(c[0], ndigits), round(c[1], ndigits)] for c in coords]


def centroid(coords: list) -> tuple[float, float]:
    n = len(coords)
    sx = sum(c[0] for c in coords)
    sy = sum(c[1] for c in coords)
    return sx / n, sy / n


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time()
    print(f"[1/5] raw 로드 중: {RAW.name}")
    with RAW.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    feats = raw.get("features", [])
    print(f"  features: {len(feats):,}")

    print(f"[2/5] 자치구 인덱스 구축")
    with GU.open("r", encoding="utf-8") as f:
        gu_gj = json.load(f)
    gu_idx = build_gu_index(gu_gj)
    print(f"  gu polygons: {sum(len(e['polygons']) for e in gu_idx)}")

    print(f"[3/5] 도로 필터 + 공간조인 + 정규화")
    out_feats = []
    gu_counts = {}
    skipped_polygon = 0
    skipped_no_gu = 0
    matched_name = 0

    for i, feat in enumerate(feats):
        if i and i % 5000 == 0:
            print(f"  ...{i:,} 처리 중 ({time()-t0:.1f}s)")

        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            skipped_polygon += 1
            continue

        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue

        props = feat.get("properties", {})
        # 도로명: name:ko 우선, 그다음 name
        road_name = props.get("name:ko") or props.get("name") or ""
        if road_name:
            matched_name += 1

        # 자치구 판정: centroid 사용
        cx, cy = centroid(coords)
        gu = find_gu(cx, cy, gu_idx)
        if not gu:
            skipped_no_gu += 1
            continue

        gu_counts[gu] = gu_counts.get(gu, 0) + 1

        out_feats.append(
            {
                "type": "Feature",
                "properties": {
                    "road_name": road_name,
                    "gu": gu,
                    "highway": props.get("highway", ""),
                    "has_name": bool(road_name),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": round_coords(coords, 5),
                },
            }
        )

    print(f"  skipped Polygon: {skipped_polygon}")
    print(f"  skipped 자치구 외: {skipped_no_gu:,}")
    print(f"  남은 도로: {len(out_feats):,}")
    print(f"  이름 있음: {matched_name:,} ({matched_name/len(out_feats)*100:.1f}%)")

    print(f"[4/5] 자치구별 분포:")
    for gu in sorted(gu_counts.keys()):
        print(f"  {gu}: {gu_counts[gu]:,}")

    print(f"[5/5] 저장: {OUT.name}")
    out_gj = {"type": "FeatureCollection", "features": out_feats}
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(out_gj, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"  파일 크기: {size_mb:.1f} MB")
    print(f"  총 소요: {time()-t0:.1f}s")


if __name__ == "__main__":
    main()
