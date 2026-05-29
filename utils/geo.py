"""지오메트리 로딩 및 도로명 매칭.

외부 데이터 파일이 있으면 사용하고, 없으면 자치구 중심점 기반 fallback.

사용자가 다음 파일을 data/ 폴더에 두면 자동으로 활용됨:
    - seoul_gu.geojson    : 서울 25개 자치구 경계 (Polygon/MultiPolygon)
    - seoul_roads.geojson : 도로명 라인 (LineString),
                            속성에 'road_name'(또는 'RN','도로명') + 'gu'(또는 'sig_kor_nm')
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import SEOUL_GU_CENTROIDS


# ──────────────────────────────────────────────────────────────────────
# 자치구 경계
# ──────────────────────────────────────────────────────────────────────

def load_gu_geojson(path: str | Path) -> dict | None:
    """자치구 경계 GeoJSON 로드. 없으면 None."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def attach_metrics_to_gu_geojson(
    gj: dict, gu_metrics: pd.DataFrame
) -> dict:
    """자치구 GeoJSON 각 feature에 집계 지표를 properties로 attach."""
    metrics_by_gu = gu_metrics.set_index("gu").to_dict(orient="index")

    # 자치구명 컬럼 후보 (출처마다 키가 다름)
    NAME_KEYS = ("SIG_KOR_NM", "sig_kor_nm", "name", "자치구", "SGG_NM")

    for feat in gj.get("features", []):
        props = feat.setdefault("properties", {})
        gu_name = next(
            (props[k] for k in NAME_KEYS if k in props and isinstance(props[k], str)),
            None,
        )
        if gu_name and gu_name in metrics_by_gu:
            props.update(metrics_by_gu[gu_name])
            props["gu"] = gu_name
        else:
            props["gu"] = gu_name or "?"
            for col in gu_metrics.columns:
                if col != "gu":
                    props.setdefault(col, 0)
    return gj


# ──────────────────────────────────────────────────────────────────────
# 도로 라인
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RoadLine:
    gu: str              # CSV 자치구 (관리주체)
    road_name: str
    coords: list[list[float]]  # [[lon, lat], ...]
    osm_gu: str = ""     # 실제 OSM 위치 자치구
    match_type: str = "exact"  # "exact" | "name_only"


def load_roads_geojson(path: str | Path) -> list[dict] | None:
    """도로 라인 GeoJSON 로드. 없으면 None.

    기대 properties: road_name(or RN/도로명), gu(or SIG_KOR_NM)
    geometry: LineString or MultiLineString (WGS84 lon/lat)
    """
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    return gj.get("features", [])


def _norm(s: Any) -> str:
    return "" if s is None else str(s).strip().replace(" ", "")


def _props_road_name(props: dict) -> str:
    for key in ("road_name", "RN", "도로명", "ROAD_NM", "rn"):
        if key in props and props[key]:
            return _norm(props[key])
    return ""


def _props_gu(props: dict) -> str:
    for key in ("gu", "SIG_KOR_NM", "sig_kor_nm", "자치구", "SGG_NM"):
        if key in props and props[key]:
            return _norm(props[key])
    return ""


def match_roads(
    df: pd.DataFrame, road_features: list[dict] | None
) -> list[RoadLine]:
    """도로 데이터프레임의 (gu, primary_road)와 라인 features를 매칭.

    2단계 매칭:
        1) (gu, road_name) 정확 매칭 — CSV 자치구와 OSM 위치가 일치
        2) 1)이 실패하면 road_name 만으로 매칭 — CSV 관리주체와 실제 위치가 다른 경우
           (예: 종로구가 관리하는 성균관로13길이 실제로는 성북구에 위치)

    매칭이 되지 않은 행은 결과에서 제외.
    """
    if not road_features:
        return []

    # 인덱스 두 종류
    by_gu_road: dict[tuple[str, str], list[dict]] = {}
    by_road: dict[str, list[dict]] = {}
    for feat in road_features:
        props = feat.get("properties", {})
        rn = _props_road_name(props)
        gu = _props_gu(props)
        if not rn:
            continue
        if gu:
            by_gu_road.setdefault((gu, rn), []).append(feat)
        by_road.setdefault(rn, []).append(feat)

    out: list[RoadLine] = []
    for _, row in df.iterrows():
        csv_gu = _norm(row["gu"])
        matched = False
        for road_name in row["road_names"]:
            rn_norm = _norm(road_name)

            # 1단계: (gu, road_name) 정확 매칭
            key1 = (csv_gu, rn_norm)
            if key1 in by_gu_road:
                for feat in by_gu_road[key1]:
                    coords = _flatten_line(feat.get("geometry", {}))
                    if coords:
                        osm_gu = _props_gu(feat.get("properties", {}))
                        out.append(
                            RoadLine(
                                gu=csv_gu,
                                road_name=road_name,
                                coords=coords,
                                osm_gu=osm_gu,
                                match_type="exact",
                            )
                        )
                matched = True
                break

            # 2단계: road_name 단독 매칭 (다른 자치구에 위치한 경우)
            if rn_norm in by_road:
                for feat in by_road[rn_norm]:
                    coords = _flatten_line(feat.get("geometry", {}))
                    if coords:
                        osm_gu = _props_gu(feat.get("properties", {}))
                        out.append(
                            RoadLine(
                                gu=csv_gu,
                                road_name=road_name,
                                coords=coords,
                                osm_gu=osm_gu,
                                match_type="name_only",
                            )
                        )
                matched = True
                break

        # matched=False면 어느 단계에서도 못 찾음 → 제외
        _ = matched
    return out


def _flatten_line(geom: dict) -> list[list[float]]:
    """LineString 또는 MultiLineString을 단일 라인 좌표 리스트로 평탄화."""
    t = geom.get("type")
    if t == "LineString":
        return geom.get("coordinates", [])
    if t == "MultiLineString":
        flat: list[list[float]] = []
        for line in geom.get("coordinates", []):
            flat.extend(line)
        return flat
    return []


# ──────────────────────────────────────────────────────────────────────
# 점-자치구 공간 매칭 (결빙구간 등 좌표 보유 데이터용)
# ──────────────────────────────────────────────────────────────────────

def _point_in_ring(x: float, y: float, ring: list) -> bool:
    """ray-casting. ring: 닫힌 외곽선 [[x,y], ...]"""
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


def build_gu_polygon_index(gj: dict) -> list[dict]:
    """자치구 GeoJSON → bbox+다각형 인덱스. find_gu_for_point 입력용."""
    idx: list[dict] = []
    NAME_KEYS = ("gu", "SIG_KOR_NM", "sig_kor_nm", "name", "자치구", "SGG_NM")
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        gu = next(
            (props[k] for k in NAME_KEYS if k in props and isinstance(props[k], str)),
            None,
        )
        if not gu:
            continue
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        if gtype == "Polygon":
            polys = [geom["coordinates"]]
        elif gtype == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue

        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        for poly in polys:
            for x, y in poly[0]:
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y

        idx.append({"gu": gu, "bbox": (minx, miny, maxx, maxy), "polygons": polys})
    return idx


def find_gu_for_point(x: float, y: float, gu_idx: list[dict]) -> str:
    """좌표 (x=lon, y=lat) 를 포함하는 자치구명. 없으면 ''. """
    for entry in gu_idx:
        minx, miny, maxx, maxy = entry["bbox"]
        if x < minx or x > maxx or y < miny or y > maxy:
            continue
        for poly in entry["polygons"]:
            if not poly:
                continue
            if not _point_in_ring(x, y, poly[0]):
                continue
            inside = True
            for hole in poly[1:]:
                if _point_in_ring(x, y, hole):
                    inside = False
                    break
            if inside:
                return entry["gu"]
    return ""


def _euclid_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """서울 위도(약 37.5°) 가정 근사 미터 거리."""
    return (
        (lat1 - lat2) ** 2 * 111_320 ** 2
        + (lon1 - lon2) ** 2 * 88_322 ** 2
    ) ** 0.5


# ──────────────────────────────────────────────────────────────────────
# 경사도 등급
# ──────────────────────────────────────────────────────────────────────

SLOPE_LEVELS = ("0_3", "3_6", "6_10", "10_plus", "unknown")
SLOPE_LABELS = {
    "0_3": "0~3%",
    "3_6": "3~6%",
    "6_10": "6~10%",
    "10_plus": "≥10%",
    "unknown": "미상",
}
# 등급별 색상 [R, G, B, A] — 회색 → 노랑 → 주황 → 빨강
SLOPE_COLORS = {
    "0_3": [180, 180, 180, 110],
    "3_6": [255, 210, 80, 200],
    "6_10": [240, 130, 40, 220],
    "10_plus": [220, 40, 40, 240],
    "unknown": [200, 200, 200, 80],
}


def filter_roads_by_slope(
    features_by_gu: dict[str, list[dict]],
    selected_gus: list[str],
    selected_levels: list[str],
) -> list[dict]:
    """선택된 자치구 + 경사도 등급에 해당하는 도로 features 반환."""
    if not features_by_gu or not selected_gus or not selected_levels:
        return []
    sel_levels = set(selected_levels)
    out: list[dict] = []
    for gu in selected_gus:
        for feat in features_by_gu.get(gu, []):
            lvl = feat.get("properties", {}).get("slope_level", "unknown")
            if lvl in sel_levels:
                out.append(feat)
    return out


def snap_icing_to_osm_paths(
    df_icing: "pd.DataFrame",
    road_features: list[dict] | None,
    max_dist_m: float = 600.0,
) -> list[dict]:
    """결빙구간을 OSM 도로 라인에 스냅 → PathLayer-ready dict 리스트.

    매칭 규칙:
        1. icing.road_name 과 OSM road_name 정규화 일치
        2. OSM feature의 centroid가 icing 중점에서 max_dist_m 이내
        결과는 매칭된 모든 segment를 각각 한 path로 반환 (길이가 긴
        도로가 여러 way로 쪼개진 경우 동시에 표시).

    매칭 실패시: 기점-종점 직선으로 fallback.
    한 icing 행이 여러 OSM segment에 매칭될 수 있어 결과 개수 ≥ 입력 개수.
    """
    import pandas as pd

    by_name: dict[str, list[dict]] = {}
    if road_features:
        for feat in road_features:
            rn = _norm(feat.get("properties", {}).get("road_name", ""))
            if rn:
                by_name.setdefault(rn, []).append(feat)

    out: list[dict] = []
    for _, r in df_icing.iterrows():
        lon_s, lat_s = float(r["lon_start"]), float(r["lat_start"])
        lon_e, lat_e = float(r["lon_end"]), float(r["lat_end"])
        mlon, mlat = (lon_s + lon_e) / 2.0, (lat_s + lat_e) / 2.0

        name_norm = _norm(r.get("road_name", ""))
        matched_coords: list[list[list[float]]] = []
        if name_norm and name_norm in by_name:
            for feat in by_name[name_norm]:
                fc = feat.get("geometry", {}).get("coordinates", [])
                if not fc:
                    continue
                clon = sum(c[0] for c in fc) / len(fc)
                clat = sum(c[1] for c in fc) / len(fc)
                if _euclid_m(mlon, mlat, clon, clat) < max_dist_m:
                    matched_coords.append(fc)

        base = {
            "seg_id": r.get("seg_id", ""),
            "road_name": r.get("road_name", "") or "",
            "road_class": r.get("road_class", ""),
            "agency": r.get("agency", ""),
            "gu": r.get("gu", ""),
            "gu_start": r.get("gu_start", ""),
            "gu_end": r.get("gu_end", ""),
            "length_km": (
                float(r["length_km"]) if pd.notna(r.get("length_km")) else None
            ),
        }

        if matched_coords:
            for coords in matched_coords:
                out.append({**base, "coords": coords, "snap": "osm"})
        else:
            out.append(
                {
                    **base,
                    "coords": [[lon_s, lat_s], [lon_e, lat_e]],
                    "snap": "straight",
                }
            )

    return out


def filter_icing_to_seoul(
    df_icing: pd.DataFrame, gu_idx: list[dict]
) -> pd.DataFrame:
    """결빙구간 DF 중 기점 좌표가 서울 자치구 내부인 행만 추출.

    추가 컬럼:
        gu_start: 기점이 속한 자치구
        gu_end  : 종점이 속한 자치구 (다를 수도 있음)
    """
    if df_icing.empty:
        out = df_icing.copy()
        out["gu_start"] = []
        out["gu_end"] = []
        return out

    gu_starts: list[str] = []
    gu_ends: list[str] = []
    for _, r in df_icing.iterrows():
        gs = find_gu_for_point(r["lon_start"], r["lat_start"], gu_idx)
        ge = find_gu_for_point(r["lon_end"], r["lat_end"], gu_idx)
        gu_starts.append(gs)
        gu_ends.append(ge)

    out = df_icing.copy()
    out["gu_start"] = gu_starts
    out["gu_end"] = gu_ends
    # 기점 또는 종점 중 하나라도 서울 자치구 안에 있으면 포함
    in_seoul = (out["gu_start"] != "") | (out["gu_end"] != "")
    out = out[in_seoul].copy()
    # 통합 자치구 라벨: 기점이 서울이면 기점 사용, 아니면 종점
    out["gu"] = out["gu_start"].where(out["gu_start"] != "", out["gu_end"])
    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Fallback: 자치구 중심점 주변에 점 분포 생성
# ──────────────────────────────────────────────────────────────────────

def fallback_points(df: pd.DataFrame, jitter_m: float = 800.0) -> pd.DataFrame:
    """매칭 실패시 시각화용 fallback 점.

    각 행마다 자치구 중심좌표 + 균일 분포의 jitter 적용.
    """
    rng = random.Random(42)  # 재현성
    rows = []

    # 위도 1도 ≒ 111_320m, 경도는 cos(lat)에 비례
    for _, r in df.iterrows():
        c = SEOUL_GU_CENTROIDS.get(r["gu"])
        if c is None:
            continue
        lat0, lon0 = c
        # 정사각 jitter — 단순화 (정확도보다 가독성 우선)
        dlat = (rng.random() - 0.5) * 2 * (jitter_m / 111_320)
        dlon = (rng.random() - 0.5) * 2 * (jitter_m / (111_320 * math.cos(math.radians(lat0))))
        rows.append(
            {
                "gu": r["gu"],
                "primary_road": r["primary_road"],
                "section": r["section"],
                "year": int(r["year"]) if pd.notna(r["year"]) else None,
                "length_m": int(r["length_m"]),
                "lat": lat0 + dlat,
                "lon": lon0 + dlon,
            }
        )
    return pd.DataFrame(rows)
