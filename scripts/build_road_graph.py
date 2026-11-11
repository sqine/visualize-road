"""도로망을 네트워크 그래프로 구축 (노드=도로링크, 엣지=실제 물리적 연결).

배경:
    "인접효과" 논의에서 나온 대로, 이번 단계에서는 지리적(위경도) 근접 매칭이
    아니라 도로 지오메트리 자체의 끝점이 실제로 맞닿아 있는지(위상적 연결)만
    기준으로 엣지를 만든다. 즉 A1→A2→A3 로 쭉 이어진 길이면 A1-A2, A2-A3 가
    연결되는 식. 서로 다른 길이 지리적으로 가까이 지나가지만 실제로 안 이어진
    경우(예: 지난번 발견한 32731 옆 내부순환로)는 이번 그래프에서는 연결로
    잡히지 않는다 — 그 부분은 추후 별도 작업으로 보류.

노드 universe:
    TBGIS road_links.geojson 중, data/flpop.csv 에 유동인구 기록이 있는
    (=기존 시급도 분석과 동일한 후보 유니버스) 도로링크. 기본 대상 행정동은
    정릉2동·정릉3동·성북동·돈암2동(4개) — 필요시 스크립트 상단 TARGET_DONGS
    수정.

노드 속성:
    slope_pct, slope_level  — data/road_links.geojson (preprocess_slope.py 결과)
    road_type               — data/road_links.geojson (classify_road_type.py 결과)
    flow_sum                — data/flpop.csv, 평일(day=1) 06~12시(time_code 1,2)
                               자동 필터, agrde_cd==0("전체") 행만 사용해 이중집계
                               버그(utils/foot_traffic.py 주석 참고) 회피
    icing / icing_dist_m    — data/icing_zones.csv 와의 최근접 거리(직선 기점-종점
                               기준)가 ICING_MAX_DIST_M 이내면 결빙도로로 플래그.
                               icing_zones 자체가 간선도로 위주 데이터라 골목 단위
                               매칭은 근사치임에 유의.
    length_m                — 도로 실제 길이(좌표 경로 길이)

엣지:
    두 도로링크의 끝점(시작/끝) 좌표가 ENDPOINT_TOL_M 이내로 겹치면 연결.
    지리적 근접(다른 길과의 위경도 기반 매칭)은 이번 단계에서 보류.

출력:
    data/road_graph.json       — networkx node_link_data 포맷 (json_graph 호환)
    data/road_graph_nodes.csv  — 확인용 평문 노드 테이블
    data/road_graph_edges.csv  — 확인용 평문 엣지 테이블
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
ROAD_LINKS = DATA / "road_links.geojson"
FLPOP = DATA / "flpop.csv"
ICING = DATA / "icing_zones.csv"

OUT_JSON = DATA / "road_graph.json"
OUT_NODES_CSV = DATA / "road_graph_nodes.csv"
OUT_EDGES_CSV = DATA / "road_graph_edges.csv"

TARGET_DONGS = ["정릉2동", "정릉3동", "성북동", "돈암2동"]
TARGET_DAY_CODE = 1          # 평일
TARGET_TIME_CODES = {1, 2}   # 06-09, 09-12 (기존 분석의 06:01~11:00 근사)
TARGET_AGE_CODE = 0          # "전체/미상" — 10~60 개별 행의 정확한 합계 (이중집계 회피)

ENDPOINT_TOL_M = 3.0         # 끝점 동일 판정 허용오차
ICING_MAX_DIST_M = 50.0      # 결빙구간 매칭 허용거리 (간선도로 데이터라 넉넉히)

_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0


def _euclid_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    return (
        (lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2
        + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2
    ) ** 0.5


def _path_length_m(coords: list[list[float]]) -> float:
    total = 0.0
    for i in range(1, len(coords)):
        total += _euclid_m(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    return total


def _point_to_segment_dist_m(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """점 P 에서 선분 AB 까지 최단거리(m, 등장방형 근사 좌표계 기준)."""
    # 로컬 평면 근사: lon,lat -> m
    def to_m(x, y):
        return (x * _M_PER_DEG_LON, y * _M_PER_DEG_LAT)

    pxm, pym = to_m(px, py)
    axm, aym = to_m(ax, ay)
    bxm, bym = to_m(bx, by)
    dx, dy = bxm - axm, bym - aym
    if dx == 0 and dy == 0:
        return math.hypot(pxm - axm, pym - aym)
    t = ((pxm - axm) * dx + (pym - aym) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = axm + t * dx, aym + t * dy
    return math.hypot(pxm - cx, pym - cy)


# ─────────────────────────────────────────────────────────────────────
# 1) 유동인구 집계 (대상 행정동 + 평일 출근~오전 + 전체연령, 이중집계 회피)
# ─────────────────────────────────────────────────────────────────────

def load_flow_by_link() -> dict[str, dict]:
    df = pd.read_csv(FLPOP, encoding="cp949")
    df = df[df["adstrd_nm"].isin(TARGET_DONGS)]
    df = df[df["dayweek_cd"] == TARGET_DAY_CODE]
    df = df[df["tmzon_cd"].isin(TARGET_TIME_CODES)]
    df = df[df["agrde_cd"] == TARGET_AGE_CODE]  # 이중집계 회피 (전체 행만 사용)
    df["rd_link_cd"] = df["rd_link_cd"].astype(str)

    grouped = df.groupby("rd_link_cd", as_index=False).agg(
        flow_sum=("flpop_co", "sum"),
        dong=("adstrd_nm", "first"),
        gu=("signgu_nm", "first"),
    )
    out: dict[str, dict] = {}
    for _, r in grouped.iterrows():
        out[r["rd_link_cd"]] = {
            "flow_sum": round(float(r["flow_sum"]), 3),
            "dong": r["dong"],
            "gu": r["gu"],
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# 2) 결빙구간 매칭 (직선 기점-종점 기준 최근접 거리)
# ─────────────────────────────────────────────────────────────────────

def load_icing_segments() -> list[dict]:
    df = pd.read_csv(ICING, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    rename = {
        "기점 위도(WGS84(4326))": "lat_start",
        "기점 경도(WGS84(4326))": "lon_start",
        "종점 위도(WGS84(4326))": "lat_end",
        "종점 경도(WGS84(4326))": "lon_end",
        "도로(노선)명": "road_name",
        "대표지역": "region",
    }
    df = df.rename(columns=rename)
    # 서울만 대략 bbox 로 1차 필터 (정밀 자치구 폴리곤 조인은 생략, bbox면 충분히 좁혀짐)
    SEOUL_BBOX = (126.76, 37.42, 127.19, 37.71)  # (minlon, minlat, maxlon, maxlat)
    df = df[
        df["lon_start"].between(SEOUL_BBOX[0], SEOUL_BBOX[2])
        & df["lat_start"].between(SEOUL_BBOX[1], SEOUL_BBOX[3])
    ]
    segs = []
    for _, r in df.iterrows():
        try:
            segs.append(
                {
                    "lon_start": float(r["lon_start"]),
                    "lat_start": float(r["lat_start"]),
                    "lon_end": float(r["lon_end"]),
                    "lat_end": float(r["lat_end"]),
                    "road_name": r.get("road_name", ""),
                }
            )
        except (TypeError, ValueError):
            continue
    return segs


def nearest_icing_dist_m(coords: list[list[float]], icing_segs: list[dict]) -> float | None:
    """도로링크 대표점(중간점) 기준, 가장 가까운 결빙구간 직선까지의 거리."""
    if not coords or not icing_segs:
        return None
    mid = coords[len(coords) // 2]
    best = None
    for seg in icing_segs:
        d = _point_to_segment_dist_m(
            mid[0], mid[1], seg["lon_start"], seg["lat_start"], seg["lon_end"], seg["lat_end"]
        )
        if best is None or d < best:
            best = d
    return best


# ─────────────────────────────────────────────────────────────────────
# 3) 끝점 공유 기반 연결성 (엣지)
# ─────────────────────────────────────────────────────────────────────

class EndpointIndex:
    """좌표를 tolerance 이내로 묶어 같은 '교차점 ID'를 부여."""

    def __init__(self, tol_m: float):
        self.tol_m = tol_m
        self.cell_lat = tol_m / _M_PER_DEG_LAT
        self.cell_lon = tol_m / _M_PER_DEG_LON
        self.junctions: list[tuple[float, float]] = []  # 대표 좌표
        self.bins: dict[tuple[int, int], list[int]] = {}

    def _key(self, lon: float, lat: float) -> tuple[int, int]:
        return (int(lon / self.cell_lon), int(lat / self.cell_lat))

    def get_or_create(self, lon: float, lat: float) -> int:
        cx, cy = self._key(lon, lat)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self.bins.get((cx + dx, cy + dy), []):
                    jlon, jlat = self.junctions[idx]
                    if _euclid_m(lon, lat, jlon, jlat) <= self.tol_m:
                        return idx
        idx = len(self.junctions)
        self.junctions.append((lon, lat))
        self.bins.setdefault((cx, cy), []).append(idx)
        return idx


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[1/5] 유동인구 집계 (평일 06~12시, 전체연령, 대상 행정동)")
    flow_by_link = load_flow_by_link()
    print(f"  유동인구 보유 링크: {len(flow_by_link):,}건")

    print("[2/5] 결빙구간 로드")
    icing_segs = load_icing_segments()
    print(f"  서울 bbox 내 결빙구간: {len(icing_segs):,}건")

    print("[3/5] road_links.geojson 로드 + 노드 구성 (유동인구 보유 링크만)")
    with ROAD_LINKS.open("r", encoding="utf-8") as f:
        rl_gj = json.load(f)
    nodes: dict[str, dict] = {}
    node_coords: dict[str, list[list[float]]] = {}
    for feat in rl_gj["features"]:
        lid = feat["properties"].get("road_lid", "")
        if lid not in flow_by_link:
            continue
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = geom["coordinates"]
        if len(coords) < 2:
            continue
        props = feat["properties"]
        icing_dist = nearest_icing_dist_m(coords, icing_segs)
        nodes[lid] = {
            "link_id": lid,
            "dong": flow_by_link[lid]["dong"],
            "gu": flow_by_link[lid]["gu"],
            "flow_sum": flow_by_link[lid]["flow_sum"],
            "slope_pct": props.get("slope_pct"),
            "slope_level": props.get("slope_level", "unknown"),
            "road_type": props.get("road_type", "unknown"),
            "length_m": round(_path_length_m(coords), 1),
            "icing": bool(icing_dist is not None and icing_dist <= ICING_MAX_DIST_M),
            "icing_dist_m": round(icing_dist, 1) if icing_dist is not None else None,
            "lon": round(coords[len(coords) // 2][0], 6),
            "lat": round(coords[len(coords) // 2][1], 6),
        }
        node_coords[lid] = coords
    print(f"  노드 수: {len(nodes):,}")

    print("[4/5] 끝점 공유 기반 엣지 구성 (tolerance={:.0f}m)".format(ENDPOINT_TOL_M))
    idx = EndpointIndex(tol_m=ENDPOINT_TOL_M)
    junction_to_links: dict[int, list[str]] = {}
    for lid, coords in node_coords.items():
        for pt in (coords[0], coords[-1]):
            jid = idx.get_or_create(pt[0], pt[1])
            junction_to_links.setdefault(jid, []).append(lid)

    edges: set[tuple[str, str]] = set()
    for jid, lids in junction_to_links.items():
        uniq = sorted(set(lids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edges.add((uniq[i], uniq[j]))
    print(f"  교차점(junction) 수: {len(idx.junctions):,}  엣지 수: {len(edges):,}")

    print("[5/5] 저장 (networkx node_link json + csv)")
    node_link_json = {
        "directed": False,
        "multigraph": False,
        "graph": {
            "source": "build_road_graph.py",
            "endpoint_tol_m": ENDPOINT_TOL_M,
            "icing_max_dist_m": ICING_MAX_DIST_M,
            "target_dongs": TARGET_DONGS,
        },
        "nodes": [{"id": lid, **attrs} for lid, attrs in nodes.items()],
        "links": [{"source": a, "target": b} for a, b in sorted(edges)],
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(node_link_json, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    with OUT_NODES_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(nodes.values())).keys()))
        w.writeheader()
        for attrs in nodes.values():
            w.writerow(attrs)

    with OUT_EDGES_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["link_id_a", "link_id_b"])
        for a, b in sorted(edges):
            w.writerow([a, b])

    print(f"  {OUT_JSON.name}, {OUT_NODES_CSV.name}, {OUT_EDGES_CSV.name} 저장 완료")

    # 연결 성분(각 노드가 속한 서로 연결된 도로 묶음) 개수 요약
    parent = {lid: lid for lid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    comp_sizes: dict[str, int] = {}
    for lid in nodes:
        r = find(lid)
        comp_sizes[r] = comp_sizes.get(r, 0) + 1
    sizes = sorted(comp_sizes.values(), reverse=True)
    isolated = sum(1 for s in sizes if s == 1)
    print(f"\n  연결 성분 수: {len(sizes)}  (고립 노드: {isolated}개)")
    print(f"  최대 연결 성분 크기: {sizes[0] if sizes else 0}개 노드")
    print(f"  상위 5개 성분 크기: {sizes[:5]}")


if __name__ == "__main__":
    main()
