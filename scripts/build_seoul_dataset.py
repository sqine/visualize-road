"""서울 전역 링크 데이터셋 영구화 — /tmp 의존 제거.

배경
----
서울 전역 수치(위험군 1,851 km · 커버리지 4.05% · 경계 생성률 75.8% ·
하류 경사 1.82배 · 25개구 상관 ρ=-0.719 · 반사실 2.01배)는 전부
/tmp/seoul_graph.pkl · /tmp/seoul_heat.pkl 에서 나왔다. 세션이 끝나면 사라져
매번 재빌드해야 했고, Streamlit 앱이 읽을 수 없었다.

이 스크립트가 같은 내용을 data/ 아래 parquet으로 고정한다.

산출
----
    data/seoul_links.parquet          링크 180,460 × 속성
    data/seoul_edges.parquet          인접 엣지 (끝점 5 m 스냅)
    data/seoul_boundary_pairs.parquet 경계쌍 (상류 보호 → 하류 비보호)

실행
----
    cd streamlit_app
    python3 scripts/build_seoul_dataset.py

주의
----
4개동 MILP는 이 파일을 쓰지 않는다. 그쪽은 road_graph.json(1,190 노드)이
별도 경로다. 이 데이터셋은 **서울 전역 진단·시각화 전용**이다.
"""

from __future__ import annotations

import collections
import heapq
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GEOCODE = DATA / "heating_geocode_all.json"

M_LAT, M_LON = 111320.0, 88322.0
SNAP_CELL = 5.0      # 끝점 스냅 간격(m) — 인접 판정
H_MIN = 2.0          # 경계 판정 고도차(m)
ICING_MAX = 50.0     # 결빙구간 좌표 → 링크 스냅 허용 반경(m)


# ── 1. 링크 + 자치구 ──────────────────────────────────────────────
def build_links() -> tuple[pd.DataFrame, list]:
    g = json.load(open(DATA / "road_links.geojson", encoding="utf-8"))
    F = g["features"]
    rows, ends = [], []
    for i, f in enumerate(F):
        p, geom = f["properties"], f["geometry"]
        c = geom["coordinates"]
        if geom["type"] != "LineString" or len(c) < 2:
            rows.append(None); ends.append(None); continue
        m = c[len(c) // 2]
        zs, ze = p.get("elev_start_m"), p.get("elev_end_m")
        rows.append(dict(
            link_id=i, lon=m[0], lat=m[1],
            slope_pct=p.get("slope_pct"), road_type=p.get("road_type"),
            road_type_highway=p.get("road_type_highway"),
            dist_m=p.get("dist_m"),
            z=None if (zs is None or ze is None) else (zs + ze) / 2,
            road_lid=p.get("road_lid"),
        ))
        ends.append((c[0], c[-1]))
    df = pd.DataFrame([r for r in rows if r])
    valid = [i for i, r in enumerate(rows) if r]

    # 자치구 배정 (링크 중점 point-in-polygon)
    gu = json.load(open(DATA / "seoul_gu.geojson", encoding="utf-8"))
    M = df[["lon", "lat"]].to_numpy()
    name = np.array([None] * len(df), dtype=object)
    for f in gu["features"]:
        nm = [v for k, v in f["properties"].items()
              if isinstance(v, str) and v.endswith("구")][0]
        geom = f["geometry"]
        rings = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for poly in rings:
            un = name == None  # noqa: E711
            if not un.any():
                break
            ins = MPath(np.array(poly[0])[:, :2]).contains_points(M[un])
            name[np.where(un)[0][ins]] = nm
    df["gu"] = name
    return df, ends


# ── 2. 인접 (끝점 스냅) ──────────────────────────────────────────
def build_edges(df: pd.DataFrame, ends: list) -> tuple[pd.DataFrame, dict]:
    cl, co = SNAP_CELL / M_LAT, SNAP_CELL / M_LON
    bins = collections.defaultdict(list)
    for i, lid in enumerate(df.link_id):
        for pt in ends[lid]:
            bins[(int(pt[0] / co), int(pt[1] / cl))].append(i)
    adj = collections.defaultdict(set)
    for v in bins.values():
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                if v[a] != v[b]:
                    adj[v[a]].add(v[b]); adj[v[b]].add(v[a])
    e = sorted({(min(u, w), max(u, w)) for u, s in adj.items() for w in s})
    return pd.DataFrame(e, columns=["u", "v"]), {k: list(s) for k, s in adj.items()}


# ── 3. 열선 라우팅 ────────────────────────────────────────────────
def route_heating(df, adj) -> tuple[dict, collections.Counter]:
    cache = json.load(open(GEOCODE, encoding="utf-8"))
    h = pd.read_csv(DATA / "heating_2026.csv", encoding="utf-8-sig")
    lcol = next(c for c in h.columns if "연장" in c)
    scol = next(c for c in h.columns if "설치위치" in c or "설치구간" in c)
    ycol = next(c for c in h.columns if "연도" in c)
    h["m"] = pd.to_numeric(h[lcol], errors="coerce")
    h["yr"] = h[ycol].astype(str).str.extract(r"(\d{4})")[0].astype(float)

    lon = df.lon.to_numpy(); lat = df.lat.to_numpy()
    dist = df.dist_m.fillna(0).to_numpy()
    CELL = 100.0; cl, co = CELL / M_LAT, CELL / M_LON
    grid = collections.defaultdict(list)
    for i in range(len(df)):
        grid[(int(lon[i] / co), int(lat[i] / cl))].append(i)

    def near(x, y, maxm=150.0):
        cx, cy = int(x / co), int(y / cl)
        best, bd = None, 1e18
        n = int(math.ceil(maxm / CELL)) + 1
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for i in grid.get((cx + dx, cy + dy), ()):
                    d = math.hypot((lat[i] - y) * M_LAT, (lon[i] - x) * M_LON)
                    if d < bd:
                        best, bd = i, d
        return (best, bd) if bd <= maxm else (None, bd)

    W = lambda i: dist[i] / 2

    def route(a, b, cap=4000.0):
        D = {a: 0.0}; pq = [(0.0, a)]; prev = {}
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                break
            if d > D.get(u, 1e18) or d > cap:
                continue
            for v in adj.get(u, ()):
                nd = d + W(u) + W(v)
                if nd < D.get(v, 1e18):
                    D[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
        if b not in D:
            return None
        p = [b]
        while p[-1] != a:
            p.append(prev[p[-1]])
        return p[::-1]

    def split(sec):
        sec = str(sec)
        if "~" not in sec:
            return None, None
        m = re.match(r"^(.*?)\(?\s*(.+?)\s*~\s*(.+?)\)?\s*$", sec)
        if not m:
            return None, None
        pre, a, b = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if pre and not re.search(r"\d", a):
            a = pre + " " + a
        return a, b

    gcol = next(c for c in h.columns if "관리기관" in c)
    heat, year, stat = {}, {}, collections.Counter()
    for _, r in h.iterrows():
        a, b = split(r[scol])
        # 캐시 키는 "서울특별시 <관리기관> <주소>" 형식
        ca = cache.get(f"서울특별시 {r[gcol]} {a}") if a else None
        cb = cache.get(f"서울특별시 {r[gcol]} {b}") if b else None
        if not ca or not cb:
            stat["지오코딩실패"] += 1; continue
        ia, da = near(ca[0], ca[1]); ib, db = near(cb[0], cb[1])
        if ia is None or ib is None:
            stat["스냅실패"] += 1; continue
        p = route(ia, ib)
        if p is None:
            stat["경로없음"] += 1
            for i in (ia, ib):
                heat[i] = True; year[i] = r["yr"]
            continue
        plen = sum(dist[i] for i in p)
        if r["m"] and plen > 2 * r["m"]:
            stat["경로과대(끝점만)"] += 1
            for i in (ia, ib):
                heat[i] = True; year[i] = r["yr"]
        else:
            stat["성공"] += 1
            for i in p:
                heat[i] = True
                year[i] = min(year.get(i, 9999), r["yr"]) if r["yr"] else year.get(i)
    return (heat, year), stat


# ── 4. 결빙취약구간 매칭 ──────────────────────────────────────────
def match_icing(df) -> set:
    ic = pd.read_csv(DATA / "icing_zones.csv", encoding="utf-8-sig")
    la = next(c for c in ic.columns if "기점" in c and "위도" in c)
    lo = next(c for c in ic.columns if "기점" in c and "경도" in c)
    la2 = next((c for c in ic.columns if "종점" in c and "위도" in c), None)
    lo2 = next((c for c in ic.columns if "종점" in c and "경도" in c), None)

    lon = df.lon.to_numpy(); lat = df.lat.to_numpy()
    CELL = 100.0; cl, co = CELL / M_LAT, CELL / M_LON
    grid = collections.defaultdict(list)
    for i in range(len(df)):
        grid[(int(lon[i] / co), int(lat[i] / cl))].append(i)

    out = set()
    pts = []
    for _, r in ic.iterrows():
        for cx, cy in [(lo, la), (lo2, la2)]:
            if cx is None:
                continue
            x, y = pd.to_numeric(r.get(cx), errors="coerce"), pd.to_numeric(r.get(cy), errors="coerce")
            if pd.notna(x) and pd.notna(y) and 126 < x < 128 and 37 < y < 38:
                pts.append((x, y))
    n = int(math.ceil(ICING_MAX / CELL)) + 1
    for x, y in pts:
        cx, cy = int(x / co), int(y / cl)
        best, bd = None, 1e18
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for i in grid.get((cx + dx, cy + dy), ()):
                    d = math.hypot((lat[i] - y) * M_LAT, (lon[i] - x) * M_LON)
                    if d < bd:
                        best, bd = i, d
        if best is not None and bd <= ICING_MAX:
            out.add(best)
    print(f"  결빙구간 좌표 {len(pts):,}점 → 매칭 링크 {len(out):,}개 "
          f"(반경 {ICING_MAX:.0f} m)")
    return out


# ── 5. 경계쌍 ────────────────────────────────────────────────────
def build_boundary(df, adj, heat) -> pd.DataFrame:
    z = df.z.to_numpy(); sl = df.slope_pct.to_numpy()
    ok = ~pd.isna(z) & ~pd.isna(sl) & (df.dist_m.fillna(0).to_numpy() > 0)
    rows = []
    for u in np.where(df.heat.to_numpy() & ok)[0]:
        for v in adj.get(u, ()):
            if df.heat.iat[v] or not ok[v]:
                continue
            d = z[u] - z[v]
            if d >= H_MIN:
                rows.append(dict(up=int(u), down=int(v), drop_m=float(d),
                                 down_slope=float(sl[v]),
                                 down_gu=df.gu.iat[v],
                                 down_road_type=df.road_type.iat[v],
                                 down_dist_m=float(df.dist_m.iat[v])))
    cols = ["up", "down", "drop_m", "down_slope", "down_gu",
            "down_road_type", "down_dist_m"]
    return pd.DataFrame(rows, columns=cols)


def main() -> None:
    t0 = time.time()
    print("[1/5] 링크 + 자치구")
    df, ends = build_links()
    print(f"  링크 {len(df):,} / 자치구 배정 {df.gu.notna().sum():,}")

    print("[2/5] 인접 엣지")
    edges, adj = build_edges(df, ends)
    print(f"  노드 {len(adj):,} / 엣지 {len(edges):,}  ({time.time()-t0:.0f}s)")

    print("[3/5] 열선 라우팅")
    (heat, year), stat = route_heating(df, adj)
    df["heat"] = df.index.isin(heat.keys())
    df["heat_year"] = [year.get(i) for i in range(len(df))]
    print(f"  {dict(stat)} → 플래그 {df.heat.sum():,}")

    print("[4/5] 결빙취약구간 매칭")
    ic = match_icing(df)
    df["icing"] = df.index.isin(ic)

    print("[5/5] 경계쌍")
    bp = build_boundary(df, adj, heat)
    print(f"  경계쌍 {len(bp):,} / 하류 고유 {bp.down.nunique():,}")

    df.to_parquet(DATA / "seoul_links.parquet", index=False)
    edges.to_parquet(DATA / "seoul_edges.parquet", index=False)
    bp.to_parquet(DATA / "seoul_boundary_pairs.parquet", index=False)
    print(f"\n저장 완료 ({time.time()-t0:.0f}s)")
    for f in ["seoul_links.parquet", "seoul_edges.parquet",
              "seoul_boundary_pairs.parquet"]:
        print(f"  data/{f}  {(DATA/f).stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
