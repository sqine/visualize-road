"""지도용 선 형상 추출 — road_links.geojson → seoul_link_geom.parquet.

왜 따로 만드나
--------------
`seoul_links.parquet` 은 링크의 **중점**(lon/lat)만 담는다. 분석에는 그걸로
충분하지만 지도는 선을 그려야 한다. 96 MB geojson 을 그림마다 다시 파싱하면
한 장에 40초씩 걸리므로, 좌표열만 뽑아 한 번 parquet 으로 굳혀 둔다.

단순화
------
도로 한 개가 정점 390개짜리인 경우도 있다. 서울 전역 그림에서는 그 정밀도가
화면에 보이지 않으므로 Douglas-Peucker 로 줄인다(tol = 약 2.2 m).
4개동 확대 그림용으로는 원본 좌표를 그대로 둔 열도 함께 저장한다.

실행: cd streamlit_app && python3 scripts/build_link_geom.py
산출: data/seoul_link_geom.parquet   (road_lid, xs, ys, xs_full, ys_full)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "road_links.geojson"
DST = ROOT / "data" / "seoul_link_geom.parquet"

#: 단순화 허용오차(도). 위도 1e-5도 ≈ 1.11 m → 2e-5 ≈ 2.2 m
TOL = 2e-5


def _dp(x: np.ndarray, y: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Douglas-Peucker — 반복 구현(재귀 깊이 한계 회피)."""
    n = len(x)
    if n <= 2:
        return x, y
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        dx, dy = x[j] - x[i], y[j] - y[i]
        den = np.hypot(dx, dy)
        seg = slice(i + 1, j)
        if den == 0:
            d = np.hypot(x[seg] - x[i], y[seg] - y[i])
        else:
            d = np.abs(dy * (x[seg] - x[i]) - dx * (y[seg] - y[i])) / den
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m)); stack.append((m, j))
    return x[keep], y[keep]


def main() -> None:
    print(f"읽는 중: {SRC.name} ({SRC.stat().st_size/1e6:.0f} MB)")
    fc = json.load(open(SRC, encoding="utf-8"))
    feats = fc["features"]

    rows = []
    n_pt_in = n_pt_out = 0
    for f in feats:
        g = f["geometry"]
        if g is None:
            continue
        if g["type"] == "LineString":
            parts = [g["coordinates"]]
        elif g["type"] == "MultiLineString":
            parts = g["coordinates"]
        else:
            continue
        c = max(parts, key=len)          # 가장 긴 선분만 사용
        a = np.asarray(c, float)
        if len(a) < 2:
            continue
        xf, yf = a[:, 0], a[:, 1]
        xs, ys = _dp(xf, yf, TOL)
        n_pt_in += len(xf); n_pt_out += len(xs)
        rows.append((str(f["properties"]["road_lid"]),
                     xs.round(6).tolist(), ys.round(6).tolist(),
                     xf.round(6).tolist(), yf.round(6).tolist()))

    d = pd.DataFrame(rows, columns=["road_lid", "xs", "ys", "xs_full", "ys_full"])
    d.to_parquet(DST, index=False)
    print(f"저장: {DST.name}  링크 {len(d):,}개")
    print(f"정점 {n_pt_in:,} → {n_pt_out:,} ({100*n_pt_out/n_pt_in:.1f}%)")

    # 링크 테이블과 결합 가능한지 확인
    l = pd.read_parquet(ROOT / "data" / "seoul_links.parquet",
                        columns=["link_id", "road_lid"])
    m = l.road_lid.astype(str).isin(set(d.road_lid))
    print(f"seoul_links 결합 가능: {m.sum():,}/{len(l):,} ({100*m.mean():.2f}%)")


if __name__ == "__main__":
    main()
