"""위험군 규모는 전역 데이터로 센다 — 유동인구 매칭과 분리.

왜 나누나
---------
MILP 는 유동인구(B030)가 붙은 링크 위에서만 풀 수 있다. 그런데 경사도는
1:1,000 수치지형도로 **모든 링크에 매칭**돼 있다. 그래서 '위험군이 얼마나
되는가'와 '현행 열선이 그중 얼마를 덮는가'는 전역 데이터로 세는 쪽이 맞다.
모형이 푸는 범위(유동인구 측정 구간)와 현황을 재는 범위(전역)를 이름으로
구분한다.

  위험군 = 골목길 ∩ 경사 5.56%(= 100/18) 이상
  커버리지 = 위험군 중 heat 플래그가 붙은 연장 / 위험군 연장

실행: cd streamlit_app && python3 scripts/build_risk_scope.py
산출: viz/data/risk_scope.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
VIZ = ROOT / "viz" / "data"
DONGS = ("정릉2동", "정릉3동", "성북동", "돈암2동")
RISK_SLOPE = 100 / 18   # 5.56% — 장애인등편의법 접근로 1/18


def tag_dong(df: pd.DataFrame) -> pd.DataFrame:
    fc = json.load(open(DATA / "seongbuk_dong.geojson", encoding="utf-8"))
    pts = np.column_stack((df.lon.values, df.lat.values))
    df = df.copy()
    df["dong"] = None
    for f in fc["features"]:
        short = f["properties"]["adm_nm"].split()[-1]
        if short not in DONGS:
            continue
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        m = np.zeros(len(df), bool)
        for poly in polys:
            m |= MPath(np.asarray(poly[0], float)).contains_points(pts)
        df.loc[m, "dong"] = short
    return df


def row(tag: str, d: pd.DataFrame) -> dict:
    gol = d[d.road_type == "골목길"]
    ris = gol[gol.slope_pct >= RISK_SLOPE]
    heat = ris[ris.heat.fillna(False).astype(bool)]
    L = ris.dist_m.sum()
    return dict(범위=tag, 도로링크=len(d), 도로연장m=round(d.dist_m.sum()),
                골목길=len(gol), 골목길연장m=round(gol.dist_m.sum()),
                위험군=len(ris), 위험군연장m=round(L),
                위험군중열선=len(heat), 위험군중열선m=round(heat.dist_m.sum()),
                커버리지=round(100 * heat.dist_m.sum() / L, 2) if L else None)


def main() -> None:
    l = pd.read_parquet(DATA / "seoul_links.parquet")
    l["dist_m"] = l.dist_m.fillna(0.0)
    rows = [row("서울 전역", l), row("성북구", l[l.gu == "성북구"])]
    f4 = tag_dong(l[l.gu == "성북구"])
    rows.append(row("성북구 4개동", f4[f4.dong.notna()]))
    for d in DONGS:
        rows.append(row(f"  {d}", f4[f4.dong == d]))
    out = pd.DataFrame(rows)
    out.to_csv(VIZ / "risk_scope.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print("\n저장: risk_scope.csv")


if __name__ == "__main__":
    main()
