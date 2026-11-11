"""성북구 20개 행정동 통계 재생성 — 전부 위험군 5.56% 기준으로 통일.

왜 다시 만드는가
----------------
기존 `data/seongbuk_dong_stats.csv` 는 생성 스크립트가 남아 있지 않고,
열의 정의가 서로 어긋나 있었다 (2026-10-11 감사).

- `급경사골목길` 열은 문서에 "경사 10% 이상"으로 적혀 있었으나 실제로는
  **5.56% 기준**에 가깝다 (돈암1동 62 = 현세대 5.56%↑ 62개와 일치).
  10% 기준 재산출값은 37개로 전혀 다르다.
- `위험군비율` 열은 어느 기준으로도 재현되지 않는다. 성북동 49.2% 인데
  현세대 골목길 기준 5.56%↑ 는 61.1%, 10%↑ 는 44.5%, 2%↑ 는 79.5% 다.
- 그 결과 발표자료 6쪽이 **왼쪽은 5.56% 정의, 오른쪽은 2% 기준 수치**
  (성북동 69.9% · 정릉3동 52.2%)를 쓰는 상태였다.

그래서 한 기준(**골목길 ∩ 경사 5.56% 이상**)으로 전부 다시 센다.
5.56% = 1/18, 장애인등편의법 접근로 기준 — 보행자 대상 유일한 법정 경사다.

행정동 배정
-----------
`seoul_links.parquet` 에는 행정동 열이 없어 링크 중점을 `seongbuk_dong.geojson`
폴리곤에 점-in-폴리곤으로 넣는다. 성북구 13,348 링크 중 **11,409개만**
어느 동 폴리곤 안에 떨어진다 (나머지는 경계선 위·도로 중앙선 오차).

최근접 동으로 붙이는 방법도 재 봤는데 **4개동 몫은 41.7% 로 동일**하고
동별 비율도 성북동만 61.1 → 62.8 로 움직인다. 경계 배정 방식에 결론이
흔들리지 않으므로, 더 보수적인 **within(폴리곤 내부)** 을 채택한다.

실행: cd streamlit_app && python3 scripts/build_seongbuk_dong_stats.py
산출: data/seongbuk_dong_stats.csv (덮어씀) · viz/data/p6b_dong_shares.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: 위험군 경사 하한(%) — 1/18, 장애인등편의법 접근로 기준
RISK_PCT = 5.56
GU = "성북구"
FOUR = ["정릉2동", "정릉3동", "성북동", "돈암2동"]

OUT_STATS = ROOT / "data" / "seongbuk_dong_stats.csv"
OUT_SHARE = ROOT / "viz" / "data" / "p6b_dong_shares.csv"


def assign_dong(d: pd.DataFrame) -> pd.DataFrame:
    """링크 중점을 행정동 폴리곤 안에 넣는다. 밖이면 버린다."""
    gj = json.load(open(ROOT / "data" / "seongbuk_dong.geojson"))
    polys, names = [], []
    for ft in gj["features"]:
        p = ft["properties"]
        nm = (p.get("adm_nm") or p.get("행정동") or p.get("name")
              or p.get("ADM_NM"))
        polys.append(shape(ft["geometry"]))
        names.append(str(nm).split()[-1])
    tree = STRtree(polys)
    pts = [Point(x, y) for x, y in zip(d.lon, d.lat)]
    hit = tree.query(pts, predicate="within")
    dong = np.full(len(d), None, dtype=object)
    dong[hit[0]] = [names[i] for i in hit[1]]
    d = d.assign(dong=dong)
    print(f"성북구 링크 {len(pts):,} → 동 배정 {d.dong.notna().sum():,} "
          f"({100 * d.dong.notna().mean():.1f}%)")
    return d[d.dong.notna()].copy()


def main() -> None:
    d = pd.read_parquet(ROOT / "data" / "seoul_links.parquet")
    d = assign_dong(d[d.gu == GU].copy())

    g = d[d.road_type == "골목길"].copy()
    g["위험군"] = g.slope_pct >= RISK_PCT

    rows = []
    for nm, s in g.groupby("dong"):
        a = d[d.dong == nm]
        L = s.dist_m.fillna(0)
        rows.append(dict(
            행정동=nm,
            링크수=len(a),
            골목길수=len(s),
            위험군골목길=int(s.위험군.sum()),
            위험군비율=round(100 * s.위험군.mean(), 1),
            위험군연장m=round(float(L[s.위험군].sum())),
            평균경사=round(float(s.slope_pct.mean()), 2),
            결빙지정=int(a.icing.sum()),
            열선=int(a.heat.sum()),
        ))
    r = pd.DataFrame(rows).sort_values("위험군골목길", ascending=False)
    r.to_csv(OUT_STATS, index=False, encoding="utf-8-sig")

    f = r[r.행정동.isin(FOUR)]
    share = []
    for col, lab in [("위험군골목길", "위험군 골목길"), ("결빙지정", "결빙취약구간 지정"),
                     ("링크수", "링크수(면적 대리)"), ("열선", "열선 설치")]:
        share.append(dict(항목=lab, 사개동=int(f[col].sum()),
                          성북구=int(r[col].sum()),
                          몫=round(100 * f[col].sum() / r[col].sum(), 1)))
    sh = pd.DataFrame(share).copy()
    sh.to_csv(OUT_SHARE, index=False, encoding="utf-8-sig")

    print(f"\n--- 기준: 골목길 ∩ 경사 {RISK_PCT}% 이상 ---")
    print(r.to_string(index=False))
    print(f"\n--- 4개동({'·'.join(FOUR)}) 몫 ---")
    print(sh.to_string(index=False))
    conc = (100 * f.위험군골목길.sum() / r.위험군골목길.sum()) / \
           (100 * f.링크수.sum() / r.링크수.sum())
    print(f"\n위험 집중도 = 위험군 몫 / 면적 몫 = {conc:.2f}배")
    print(f"→ {OUT_STATS.relative_to(ROOT)} · {OUT_SHARE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
