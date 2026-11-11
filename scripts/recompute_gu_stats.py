"""자치구별 통계 재산출 — 5.56% 기준 통일.

대상
----
1. golmok_slope_class_25gu.csv   자치구별 골목길·위험군·급경사 수와 순위
   → 기존 파일은 위험군을 **경사 2%** 로 잡았다. 보고서 채택 기준인
     **5.56%**(장애인등편의법 접근로 1/18)로 재산출한다.

2. icing_capture_by_gu.csv       결빙취약구간 지정이 급경사 골목길을 포착하는가
   → obs = 지정 매칭 링크 중 (골목길 ∩ 경사 ≥ θ) 비율
     base = 그 자치구 전체 링크 중 같은 비율 (무작위 지정 시 기대값)
     ratio = obs / base   (1 미만이면 지형을 덜 반영)
   → 2% 와 5.56% 두 기준을 함께 산출해 비교한다.

   ⚠️ 행정동 단위(icing_capture_seongbuk_dong.csv)는 동당 매칭이 3~5건에
      불과해 통계적으로 성립하지 않는다. 폐기 대상이며 재산출하지 않는다.

입력  data/seoul_links.parquet  (scripts/build_seoul_dataset.py 산출)
실행  cd streamlit_app && python3 scripts/recompute_gu_stats.py
산출  viz/data/golmok_slope_class_25gu_v2.csv
      viz/data/icing_capture_by_gu_v2.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # 원자료·파이프라인 산출물
VIZ = ROOT / "viz" / "data"   # 그림 입력 전용 집계표

T_LEGAL = 100 / 18     # 5.56% — 장애인등편의법 접근로
T_OLD = 2.0            # 기존 파일 기준
T_STEEP = 10.0         # 급경사


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "seoul_links.parquet")
    return df[df.slope_pct.notna() & df.dist_m.notna() & df.gu.notna()].copy()


def golmok_class(df: pd.DataFrame) -> pd.DataFrame:
    """자치구별 골목길 규모와 위험군 규모 — 전부 5.56% 단일 기준.

    기존 버전에 있던 '급경사(10%↑)' 열은 제거했다. 보고서 전체가 5.56%
    (장애인등편의법 접근로 1/18)를 쓰는데 그 열만 10%라 기준이 섞였다.
    대신 '위험군 중 골목길 비중'(모든 도로유형 위험군 대비)을 넣어
    "위험은 골목길에 있다"를 자치구 단위로도 보이게 했다.
    """
    rows = []
    for gu, s in df.groupby("gu"):
        g = s[s.road_type == "골목길"]
        if not len(g):
            continue
        gr = g[g.slope_pct >= T_LEGAL]              # 위험군 골목길
        ar = s[s.slope_pct >= T_LEGAL]              # 위험군 전체(모든 유형)
        rows.append(dict(
            자치구=gu,
            골목길=len(g),
            골목길연장m=round(g.dist_m.sum()),
            위험군=len(gr),
            위험군연장m=round(gr.dist_m.sum()),
            위험군율=round(100 * gr.dist_m.sum() / g.dist_m.sum(), 1),
            위험군중골목길비중=round(100 * gr.dist_m.sum() / ar.dist_m.sum(), 1)
            if ar.dist_m.sum() else None,
            위험군중골목길비중_구간=round(100 * len(gr) / len(ar), 1)
            if len(ar) else None,
        ))
    d = pd.DataFrame(rows)
    for c in ["골목길", "골목길연장m", "위험군", "위험군연장m", "위험군율",
              "위험군중골목길비중"]:
        d["R" + c] = d[c].rank(ascending=False).astype(int)
    return d.sort_values("R위험군연장m")


def _z(obs_k, n, p0):
    """이항 비율 검정 z (기대비율 p0 대비)."""
    if n == 0 or p0 in (0, 1):
        return None
    se = math.sqrt(p0 * (1 - p0) / n)
    return (obs_k / n - p0) / se if se else None


def icing_capture(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gu, s in df.groupby("gu"):
        m = s[s.icing]
        r = dict(자치구=gu, 전체링크=len(s), 매칭=len(m))
        for tag, th in [("2", T_OLD), ("556", T_LEGAL)]:
            tgt = (s.road_type == "골목길") & (s.slope_pct >= th)
            base = tgt.mean()
            if len(m):
                mt = (m.road_type == "골목길") & (m.slope_pct >= th)
                obs = mt.mean()
                r[f"obs{tag}"] = round(obs, 4)
                r[f"base{tag}"] = round(base, 4)
                r[f"ratio{tag}"] = round(obs / base, 3) if base else None
                z = _z(mt.sum(), len(m), base)
                r[f"z{tag}"] = round(z, 2) if z is not None else None
            else:
                r.update({f"obs{tag}": None, f"base{tag}": round(base, 4),
                          f"ratio{tag}": None, f"z{tag}": None})
        rows.append(r)
    return pd.DataFrame(rows).sort_values("매칭", ascending=False)


def main() -> None:
    df = load()
    print(f"유효 링크 {len(df):,} / 결빙 매칭 {int(df.icing.sum()):,}\n")

    g = golmok_class(df)
    g.to_csv(VIZ / "golmok_slope_class_25gu_v2.csv", index=False,
             encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("=== golmok_slope_class_25gu_v2 (전부 경사 5.56%↑ 단일 기준) 상위 8 ===")
    print(g.head(8)[["자치구", "골목길", "골목길연장m", "위험군", "위험군연장m",
                     "R위험군연장m", "위험군율", "R위험군율",
                     "위험군중골목길비중"]].to_string(index=False))
    s = g[g.자치구 == "성북구"].iloc[0]
    print(f"\n성북구: 골목길 {s.골목길:,}개({s.R골목길}위) / {s.골목길연장m:,}m "
          f"· 위험군 {s.위험군:,}개({s.R위험군}위) / {s.위험군연장m:,}m"
          f"({s.R위험군연장m}위) · 위험군율 {s.위험군율}%({s.R위험군율}위) "
          f"· 위험군 중 골목길 {s.위험군중골목길비중}%")

    ic = icing_capture(df)
    ic.to_csv(VIZ / "icing_capture_by_gu_v2.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== icing_capture_by_gu_v2 — 매칭 상위 10 ===")
    print(ic.head(10)[["자치구", "매칭", "obs2", "base2", "ratio2",
                       "obs556", "base556", "ratio556"]].to_string(index=False))

    big = ic[ic.매칭 >= 10]
    print(f"\n매칭 10건 이상 자치구 {len(big)}개만 비교")
    for tag in ["2", "556"]:
        b = big.dropna(subset=[f"ratio{tag}"]).sort_values(f"ratio{tag}")
        th = "2%" if tag == "2" else "5.56%"
        sb = b[b.자치구 == "성북구"]
        rank = (list(b.자치구).index("성북구") + 1) if len(sb) else None
        print(f"  [{th:>5} 기준] 성북구 ratio "
              f"{float(sb[f'ratio{tag}'].iloc[0]) if len(sb) else float('nan'):.3f} "
              f"— {len(b)}개 중 {rank}위(낮은 순) · "
              f"최저 {b.자치구.iloc[0]} {b[f'ratio{tag}'].iloc[0]:.3f}")

    print("\n※ 매칭 1~2건 자치구(금천·성동·강남 등)는 비율이 0 또는 극단값으로 "
          "튀므로 순위 비교에서 제외해야 한다.")


if __name__ == "__main__":
    main()
