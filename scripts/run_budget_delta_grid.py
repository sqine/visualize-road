"""예산 × δ 격자 — 효율적 예산과 그 예산의 권장 δ를 찾는다.

설계
----
고정 예산에서 δ를 올리면 경계는 줄고 편익은 깎인다(1차원 트레이드오프).
예산을 바꾸면 그 곡선 자체가 바깥으로 밀린다. 두 축을 같이 보면
"예산을 정하면 δ가 따라오고, δ까지 최적화한 예산 곡선에서 효율점을 고른다"가
성립한다.

예산 격자
---------
후보군 847곳 총연장 39,860 m 전면 시공비 = **351.7억**
그 대비 비율로 예산을 잡는다. 기준선 2개를 함께 표시한다.
    모형 예산 8.81억  = 2.5%
    현행 투입 18.53억 = 5.3%

실행
----
    cd streamlit_app
    python3 scripts/run_budget_delta_grid.py

산출
----
    viz/data/p11_budget_delta_grid.csv   격자 전체
    viz/data/p11_budget_frontier.csv     예산별 권장 δ와 그 해
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402

VIZ = ROOT / "viz" / "data"

#: 전면 시공비 대비 예산 비율(%)
BUDGET_PCT = [1, 2, 2.5, 3, 4, 5.3, 7, 10, 15, 20, 30]
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]


def knee(df: pd.DataFrame) -> float:
    """트레이드오프 곡선의 파레토 전환점 — 양 끝점을 잇는 현에서 가장 먼 점(Kneedle)."""
    base = df.loc[df.delta == 0, "경계노출"].iloc[0]
    if base <= 0:
        return 0.0
    x = df["편익손실%"].to_numpy(float)
    y = (100 * (1 - df["경계노출"] / base)).to_numpy(float)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    n = (dx * dx + dy * dy) ** 0.5
    if n == 0:
        return 0.0
    return float(df["delta"].iloc[int((np.abs(dy * (x - x[0]) - dx * (y - y[0])) / n).argmax())])


def main() -> None:
    t0 = time.time()
    G = M.load_graph()
    cand = M.candidates(G)
    risk, flow, slope = M.risk_flow(G)
    D = M.direction_set(G, cand)
    ln_all = sum(G.nodes[n].get("length_m") or 0 for n in cand)
    FULL = ln_all * M.COST_PER_M
    total_risk = sum(risk[n] * flow[n] for n in cand)
    print(f"후보 {len(cand)} / 총연장 {ln_all:,.0f} m / 전면 시공비 {FULL/1e8:.1f}억")
    print(f"후보군 총위험 {total_risk:,.1f}건 / 방향엣지 {len(D)}\n")

    rows = []
    for pct in BUDGET_PCT:
        B = FULL * pct / 100
        base = M.solve_msa(G, cand, D, risk, flow, B, 0.0)
        for d in DELTAS:
            res = base if d == 0 else M.solve_msa(G, cand, D, risk, flow, B, d)
            r = M.summarize(G, res, risk, flow, slope, B,
                            base_benefit=base["benefit"],
                            base_boundary=base["boundary"] or None, D_all=D)
            r.update(예산비율=pct, 예산억=round(B / 1e8, 2), delta=d,
                     위험비율=round(100 * res["benefit"] / total_risk, 1))
            rows.append(r)
        b0 = rows[-len(DELTAS)]
        print(f"  {pct:>4}% ({B/1e8:>5.1f}억)  δ=0 선정 {b0['선정수']:>3} "
              f"방지 {b0['기대방지']:>7.1f} 경계 {b0['잔여경계']:>3}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    g = pd.DataFrame(rows)
    cols = ["예산비율", "예산억", "delta", "선정수", "선정연장m", "집행액억",
            "기대방지", "위험비율", "편익손실%", "잔여경계", "잔여경계_D전체",
            "경계감소%", "경계노출", "평균경사", "평균통행", "MSA수렴반복"]
    g = g[[c for c in cols if c in g.columns]]
    g.to_csv(VIZ / "p11_budget_delta_grid.csv", index=False, encoding="utf-8-sig")

    # 예산별 권장 δ
    fr = []
    for pct, s in g.groupby("예산비율"):
        s = s.sort_values("delta").reset_index(drop=True)
        k = knee(s)
        r = s[s.delta == k].iloc[0]
        b = s.iloc[0]
        fr.append(dict(
            예산비율=pct, 예산억=r.예산억, 권장delta=k,
            선정수=int(r.선정수), 기대방지=r.기대방지, 위험비율=r.위험비율,
            잔여경계=int(r.잔여경계), 경계_델타0=int(b.잔여경계),
            경계감소pct=round(100 * (1 - r.잔여경계 / b.잔여경계), 1)
            if b.잔여경계 else None,
            편익손실pct=r["편익손실%"],
            한계효율_건per억=None))
    f = pd.DataFrame(fr).sort_values("예산비율").reset_index(drop=True)
    # 한계효율 = Δ기대방지 / Δ예산
    f["한계효율_건per억"] = (f.기대방지.diff() / f.예산억.diff()).round(1)
    f["평균효율_건per억"] = (f.기대방지 / f.예산억).round(1)
    f.to_csv(VIZ / "p11_budget_frontier.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 예산별 권장 δ와 그 해 ===")
    print(f.to_string(index=False))
    print(f"\n저장: viz/data/p11_budget_delta_grid.csv ({len(g)}행) / "
          f"p11_budget_frontier.csv ({len(f)}행)  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
