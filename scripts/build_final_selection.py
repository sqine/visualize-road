"""운영점 확정 — δ 확정값으로 최종 선정·전략비교 재생성.

100×100 격자(`run_grid_hires.py`)에서 현행 예산선의 전환점이 δ=1.69 로 나왔고,
그 해는 δ=2.06 과 완전히 같다. 예산선 100개의 전환점 δ 중위도 2.20 이다.
따라서 **δ = 2.0** 을 운영점으로 확정하고, 이 값으로 최종 산출물을 굳힌다.

산출
----
    viz/data/p12_selection.csv     링크별 선정 여부 (δ=0 / δ=2 / 현행)
    viz/data/p12_boundary.csv      잔여 경계쌍과 하류 노출
    viz/data/p12_summary.csv       전략별 요약 한 줄씩

실행: cd streamlit_app && python3 scripts/build_final_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402

VIZ = ROOT / "viz" / "data"

#: 확정 운영점
DELTA = 2.0


def main() -> None:
    G = M.load_graph()
    cand = M.candidates(G)
    risk, flow, slope = M.risk_flow(G)
    D = M.direction_set(G, cand)
    # 현행 열선은 후보에서 빠지므로 D 의 상류로 등장할 수 없다.
    # 현행을 같은 잣대로 평가하려면 상류 제한이 없는 방향집합이 따로 필요하다.
    D_ALL = M.direction_set(G, list(G.nodes()))
    B, cur_len = M.current_budget(G)
    print(f"현행 예산 {B/1e8:.2f}억 (골목길 열선 {cur_len:,.0f} m)")

    inst = set(M.installed_links(G))
    out = {}
    for name, dl in [("delta0", 0.0), ("proposed", DELTA)]:
        r = M.solve_msa(G, cand, D, risk, flow, B, dl)
        out[name] = r
        print(f"  {name:>9} δ={dl:<4} 선정 {len(r['selected']):>3}  "
              f"편익 {r['benefit']:>7.1f}  경계 {r['boundary']:>3}  "
              f"노출 {r['boundary_exposure']:>6.1f}")

    S0, S2 = set(out["delta0"]["selected"]), set(out["proposed"]["selected"])

    # ── 링크별 선정표 ────────────────────────────────────────
    rows = []
    for n in G.nodes():
        a = G.nodes[n]
        rows.append(dict(
            node=n, link_id=a.get("link_id"), dong=a.get("dong"),
            lon=a.get("lon"), lat=a.get("lat"),
            length_m=a.get("length_m"), slope_pct=a.get("slope_pct"),
            road_type=a.get("road_type"), flow=flow[n], risk=risk[n],
            현행열선=n in inst, 후보=n in set(cand),
            선정_delta0=n in S0, 선정_제안=n in S2))
    sel = pd.DataFrame(rows)
    sel.to_csv(VIZ / "p12_selection.csv", index=False, encoding="utf-8-sig")
    print(f"저장: p12_selection.csv ({len(sel):,}행)")

    # ── 잔여 경계쌍 ──────────────────────────────────────────
    brows = []
    for tag, S, DD in [("현행", inst, D_ALL), ("delta0", S0, D), ("제안", S2, D)]:
        for i, j in DD:
            if i in S and j not in S:
                brows.append(dict(
                    전략=tag, 상류=i, 하류=j,
                    상류lon=G.nodes[i]["lon"], 상류lat=G.nodes[i]["lat"],
                    하류lon=G.nodes[j]["lon"], 하류lat=G.nodes[j]["lat"],
                    낙차m=round(G.nodes[i]["z"] - G.nodes[j]["z"], 2),
                    하류노출=round(risk[j] * flow[j], 3)))
    bd = pd.DataFrame(brows)
    bd.to_csv(VIZ / "p12_boundary.csv", index=False, encoding="utf-8-sig")
    print(f"저장: p12_boundary.csv ({len(bd):,}행)")

    # ── 전략 요약 ────────────────────────────────────────────
    def row(tag, S, DD=None):
        S = set(S)
        bdl = [(i, j) for (i, j) in (DD or D) if i in S and j not in S]
        ben = sum(risk[n] * flow[n] for n in S)
        exp = sum(risk[j] * flow[j] for (_, j) in bdl)
        L = sum(G.nodes[n].get("length_m") or 0.0 for n in S)
        return dict(전략=tag, 선정수=len(S), 연장m=round(L),
                    투입액억=round(L * M.COST_PER_M / 1e8, 2),
                    기대방지=round(ben, 1), 경계수=len(bdl),
                    경계노출=round(exp, 1),
                    노출편익비=round(100 * exp / ben, 1) if ben else None,
                    평균경사=round(float(np.mean([slope[n] for n in S])), 2),
                    평균통행=round(float(np.mean([flow[n] for n in S])), 1))

    s = pd.DataFrame([row("현행 실제", inst, D_ALL),
                      row("경계 미고려 (δ=0)", S0),
                      row(f"제안 (δ={DELTA:g})", S2)])
    s.to_csv(VIZ / "p12_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n" + s.to_string(index=False))

    j = len(S0 & S2) / len(S0 | S2)
    print(f"\nδ=0 ↔ 제안 Jaccard {j:.3f} · 교체 {len(S0-S2)}개 "
          f"(전체 {len(S0)}개 중 {100*len(S0-S2)/len(S0):.1f}%)")


if __name__ == "__main__":
    main()
