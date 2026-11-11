"""선택 기준 5종을 같은 솔버·같은 예산으로 풀어 한 평면에 세운다.

왜 다시 푸는가
--------------
기존 p10_strategy_v3.csv 는 ①②③ 이 기준값 순 그리디, ④⑤ 가 MILP 였다.
그러면 "③이 ④보다 못한 것은 기준 때문인가 알고리즘 때문인가"를 가릴 수 없다.
여기서는 **다섯 가지를 모두 MILP 로** 푼다. 예산·후보군·평가식이 전부 같으므로
차이는 오직 '무엇을 보고 골랐는가'에서 온다.

  ① 경사(위험)만   max Σ slope_i · x_i
  ② 통행만         max Σ flow_i · x_i
  ③ 위험 × 통행    max Σ risk_i · flow_i · x_i            (= 제안에서 δ=0)
  ④ 제안           max Σ risk_i·flow_i·x_i − δ Σ w_ij b_ij  (δ=2)
  현행 실제        지금 깔려 있는 열선 34구간

모두 같은 예산 B(현행 18.53억)와 같은 평가식(기대방지·경계수·경계노출)으로
비교한다. 현행은 후보군에 없으므로 상류 제한 없는 방향집합 D_ALL 로 잰다.

실행: cd streamlit_app && python3 scripts/build_strategy_balance.py
산출: viz/data/p10_balance.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402

VIZ = ROOT / "viz" / "data"
DELTA = 2.0


def knapsack(G, cand, score: dict, B: float) -> set:
    """예산 제약 아래 score 합을 최대화하는 0-1 배낭 (CBC)."""
    ln = {n: (G.nodes[n].get("length_m") or 0.0) for n in cand}
    p = pulp.LpProblem("single_criterion", pulp.LpMaximize)
    x = {n: pulp.LpVariable(f"x{n}", cat="Binary") for n in cand}
    p += pulp.lpSum(score[n] * x[n] for n in cand)
    p += pulp.lpSum(ln[n] * M.COST_PER_M * x[n] for n in cand) <= B
    p.solve(pulp.PULP_CBC_CMD(msg=0))
    return {n for n in cand if x[n].value() and x[n].value() > 0.5}


def main() -> None:
    G = M.load_graph()
    cand = M.candidates(G)
    risk, flow, slope = M.risk_flow(G)
    D = M.direction_set(G, cand)
    D_ALL = M.direction_set(G, list(G.nodes()))
    B, cur_len = M.current_budget(G)
    inst = set(M.installed_links(G))
    print(f"예산 {B/1e8:.2f}억 · 후보 {len(cand):,} · 방향쌍 {len(D):,}")

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

    rows = [row("현행 실제", inst, D_ALL)]

    S1 = knapsack(G, cand, slope, B)
    rows.append(row("① 경사(위험)만", S1))

    S2 = knapsack(G, cand, flow, B)
    rows.append(row("② 통행만", S2))

    r3 = M.solve_msa(G, cand, D, risk, flow, B, 0.0)
    rows.append(row("③ 위험 × 통행", r3["selected"]))

    # 이 예산선의 파레토 전환점 δ ≈ 2.1 — 편익을 가장 지키는 하한
    r4 = M.solve_msa(G, cand, D, risk, flow, B, 2.0)
    rows.append(row("④ 경계까지 — 전환점 (δ=2)", r4["selected"]))

    # 최종 채택안 S1 — 전환점에서 δ 를 더 올려 경계를 한 번 더 줄인 지점
    r5 = M.solve_msa(G, cand, D, risk, flow, B, 4.0)
    rows.append(row("⑤ 제안 채택안 (δ=4)", r5["selected"]))

    out = pd.DataFrame(rows)
    out.to_csv(VIZ / "p10_balance.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n저장: p10_balance.csv")


if __name__ == "__main__":
    main()
