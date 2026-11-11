"""예산선별 효율 비교 — 현행 관행 대리모델 vs 제안 모델 [2026-09-23].

세 전략을 동일 후보군·동일 예산에서 비교한다.
  (A) 현행 대리(RISK-ONLY)   : 4장 ④-2에서 확인한 현행 설치 패턴(유동인구 미반영)을 모사.
                               점수 = risk_i 만, 비용효율 순으로 예산 소진까지.
  (B) 제안-순위만(FLOW)      : 점수 = risk_i x flow_i (Bonus 없음), 비용효율 순.
  (C) 제안-MILP(FLOW+BONUS)  : 목적함수 = sum P_i x_i + sum Bonus_ij y_ij, 예산제약 MILP.

공통 성과지표(어느 전략의 목적함수도 아닌 중립 지표로도 함께 평가):
  - 보호 통행량      : 선정 구간의 flow 합
  - 보호 위험량      : 선정 구간의 risk 합
  - 위험가중 통행량  : sum(risk_i * flow_i)   <- (B)(C)의 목적함수와 같은 축이므로 참고용
  - 선정 구간 수 / 총연장
"""
import json
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

EPS = 1e-6
A_CENTRAL = 0.25
COST_PER_M = 1_500_000_000 / 1700
DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
BUDGET_PCTS = [0.02, 0.05, 0.10, 0.15, 0.25]


def slope_risk(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0.20
    if s >= 10: return 1.00
    if s >= 6: return 0.70
    if s >= 3: return 0.40
    return 0.10


def build_L(G):
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}
    L = lil_matrix((len(nodes), len(nodes)))
    for u, v in G.edges():
        c = ((G.nodes[u].get("flow_sum", 0) or 0) + (G.nodes[v].get("flow_sum", 0) or 0)) / 2.0 + EPS
        i, j = idx[u], idx[v]
        L[i, i] += c; L[j, j] += c; L[i, j] -= c; L[j, i] -= c
    return csr_matrix(L), nodes, idx


def solve(L, nodes, idx, inj, ref):
    n = len(nodes); b = np.zeros(n)
    for node, amt in inj.items():
        b[idx[node]] = amt
    ri = idx[ref]; keep = [i for i in range(n) if i != ri]
    vr = spsolve(L[keep, :][:, keep].tocsc(), b[keep])
    v = np.zeros(n)
    for k, i in enumerate(keep):
        v[i] = vr[k]
    return {nodes[i]: v[i] for i in range(n)}


def edge_cur(G, v, u, w):
    c = ((G.nodes[u].get("flow_sum", 0) or 0) + (G.nodes[w].get("flow_sum", 0) or 0)) / 2.0 + EPS
    return c * (v[u] - v[w])


def disp_inj(G, node, a):
    out = (1 - a) * (G.nodes[node].get("flow_sum", 0) or 0)
    others = [nd for nd in G.nodes() if nd != node]
    w = np.array([G.nodes[nd].get("flow_sum", 0) or 0 for nd in others], dtype=float) + EPS
    w = w / w.sum()
    inj = {node: out}
    for nd, ww in zip(others, w):
        inj[nd] = inj.get(nd, 0) - out * ww
    return inj, out


def greedy_by_score(score, length, B):
    order = sorted(score, key=lambda n: score[n] / max(length[n], 1), reverse=True)
    sel, left = [], B
    for n in order:
        c = length[n] * COST_PER_M
        if c <= left:
            sel.append(n); left -= c
    return set(sel)


def evaluate(G, sel, risk, flow, length):
    return dict(n=len(sel),
                len_m=round(sum(length[n] for n in sel), 1),
                flow=round(sum(flow[n] for n in sel), 1),
                risk=round(sum(risk[n] for n in sel), 2),
                riskflow=round(sum(risk[n] * flow[n] for n in sel), 1))


def main():
    t0 = time.time()
    with open(f"{DATA}/road_graph.json") as f:
        G = nx.node_link_graph(json.load(f), edges="links")
    Gg = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    L, nodes, idx = build_L(Gg)
    giant = set(Gg.nodes())

    gap = [n for n in G.nodes() if (G.nodes[n].get("slope_pct") or -1) >= 6.0
           and not G.nodes[n].get("heating_installed_precise")
           and G.nodes[n].get("road_type") == "골목길"]
    gs = set(gap)
    risk = {n: slope_risk(G.nodes[n].get("slope_pct")) for n in gap}
    flow = {n: (G.nodes[n].get("flow_sum", 0) or 0) for n in gap}
    length = {n: G.nodes[n].get("length_m", 0) or 0 for n in gap}
    P = {n: risk[n] * (1 - A_CENTRAL) * flow[n] for n in gap}
    total_len = sum(length.values())
    print(f"후보 {len(gap)}개, 총연장 {total_len:.0f}m ({time.time()-t0:.1f}s)", flush=True)

    Bonus = {}
    for g, h in [(u, v) for u, v in G.edges() if u in gs and v in gs]:
        if g not in giant or h not in giant:
            continue
        ig, og = disp_inj(Gg, g, A_CENTRAL); vg = solve(L, nodes, idx, ig, nodes[0])
        ih, oh = disp_inj(Gg, h, A_CENTRAL); vh = solve(L, nodes, idx, ih, nodes[0])
        cn = (set(Gg.neighbors(g)) & set(Gg.neighbors(h))) - {g, h}
        if not cn:
            continue
        bg = sum(abs(edge_cur(Gg, vg, g, k)) for k in cn if k in Gg[g])
        bh = sum(abs(edge_cur(Gg, vh, h, k)) for k in cn if k in Gg[h])
        tot = og + oh
        others = [nd for nd in Gg.nodes() if nd not in (g, h)]
        w = np.array([Gg.nodes[nd].get("flow_sum", 0) or 0 for nd in others]) + EPS
        w = w / w.sum()
        igh = {g: og, h: oh}
        for nd, ww in zip(others, w):
            igh[nd] = igh.get(nd, 0) - tot * ww
        vgh = solve(L, nodes, idx, igh, nodes[0])
        bj = sum(abs(edge_cur(Gg, vgh, g, k)) for k in cn if k in Gg[g]) + \
             sum(abs(edge_cur(Gg, vgh, h, k)) for k in cn if k in Gg[h])
        rg = (risk.get(g, 0.2) * risk.get(h, 0.2)) ** 0.5
        Bonus[(g, h)] = max(0.0, rg * (bj - (bg + bh)))
    print(f"Bonus 쌍 {len(Bonus)}개 ({time.time()-t0:.1f}s)", flush=True)

    import pulp
    rows = []
    for pct in BUDGET_PCTS:
        B = pct * total_len * COST_PER_M
        selA = greedy_by_score(risk, length, B)
        selB = greedy_by_score(P, length, B)

        prob = pulp.LpProblem(f"b{int(pct*100)}", pulp.LpMaximize)
        x = {n: pulp.LpVariable(f"x_{n}", cat="Binary") for n in gap}
        y = {k: pulp.LpVariable(f"y_{k[0]}_{k[1]}", cat="Binary") for k in Bonus}
        prob += pulp.lpSum(P[n] * x[n] for n in gap) + pulp.lpSum(Bonus[k] * y[k] for k in Bonus)
        prob += pulp.lpSum(length[n] * COST_PER_M * x[n] for n in gap) <= B
        for (g, h) in Bonus:
            prob += y[(g, h)] <= x[g]; prob += y[(g, h)] <= x[h]
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        selC = {n for n in gap if x[n].value() > 0.5}

        for label, sel in [("A_현행대리(risk만)", selA), ("B_제안순위(risk×flow)", selB), ("C_제안MILP(+Bonus)", selC)]:
            ev = evaluate(G, sel, risk, flow, length)
            realized = sum(v for k, v in Bonus.items() if k[0] in sel and k[1] in sel)
            rows.append(dict(budget_pct=pct, budget_억=round(B / 1e8, 2), strategy=label,
                             **ev, bonus_realized=round(realized, 2)))
        print(f"예산 {pct*100:.0f}%({B/1e8:.2f}억) 완료 ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("/tmp/budget_efficiency.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))

    # 효율 비교: 같은 예산에서 성과 배율 / 같은 성과를 내는 데 필요한 예산
    print("\n=== 같은 예산에서 성과 배율 (A 대비) ===")
    for pct in BUDGET_PCTS:
        a = df[(df.budget_pct == pct) & (df.strategy.str.startswith("A"))].iloc[0]
        c = df[(df.budget_pct == pct) & (df.strategy.str.startswith("C"))].iloc[0]
        print(f"  예산 {pct*100:>2.0f}%: 보호통행량 {a['flow']:8.1f} -> {c['flow']:8.1f} "
              f"({c['flow']/a['flow']:.2f}배)  | 위험가중통행량 {a['riskflow']:8.1f} -> {c['riskflow']:8.1f} "
              f"({c['riskflow']/a['riskflow']:.2f}배)")

    print("\n=== A가 최대예산(25%)에서 낸 성과를, C는 몇 % 예산으로 달성하는가 ===")
    a_max = df[(df.budget_pct == 0.25) & (df.strategy.str.startswith("A"))].iloc[0]
    for metric in ["flow", "riskflow"]:
        target = a_max[metric]
        cands = df[(df.strategy.str.startswith("C")) & (df[metric] >= target)]
        if len(cands):
            need = cands.budget_pct.min()
            print(f"  {metric}: A는 25% 예산으로 {target:.1f} → C는 {need*100:.0f}% 예산으로 달성 "
                  f"(예산 {100-need/0.25*100:.0f}% 절감)")
        else:
            print(f"  {metric}: A의 25% 성과({target:.1f})를 C가 25% 이내 예산으로는 달성 못함")


if __name__ == "__main__":
    main()
