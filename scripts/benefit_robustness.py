"""편익 측정 강건성 검증 [2026-09-23].

두 가지 자유도가 결과를 바꾸는지 확인한다.
  (1) a(증발률) = 0 / 0.3 / 0.5 / 1.0
      - P_i = risk_i x (1-a) x flow_i 이므로, a가 커질수록 P_i가 작아지고
        목적함수에서 Bonus_ij의 상대 비중이 커진다. a=1이면 P_i=0.
  (2) 편익 지표 = 보호통행량(sum flow) / 위험가중통행량(sum risk*flow) / 보호위험량(sum risk)
      - 어느 것도 특정 전략의 목적함수가 아니어야 공정하다.

전략:
  A 현행대리 : score = risk_i            (M5에서 확인한 현행 패턴 모사)
  B 제안순위 : score = risk_i * flow_i   (Bonus 없음)
  C 제안MILP : max sum P_i x_i + sum Bonus_ij y_ij
"""
import json
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

EPS = 1e-6
COST_PER_M = 1_500_000_000 / 1700
DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
A_VALUES = [0.0, 0.3, 0.5, 1.0]
BUDGET_PCTS = [0.05, 0.10, 0.25]


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


def compute_bonus(G, Gg, L, nodes, idx, gap_set, risk, a):
    giant = set(Gg.nodes())
    Bonus = {}
    for g, h in [(u, v) for u, v in G.edges() if u in gap_set and v in gap_set]:
        if g not in giant or h not in giant:
            continue
        ig, og = disp_inj(Gg, g, a); vg = solve(L, nodes, idx, ig, nodes[0])
        ih, oh = disp_inj(Gg, h, a); vh = solve(L, nodes, idx, ih, nodes[0])
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
    return Bonus


def greedy(score, length, B):
    order = sorted(score, key=lambda n: score[n] / max(length[n], 1), reverse=True)
    sel, left = set(), B
    for n in order:
        c = length[n] * COST_PER_M
        if c <= left:
            sel.add(n); left -= c
    return sel


def main():
    t0 = time.time()
    with open(f"{DATA}/road_graph.json") as f:
        G = nx.node_link_graph(json.load(f), edges="links")
    Gg = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    L, nodes, idx = build_L(Gg)

    gap = [n for n in G.nodes() if (G.nodes[n].get("slope_pct") or -1) >= 6.0
           and not G.nodes[n].get("heating_installed_precise")
           and G.nodes[n].get("road_type") == "골목길"]
    gs = set(gap)
    risk = {n: slope_risk(G.nodes[n].get("slope_pct")) for n in gap}
    flow = {n: (G.nodes[n].get("flow_sum", 0) or 0) for n in gap}
    length = {n: G.nodes[n].get("length_m", 0) or 0 for n in gap}
    total_len = sum(length.values())
    print(f"후보 {len(gap)}개, 총연장 {total_len:.0f}m", flush=True)

    # 전략 A, B는 a와 무관 (risk만 / risk*flow) -> 예산별로 1회만 계산
    selA = {pct: greedy(risk, length, pct * total_len * COST_PER_M) for pct in BUDGET_PCTS}
    selB = {pct: greedy({n: risk[n] * flow[n] for n in gap}, length,
                        pct * total_len * COST_PER_M) for pct in BUDGET_PCTS}

    def benefit(sel):
        return dict(
            보호통행량=sum(flow[n] for n in sel),
            위험가중통행량=sum(risk[n] * flow[n] for n in sel),
            보호위험량=sum(risk[n] for n in sel),
            구간수=len(sel),
        )

    import pulp
    rows = []
    for a in A_VALUES:
        Bonus = compute_bonus(G, Gg, L, nodes, idx, gs, risk, a)
        P = {n: risk[n] * (1 - a) * flow[n] for n in gap}
        for pct in BUDGET_PCTS:
            B = pct * total_len * COST_PER_M
            prob = pulp.LpProblem(f"m{int(a*100)}_{int(pct*100)}", pulp.LpMaximize)
            x = {n: pulp.LpVariable(f"x_{n}", cat="Binary") for n in gap}
            y = {k: pulp.LpVariable(f"y_{k[0]}_{k[1]}", cat="Binary") for k in Bonus}
            prob += pulp.lpSum(P[n] * x[n] for n in gap) + pulp.lpSum(Bonus[k] * y[k] for k in Bonus)
            prob += pulp.lpSum(length[n] * COST_PER_M * x[n] for n in gap) <= B
            for (g, h) in Bonus:
                prob += y[(g, h)] <= x[g]; prob += y[(g, h)] <= x[h]
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            selC = {n for n in gap if x[n].value() > 0.5}

            bA, bB, bC = benefit(selA[pct]), benefit(selB[pct]), benefit(selC)
            for metric in ["보호통행량", "위험가중통행량", "보호위험량"]:
                rows.append(dict(a=a, budget_pct=pct, metric=metric,
                                 A=round(bA[metric], 1), B=round(bB[metric], 1), C=round(bC[metric], 1),
                                 C_over_A=round(bC[metric] / bA[metric], 3) if bA[metric] else None,
                                 C_over_B=round(bC[metric] / bB[metric], 3) if bB[metric] else None,
                                 nA=bA["구간수"], nB=bB["구간수"], nC=bC["구간수"],
                                 bonus_sum=round(sum(Bonus.values()), 1)))
            print(f"a={a:.1f} 예산{pct*100:.0f}% : C선정 {len(selC)}개, "
                  f"보호통행량 A={bA['보호통행량']:.0f} B={bB['보호통행량']:.0f} C={bC['보호통행량']:.0f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("/tmp/benefit_robustness.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n=== 전체 결과 ===")
    print(df.to_string(index=False))

    print("\n=== C/A 배율: a x 지표 (예산 5%) ===")
    piv = df[df.budget_pct == 0.05].pivot(index="a", columns="metric", values="C_over_A")
    print(piv.to_string())
    print("\n=== C/A 배율: a x 지표 (예산 25%) ===")
    piv2 = df[df.budget_pct == 0.25].pivot(index="a", columns="metric", values="C_over_A")
    print(piv2.to_string())


if __name__ == "__main__":
    main()
