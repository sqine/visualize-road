"""증발률 a 스윕 — "통행량 부분감소" 모델 vs "통행 완전차단" 모델 비교 [2026-09-23].

8장 전파모형에서 a는 결빙 구간의 통행량 중 **사라지는(통행 포기) 비율**이고,
(1-a)가 인접 경로로 **우회하는 비율**이다. 해석은 양 극단에서 완전히 갈린다.

  a = 0.0  : 통행량이 하나도 사라지지 않고 전량 우회  = "길이 막히면 반드시 돌아간다"
  a = 0.3  : 30% 포기 + 70% 우회 (노르웨이 조사의 '눈·결빙 시 고령자 보행 1/3 감소'와 정합)
  a = 1.0  : 전량 포기, 우회 없음                      = "길이 막히면 아예 안 나간다"

a=1.0이면 재분배량이 0이 되어 Bonus_ij(연결효과)가 정의상 사라진다 — 즉 이 파라미터는
'연결효과가 존재하는가' 자체를 좌우한다. 그래서 선정 결과가 a에 얼마나 민감한지 확인한다.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

EPS = 1e-6
COST_PER_M = 1_500_000_000 / 1700
DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"


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


def run(G, a):
    Gg = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    L, nodes, idx = build_L(Gg)
    giant = set(Gg.nodes())
    gap = [n for n in G.nodes() if (G.nodes[n].get("slope_pct") or -1) >= 6.0
           and not G.nodes[n].get("heating_installed_precise")
           and G.nodes[n].get("road_type") == "골목길"]
    gs = set(gap)
    P = {n: slope_risk(G.nodes[n].get("slope_pct")) * (1 - 0.25) * (G.nodes[n].get("flow_sum", 0) or 0)
         for n in gap}   # P_i의 (1-a)는 기존 중심값 0.25 고정(목적함수 스케일 통일)
    length = {n: G.nodes[n].get("length_m", 0) or 0 for n in gap}

    Bonus = {}
    for g, h in [(u, v) for u, v in G.edges() if u in gs and v in gs]:
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
        rg = (slope_risk(Gg.nodes[g].get("slope_pct")) * slope_risk(Gg.nodes[h].get("slope_pct"))) ** 0.5
        Bonus[(g, h)] = max(0.0, rg * (bj - (bg + bh)))

    B = 0.05 * sum(length.values()) * COST_PER_M
    import pulp
    prob = pulp.LpProblem(f"h_{int(a*100)}", pulp.LpMaximize)
    x = {n: pulp.LpVariable(f"x_{n}", cat="Binary") for n in gap}
    y = {k: pulp.LpVariable(f"y_{k[0]}_{k[1]}", cat="Binary") for k in Bonus}
    prob += pulp.lpSum(P[n] * x[n] for n in gap) + pulp.lpSum(Bonus[k] * y[k] for k in Bonus)
    prob += pulp.lpSum(length[n] * COST_PER_M * x[n] for n in gap) <= B
    for (g, h) in Bonus:
        prob += y[(g, h)] <= x[g]; prob += y[(g, h)] <= x[h]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    sel = {n for n in gap if x[n].value() > 0.5}
    bsum = sum(Bonus.values())
    realized = sum(v for k, v in Bonus.items() if k[0] in sel and k[1] in sel)
    return dict(a=a, n_sel=len(sel), sel=sel, bonus_total=bsum, bonus_realized=realized,
                n_pairs_used=sum(1 for k in Bonus if k[0] in sel and k[1] in sel),
                obj=pulp.value(prob.objective))


def main():
    with open(f"{DATA}/road_graph.json") as f:
        G = nx.node_link_graph(json.load(f), edges="links")
    results = []
    t0 = time.time()
    for a in [0.0, 0.3, 0.5, 0.75, 1.0]:
        r = run(G, a)
        results.append(r)
        print(f"a={a:.2f} | 선정 {r['n_sel']:>3}개 | Bonus합={r['bonus_total']:9.2f} | "
              f"실현보너스={r['bonus_realized']:8.2f} | 인접쌍선정={r['n_pairs_used']:>3} | "
              f"obj={r['obj']:9.2f}  ({time.time()-t0:.1f}s)", flush=True)

    base = next(r for r in results if r["a"] == 0.3)["sel"]
    print("\n=== a=0.3(부분감소, 문헌 정합) 기준 선정집합과의 중복도 ===")
    for r in results:
        inter = len(base & r["sel"]); union = len(base | r["sel"])
        print(f"  a={r['a']:.2f}: 공통 {inter:>3}/{union:>3}  Jaccard={inter/union:.3f}")

    pd.DataFrame([{k: v for k, v in r.items() if k != "sel"} for r in results]).to_csv(
        "/tmp/diversion_model_compare.csv", index=False, encoding="utf-8-sig")
    print("\n저장: /tmp/diversion_model_compare.csv")


if __name__ == "__main__":
    main()
