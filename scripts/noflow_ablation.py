"""추가검증(보고서 7번): 유동인구를 반영하지 않으면 배치 결과가 어떻게 달라지는가.

두 시나리오를 동일 예산·동일 후보군·동일 알고리즘으로 돌려 비교한다.
  (A) WITH-FLOW  : 현행 — P_i = risk_i x (1-a) x flow_i, 전파모형 전도도도 유동인구 가중
  (B) NO-FLOW    : 유동인구를 모두 1로 균일화 — P_i = risk_i (경사도만), 전파모형도 무가중

출력: 선정 결과 중복도(Jaccard), 선정 집합의 경사/유동 프로파일, 목적함수, /tmp/noflow_ablation.csv
"""
import json
import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

EPS = 1e-6
A_CENTRAL = 0.25
COST_PER_M = 1_500_000_000 / 1700
DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"


def slope_risk(slope_pct):
    if slope_pct is None or (isinstance(slope_pct, float) and np.isnan(slope_pct)):
        return 0.20
    if slope_pct >= 10: return 1.00
    if slope_pct >= 6: return 0.70
    if slope_pct >= 3: return 0.40
    return 0.10


def flow_of(G, n, use_flow):
    if not use_flow:
        return 1.0
    return G.nodes[n].get("flow_sum", 0) or 0


def build_weighted_laplacian(G, use_flow):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    L = lil_matrix((n, n))
    for u, v in G.edges():
        c = (flow_of(G, u, use_flow) + flow_of(G, v, use_flow)) / 2.0 + EPS
        i, j = idx[u], idx[v]
        L[i, i] += c; L[j, j] += c; L[i, j] -= c; L[j, i] -= c
    return csr_matrix(L), nodes, idx


def solve_injection(L, nodes, idx, injections, ref_node):
    n = len(nodes)
    b = np.zeros(n)
    for node, amt in injections.items():
        b[idx[node]] = amt
    ref_i = idx[ref_node]
    keep = [i for i in range(n) if i != ref_i]
    Lr = L[keep, :][:, keep]
    vr = spsolve(Lr.tocsc(), b[keep])
    v = np.zeros(n)
    for k, i in enumerate(keep):
        v[i] = vr[k]
    return {nodes[i]: v[i] for i in range(n)}


def edge_current(G, v, u, w, use_flow):
    c = (flow_of(G, u, use_flow) + flow_of(G, w, use_flow)) / 2.0 + EPS
    return c * (v[u] - v[w])


def displacement_injection(G, node, a, use_flow):
    flow_i = flow_of(G, node, use_flow)
    total_out = (1 - a) * flow_i
    others = [nd for nd in G.nodes() if nd != node]
    weights = np.array([flow_of(G, nd, use_flow) for nd in others], dtype=float) + EPS
    weights = weights / weights.sum()
    inj = {node: total_out}
    for nd, w in zip(others, weights):
        inj[nd] = inj.get(nd, 0) - total_out * w
    return inj, total_out


def run_scenario(G, use_flow, label):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    Gg = G.subgraph(comps[0]).copy()
    L, nodes, idx = build_weighted_laplacian(Gg, use_flow)
    giant_set = set(Gg.nodes())

    gap_nodes = [n for n in G.nodes() if (G.nodes[n].get("slope_pct") or -1) >= 6.0
                 and not G.nodes[n].get("heating_installed_precise")
                 and G.nodes[n].get("road_type") == "골목길"]
    gap_set = set(gap_nodes)

    P, length = {}, {}
    for n in gap_nodes:
        d = G.nodes[n]
        risk_i = slope_risk(d.get("slope_pct"))
        impact_i = (1 - A_CENTRAL) * flow_of(G, n, use_flow)
        P[n] = risk_i * impact_i
        length[n] = d.get("length_m", 0) or 0

    gap_gap_edges = [(u, v) for u, v in G.edges() if u in gap_set and v in gap_set]
    Bonus = {}
    for g, h in gap_gap_edges:
        if g not in giant_set or h not in giant_set:
            continue
        inj_g, out_g = displacement_injection(Gg, g, A_CENTRAL, use_flow)
        v_g = solve_injection(L, nodes, idx, inj_g, nodes[0])
        inj_h, out_h = displacement_injection(Gg, h, A_CENTRAL, use_flow)
        v_h = solve_injection(L, nodes, idx, inj_h, nodes[0])
        common_nb = (set(Gg.neighbors(g)) & set(Gg.neighbors(h))) - {g, h}
        if not common_nb:
            continue
        burden_g = sum(abs(edge_current(Gg, v_g, g, k, use_flow)) for k in common_nb if k in Gg[g])
        burden_h = sum(abs(edge_current(Gg, v_h, h, k, use_flow)) for k in common_nb if k in Gg[h])
        total_gh = out_g + out_h
        others = [nd for nd in Gg.nodes() if nd not in (g, h)]
        weights = np.array([flow_of(Gg, nd, use_flow) for nd in others]) + EPS
        weights = weights / weights.sum()
        inj_gh = {g: out_g, h: out_h}
        for nd, w in zip(others, weights):
            inj_gh[nd] = inj_gh.get(nd, 0) - total_gh * w
        v_gh = solve_injection(L, nodes, idx, inj_gh, nodes[0])
        burden_joint = sum(abs(edge_current(Gg, v_gh, g, k, use_flow)) for k in common_nb if k in Gg[g]) + \
                       sum(abs(edge_current(Gg, v_gh, h, k, use_flow)) for k in common_nb if k in Gg[h])
        synergy = burden_joint - (burden_g + burden_h)
        risk_gh = (slope_risk(Gg.nodes[g].get("slope_pct")) * slope_risk(Gg.nodes[h].get("slope_pct"))) ** 0.5
        Bonus[(g, h)] = max(0.0, risk_gh * synergy)

    total_len = sum(length.values())
    B = 0.05 * total_len * COST_PER_M

    # MILP
    import pulp
    prob = pulp.LpProblem(f"heating_{label}", pulp.LpMaximize)
    x = {n: pulp.LpVariable(f"x_{n}", cat="Binary") for n in gap_nodes}
    y = {(g, h): pulp.LpVariable(f"y_{g}_{h}", cat="Binary") for (g, h) in Bonus}
    prob += pulp.lpSum(P[n] * x[n] for n in gap_nodes) + pulp.lpSum(Bonus[k] * y[k] for k in Bonus)
    prob += pulp.lpSum(length[n] * COST_PER_M * x[n] for n in gap_nodes) <= B
    for (g, h) in Bonus:
        prob += y[(g, h)] <= x[g]
        prob += y[(g, h)] <= x[h]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    milp_selected = [n for n in gap_nodes if x[n].value() > 0.5]

    print(f"\n[{label}] 후보 {len(gap_nodes)}개, 예산 {B/1e8:.2f}억, Bonus쌍 {len(Bonus)}")
    print(f"  MILP 선정 {len(milp_selected)}개, 목적함수={pulp.value(prob.objective):.2f}")

    sel_slopes = [G.nodes[n].get("slope_pct") for n in milp_selected]
    sel_flows = [G.nodes[n].get("flow_sum", 0) or 0 for n in milp_selected]
    all_flows = [G.nodes[n].get("flow_sum", 0) or 0 for n in gap_nodes]
    all_slopes = [G.nodes[n].get("slope_pct") for n in gap_nodes]
    print(f"  선정집합 평균경사={np.mean(sel_slopes):.2f}% (후보전체 {np.mean(all_slopes):.2f}%)")
    print(f"  선정집합 평균유동={np.mean(sel_flows):.1f} (후보전체 {np.mean(all_flows):.1f})")
    print(f"  선정집합 총연장={sum(length[n] for n in milp_selected):.0f}m")
    return dict(label=label, selected=set(milp_selected), gap_nodes=gap_nodes, P=P, length=length,
                n_sel=len(milp_selected), obj=pulp.value(prob.objective),
                mean_slope=float(np.mean(sel_slopes)), mean_flow=float(np.mean(sel_flows)),
                total_len=float(sum(length[n] for n in milp_selected)))


def main():
    with open(f"{DATA}/road_graph.json") as f:
        G = nx.node_link_graph(json.load(f), edges="links")

    a = run_scenario(G, True, "WITH-FLOW(현행)")
    b = run_scenario(G, False, "NO-FLOW(경사도만)")

    inter = a["selected"] & b["selected"]
    union = a["selected"] | b["selected"]
    print(f"\n=== 두 시나리오 비교 ===")
    print(f"  공통 선정: {len(inter)}개, 합집합 {len(union)}개, Jaccard={len(inter)/len(union):.3f}")
    print(f"  현행에만: {len(a['selected']-b['selected'])}개 / 경사도만에만: {len(b['selected']-a['selected'])}개")
    print(f"  선정 수: 현행 {a['n_sel']} vs 경사도만 {b['n_sel']}")
    print(f"  평균경사: 현행 {a['mean_slope']:.2f}% vs 경사도만 {b['mean_slope']:.2f}%")
    print(f"  평균유동: 현행 {a['mean_flow']:.1f} vs 경사도만 {b['mean_flow']:.1f}")

    # 현행 목적함수(=실제 편익 기준)로 두 해를 평가 -> 유동인구 무시의 '손실' 정량화
    P_true = a["P"]
    val_a = sum(P_true[n] for n in a["selected"])
    val_b = sum(P_true[n] for n in b["selected"] if n in P_true)
    print(f"\n  [현행 P_i 기준 평가] 현행해={val_a:.1f} vs 경사도만해={val_b:.1f} "
          f"({val_b/val_a*100:.1f}% 수준, 손실 {100-val_b/val_a*100:.1f}%)")

    rows = []
    for n in union:
        rows.append(dict(node=n, in_withflow=n in a["selected"], in_noflow=n in b["selected"],
                         slope_pct=G.nodes[n].get("slope_pct"),
                         flow_sum=G.nodes[n].get("flow_sum", 0) or 0,
                         length_m=G.nodes[n].get("length_m", 0) or 0,
                         dong=G.nodes[n].get("dong")))
    pd.DataFrame(rows).to_csv("/tmp/noflow_ablation.csv", index=False, encoding="utf-8-sig")
    print("\n저장: /tmp/noflow_ablation.csv")


if __name__ == "__main__":
    main()
