"""재결빙 경계효과를 고려한 열선 배치 MILP — 단일 출처.

    max  Σ_i risk_i · flow_i · x_i  −  δ · Σ_(i,j)∈D  w_ij · b_ij
    s.t. Σ_i length_i · COST_PER_M · x_i ≤ B          (예산)
         b_ij ≥ x_i − x_j            ∀(i,j) ∈ D       (경계 발생 선형화)
         x_i ∈ {0,1},  b_ij ∈ [0,1]

D = {(i,j) : i~j 인접 AND z_i − z_j ≥ h_min}  — i가 상류

경계가중치 w_ij는 내생적이다. 배치가 바뀌면 통행이 바뀌고, 통행이 바뀌면
경계를 넘는 사람 수도 바뀐다. current-flow(가중 라플라시안) 엣지전류로
w_ij를 만들고 MSA(λ=1/(k+1))로 고정점을 찾는다.

이 모듈은 배치 스크립트(scripts/run_delta_experiment.py)와 Streamlit 앱이
**함께 호출**한다. 발표 숫자와 화면이 어긋나지 않게 하기 위해서다.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pulp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized, splu

from utils.risk import obs_risk

# ── 상수 ─────────────────────────────────────────────────────────────
#: 15억 원 / 1,700 m (3 m 폭 1개 차로 기준) = 882,353 원/m
COST_PER_M = 1_500_000_000 / 1700
EPS = 1e-6
#: 결빙 구간의 잔존 통행 비율(= 증발률 0.25의 보수). 0.05~0.75에서 해 불변
A_RES = 0.75
#: 경계 판정 고도차(m)
H_MIN = 2.0
#: current-flow 전파의 source 수. None이면 연결성분 전체 노드를 source로 쓴다.
#: 전체(775개)여도 1초 미만이라 근사할 이유가 없다. 상위 200개만 쓰면
#: 전체 대비 ρ=0.990 / 상위 20% 일치 92.5%로 미세하게 어긋난다.
N_SRC = None

_DATA = Path(__file__).resolve().parent.parent / "data"


# ── 그래프 ───────────────────────────────────────────────────────────
def load_graph(data_dir: Path | str | None = None) -> nx.Graph:
    """4개동 도로망 그래프 + 링크 중점 고도(z)."""
    d = Path(data_dir) if data_dir else _DATA
    G = nx.node_link_graph(json.load(open(d / "road_graph.json")), edges="links")
    z = pd.read_csv(d / "road_graph_nodes_elev.csv")
    z["id"] = z["id"].astype(str)
    zmap = dict(zip(z.id, z.z))
    for n in G.nodes():
        G.nodes[n]["z"] = zmap.get(str(n))
    return G


def candidates(G: nx.Graph, slope_min: float | None = None) -> list:
    """후보군 = 골목길 ∩ 열선 미설치. 기본은 경사 하한 없음."""
    out = []
    for n in G.nodes():
        if G.nodes[n].get("heating_installed_precise"):
            continue
        if G.nodes[n].get("road_type") != "골목길":
            continue
        if slope_min is not None and (G.nodes[n].get("slope_pct") or -1) < slope_min:
            continue
        out.append(n)
    return out


def installed_links(G: nx.Graph, golmok_only: bool = True) -> list:
    """현행 열선 설치 링크. 현행 예산 산정에 쓴다."""
    out = []
    for n in G.nodes():
        if not G.nodes[n].get("heating_installed_precise"):
            continue
        if golmok_only and G.nodes[n].get("road_type") != "골목길":
            continue
        out.append(n)
    return out


def current_budget(G: nx.Graph, golmok_only: bool = True) -> tuple[float, float]:
    """(현행 투입액 원, 현행 투입 연장 m). 골목길분 기준 2,101 m = 18.53억."""
    links = installed_links(G, golmok_only)
    length = sum(G.nodes[n].get("length_m") or 0.0 for n in links)
    return length * COST_PER_M, length


def model_budget(G: nx.Graph, ratio: float = 0.05, slope_min: float = 6.0) -> float:
    """보고서 기준 예산 — 구 후보군(경사 6%↑) 총연장의 5%. 8.81억."""
    c = candidates(G, slope_min)
    return ratio * sum(G.nodes[n].get("length_m") or 0.0 for n in c) * COST_PER_M


def direction_set(G: nx.Graph, cand: list, h_min: float = H_MIN) -> list[tuple]:
    """하류 방향 엣지집합 D. i가 상류이고 낙차가 h_min 이상인 인접 쌍."""
    D, cs = [], set(cand)
    for u, v in G.edges():
        zu, zv = G.nodes[u].get("z"), G.nodes[v].get("z")
        if zu is None or zv is None:
            continue
        d = zu - zv
        if abs(d) < h_min:
            continue
        i, j = (u, v) if d > 0 else (v, u)
        if i in cs:
            D.append((i, j))
    return D


def risk_flow(G: nx.Graph) -> tuple[dict, dict, dict]:
    """(risk, flow, slope) 사전. risk는 Observed MAA EDF."""
    risk = {n: obs_risk(G.nodes[n].get("slope_pct")) for n in G.nodes()}
    flow = {n: (G.nodes[n].get("flow_sum") or 0.0) for n in G.nodes()}
    slope = {n: (G.nodes[n].get("slope_pct") or 0.0) for n in G.nodes()}
    return risk, flow, slope


# ── MILP ─────────────────────────────────────────────────────────────
def solve(G, cand, D, risk, flow, B, delta, pen=None,
          time_limit: float | None = None) -> tuple[set, int]:
    """1회 풀이. pen을 주지 않으면 risk_j×flow_j를 쓰지만, 채택 모형은
    항상 current-flow 기반 w를 넘긴다(solve_msa 참조).

    time_limit(초)을 주면 CBC가 그 시간 안의 최선해를 반환한다. 기본값 None은
    기존 동작(최적해까지 풀이)과 같다. 큰 지역(후보 수천 개) 대응용."""
    ln = {n: (G.nodes[n].get("length_m") or 0.0) for n in cand}
    cs = set(cand)
    if pen is None:
        pen = {k: risk[k[1]] * flow[k[1]] for k in D}
    p = pulp.LpProblem("heating", pulp.LpMaximize)
    x = {n: pulp.LpVariable(f"x{n}", cat="Binary") for n in cand}
    b = {k: pulp.LpVariable("b%s_%s" % k, lowBound=0, upBound=1) for k in D}
    p += (pulp.lpSum(risk[n] * flow[n] * x[n] for n in cand)
          - delta * pulp.lpSum(pen[k] * b[k] for k in D))
    p += pulp.lpSum(ln[n] * COST_PER_M * x[n] for n in cand) <= B
    for (i, j) in D:
        p += b[(i, j)] >= (x[i] - x[j] if j in cs else x[i])
    if time_limit:
        p.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    else:
        p.solve(pulp.PULP_CBC_CMD(msg=0))
    S = {n for n in cand if x[n].value() and x[n].value() > 0.5}
    ob = sum(1 for (i, j) in D if i in S and j not in S)
    return S, ob


# ── current-flow 가중치 ──────────────────────────────────────────────
_USE_DEFAULT = object()


def edge_currents(G, nodes, idx, edges, fl, ice=frozenset(),
                  n_src=_USE_DEFAULT) -> np.ndarray:
    """가중 라플라시안 기반 엣지 전류. 결빙(ice) 구간은 컨덕턴스를 A_RES배.

    n_src를 생략하면 모듈 상수 N_SRC를 쓴다(기존 동작). 노드가 수천 개인
    지역에서는 (N × source) 행렬이 커지므로 호출부가 상한을 넘길 수 있다."""
    if n_src is _USE_DEFAULT:
        n_src = N_SRC
    N = len(nodes)
    EC = []
    for u, v in edges:
        c = ((G.nodes[u].get("flow_sum") or 0) + (G.nodes[v].get("flow_sum") or 0)) / 2 + EPS
        if u in ice or v in ice:
            c *= A_RES
        EC.append(c)
    EC = np.array(EC)
    L = lil_matrix((N, N))
    for k, (u, v) in enumerate(edges):
        i, j = idx[u], idx[v]
        L[i, i] += EC[k]; L[j, j] += EC[k]
        L[i, j] -= EC[k]; L[j, i] -= EC[k]
    L = csr_matrix(L)
    keep = [i for i in range(N) if i != 0]
    lu = splu(L[keep, :][:, keep].tocsc())
    w = fl + EPS
    w = w / w.sum()
    EU = np.array([idx[u] for u, v in edges])
    EV = np.array([idx[v] for u, v in edges])
    order = np.argsort(-fl)
    order = order[fl[order] > 0]          # 통행 0인 source는 우변이 0 → 기여 없음
    srcs = order if n_src is None else order[:n_src]

    # 모든 source 의 우변을 한 번에 풀어 back-substitution 을 벡터화한다.
    # (source 마다 따로 풀면 1,956회 반복이 되어 수십 배 느리다)
    K = len(srcs)
    BB = -np.outer(w, fl[srcs])          # (N, K) 각 열이 한 source 의 우변
    BB[srcs, np.arange(K)] += fl[srcs]   # s 에 fl[s] 주입 → 열 합 0
    VV = np.zeros((N, K))
    VV[keep, :] = lu.solve(BB[keep, :])
    acc = np.abs(EC[:, None] * (VV[EU, :] - VV[EV, :])).sum(axis=1)
    return acc


def _edge_currents_loop(G, nodes, idx, edges, fl, ice=frozenset()):
    """참조용 루프 구현 — 벡터화본 검증에만 쓴다."""
    N = len(nodes)
    EC = []
    for u, v in edges:
        c = ((G.nodes[u].get("flow_sum") or 0) + (G.nodes[v].get("flow_sum") or 0)) / 2 + EPS
        if u in ice or v in ice:
            c *= A_RES
        EC.append(c)
    EC = np.array(EC)
    L = lil_matrix((N, N))
    for k, (u, v) in enumerate(edges):
        i, j = idx[u], idx[v]
        L[i, i] += EC[k]; L[j, j] += EC[k]
        L[i, j] -= EC[k]; L[j, i] -= EC[k]
    L = csr_matrix(L)
    keep = [i for i in range(N) if i != 0]
    sol = factorized(L[keep, :][:, keep].tocsc())
    w = fl + EPS; w = w / w.sum()
    EU = np.array([idx[u] for u, v in edges])
    EV = np.array([idx[v] for u, v in edges])
    acc = np.zeros(len(edges))
    order = np.argsort(-fl)
    srcs = order if N_SRC is None else order[:N_SRC]
    for s in srcs:
        # s에 fl[s]를 주입하고 모든 노드에서 통행량 비례로 회수한다.
        # 주입 벡터의 합은 0이어야 한다(전류 보존). 이전 구현은 bb[s]에
        # fl[s]*w[s]를 한 번 더 더해 합이 fl[s]*w[s]만큼 남았고, 그 잉여가
        # 접지 노드로 흘러 전류를 미세하게 왜곡했다.
        bb = -fl[s] * w.copy()
        bb[s] += fl[s]
        vv = np.zeros(N)
        vv[keep] = sol(bb[keep])
        acc += np.abs(EC * (vv[EU] - vv[EV]))
    return acc


#: 전파모형 전용 버퍼 홉 수. 0이면 버퍼 없이 4개동만(구버전 동작).
#: 2홉에서 방향엣지 포함률 97%(492/508)이고, Dg 기준 경계수와 D 전체 기준
#: 경계수가 11로 일치한다(패널티 밖에서 새 경계가 생기지 않음). 3홉은 2홉과
#: 결과가 같아 수렴했다. 검증: 04_문서/데이터검증/03_전파모형_검증기록.txt
FLOW_BUFFER_HOPS = 2


def build_flow_graph(G, hops: int = FLOW_BUFFER_HOPS, data_dir=None):
    """전파모형 전용 확장 그래프 — 4개동 + k홉 버퍼.

    왜 필요한가
    -----------
    4개동 링크 1,190개는 **서울 전역 그래프에서는 하나로 연결**되어 있다
    (180,350개 거대 성분 안에 100% 포함). 그런데 행정동 경계로 잘라내면
    "4개동 → 동 밖 한 구간 → 4개동"으로 이어지던 경로가 끊겨 74개 성분으로
    쪼개진다. 최대 성분만 쓰면 방향엣지 508개 중 343개(67.5%)에만
    가중치가 간다.

    버퍼는 **전류가 지나갈 길만 복원**한다. 후보군·예산·risk·flow·경계 집계는
    전부 원래 4개동 그래프 G 기준으로 유지되며, 버퍼 링크는 선정 대상이
    아니고 편익에도 기여하지 않는다.

    버퍼 링크의 통행량
    ------------------
    유동인구 실측은 4개동 한정이므로 버퍼 링크의 flow는 0이다.
    엣지 컨덕턴스가 ``(flow_u + flow_v)/2`` 이므로, 4개동 링크에 붙은
    버퍼 링크는 그 절반의 컨덕턴스를 갖는다. 즉 "골목이 이어진다"는
    사실만 반영하고 통행량을 지어내지 않는다.

    Returns
    -------
    (Gg, nodes, idx, edges, ep, Dg) — 기존 _largest_component 와 같은 형태.
    Gg 의 노드 중 코어(4개동)는 G 와 같은 id 를, 버퍼는 ``("_buf", i)`` 를 쓴다.
    """
    if not hops:
        return _largest_component(G, D=[])

    d = Path(data_dir) if data_dir else _DATA
    lp, ep_ = d / "seoul_links.parquet", d / "seoul_edges.parquet"
    if not (lp.exists() and ep_.exists()):
        raise FileNotFoundError(
            "버퍼 확장에는 seoul_links/seoul_edges parquet 이 필요하다. "
            "scripts/build_seoul_dataset.py 를 먼저 실행할 것.")

    L = pd.read_parquet(lp, columns=["road_lid"])
    E = pd.read_parquet(ep_)
    rl2ix = {str(v): i for i, v in enumerate(L.road_lid)}

    core_ix, ix2core = {}, {}
    for n in G.nodes():
        key = str(G.nodes[n].get("link_id") or n)
        i = rl2ix.get(key)
        if i is not None:
            core_ix[n] = i
            ix2core[i] = n

    adj = collections.defaultdict(list)
    for u, v in E.itertuples(index=False, name=None):
        adj[u].append(v); adj[v].append(u)

    frontier = set(ix2core)
    keep = set(frontier)
    for _ in range(hops):
        nxt = {w for u in frontier for w in adj[u]} - keep
        keep |= nxt
        frontier = nxt

    Gg = nx.Graph()
    for i in keep:
        n = ix2core.get(i)
        if n is not None:
            Gg.add_node(n, flow_sum=G.nodes[n].get("flow_sum") or 0.0)
        else:
            Gg.add_node(("_buf", int(i)), flow_sum=0.0)
    lab = lambda i: ix2core.get(i, ("_buf", int(i)))
    for u, v in E.itertuples(index=False, name=None):
        if u in keep and v in keep:
            Gg.add_edge(lab(u), lab(v))

    comps = sorted(nx.connected_components(Gg), key=len, reverse=True)
    Gg = Gg.subgraph(comps[0]).copy()
    nodes = list(Gg.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = list(Gg.edges())
    ep = {}
    for k, (u, v) in enumerate(edges):
        ep[(u, v)] = k; ep[(v, u)] = k
    return Gg, nodes, idx, edges, ep


def _largest_component(G, D):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    Gg = G.subgraph(comps[0]).copy()
    nodes = list(Gg.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = list(Gg.edges())
    ep = {}
    for k, (u, v) in enumerate(edges):
        ep[(u, v)] = k
        ep[(v, u)] = k
    Dg = [(i, j) for (i, j) in D if (i, j) in ep]
    return Gg, nodes, idx, edges, ep, Dg


def solve_msa(G, cand, D, risk, flow, B, delta, iters: int = 8) -> dict:
    """MSA 고정점까지 반복해 채택 해를 반환.

    반환 키: selected, boundary, benefit, boundary_exposure, iters,
             converged, D_used
    """
    if FLOW_BUFFER_HOPS:
        # 전파 전용 확장 그래프. 후보·예산·risk·flow·경계 집계는 G 기준 유지.
        Gg, nodes, idx, edges, ep = build_flow_graph(G, FLOW_BUFFER_HOPS)
        Dg = [(i, j) for (i, j) in D if (i, j) in ep]
    else:
        Gg, nodes, idx, edges, ep, Dg = _largest_component(G, D)
    fl = np.array([(Gg.nodes[n].get("flow_sum") or 0.0) for n in nodes])
    ref = float(np.mean([risk[j] * flow[j] for (i, j) in Dg]))

    def make_W(ice):
        acc = edge_currents(Gg, nodes, idx, edges, fl, ice)
        raw = {k: risk[k[1]] * acc[ep[k]] for k in Dg}
        s = float(np.mean(list(raw.values())))
        return {k: v * ref / s for k, v in raw.items()}

    w = make_W(frozenset())
    prev, converged, it_used = None, False, iters
    S, ob = set(), 0
    for it in range(iters):
        S, ob = solve(G, cand, Dg, risk, flow, B, delta, pen=w)
        if prev is not None and S == prev:
            converged, it_used = True, it
            break
        prev = S
        ice = frozenset(n for n in cand if n not in S and n in Gg)
        wn = make_W(ice)
        lam = 1.0 / (it + 2)
        w = {k: (1 - lam) * w[k] + lam * wn[k] for k in Dg}

    exposure = sum(risk[j] * flow[j] for (i, j) in Dg if i in S and j not in S)
    return {
        "w": w,
        "selected": S,
        "boundary": ob,
        "benefit": sum(risk[n] * flow[n] for n in S),
        "boundary_exposure": exposure,
        "iters": it_used,
        "converged": converged,
        "D_used": Dg,
    }


def summarize(G, res: dict, risk: dict, flow: dict, slope: dict,
              B: float, base_benefit: float | None = None,
              base_boundary: int | None = None, D_all=None) -> dict:
    """δ 한 점의 결과를 표 한 행으로."""
    S = res["selected"]
    ln = sum(G.nodes[n].get("length_m") or 0.0 for n in S)
    ben = res["benefit"]
    row = {
        "선정수": len(S),
        "선정연장m": round(ln),
        "집행액억": round(ln * COST_PER_M / 1e8, 2),
        "기대방지": round(ben, 1),
        "잔여경계": res["boundary"],
        "경계노출": round(res["boundary_exposure"], 1),
        "평균경사": round(float(np.mean([slope[n] for n in S])), 2) if S else None,
        "평균통행": round(float(np.mean([flow[n] for n in S])), 1) if S else None,
        "건당만원": round(B / ben / 1e4) if ben else None,
        "MSA수렴반복": res["iters"],
    }
    if D_all is not None:
        # 패널티 대상(Dg) 밖 경계까지 포함한 집계 — 보고용 보수 수치
        row["잔여경계_D전체"] = sum(
            1 for (i, j) in D_all if i in S and j not in S)
    if base_benefit:
        row["편익손실%"] = round(100 * (1 - ben / base_benefit), 2)
    if base_boundary:
        row["경계감소%"] = round(100 * (1 - res["boundary"] / base_boundary), 1)
    return row
