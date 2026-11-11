"""/tmp/citywide_topo_graph.pkl 체크포인트를 읽어 (자치구 x 정밀/비정밀 계층)
셀별로 표준 네트워크 통계량을 계산. 정밀 계층끼리만, 비정밀 계층끼리만 비교.
"""
import pickle
import time

import networkx as nx
import pandas as pd

MIN_N = 100  # 이보다 노드가 적은 셀은 표본 부족으로 제외(참고용으로만 표시)


def stats_for_subgraph(G, node_ids):
    H = G.subgraph(node_ids)
    N = H.number_of_nodes()
    M = H.number_of_edges()
    if N == 0:
        return None
    avg_deg = 2 * M / N if N else 0
    density = nx.density(H)
    comps = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    giant_frac = comps[0] / N if comps else 0
    n_comps = len(comps)
    isolated = sum(1 for c in comps if c == 1)
    try:
        avg_clust = nx.average_clustering(H)
    except Exception:
        avg_clust = None
    try:
        transitivity = nx.transitivity(H)
    except Exception:
        transitivity = None
    core = nx.core_number(H)
    avg_core = sum(core.values()) / N if N else 0
    max_core = max(core.values()) if core else 0
    try:
        assort = nx.degree_assortativity_coefficient(H)
    except Exception:
        assort = None
    return dict(
        N=N, M=M, avg_degree=round(avg_deg, 3), density=round(density, 5),
        n_components=n_comps, isolated_nodes=isolated,
        giant_component_frac=round(giant_frac, 3),
        avg_clustering=round(avg_clust, 4) if avg_clust is not None else None,
        transitivity=round(transitivity, 4) if transitivity is not None else None,
        avg_kcore=round(avg_core, 3), max_kcore=max_core,
        degree_assortativity=round(assort, 4) if assort is not None else None,
    )


def main():
    t0 = time.time()
    with open('/tmp/citywide_topo_graph.pkl', 'rb') as f:
        d = pickle.load(f)
    G = nx.Graph()
    G.add_nodes_from(d['nodes'])
    G.add_edges_from(d['edges'])
    print(f"그래프 로드: N={G.number_of_nodes():,} M={G.number_of_edges():,} ({time.time()-t0:.1f}s)", flush=True)

    by_gu_tier = {}
    for n, attrs in G.nodes(data=True):
        key = (attrs['gu'], attrs['source_1000'])
        by_gu_tier.setdefault(key, []).append(n)

    rows = []
    for (gu, tier), node_ids in sorted(by_gu_tier.items(), key=lambda x: (x[0][0], not x[0][1])):
        s = stats_for_subgraph(G, node_ids)
        if s is None:
            continue
        s['gu'] = gu
        s['tier'] = '정밀(1:1,000)' if tier else '비정밀(기존)'
        rows.append(s)
        flag = '' if s['N'] >= MIN_N else '  [표본<100, 참고용]'
        print(f"{gu:>6} / {s['tier']:>12}: N={s['N']:>6,} <k>={s['avg_degree']:.2f} "
              f"밀도={s['density']:.5f} 거대성분비율={s['giant_component_frac']:.3f} "
              f"군집계수={s['avg_clustering']} k-core평균={s['avg_kcore']:.2f} "
              f"동류성={s['degree_assortativity']}{flag}", flush=True)

    df = pd.DataFrame(rows)
    cols = ['gu', 'tier', 'N', 'M', 'avg_degree', 'density', 'n_components',
            'isolated_nodes', 'giant_component_frac', 'avg_clustering',
            'transitivity', 'avg_kcore', 'max_kcore', 'degree_assortativity']
    df = df[cols]
    df.to_csv('/tmp/network_stats_by_tier_gu.csv', index=False, encoding='utf-8-sig')
    print(f"\n저장: /tmp/network_stats_by_tier_gu.csv ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
