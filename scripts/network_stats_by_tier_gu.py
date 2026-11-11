"""정밀/비정밀 계층 분리 + 자치구별 네트워크 통계량 산출.

방법:
  1) road_links.geojson(citywide, 180,460개 링크)에 각 링크 중점 좌표 기준으로
     seoul_gu.geojson과 공간조인해서 'gu' 속성 부여(벡터화 point-in-polygon).
  2) 1:1,000 정밀 커버리지가 실질적으로 존재하는 9개 자치구(성북/서대문/종로/중구/
     용산/성동/광진/동대문/중랑)만 대상으로, 링크 끝점 좌표를 grid quantization(5m
     셀)으로 묶어 '교차점'을 만들고, 같은 교차점을 공유하는 링크끼리 위상적으로
     연결된 것으로 보는 그래프를 구축(build_road_graph.py와 동일 원리, citywide
     규모에서도 돌아가도록 벡터화).
  3) source_1000(정밀 재계산 여부) True/False로 노드를 나눠, 정밀 노드끼리만 연결된
     유도부분그래프(induced subgraph), 비정밀 노드끼리만 연결된 유도부분그래프를
     각각 만들고 - 절대 섞지 않음 - 표준 네트워크 통계량을 자치구별로 계산.
"""
import json
import pickle
import time
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
import shapely
from shapely.geometry import shape

HERE_DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"

TARGET_GU = ["성북구", "서대문구", "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구"]
CELL_M = 5.0
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0


def main():
    t0 = time.time()
    with open(f"{HERE_DATA}/road_links.geojson") as f:
        rl = json.load(f)
    feats = rl["features"]
    n = len(feats)
    print(f"전체 링크: {n:,}", flush=True)

    lon_s = np.zeros(n); lat_s = np.zeros(n)
    lon_e = np.zeros(n); lat_e = np.zeros(n)
    mid_lon = np.zeros(n); mid_lat = np.zeros(n)
    source_1000 = np.zeros(n, dtype=bool)
    road_type = [None] * n
    slope_level = [None] * n
    for i, feat in enumerate(feats):
        c = feat["geometry"]["coordinates"]
        lon_s[i], lat_s[i] = c[0]
        lon_e[i], lat_e[i] = c[-1]
        mid = c[len(c) // 2]
        mid_lon[i], mid_lat[i] = mid
        p = feat["properties"]
        source_1000[i] = bool(p.get("source_1000"))
        road_type[i] = p.get("road_type")
        slope_level[i] = p.get("slope_level")

    # ---- gu 공간조인 ----
    with open(f"{HERE_DATA}/seoul_gu.geojson") as f:
        gu_gj = json.load(f)
    gu_polys = [shape(ft["geometry"]) for ft in gu_gj["features"]]
    gu_names = [ft["properties"]["gu"] for ft in gu_gj["features"]]
    pts = shapely.points(mid_lon, mid_lat)
    gu_arr = np.array([None] * n, dtype=object)
    for name, poly in zip(gu_names, gu_polys):
        mask = shapely.contains(poly, pts)
        gu_arr[mask] = name
    print(f"gu 공간조인 완료 ({time.time()-t0:.1f}s)", flush=True)

    # 정밀 계층 비교는 9개구로 한정하지만, 비정밀 계층 비교는 표본 확보를 위해
    # 전체 25개 자치구(gu 판정 성공분)를 다 포함해서 그래프를 만든다.
    idxs = np.where(gu_arr != None)[0]  # noqa: E711
    print(f"gu 판정된 전체 링크: {len(idxs):,}개", flush=True)

    # ---- grid quantization 기반 교차점 결합 ----
    t1 = time.time()
    cell_lat = CELL_M / _M_PER_DEG_LAT
    cell_lon = CELL_M / _M_PER_DEG_LON

    def qkey(lon, lat):
        return (np.round(lon / cell_lon).astype(np.int64), np.round(lat / cell_lat).astype(np.int64))

    ks_lon, ks_lat = qkey(lon_s[idxs], lat_s[idxs])
    ke_lon, ke_lat = qkey(lon_e[idxs], lat_e[idxs])

    junction_links = defaultdict(list)
    for local_i, gi in enumerate(idxs):
        junction_links[(ks_lon[local_i], ks_lat[local_i])].append(gi)
        junction_links[(ke_lon[local_i], ke_lat[local_i])].append(gi)
    print(f"교차점 그룹화 완료 ({time.time()-t1:.1f}s): 교차점 {len(junction_links):,}개", flush=True)

    sizes = sorted((len(set(v)) for v in junction_links.values()), reverse=True)
    print(f"교차점 크기 분포 상위 10개: {sizes[:10]}", flush=True)

    t1b = time.time()
    G = nx.Graph()
    for gi in idxs:
        G.add_node(
            int(gi),
            gu=gu_arr[gi],
            source_1000=bool(source_1000[gi]),
            road_type=road_type[gi],
            slope_level=slope_level[gi],
        )
    print(f"노드 추가 완료 ({time.time()-t1b:.1f}s)", flush=True)

    t1c = time.time()
    edge_set = set()
    MAX_JUNCTION = 40  # 이보다 큰 교차점(그리드 셀 병합 아티팩트 가능성)은 스타 형태로만 연결해 폭증 방지
    for jid, lids in junction_links.items():
        uniq = sorted(set(lids))
        if len(uniq) < 2:
            continue
        if len(uniq) > MAX_JUNCTION:
            hub = uniq[0]
            for other in uniq[1:]:
                a, b = (hub, other) if hub < other else (other, hub)
                edge_set.add((a, b))
            continue
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                edge_set.add((uniq[a], uniq[b]))
    print(f"엣지 집합 구축 완료 ({time.time()-t1c:.1f}s): {len(edge_set):,}개 (dedup 전 중복 포함 가능)", flush=True)

    t1d = time.time()
    G.add_edges_from(edge_set)
    print(f"위상 그래프 구축 완료 ({time.time()-t1:.1f}s, 엣지추가 {time.time()-t1d:.1f}s): N={G.number_of_nodes():,} M={G.number_of_edges():,}", flush=True)

    with open("/tmp/citywide_topo_graph.pkl", "wb") as f:
        pickle.dump({"nodes": list(G.nodes(data=True)), "edges": list(G.edges())}, f)
    print(f"체크포인트 저장: /tmp/citywide_topo_graph.pkl  (총 {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
