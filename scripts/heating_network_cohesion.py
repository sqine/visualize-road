"""서대문구 vs 성북구: 열선설치(도로명 매칭 근사)가 도로망에서 얼마나 응집돼
있는지(E-I index, conductance, Louvain modularity+enrichment) 비교.
- classify_road_type.py와 동일한 방식(OSM 25m 리샘플 그리드 + 30m 반경 최근접
  매칭)으로 road_links.geojson 각 링크에 OSM road_name을 부여.
- heating_2026.csv에서 도로명 추출(로/길/가 접미사 기준) 후, 매칭된 road_name과
  같은 링크를 heating≈True 로 플래그.
- 5m 격자스냅 위상그래프(N=gu 전체) 구축 후 heating≈True 그룹에 대해
  E-I index / conductance / Louvain modularity+enrichment 계산.
"""
import json
import math
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import shapely
from shapely.geometry import shape

HERE_DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 88_322.0
CELL_M = 5.0

GU_CONFIG = {
    "서대문구": "seodaemun_dong.geojson",
    "성북구": "seongbuk_dong.geojson",
}


def _euclid_m(lon1, lat1, lon2, lat2):
    return ((lat1 - lat2) ** 2 * _M_PER_DEG_LAT ** 2 + (lon1 - lon2) ** 2 * _M_PER_DEG_LON ** 2) ** 0.5


def extract_names(loc):
    parts = re.split(r"[~∼]", str(loc))
    names = []
    for p in parts:
        p = p.strip()
        m = re.match(r"^([가-힣0-9]+(?:로|길|가))(?=\s|$)", p)
        if m:
            names.append(m.group(1))
    return names


class NameGridIndex:
    def __init__(self, cell_m=30.0):
        self.cell_lat = cell_m / _M_PER_DEG_LAT
        self.cell_lon = cell_m / _M_PER_DEG_LON
        self.bins = {}
        self.lon = []; self.lat = []; self.name = []

    def _key(self, lon, lat):
        return (int(lon / self.cell_lon), int(lat / self.cell_lat))

    def add(self, lon, lat, name):
        idx = len(self.lon)
        self.lon.append(lon); self.lat.append(lat); self.name.append(name)
        self.bins.setdefault(self._key(lon, lat), []).append(idx)

    def nearest(self, lon, lat, max_m=30.0):
        cx, cy = self._key(lon, lat)
        cl = int(math.ceil(max_m / _M_PER_DEG_LON / self.cell_lon))
        ca = int(math.ceil(max_m / _M_PER_DEG_LAT / self.cell_lat))
        best_d, best_name = None, None
        for dx in range(-cl, cl + 1):
            for dy in range(-ca, ca + 1):
                for i in self.bins.get((cx + dx, cy + dy), []):
                    d = _euclid_m(lon, lat, self.lon[i], self.lat[i])
                    if best_d is None or d < best_d:
                        best_d, best_name = d, self.name[i]
        if best_d is not None and best_d <= max_m:
            return best_name
        return None


def resample_line(coords, interval_m=25.0):
    if not coords:
        return []
    pts = [(coords[0][0], coords[0][1])]
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]; lon2, lat2 = coords[i]
        seg_len = _euclid_m(lon1, lat1, lon2, lat2)
        if seg_len <= 0:
            continue
        d = 0.0
        while d + interval_m <= seg_len:
            d += interval_m
            t = d / seg_len
            pts.append((lon1 + (lon2 - lon1) * t, lat1 + (lat2 - lat1) * t))
    pts.append((coords[-1][0], coords[-1][1]))
    return pts


def ei_index(G, in_set):
    internal, external = 0, 0
    for u, v in G.edges():
        u_in, v_in = u in in_set, v in in_set
        if u_in and v_in:
            internal += 1
        elif u_in or v_in:
            external += 1
    total = internal + external
    return (external - internal) / total if total else None


def main():
    t0 = time.time()
    with open(f"{HERE_DATA}/seoul_roads.geojson") as f:
        osm_gj = json.load(f)
    name_idx = NameGridIndex(cell_m=30.0)
    for feat in osm_gj["features"]:
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        name = feat.get("properties", {}).get("road_name")
        if not name:
            continue
        for lon, lat in resample_line(geom["coordinates"], interval_m=25.0):
            name_idx.add(lon, lat, name)
    print(f"OSM road_name 그리드 구축 완료 ({time.time()-t0:.1f}s, {len(name_idx.lon):,}점)", flush=True)

    heat = pd.read_csv(f"{HERE_DATA}/heating_2026.csv", encoding="utf-8-sig")

    with open(f"{HERE_DATA}/road_links.geojson") as f:
        rl = json.load(f)
    feats = rl["features"]

    for gu, dong_file in GU_CONFIG.items():
        t1 = time.time()
        print(f"\n================ {gu} ================", flush=True)
        with open(f"{HERE_DATA}/{dong_file}", encoding="utf-8") as f:
            dong_gj = json.load(f)
        dong_polys = [shape(ft["geometry"]) for ft in dong_gj["features"]]

        # 이 gu 범위 내 링크만 추출(행정동 폴리곤 union으로 대략 필터)
        union_poly = shapely.unary_union(dong_polys)
        cand = []
        for feat in feats:
            c = feat["geometry"]["coordinates"]
            mid = c[len(c) // 2]
            cand.append((feat, mid[0], mid[1]))
        lons = np.array([c[1] for c in cand]); lats = np.array([c[2] for c in cand])
        pts = shapely.points(lons, lats)
        mask = shapely.contains(union_poly, pts)
        gu_feats = [cand[i][0] for i in range(len(cand)) if mask[i]]
        print(f"  {gu} 내 링크: {len(gu_feats):,}개 ({time.time()-t1:.1f}s)", flush=True)

        # 열선 도로명 집합
        sub_heat = heat[heat["관리기관"] == gu].copy()
        sub_heat["names"] = sub_heat["설치위치 (시점 ~ 종점)"].apply(extract_names)
        heating_names = set(n for ns in sub_heat["names"] for n in ns)

        # 각 링크에 OSM road_name 매칭 + heating 플래그
        t2 = time.time()
        node_attrs = {}
        lon_s = {}; lat_s = {}; lon_e = {}; lat_e = {}
        for i, feat in enumerate(gu_feats):
            c = feat["geometry"]["coordinates"]
            mid = c[len(c) // 2]
            name = name_idx.nearest(mid[0], mid[1], max_m=30.0)
            is_heat = name in heating_names if name else False
            gid = i
            node_attrs[gid] = {"road_name": name, "heating": is_heat}
            lon_s[gid], lat_s[gid] = c[0]
            lon_e[gid], lat_e[gid] = c[-1]
        n_heat = sum(1 for v in node_attrs.values() if v["heating"])
        print(f"  OSM road_name 매칭 완료 ({time.time()-t2:.1f}s): heating≈True 링크 {n_heat}개", flush=True)

        # 위상그래프(5m 격자스냅)
        t3 = time.time()
        cell_lat = CELL_M / _M_PER_DEG_LAT
        cell_lon = CELL_M / _M_PER_DEG_LON

        def qkey(lon, lat):
            return (round(lon / cell_lon), round(lat / cell_lat))

        junction_links = defaultdict(list)
        for gid in node_attrs:
            junction_links[qkey(lon_s[gid], lat_s[gid])].append(gid)
            junction_links[qkey(lon_e[gid], lat_e[gid])].append(gid)

        G = nx.Graph()
        for gid, attrs in node_attrs.items():
            G.add_node(gid, **attrs)
        edge_set = set()
        MAX_J = 40
        for jid, lids in junction_links.items():
            uniq = sorted(set(lids))
            if len(uniq) < 2:
                continue
            if len(uniq) > MAX_J:
                hub = uniq[0]
                for other in uniq[1:]:
                    a, b = (hub, other) if hub < other else (other, hub)
                    edge_set.add((a, b))
                continue
            for a in range(len(uniq)):
                for b in range(a + 1, len(uniq)):
                    edge_set.add((uniq[a], uniq[b]))
        G.add_edges_from(edge_set)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        Gg = G.subgraph(comps[0]).copy()
        print(f"  위상그래프: N={G.number_of_nodes():,} M={G.number_of_edges():,} "
              f"거대성분 N={Gg.number_of_nodes():,} ({time.time()-t3:.1f}s)", flush=True)

        heat_set = set(n for n in Gg.nodes() if Gg.nodes[n]["heating"])
        n_heat_giant = len(heat_set)
        if n_heat_giant < 5:
            print(f"  [경고] 거대성분 내 heating≈True 노드가 {n_heat_giant}개뿐 — 통계 신뢰도 낮음", flush=True)

        ei = ei_index(Gg, heat_set)
        cond = nx.conductance(Gg, heat_set, weight=None) if 0 < n_heat_giant < Gg.number_of_nodes() else None
        print(f"  heating≈True 그룹(N={n_heat_giant}): E-I index={ei}, conductance={cond}", flush=True)

        # Louvain
        t4 = time.time()
        communities = nx.algorithms.community.louvain_communities(Gg, seed=42, resolution=1.0)
        Q = nx.algorithms.community.modularity(Gg, communities)
        node_comm = {}
        for cid, comm in enumerate(communities):
            for n in comm:
                node_comm[n] = cid
        comm_heat_ratio = {}
        for cid, comm in enumerate(communities):
            n_h = sum(1 for n in comm if Gg.nodes[n]["heating"])
            comm_heat_ratio[cid] = (n_h, len(comm), n_h / len(comm) if comm else 0)
        base_rate = n_heat_giant / Gg.number_of_nodes()
        top_enriched = sorted(comm_heat_ratio.items(), key=lambda x: -x[1][2])[:5]
        print(f"  Louvain: 커뮤니티 {len(communities)}개 Q={Q:.3f} ({time.time()-t4:.1f}s)", flush=True)
        print(f"  전체 heating 비율(baseline)={base_rate*100:.2f}%", flush=True)
        print(f"  heating 비율 상위 5개 커뮤니티:")
        for cid, (nh, sz, rate) in top_enriched:
            if sz >= 20:
                print(f"    커뮤니티{cid}: N={sz}, heating={nh}개({rate*100:.1f}%, baseline대비 {rate/base_rate:.1f}배)", flush=True)


if __name__ == "__main__":
    main()
