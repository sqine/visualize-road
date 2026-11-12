"""지역별 골목길 열선 최적 배치 — 데이터 점검 → 위험구간 → MILP.

발표자료(제22회 산업공학 프로젝트 경진대회) 9~12쪽 모형을 지역 단위로 돌린다.

    max  Σ risk_i·flow_i·x_i − δ Σ_(i,j)∈D w_ij·b_ij
    s.t. Σ length_i·COST_PER_M·x_i ≤ B,  b_ij ≥ x_i − x_j,  x∈{0,1}, b∈[0,1]

단계
----
0  경사·표고 없음      → 열선 위치만 안내
1  경사·표고(고도차)   → 위험구간 시각화
2  + 유동인구          → MILP (δ 기본 1, 예산 슬라이더)
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

from utils import milp as M  # noqa: E402
from utils import region as R  # noqa: E402
from utils.data_bootstrap import extract_archive_to, save_uploaded_to  # noqa: E402


# 의미색 — 발표자료와 같은 계열
C_BG = [185, 194, 206, 90]
C_HEAT = [31, 111, 235, 230]       # 현행 열선
C_SEL = [224, 103, 74, 240]        # 제안 선정
C_BND = [194, 59, 34]              # 잔여 경계(하류)
C_DONG = [80, 90, 110, 160]
LEVEL_TEXT = {
    0: "경사·표고 자료가 부족해 위험 분석을 할 수 없습니다. 열선 위치만 표시합니다.",
    1: "경사도·고도차까지 결합됐습니다. **위험구간 시각화**까지 가능합니다. "
       "유동인구 자료(B030)를 추가하면 MILP 최적화가 열립니다.",
    2: "경사도·고도차·유동인구가 모두 결합됐습니다. **위험구간 시각화 + MILP 최적 배치**가 가능합니다.",
}


# ══════════════════════════════════════════════════════════════════════
# 캐시 로더
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _scan(root: str) -> tuple[dict, list[dict]]:
    roots = [Path(root)] if root else []
    return R.find_seoul_files(roots), R.scan_flow_files(roots)


@st.cache_resource(show_spinner="서울 전역 링크 읽는 중...")
def _seoul(paths: tuple):
    files = dict(paths)
    L, E, geom = R.load_seoul(files)
    polys = R.load_dong_polygons()
    return L, E, geom, polys


@st.cache_data(show_spinner="유동인구 결합 중...")
def _flow(paths: tuple[str, ...]):
    return R.load_flow(list(paths))


@st.cache_data(show_spinner=False)
def _dong(seoul_key: tuple, flow_key: tuple[str, ...]) -> pd.Series:
    L, E, geom, polys = _seoul(seoul_key)
    flow = _flow(flow_key)[0] if flow_key else None
    return R.assign_dong(L, polys, flow)


@st.cache_resource(show_spinner="지역 그래프 구성 중...")
def _graph(seoul_key, flow_key, gu, dongs, preset, flow_only, h_min):
    L, E, geom, polys = _seoul(seoul_key)
    flow = _flow(flow_key)[0] if flow_key else None
    dong = _dong(seoul_key, flow_key)
    G = R.build_region_graph(R.Region(gu, dongs, preset), L, E, dong, flow,
                             flow_links_only=flow_only)
    return G, R.check_availability(G, h_min)


@st.cache_resource(show_spinner="최적화 준비 중...")
def _ctx(graph_key: tuple, h_min, cost, mode, iters, tlimit, n_src):
    G, _ = _graph(*graph_key)
    cfg = R.OptConfig(h_min=h_min, cost_per_m=cost, weight_mode=mode,
                      msa_iters=iters, time_limit=tlimit or None, n_src=n_src or None)
    return R.prepare(G, cfg)


# ══════════════════════════════════════════════════════════════════════
# 지도
# ══════════════════════════════════════════════════════════════════════
def _paths(G, geom, nodes, color, width, extra=None) -> list[dict]:
    out = []
    for n in nodes:
        a = G.nodes[n]
        lid = str(a.get("link_id") or n)
        if geom is not None and lid in geom.index:
            g = geom.loc[lid]
            path = [[float(x), float(y)] for x, y in zip(g["xs"], g["ys"])]
        else:
            path = [[a["lon"], a["lat"]], [a["lon"] + 1e-5, a["lat"] + 1e-5]]
        d = {"path": path, "color": color, "width": width, "lid": lid,
             "dong": a.get("dong") or "-",
             "slope": "-" if a.get("slope_pct") is None else f"{a['slope_pct']:.1f}",
             "flow": f"{a.get('flow_sum') or 0:.1f}" if a.get("has_flow") else "-",
             "len": f"{a.get('length_m') or 0:.0f}", "tag": ""}
        if extra:
            d.update(extra(n))
        out.append(d)
    return out


def _path_layer(data, lid="paths") -> pdk.Layer:
    # pydeck 은 문자열 인자를 JS 식으로 해석한다 — 상수 문자열은 따옴표를 한 번 더 감싼다
    return pdk.Layer("PathLayer", data=data, id=lid, get_path="path",
                     get_color="color", get_width="width", width_units='"pixels"',
                     width_min_pixels=1, cap_rounded=True, joint_rounded=True,
                     pickable=True)


def _dong_layer(polys, gu, dongs) -> pdk.Layer | None:
    data = [{"polygon": r.tolist(), "name": d}
            for (g, d), rings in polys.items()
            if g == gu and (not dongs or d in dongs) for r in rings]
    if not data:
        return None
    return pdk.Layer("PolygonLayer", data=data, id="dong", get_polygon="polygon",
                     filled=False, stroked=True, get_line_color=C_DONG,
                     line_width_min_pixels=1.5, pickable=False)


def _deck(G, layers) -> pdk.Deck:
    lat = np.mean([G.nodes[n]["lat"] for n in G.nodes()])
    lon = np.mean([G.nodes[n]["lon"] for n in G.nodes()])
    span = max(np.ptp([G.nodes[n]["lat"] for n in G.nodes()]),
               np.ptp([G.nodes[n]["lon"] for n in G.nodes()]) * 0.8, 1e-3)
    zoom = float(np.clip(np.log2(0.9 / span) + 8.4, 11, 17))
    tip = {"html": "<b>{lid}</b> {tag}<br/>{dong} · 경사 {slope}% · 통행 {flow} · {len} m",
           "style": {"fontSize": "12px"}}
    return pdk.Deck(layers=[l for l in layers if l is not None],
                    initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
                    map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
                    tooltip=tip)


def _ramp(v: float) -> list[int]:
    """0~1 → 연노랑→진빨강 (미끄러짐 확률)."""
    v = float(np.clip(v, 0, 1))
    stops = [(0.0, (253, 231, 179)), (0.4, (247, 160, 90)),
             (0.7, (220, 75, 50)), (1.0, (130, 20, 30))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return [int(ca[k] + t * (cb[k] - ca[k])) for k in range(3)] + [220]
    return list(stops[-1][1]) + [220]


# ══════════════════════════════════════════════════════════════════════
# 1. 데이터 소스 (사이드바)
# ══════════════════════════════════════════════════════════════════════
st.title("지역별 골목길 열선 최적 배치")
st.caption(
    "열선 설치구간·경사도·고도차를 지역 단위로 추출하고, 유동인구가 결합되면 "
    "하류 위험전이를 고려한 MILP(발표자료 9쪽)로 예산별 최적 배치를 본다."
)

with st.sidebar:
    st.header("① 데이터 소스")
    local_root = st.text_input(
        "로컬 데이터 폴더",
        value=str(R.PROJECT_ROOT),
        help="앱 data/ 에 없는 자료를 이 폴더 아래(4단계 깊이)에서 찾는다. "
             "다운로드 받은 B030 유동인구 CSV, seoul_*.parquet 를 넣어두면 자동 인식.",
    )
    if st.button("다시 스캔", use_container_width=True):
        _scan.clear()

    with st.expander("파일 추가 (업로드)"):
        up_flow = st.file_uploader("유동인구 CSV (B030 형식)", type=["csv"],
                                   accept_multiple_files=True, key="up_flow")
        if up_flow and st.button("유동인구 저장", key="save_flow"):
            for f in up_flow:
                save_uploaded_to(f, f.name, R.FLOW_UPLOAD_DIR).rename(
                    R.FLOW_UPLOAD_DIR / f.name)
            _scan.clear()
            st.success(f"{len(up_flow)}개 저장 → data/flow_uploads/")
        up_seoul = st.file_uploader("서울 링크 자료 (zip · parquet)",
                                    type=["zip", "parquet"], accept_multiple_files=True,
                                    key="up_seoul")
        if up_seoul and st.button("링크 자료 설치", key="save_seoul"):
            placed = []
            for f in up_seoul:
                tmp = save_uploaded_to(f, f.name, R.DATA)
                placed += extract_archive_to(tmp, R.DATA)
                tmp.unlink(missing_ok=True)
            _scan.clear(); _seoul.clear()
            st.success(f"배치: {', '.join(placed)}")

seoul_files, flow_files = _scan(local_root)

with st.sidebar:
    st.markdown("**서울 링크 자료** (열선·경사·표고)")
    for name, p in seoul_files.items():
        where = "—" if p is None else ("data/" if p.parent == R.DATA else f"로컬: {p.parent.name}/")
        st.caption(f"{'있음' if p else '없음'} · `{name}` {where}")
    st.markdown("**유동인구 자료** (B030)")
    flow_opts = [str(f["path"]) for f in flow_files]
    rel = lambda s: s.replace(str(R.PROJECT_ROOT) + "/", "")
    flow_sel = st.multiselect(
        "결합할 파일", flow_opts, default=flow_opts, format_func=rel,
        help="같은 자료가 여러 벌 있어도 (링크·요일·연령·시간대) 키로 중복 제거한다.",
    ) if flow_opts else []
    if not flow_opts:
        st.caption("찾지 못함 — 위 폴더에 B030 CSV를 두거나 업로드")

if not seoul_files["seoul_links.parquet"] or not seoul_files["seoul_edges.parquet"]:
    st.error(
        "서울 링크 자료(`seoul_links.parquet`, `seoul_edges.parquet`)를 앱 data/ 와 "
        f"로컬 폴더(`{local_root}`) 어디에서도 찾지 못했습니다.\n\n"
        "- 사이드바 **파일 추가**에서 zip/parquet 업로드, 또는\n"
        "- `road_links.geojson` 이 있으면 `python3 scripts/build_seoul_dataset.py` 로 생성"
    )
    st.stop()

seoul_key = tuple(sorted((k, str(v)) for k, v in seoul_files.items() if v))
flow_key = tuple(sorted(flow_sel))
L, E, geom, polys = _seoul(seoul_key)
flow_df, flow_note = _flow(flow_key) if flow_key else (None, {"files": [], "filters": []})
dong_s = _dong(seoul_key, flow_key)


# ══════════════════════════════════════════════════════════════════════
# 2. 지역 선택
# ══════════════════════════════════════════════════════════════════════
st.subheader("② 지역 선택")
catalog = R.region_catalog(L, dong_s, flow_df)
c1, c2, c3 = st.columns([1.2, 1, 2])
mode = c1.radio("대상", ["발표 대상지", "직접 선택"], horizontal=True,
                help="발표 대상지는 road_graph.json(정밀 열선 매칭)을 그대로 써서 발표 수치를 재현한다.")
if mode == "발표 대상지":
    gu, dongs, preset, flow_only = R.PRESET_GU, R.PRESET_DONGS, True, False
    c2.markdown(f"**{R.PRESET_GU}**")
    c3.markdown("· ".join(R.PRESET_DONGS))
else:
    gus = sorted(catalog.gu.unique())
    has_flow_gu = set(catalog.loc[catalog.유동인구링크 > 0, "gu"])
    gu = c2.selectbox("자치구", gus,
                      index=gus.index(R.PRESET_GU) if R.PRESET_GU in gus else 0,
                      format_func=lambda g: f"{g} (유동인구 있음)" if g in has_flow_gu else g)
    sub = catalog[catalog.gu == gu].sort_values("골목길", ascending=False)
    lab = {r.dong: f"{r.dong} (골목 {r.골목길} · 유동 {r['유동인구커버%']:.0f}%)"
           for _, r in sub.iterrows()}
    dongs = tuple(c3.multiselect("행정동 (비우면 구 전체)", list(lab), format_func=lab.get))
    preset = False
    any_flow = bool(sub.loc[sub.dong.isin(dongs) if dongs else slice(None), "유동인구링크"].sum())
    flow_only = st.checkbox(
        "유동인구 자료가 있는 링크만 분석 (발표 방식)", value=any_flow, disabled=not any_flow,
        help="B030 이 잡지 않은 링크를 통행 0으로 두면 MILP가 그 구간을 고르지 않는다. "
             "발표 모형은 유동인구가 있는 링크만 노드로 썼다.",
    )

h_min = st.session_state.get("h_min", M.H_MIN)
graph_key = (seoul_key, flow_key, gu, tuple(dongs), preset, flow_only, h_min)
G, av = _graph(*graph_key)
if G.number_of_nodes() == 0:
    st.warning("선택한 지역에 링크가 없습니다.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# 3. 데이터 점검
# ══════════════════════════════════════════════════════════════════════
st.subheader("③ 데이터 결합 점검")
ok = lambda c: "충분" if c >= R.LEVEL_MIN_COVERAGE else ("부족" if c > 0 else "없음")
rows = [
    {"항목": "도로 링크", "상태": "충분", "결합": f"{av['links']:,}개 (골목길 {av['golmok']:,})",
     "출처": "seoul_links.parquet" + (" + road_graph.json" if preset else "")},
    {"항목": "열선 설치구간", "상태": "충분",
     "결합": f"{av['heat_links']}개 링크 · {av['heat_len_m']:,.0f} m",
     "출처": "서울시 열선 설치현황 → 링크 매칭" + (" (정밀)" if preset else "")},
    {"항목": "경사도", "상태": ok(av["slope_cov"]), "결합": f"골목길의 {100*av['slope_cov']:.1f}%",
     "출처": "수치지형도 1:1,000 등고선 → IDW"},
    {"항목": "고도차(표고)", "상태": ok(av["z_cov"]),
     "결합": f"골목길의 {100*av['z_cov']:.1f}% · 낙차 ≥{h_min:g} m 인접쌍 {av['downhill_pairs']:,}",
     "출처": "링크 중점 표고"},
    {"항목": "유동인구", "상태": ok(av["flow_cov"]), "결합": f"골목길의 {100*av['flow_cov']:.1f}%",
     "출처": (", ".join(Path(p).name for p in flow_key) or "없음")
             + (f" [{' · '.join(flow_note['filters'])}]" if flow_note["filters"] else "")},
]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
lvl = av["level"]
(st.success if lvl == 2 else st.info if lvl == 1 else st.warning)(
    f"**{lvl}단계** — {LEVEL_TEXT[lvl]}")
if lvl == 1 and flow_df is not None and len(flow_df):
    cov = catalog[(catalog.유동인구링크 > 0)].sort_values("유동인구링크", ascending=False)
    st.caption("유동인구가 있는 행정동: " + ", ".join(
        f"{r.gu} {r.dong}({r.유동인구링크})" for _, r in cov.head(8).iterrows())
        + " — 직접 선택에서 이 동을 고르고 '유동인구 자료가 있는 링크만'을 켜면 2단계가 됩니다.")

if lvl == 0:
    heat = [n for n in G.nodes() if G.nodes[n].get("heating_installed_precise")]
    st.pydeck_chart(_deck(G, [
        _path_layer(_paths(G, geom, list(G.nodes()), C_BG, 1), "bg"),
        _path_layer(_paths(G, geom, heat, C_HEAT, 4), "heat")]))
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# 4. 분석 탭
# ══════════════════════════════════════════════════════════════════════
tabs = st.tabs(["위험구간", "MILP 최적 배치"] if lvl == 2 else ["위험구간"])

# ── 4-1. 위험구간 ─────────────────────────────────────────────────────
with tabs[0]:
    rt = R.risk_table(G, h_min)
    bp = R.boundary_pair_table(G, h_min)
    gm = rt[rt.road_type == "골목길"]
    steep = gm[gm.steep]
    steep_heat = steep[steep.heat]
    k = st.columns(5)
    k[0].metric("골목길 연장", f"{gm.length_m.sum()/1000:,.1f} km")
    k[1].metric("위험군(경사 ≥5.56%)", f"{steep.length_m.sum()/1000:,.1f} km",
                f"{100*steep.length_m.sum()/max(gm.length_m.sum(),1):.1f}%", delta_color="off")
    k[2].metric("위험군 열선 커버리지", f"{100*steep_heat.length_m.sum()/max(steep.length_m.sum(),1):.1f}%")
    k[3].metric(f"낙차 ≥{h_min:g} m 인접쌍", f"{av['downhill_pairs']:,}")
    k[4].metric("현행 경계쌍 (열선 위·미설치 아래)", f"{len(bp)}곳")

    color_opts = ["미끄러짐 확률 P(MAA<θ)", "경사도 등급"]
    if rt.flow.notna().any():
        color_opts.append("기대 미끄러짐 (확률×통행)")
    cby = st.radio("색 기준", color_opts, horizontal=True, key="risk_color")
    only_golmok = st.checkbox("골목길만 표시", value=True)

    rti = rt.set_index("link_id")
    em = rt.expected_slips.quantile(0.98) if rt.expected_slips.notna().any() else 1

    def _rcolor(n):
        r = rti.loc[str(G.nodes[n].get("link_id") or n)]
        if cby.startswith("미끄러짐"):
            c = _ramp(r.risk) if r.risk is not None and not pd.isna(r.risk) else C_BG
        elif cby.startswith("경사"):
            s = r.slope_pct
            c = (C_BG if s is None or pd.isna(s) else
                 [120, 190, 120, 200] if s < 2 else [250, 210, 90, 220] if s < R.STEEP_PCT
                 else [235, 120, 60, 230] if s < 10 else [170, 30, 40, 240])
        else:
            v = r.expected_slips
            c = C_BG if v is None or pd.isna(v) else _ramp(v / max(em, 1e-9))
        return {"color": c, "tag": "[현행 열선]" if r.heat else ""}

    show = [n for n in G.nodes()
            if not only_golmok or G.nodes[n].get("road_type") == "골목길"]
    heat_n = [n for n in G.nodes() if G.nodes[n].get("heating_installed_precise")]
    layers = [
        _dong_layer(polys, gu, dongs),
        _path_layer(_paths(G, geom, show, C_BG, 2, _rcolor), "risk"),
        _path_layer(_paths(G, geom, heat_n, C_HEAT, 5, lambda n: {"tag": "[현행 열선]"}), "heat"),
    ]
    if len(bp):
        bpd = bp.assign(r=4 + 7 * bp.drop_m.clip(upper=10) / 10)
        layers.append(pdk.Layer("ScatterplotLayer", data=bpd, id="bnd",
                                get_position="[lon, lat]", get_radius="r",
                                radius_units='"pixels"', get_fill_color=C_BND + [200],
                                stroked=True, get_line_color=[255, 255, 255],
                                line_width_min_pixels=1.5, pickable=False))
    st.pydeck_chart(_deck(G, layers), use_container_width=True)
    st.caption(
        "선 색 = " + cby + " · 굵은 파랑 = 현행 열선 · 빨간 원 = 현행 경계쌍의 하류 구간"
        "(원 크기 = 낙차). 미끄러짐 확률은 WinterLab 빙판 보행 실측 MAA 경험분포(58세션)."
    )

    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("##### 행정동별 위험군")
        byd = (gm.assign(dong=gm.dong.fillna("(미상)"),
                         steep_len=gm.length_m.where(gm.steep, 0),
                         steep_heat=gm.length_m.where(gm.steep & gm.heat, 0))
                 .groupby("dong").agg(골목km=("length_m", lambda s: s.sum() / 1000),
                                      위험군km=("steep_len", lambda s: s.sum() / 1000),
                                      열선km=("steep_heat", lambda s: s.sum() / 1000))
                 .round(2).reset_index())
        byd["커버리지%"] = (100 * byd.열선km / byd.위험군km.replace(0, np.nan)).round(1)
        st.dataframe(byd, hide_index=True, use_container_width=True)
    with cB:
        st.markdown("##### 현행 경계쌍")
        if len(bp):
            st.dataframe(bp[["up", "down", "dong", "drop_m", "down_slope", "down_risk", "down_flow"]]
                         .rename(columns={"up": "상류(열선)", "down": "하류(미설치)",
                                          "drop_m": "낙차m", "down_slope": "하류경사%",
                                          "down_risk": "하류 미끄러짐확률", "down_flow": "하류 통행"}),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("경계쌍 없음")

    st.download_button("⬇️ 링크별 위험 지표 CSV", rt.drop(columns=["_ix"]).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"risk_links_{gu}.csv", mime="text/csv")


# ── 4-2. MILP ─────────────────────────────────────────────────────────
def _solution_view(ctx, res, res0, B, delta, kp, show_net_default=False):
    """한 해(예산 B, δ)의 지도 + 통계. 예산 탐색·δ 탐색 탭이 함께 쓴다."""
    cur = ctx.current
    mcol, scol = st.columns([1.75, 1])
    with mcol:
        t = st.columns(2)
        show_cur = t[0].toggle("현행 열선 함께 표시", value=True, key=f"{kp}_cur")
        show_net = t[1].toggle("경계 통행량(네트워크 전이) 표시", value=show_net_default,
                               key=f"{kp}_net",
                               help="낙차 2 m 이상 인접쌍의 상류→하류 연결을 경계 통행량(엣지전류)에 비례한 굵기로 그린다.")
        layers = [_dong_layer(polys, gu, dongs),
                  _path_layer(_paths(G, geom, list(G.nodes()), C_BG, 1), "bg")]
        if show_net:
            cur_d = R.boundary_currents(ctx)
            if cur_d:
                q = max(np.quantile(list(cur_d.values()), 0.98), 1e-9)
                lines = [{"path": [[G.nodes[i]["lon"], G.nodes[i]["lat"]],
                                   [G.nodes[j]["lon"], G.nodes[j]["lat"]]],
                          "color": [70, 90, 130, int(35 + 110 * min(c / q, 1))],
                          "width": 1 + 3 * min(c / q, 1), "lid": f"{i}→{j}", "dong": "-",
                          "slope": "-", "flow": f"{c:.1f}", "len": "-", "tag": "[경계 통행량]"}
                         for (i, j), c in cur_d.items()]
                layers.append(_path_layer(lines, "net"))
        if show_cur:
            hn = [n for n in G.nodes() if G.nodes[n].get("heating_installed_precise")]
            layers.append(_path_layer(_paths(G, geom, hn, C_HEAT, 4,
                                             lambda n: {"tag": "[현행 열선]"}), "heat"))
        layers.append(_path_layer(_paths(
            G, geom, res["selected"], C_SEL, 5,
            lambda n: {"tag": f"[선정] 기대방지 {ctx.risk[n]*ctx.flow[n]:.1f}건"}), "sel"))
        _, contrib = R.edge_exposure(ctx, res["selected"])
        by_down = {}
        for (i, j) in res["boundaries"]:
            by_down[j] = by_down.get(j, 0) + (contrib.get((i, j)) or ctx.risk[j] * ctx.flow[j])
        if by_down:
            emax = max(by_down.values()) or 1
            pts = pd.DataFrame([{"lon": G.nodes[j]["lon"], "lat": G.nodes[j]["lat"],
                                 "a": int(90 + 160 * e / emax), "r": 3 + 7 * (e / emax) ** 0.5}
                                for j, e in by_down.items()])
            pts["c"] = pts.a.apply(lambda a: C_BND + [a])
            layers.append(pdk.Layer("ScatterplotLayer", data=pts, id="rb",
                                    get_position="[lon, lat]", get_radius="r",
                                    radius_units='"pixels"', get_fill_color="c",
                                    stroked=True, get_line_color=[255, 255, 255],
                                    line_width_min_pixels=1, pickable=False))
        st.pydeck_chart(_deck(G, layers), use_container_width=True)
        st.caption("주황 = 선정 구간 · 파랑 = 현행 열선 · 빨간 원 = 잔여 경계의 하류(진하고 클수록 하류 전이위험이 큼)"
                   + (" · 남색 선 = 경계 통행량" if show_net else ""))

    with scol:
        st.markdown(f"##### 예산 {B/1e8:,.2f}억 · δ={delta:g}")
        m1, m2 = st.columns(2)
        m1.metric("선정 구간", f"{res['n']}곳", f"{res['length_m']:,.0f} m", delta_color="off")
        m2.metric("집행액", f"{res['spent_won']/1e8:,.2f}억",
                  f"잔액 {(B-res['spent_won'])/1e8:,.2f}억", delta_color="off")
        m1.metric("경사위험 방지", f"{res['benefit']:,.1f}건",
                  f"현행 대비 {res['benefit']/cur['benefit']:.1f}배" if cur["benefit"] else None,
                  delta_color="off", help="Σ 선정 구간 P(MAA<θ) × 유동인구 = 기대 미끄러짐 방지 건수")
        m2.metric("위험 커버율", f"{res['coverage_risk_pct']:.1f}%",
                  f"연장 커버 {res['coverage_len_pct']:.1f}%", delta_color="off",
                  help="후보군 전체 기대 미끄러짐 중 선정 구간이 막는 비율")
        m1.metric("위험경계", f"{res['boundary']}곳",
                  f"{res['boundary']-res0['boundary']:+d} vs δ=0", delta_color="inverse")
        e, e0 = res["exposure_edge"], res0["exposure_edge"]
        m2.metric("하류 전이위험", f"{e:,.1f}" if e is not None else "-",
                  f"{100*(e/e0-1):+.0f}% vs δ=0" if e and e0 else None, delta_color="inverse",
                  help="Σ 잔여 경계의 경계 통행량(엣지전류) × 하류 미끄러짐 확률. 발표 10쪽 정의·목적함수와 같은 양")
        loss = 100 * (1 - res["benefit"] / res0["benefit"]) if res0["benefit"] else 0
        st.caption(f"δ=0 대비 편익 {loss:.1f}% 양보 · 건당 {res['won_per_slip']/1e4:,.0f}만원 · "
                   f"하류 통행 기준 전이위험 {res['exposure']:,.1f} (발표 표 수치)"
                   if res["won_per_slip"] else "")
        cmp = pd.DataFrame([
            {"": "현행 실제", "구간": cur["n"], "편익": round(cur["benefit"], 1),
             "경계": cur["boundary"], "전이위험": R._r1(cur["exposure_edge"])},
            {"": "위험×통행 (δ=0)", "구간": res0["n"], "편익": round(res0["benefit"], 1),
             "경계": res0["boundary"], "전이위험": R._r1(e0)},
            {"": f"제안 (δ={delta:g})", "구간": res["n"], "편익": round(res["benefit"], 1),
             "경계": res["boundary"], "전이위험": R._r1(e)},
        ])
        st.dataframe(cmp, hide_index=True, use_container_width=True)
        if not res["converged"] and ctx.cfg.weight_mode == "current":
            st.caption(f"MSA {ctx.cfg.msa_iters}회 안에 고정점 미수렴 — 마지막 해를 표시")

    with st.expander("같은 예산, 선정 기준별 비교 (발표자료 10쪽)"):
        st.dataframe(R.strategy_compare(ctx, B), hide_index=True, use_container_width=True)
        st.caption("전이위험(경계통행) = 경계 통행량(엣지전류) × 하류 확률 — 발표 정의. "
                   "전이위험(하류통행) = 하류 구간 유동인구 × 확률 — 발표 10쪽 표의 수치. "
                   "경사·통행량 기준은 값이 큰 순으로 예산이 찰 때까지 담는 그리디.")
    sel_df = pd.DataFrame([{
        "link_id": str(G.nodes[n].get("link_id") or n), "dong": G.nodes[n].get("dong"),
        "length_m": G.nodes[n].get("length_m"), "slope_pct": ctx.slope[n],
        "risk": round(ctx.risk[n], 3), "flow": ctx.flow[n],
        "expected_slips": round(ctx.risk[n] * ctx.flow[n], 2),
        "lon": G.nodes[n]["lon"], "lat": G.nodes[n]["lat"],
    } for n in res["selected"]])
    if len(sel_df):
        sel_df = sel_df.sort_values("expected_slips", ascending=False)
    with st.expander(f"선정 구간 {len(sel_df)}곳"):
        st.dataframe(sel_df, hide_index=True, use_container_width=True)
        st.download_button(
            "선정 구간 CSV 내려받기", sel_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"selection_{gu}_B{B/1e8:.1f}억_d{delta:g}.csv", mime="text/csv",
            key=f"{kp}_dl")


REPO = "https://github.com/sqine/visualize-road"
GRID_KEYS = ("region", "n_cand", "h_min", "cost_per_m", "weight_mode", "msa_iters")


@st.cache_data(show_spinner=False)
def _grid_file(mtime: float):
    return R.load_grid()


def _read_grid_upload(files) -> tuple[pd.DataFrame | None, dict]:
    df, meta = None, {}
    for f in files or []:
        name = f.name.lower()
        if name.endswith(".json"):
            meta = json.loads(f.getvalue().decode("utf-8"))
        elif name.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(f.getvalue()))
        elif name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(f.getvalue()))
    return df, meta


def _grid_other() -> tuple[pd.DataFrame | None, dict]:
    """다른 지역·설정 — GitHub 스크립트 안내와 결과 파일 올리기."""
    with st.expander("다른 지역이나 설정으로 계산하려면"):
        st.markdown(
            "이 탭에 기본으로 나오는 결과는 발표 대상지(성북구 4개동)를 앱 기본 설정으로 미리 계산한 것입니다. "
            "다른 행정동이나 다른 설정으로 보려면 저장소의 스크립트로 계산합니다. 저장소를 내려받아 "
            "`streamlit_app` 폴더에서 실행하면 결과 파일(`.parquet`), 계산 조건(`.json`), 그림 보고서(`.html`)가 "
            "만들어지며, 보고서는 브라우저에서 바로 열립니다. 자세한 사용법은 README 4.6절에 있습니다.")
        st.code("python3 scripts/run_sensitivity_grid.py --gu 강북구 --dongs 수유1동,수유2동", language="bash")
        c = st.columns(2)
        c[0].link_button("스크립트 보기 (GitHub)", f"{REPO}/blob/main/{R.GRID_SCRIPT}", use_container_width=True)
        c[1].link_button("사용법 보기 (README)", f"{REPO}#readme", use_container_width=True)
        ups = st.file_uploader("계산한 결과를 이 탭에서 보려면 .parquet와 .json을 함께 올립니다",
                               type=["parquet", "json", "csv"], accept_multiple_files=True, key="g_up")
    return _read_grid_upload(ups)


def _grid_tab(ctx, cur, region_label):
    from utils import sensitivity_viz as V
    sig = R.grid_signature(ctx, region_label)
    st.caption("예산과 δ를 함께 바꿔 가며 경사위험 방지와 하류 전이위험의 교환을 봅니다(발표자료 11쪽). "
               "칸마다 MILP를 풀어야 해서 결과는 미리 계산해 두었고, 이 탭은 그 결과를 읽어 그립니다.")
    gdf, gmeta = _grid_other()
    if gdf is None and R.GRID_FILE.exists():
        gdf, gmeta = _grid_file(R.GRID_FILE.stat().st_mtime)
        gmeta = gmeta or {}
    if gdf is None or not len(gdf):
        st.info("미리 계산한 결과(data/sensitivity_grid.parquet)가 없습니다. 위 안내대로 스크립트로 계산해 주세요.")
        return
    need = {"_B", "δ", "경사위험방지", "편익손실%", "감소%_경계통행", "선정링크"}
    if need - set(gdf.columns):
        st.error("이중 탐색 결과 파일이 아닙니다 (빠진 열: " + ", ".join(sorted(need - set(gdf.columns))) + ").")
        return

    diff = [k for k in GRID_KEYS if k in gmeta and gmeta[k] != sig[k]]
    same = bool(gmeta) and not diff
    if not gmeta:
        st.warning("계산 조건(.json)이 없어 현재 설정과 같은 조건인지 확인할 수 없습니다. 그림만 표시합니다.")
    elif diff:
        st.warning("결과 파일의 계산 조건이 현재 설정과 다릅니다: " +
                   ", ".join(f"{k} {gmeta[k]} ↔ {sig[k]}" for k in diff) +
                   ". 그림은 파일 기준이며, 지도 배치는 같은 조건일 때만 표시합니다.")
    elif gmeta.get("created"):
        st.caption(f"계산 조건 일치 · {gmeta.get('created')} 계산 · {gmeta.get('cells', len(gdf))}칸")

    metric = st.radio("전이위험 기준", ["경계통행", "하류통행"], horizontal=True, key="g_metric",
                      format_func={"경계통행": "경계 통행량 × 하류 확률 (발표 정의)",
                                   "하류통행": "하류 유동인구 × 확률 (발표 10쪽 표)"}.get)
    ycol = f"감소%_{metric}"
    summ = R.grid_summary(gdf, ycol)
    budgets = sorted(gdf["_B"].unique())
    curB = min(budgets, key=lambda b: abs(b - cur["spent_won"])) if cur["spent_won"] else budgets[len(budgets) // 2]
    srow = summ.set_index("_B").loc[curB]
    k = st.columns(4)
    k[0].metric("격자", f"예산 {len(budgets)} × δ {gdf['δ'].nunique()}")
    k[1].metric(f"{curB/1e8:,.2f}억 권장 δ 범위",
                f"{srow['하한δ']:g} ~ {srow['상한δ']:g}" if pd.notna(srow["하한δ"]) and pd.notna(srow["상한δ"]) else "-")
    k[2].metric("범위가 성립하는 예산선", f"{int(summ['범위성립'].sum())} / {len(summ)}")
    k[3].metric("전환점 δ 최빈값", f"{summ['하한δ'].mode().iloc[0]:g}" if summ["하한δ"].notna().any() else "-")

    s = st.columns([2, 1.4])
    B = s[0].select_slider("예산", options=budgets, value=curB, key="g_B",
                           format_func=lambda x: _budget_label(x, cur))
    line = gdf[gdf["_B"] == B].sort_values("δ")
    knee = summ.set_index("_B").at[B, "하한δ"]
    dopts = list(line["δ"])
    dv = s[1].select_slider("δ", options=dopts, value=knee if knee in dopts else dopts[0], key="g_d",
                            format_func=lambda v: f"{v:g}" + (" (전환점)" if v == knee else ""))
    hm = st.selectbox("히트맵 지표", [ycol, "편익손실%", "경사위험방지", "위험경계", f"전이위험_{metric}"],
                      format_func=lambda c: V.METRICS.get(c, c), key="g_hm")
    st.altair_chart(V.heatmap(gdf, summ, hm, sel=(B, dv)), use_container_width=True)
    st.caption("진한 테두리는 예산선마다의 파레토 전환점(δ 하한), 점선 테두리는 편익 5% 손실 이내 최대 δ(상한), "
               "흰 점은 아래 지도에 표시한 칸입니다.")
    c = st.columns(2)
    c[0].markdown("##### 예산선별 파레토 프런트")
    c[0].altair_chart(V.fronts(gdf, summ, f"전이위험_{metric}", sel_B=B, height=300), use_container_width=True)
    c[1].markdown("##### 예산별 권장 δ 범위")
    c[1].altair_chart(V.band(summ, cur["spent_won"] / 1e8 if cur["spent_won"] else None, height=300),
                      use_container_width=True)
    st.caption("왼쪽은 예산이 클수록 진한 선이며 오른쪽 아래로 갈수록 좋은 배치입니다(편익은 크고 전이위험은 작음). "
               "◆는 각 예산선의 전환점입니다. 오른쪽 띠는 예산마다 고를 만한 δ의 범위입니다.")

    if same:
        row = line[line["δ"] == dv].iloc[0]
        row0 = line[line["δ"] == 0].iloc[0]
        res = R.plan_metrics(ctx, R.grid_selection(ctx, row), B)
        res.update(delta=float(dv), msa_iters=int(row.get("MSA반복", 0)), converged=bool(row.get("수렴", True)))
        res0 = R.plan_metrics(ctx, R.grid_selection(ctx, row0), B)
        res0.update(delta=0.0, msa_iters=0, converged=True)
        _solution_view(ctx, res, res0, B, float(dv), "g", show_net_default=False)
    else:
        st.info("지도 배치는 결과 파일이 현재 지역·설정과 같은 조건으로 계산됐을 때 표시합니다.")
    with st.expander("격자 표"):
        show = gdf.drop(columns=["_B", "_ben", "선정링크"], errors="ignore")
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.dataframe(summ.drop(columns=["_B"]), hide_index=True, use_container_width=True)
        st.download_button("격자 CSV 내려받기", show.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sensitivity_grid.csv", mime="text/csv")


def _budget_label(x, cur):
    return (f"{x/1e8:,.2f}억" + (" (현행)" if abs(x - cur["spent_won"]) < 1 else
                                 " (현행+30%)" if abs(x - 1.3 * cur["spent_won"]) < 1 else ""))


if lvl == 2:
    with tabs[1]:
        st.latex(r"\max\ \sum_i risk_i\,flow_i\,x_i-\delta\!\!\sum_{(i,j)\in D}\! w_{ij}\,b_{ij}"
                 r"\quad\text{s.t.}\ \sum_i \ell_i\,c\,x_i\le B,\ \ b_{ij}\ge x_i-x_j")
        p = st.columns([1, 1, 1.6])
        h_new = p[0].number_input("경계 낙차 기준 h_min (m)", 0.5, 10.0, float(h_min), 0.5,
                                  help="D = 인접하고 z_i − z_j ≥ h_min 인 쌍")
        if h_new != h_min:
            st.session_state["h_min"] = h_new
            st.rerun()
        cost = p[1].number_input("미터당 설치비 (원)", 100_000, 5_000_000,
                                 int(round(M.COST_PER_M)), 10_000,
                                 help="기본 15억 원 / 1.7 km (3 m 1개 차로)")
        wmode = p[2].selectbox(
            "전이 가중치 w_ij", ["current", "simple"],
            format_func={"current": "경계 통행량(엣지전류·MSA) × 하류 확률 — 발표 모형",
                         "simple": "하류 risk×flow (빠른 근사)"}.get)
        with st.expander("고급 설정"):
            a = st.columns(3)
            iters = a[0].number_input("MSA 최대 반복", 1, 40, 15)
            tlimit = a[1].number_input("MILP 시간 제한(초, 0=없음)", 0, 600,
                                       0 if av["golmok"] < 3000 else 30)
            n_src = a[2].number_input("엣지전류 source 상한 (0=전체)", 0, 5000,
                                      0 if G.number_of_nodes() < R.LARGE_REGION_NODES
                                      else R.LARGE_REGION_N_SRC)

        ctx = _ctx(graph_key, h_min, float(cost), wmode, int(iters), float(tlimit), int(n_src))
        cur = ctx.current
        auto = len(ctx.cand) <= 1500
        st.caption(
            f"후보 {len(ctx.cand):,}개 · 전면 시공비 {ctx.full_cost/1e8:,.1f}억 · "
            f"현행 골목 열선 {cur['length_m']:,.0f} m = **{cur['spent_won']/1e8:.2f}억** · "
            f"현행 경계 {cur['boundary']}곳, 하류 전이위험 {R._r1(cur['exposure_edge'])}")
        base_key = (graph_key, h_min, cost, wmode, iters, tlimit, n_src)
        tb, td, tg = st.tabs(["예산 탐색 (δ 고정)", "δ 탐색 (예산 고정)", "이중 탐색 (예산 × δ)"])

        # ── 예산 탐색 ───────────────────────────────────────────────
        with tb:
            dflt = R.default_budgets(ctx, 16)
            b = st.columns([1, 1, 1, 1])
            delta = b[0].number_input("δ (전이위험 가중치)", min_value=0.0, max_value=50.0,
                                      value=1.0, step=0.5, key="delta",
                                      help="0이면 경계를 무시(위험×통행량 최대화). 클수록 하류 경계를 강하게 피한다.")
            bmin = b[1].number_input("예산 하한 (억)", 0.1, 10_000.0, round(dflt[0] / 1e8, 2), 0.5)
            bmax = b[2].number_input("예산 상한 (억)", 0.2, 10_000.0, round(dflt[-1] / 1e8, 2), 0.5)
            npts = b[3].number_input("격자 수", 3, 60, 16)
            grid = [round(float(x), -6) for x in np.linspace(bmin * 1e8, bmax * 1e8, int(npts))]
            grid = R.with_anchor_budgets(grid, cur["spent_won"])   # 현행·현행+30% 정확히
            sweep_key = base_key + (float(delta), tuple(grid))
            cached = st.session_state.get("sweep_key") == sweep_key
            if st.button("예산 스윕 실행", type="primary", disabled=cached, key="run_b") or (auto and not cached):
                bar = st.progress(0.0, text="MILP 풀이 중...")
                df = R.sweep(ctx, grid, progress=lambda f, t: bar.progress(f, text=t))
                bar.empty()
                st.session_state["sweep_key"], st.session_state["sweep_df"] = sweep_key, df
            if st.session_state.get("sweep_key") != sweep_key:
                st.info("설정을 바꿨습니다. **예산 스윕 실행**을 누르면 예산별 해를 계산합니다.")
            else:
                sw = st.session_state["sweep_df"]
                opts = list(sw._B)
                init = min(opts, key=lambda x: abs(x - cur["spent_won"])) if cur["spent_won"] else opts[len(opts) // 2]
                B = st.select_slider("예산 — 드래그해서 예산별 배치를 본다", options=opts, value=init,
                                     format_func=lambda x: _budget_label(x, cur), key="B_slider")
                _solution_view(ctx, R.solve_budget(ctx, B, float(delta)), R.solve_budget(ctx, B, 0.0),
                               B, float(delta), "b")

                st.markdown("##### 예산별 결과")
                cols3 = [("경사위험방지", "경사위험 방지 (건)"), ("위험경계", "위험경계 (곳)"),
                         ("전이위험_경계통행", "하류 전이위험")]
                long = pd.concat([
                    sw.assign(모형=f"제안 δ={delta:g}")[["예산억", "모형"] + [c for c, _ in cols3]],
                    sw[["예산억"] + ["δ0_" + c for c, _ in cols3]]
                      .rename(columns={"δ0_" + c: c for c, _ in cols3}).assign(모형="δ=0")
                      [["예산억", "모형"] + [c for c, _ in cols3]],
                ])
                rule = alt.Chart(pd.DataFrame({"x": [B / 1e8]})).mark_rule(
                    color="#2B3348", strokeDash=[4, 3]).encode(x="x:Q")
                colr = alt.Color("모형:N", scale=alt.Scale(domain=[f"제안 δ={delta:g}", "δ=0"],
                                                           range=["#E0674A", "#9AA5B4"]),
                                 legend=alt.Legend(orient="top", title=None))
                cc = st.columns(3)
                for c_, (col, ttl) in zip(cc, cols3):
                    base = alt.Chart(long).encode(
                        x=alt.X("예산억:Q", title="예산 (억)"), y=alt.Y(f"{col}:Q", title=ttl),
                        color=colr, tooltip=["모형", "예산억", col])
                    c_.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(size=30, filled=True)
                                     + rule).properties(height=220), use_container_width=True)
                with st.expander("예산별 표"):
                    st.dataframe(sw.drop(columns=["_B"]), hide_index=True, use_container_width=True)

        # ── δ 탐색 ─────────────────────────────────────────────────
        with td:
            st.caption("예산을 하나 고정하고 δ를 바꿔 가며 경사위험 방지(편익)와 하류 전이위험의 교환을 본다 "
                       "(발표자료 11쪽의 한 예산선). 권장 δ = 편익 손실–전이위험 감소 곡선의 파레토 전환점.")
            d0 = st.columns([1.2, 2])
            bopts = sorted(set(R.default_budgets(ctx, 16)))
            Bd = d0[0].select_slider("고정 예산", options=bopts,
                                     value=min(bopts, key=lambda x: abs(x - cur["spent_won"])) if cur["spent_won"] else bopts[len(bopts)//2],
                                     format_func=lambda x: _budget_label(x, cur), key="Bd")
            dtxt = d0[1].text_input("δ 격자 (쉼표로 구분)", "0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10, 12")
            try:
                dgrid = sorted({float(v) for v in dtxt.replace(" ", "").split(",") if v} | {0.0})
            except ValueError:
                st.error("δ 격자는 숫자를 쉼표로 구분해 적어 주세요.")
                dgrid = [0.0, 1.0]
            dkey = base_key + (float(Bd), tuple(dgrid))
            dcached = st.session_state.get("dsweep_key") == dkey
            if st.button("δ 스윕 실행", type="primary", disabled=dcached, key="run_d") or (auto and not dcached):
                bar = st.progress(0.0, text="MILP 풀이 중...")
                ddf = R.delta_sweep(ctx, Bd, dgrid, progress=lambda f, t: bar.progress(f, text=t))
                bar.empty()
                st.session_state["dsweep_key"], st.session_state["dsweep_df"] = dkey, ddf
            if st.session_state.get("dsweep_key") != dkey:
                st.info("설정을 바꿨습니다. **δ 스윕 실행**을 누르면 δ별 해를 계산합니다.")
            else:
                ddf = st.session_state["dsweep_df"]
                metric = st.radio("전이위험 기준", ["경계통행", "하류통행"], horizontal=True,
                                  format_func={"경계통행": "경계 통행량 × 하류 확률 (발표 정의)",
                                               "하류통행": "하류 유동인구 × 확률 (발표 10쪽 표)"}.get,
                                  key="d_metric")
                ycol = f"감소%_{metric}"
                knee = R.knee_delta(ddf, ycol)
                ok = ddf[ddf["편익손실%"] <= 5.0]
                upper = float(ok["δ"].max()) if len(ok) else None
                k = st.columns(3)
                k[0].metric("권장 δ (파레토 전환점, 하한)", f"{knee:g}" if knee is not None else "-")
                k[1].metric("편익 5% 손실 이내 최대 δ (상한)", f"{upper:g}" if upper is not None else "-")
                k[2].metric("현행 대비 (전환점 δ)",
                            f"{float(ddf.loc[ddf['δ']==knee,'경사위험방지'].iloc[0])/cur['benefit']:.1f}배"
                            if knee is not None and cur["benefit"] else "-")
                dsel = st.select_slider("δ — 드래그해서 δ별 배치를 본다", options=list(ddf["δ"]),
                                        value=knee if knee is not None else 0.0,
                                        format_func=lambda v: f"{v:g}" + (" (전환점)" if v == knee else ""),
                                        key="d_slider")
                d = ddf.assign(권장=ddf["δ"] == knee, 선택=ddf["δ"] == dsel)
                line = alt.Chart(d).mark_line(color="#E0674A", strokeWidth=2.5).encode(
                    x=alt.X("편익손실%:Q", title="편익 손실 (%)"),
                    y=alt.Y(f"{ycol}:Q", title="하류 전이위험 감소 (%)"))
                pts = alt.Chart(d).mark_point(filled=True).encode(
                    x="편익손실%:Q", y=f"{ycol}:Q",
                    size=alt.condition(alt.datum.선택, alt.value(260), alt.value(70)),
                    color=alt.condition(alt.datum.권장, alt.value("#C23B22"), alt.value("#E0674A")),
                    tooltip=["δ", "선정", "경사위험방지", "편익손실%", "위험경계",
                             "전이위험_경계통행", "전이위험_하류통행"])
                lab = alt.Chart(d).mark_text(dy=-12, fontSize=11, color="#2B3348").encode(
                    x="편익손실%:Q", y=f"{ycol}:Q", text=alt.Text("δ:Q", format="~g"))
                five = alt.Chart(pd.DataFrame({"x": [5.0]})).mark_rule(
                    color="#9AA5B4", strokeDash=[4, 3]).encode(x="x:Q")
                st.altair_chart((line + pts + lab + five).properties(height=300), use_container_width=True)
                st.caption("곡선이 가파른 구간은 편익을 조금만 양보해도 전이위험이 크게 준다. 진한 빨강 = 파레토 전환점, "
                           "회색 점선 = 편익 5% 손실(발표의 δ 상한).")
                _solution_view(ctx, R.solve_budget(ctx, Bd, float(dsel)), R.solve_budget(ctx, Bd, 0.0),
                               Bd, float(dsel), "d", show_net_default=True)
                with st.expander("δ별 표"):
                    st.dataframe(ddf, hide_index=True, use_container_width=True)

        # ── 이중 탐색 (예산 × δ) ─────────────────────────────────────
        with tg:
            _grid_tab(ctx, cur, R.Region(gu, tuple(dongs), preset).label)
