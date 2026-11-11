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

st.set_page_config(page_title="열선 최적 배치", layout="wide")

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
if lvl == 2:
    with tabs[1]:
        st.latex(r"\max\ \sum_i risk_i\,flow_i\,x_i-\delta\!\!\sum_{(i,j)\in D}\! w_{ij}\,b_{ij}"
                 r"\quad\text{s.t.}\ \sum_i \ell_i\,c\,x_i\le B,\ \ b_{ij}\ge x_i-x_j")
        p = st.columns([1, 1, 1, 1.4])
        delta = p[0].number_input("δ (전이위험 가중치)", min_value=0.0, max_value=50.0,
                                  value=1.0, step=0.5, key="delta",
                                  help="0이면 경계를 무시(위험×통행량 최대화). 클수록 하류 경계를 강하게 피한다.")
        h_new = p[1].number_input("경계 낙차 기준 h_min (m)", 0.5, 10.0, float(h_min), 0.5,
                                  help="D = 인접하고 z_i − z_j ≥ h_min 인 쌍")
        if h_new != h_min:
            st.session_state["h_min"] = h_new
            st.rerun()
        cost = p[2].number_input("미터당 설치비 (원)", 100_000, 5_000_000,
                                 int(round(M.COST_PER_M)), 10_000,
                                 help="기본 15억 원 / 1.7 km (3 m 1개 차로)")
        wmode = p[3].selectbox(
            "전이 가중치 w_ij", ["current", "simple"],
            format_func={"current": "경계 통행량(엣지전류·MSA) × 하류 확률 — 발표 모형",
                         "simple": "하류 risk×flow (빠른 근사)"}.get)
        with st.expander("고급 설정"):
            a = st.columns(3)
            iters = a[0].number_input("MSA 최대 반복", 1, 20, 8)
            tlimit = a[1].number_input("MILP 시간 제한(초, 0=없음)", 0, 600,
                                       0 if av["golmok"] < 3000 else 30)
            n_src = a[2].number_input("엣지전류 source 상한 (0=전체)", 0, 5000,
                                      0 if G.number_of_nodes() < R.LARGE_REGION_NODES
                                      else R.LARGE_REGION_N_SRC)

        ctx = _ctx(graph_key, h_min, float(cost), wmode, int(iters), float(tlimit), int(n_src))
        ctx.cfg.delta = float(delta)
        cur = ctx.current

        # 예산 격자
        dflt = R.default_budgets(ctx, 16)
        b = st.columns([1, 1, 1, 1.4])
        bmin = b[0].number_input("예산 하한 (억)", 0.1, 10_000.0, round(dflt[0] / 1e8, 2), 0.5)
        bmax = b[1].number_input("예산 상한 (억)", 0.2, 10_000.0, round(dflt[-1] / 1e8, 2), 0.5)
        npts = b[2].number_input("격자 수", 3, 60, 16)
        b[3].caption(
            f"후보 {len(ctx.cand):,}개 · 전면 시공비 {ctx.full_cost/1e8:,.1f}억 · "
            f"현행 골목 열선 {cur['length_m']:,.0f} m = **{cur['spent_won']/1e8:.2f}억** "
            f"(격자에 현행·현행×1.3 포함)")
        grid = [round(float(x), -6) for x in np.linspace(bmin * 1e8, bmax * 1e8, int(npts))]
        if cur["spent_won"] > 0:   # 현행·현행+30% 는 반올림 없이 정확히 넣는다
            grid += [x for x in (cur["spent_won"], 1.3 * cur["spent_won"])
                     if bmin * 1e8 <= x <= bmax * 1e8]
        grid = sorted(set(grid))

        sweep_key = (graph_key, h_min, cost, wmode, iters, tlimit, n_src, float(delta), tuple(grid))
        cached = st.session_state.get("sweep_key") == sweep_key
        auto = len(ctx.cand) <= 1500
        run = st.button("예산 스윕 실행", type="primary", disabled=cached) or (auto and not cached)
        if run:
            bar = st.progress(0.0, text="MILP 풀이 중...")
            df = R.sweep(ctx, grid, progress=lambda f, t: bar.progress(f, text=t))
            bar.empty()
            st.session_state["sweep_key"], st.session_state["sweep_df"] = sweep_key, df
        if st.session_state.get("sweep_key") != sweep_key:
            st.info("설정을 바꿨습니다. **예산 스윕 실행**을 누르면 예산별 해를 계산합니다.")
            st.stop()
        sw = st.session_state["sweep_df"]

        # 예산 슬라이더
        opts = list(sw._B)
        init = min(opts, key=lambda x: abs(x - cur["spent_won"])) if cur["spent_won"] else opts[len(opts) // 2]
        B = st.select_slider(
            "예산 — 드래그해서 예산별 배치를 본다", options=opts, value=init,
            format_func=lambda x: f"{x/1e8:,.2f}억"
                                  + (" (현행)" if abs(x - cur['spent_won']) < 1 else
                                     " (현행+30%)" if abs(x - 1.3 * cur['spent_won']) < 1 else ""))
        res = R.solve_budget(ctx, B)
        res0 = R.solve_budget(ctx, B, 0.0)

        mcol, scol = st.columns([1.75, 1])
        with mcol:
            sel, bnd = res["selected"], res["boundaries"]
            exp_by_down = {}
            for i, j in bnd:
                exp_by_down[j] = exp_by_down.get(j, 0) + ctx.risk[j] * ctx.flow[j]
            emax = max(exp_by_down.values(), default=1) or 1
            show_cur = st.toggle("현행 열선 함께 표시", value=True)
            layers = [
                _dong_layer(polys, gu, dongs),
                _path_layer(_paths(G, geom, list(G.nodes()), C_BG, 1), "bg"),
            ]
            if show_cur:
                hn = [n for n in G.nodes() if G.nodes[n].get("heating_installed_precise")]
                layers.append(_path_layer(_paths(G, geom, hn, C_HEAT, 4,
                                                 lambda n: {"tag": "[현행 열선]"}), "heat"))
            layers.append(_path_layer(_paths(
                G, geom, sel, C_SEL, 5,
                lambda n: {"tag": f"[선정] 기대방지 {ctx.risk[n]*ctx.flow[n]:.1f}건"}), "sel"))
            if exp_by_down:
                pts = pd.DataFrame([{"lon": G.nodes[j]["lon"], "lat": G.nodes[j]["lat"],
                                     "e": e, "a": int(90 + 160 * e / emax),
                                     "r": 3 + 7 * (e / emax) ** 0.5}
                                    for j, e in exp_by_down.items()])
                pts["c"] = pts.a.apply(lambda a: C_BND + [a])
                layers.append(pdk.Layer("ScatterplotLayer", data=pts, id="rb",
                                        get_position="[lon, lat]", get_radius="r",
                                        radius_units='"pixels"', get_fill_color="c",
                                        stroked=True, get_line_color=[255, 255, 255],
                                        line_width_min_pixels=1, pickable=False))
            st.pydeck_chart(_deck(G, layers), use_container_width=True)
            st.caption("주황 = 선정 구간 · 파랑 = 현행 열선 · 빨간 원 = 잔여 경계의 하류(색이 진하고 클수록 하류 노출이 큼)")

        with scol:
            st.markdown(f"##### 예산 {B/1e8:,.2f}억 · δ={delta:g}")
            m1, m2 = st.columns(2)
            m1.metric("선정 구간", f"{res['n']}곳", f"{res['length_m']:,.0f} m", delta_color="off")
            m2.metric("집행액", f"{res['spent_won']/1e8:,.2f}억",
                      f"잔액 {(B-res['spent_won'])/1e8:,.2f}억", delta_color="off")
            m1.metric("경사위험 방지 (기대 미끄러짐)", f"{res['benefit']:,.1f}건",
                      f"현행 대비 {res['benefit']/cur['benefit']:.1f}배" if cur["benefit"] else None,
                      delta_color="off")
            m2.metric("위험 커버율", f"{res['coverage_risk_pct']:.1f}%",
                      f"연장 커버 {res['coverage_len_pct']:.1f}%", delta_color="off")
            m1.metric("위험경계", f"{res['boundary']}곳",
                      f"{res['boundary']-res0['boundary']:+d} vs δ=0", delta_color="inverse")
            m2.metric("하류 전이위험", f"{res['exposure']:,.1f}",
                      f"{100*(res['exposure']/res0['exposure']-1):+.0f}% vs δ=0" if res0["exposure"] else None,
                      delta_color="inverse")
            loss = 100 * (1 - res["benefit"] / res0["benefit"]) if res0["benefit"] else 0
            st.caption(f"δ=0 대비 편익 {loss:.1f}% 양보 · 건당 {res['won_per_slip']/1e4:,.0f}만원"
                       if res["won_per_slip"] else "")
            cmp = pd.DataFrame([
                {"": "현행 실제", "구간": cur["n"], "편익": round(cur["benefit"], 1),
                 "경계": cur["boundary"], "전이위험": round(cur["exposure"], 1)},
                {"": "위험×통행 (δ=0)", "구간": res0["n"], "편익": round(res0["benefit"], 1),
                 "경계": res0["boundary"], "전이위험": round(res0["exposure"], 1)},
                {"": f"제안 (δ={delta:g})", "구간": res["n"], "편익": round(res["benefit"], 1),
                 "경계": res["boundary"], "전이위험": round(res["exposure"], 1)},
            ])
            st.dataframe(cmp, hide_index=True, use_container_width=True)
            if not res["converged"] and wmode == "current":
                st.caption(f"MSA {iters}회 안에 고정점 미수렴 — 마지막 해를 표시")

        # 예산 곡선
        st.markdown("##### 예산별 결과")
        long = pd.concat([
            sw.assign(모형=f"제안 δ={delta:g}")[["예산억", "모형", "경사위험방지", "위험경계", "하류전이위험"]],
            sw[["예산억", "δ0_경사위험방지", "δ0_위험경계", "δ0_하류전이위험"]]
              .rename(columns={"δ0_경사위험방지": "경사위험방지", "δ0_위험경계": "위험경계",
                               "δ0_하류전이위험": "하류전이위험"})
              .assign(모형="δ=0")[["예산억", "모형", "경사위험방지", "위험경계", "하류전이위험"]],
        ])
        rule = alt.Chart(pd.DataFrame({"x": [B / 1e8]})).mark_rule(
            color="#2B3348", strokeDash=[4, 3]).encode(x="x:Q")
        colr = alt.Color("모형:N", scale=alt.Scale(domain=[f"제안 δ={delta:g}", "δ=0"],
                                                   range=["#E0674A", "#9AA5B4"]),
                         legend=alt.Legend(orient="top", title=None))
        ch = []
        for col, ttl in [("경사위험방지", "경사위험 방지 (건)"), ("위험경계", "위험경계 (곳)"),
                         ("하류전이위험", "하류 전이위험")]:
            base = alt.Chart(long).encode(
                x=alt.X("예산억:Q", title="예산 (억)"), y=alt.Y(f"{col}:Q", title=ttl), color=colr,
                tooltip=["모형", "예산억", col])
            ch.append((base.mark_line(strokeWidth=2) + base.mark_point(size=30, filled=True) + rule)
                      .properties(height=220))
        cc = st.columns(3)
        for c_, chart in zip(cc, ch):
            c_.altair_chart(chart, use_container_width=True)
        with st.expander("예산별 표"):
            st.dataframe(sw.drop(columns=["_B"]), hide_index=True, use_container_width=True)

        with st.expander("같은 예산, 선정 기준별 비교 (발표자료 10쪽)"):
            st.dataframe(R.strategy_compare(ctx, B), hide_index=True, use_container_width=True)
            st.caption("경사·통행량 기준은 해당 값이 큰 순으로 예산이 찰 때까지 담는 그리디.")

        sel_df = pd.DataFrame([{
            "link_id": str(G.nodes[n].get("link_id") or n), "dong": G.nodes[n].get("dong"),
            "length_m": G.nodes[n].get("length_m"), "slope_pct": ctx.slope[n],
            "risk": round(ctx.risk[n], 3), "flow": ctx.flow[n],
            "expected_slips": round(ctx.risk[n] * ctx.flow[n], 2),
            "lon": G.nodes[n]["lon"], "lat": G.nodes[n]["lat"],
        } for n in res["selected"]]).sort_values("expected_slips", ascending=False)
        with st.expander(f"선정 구간 {len(sel_df)}곳"):
            st.dataframe(sel_df, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ 선정 구간 CSV", sel_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"selection_{gu}_B{B/1e8:.1f}억_d{delta:g}.csv", mime="text/csv")
