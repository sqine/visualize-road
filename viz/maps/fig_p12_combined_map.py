"""p12 — 최종 배치 지도 (1패널, 두 시나리오를 색으로 겹쳐 표시).

2패널로 나란히 두면 "어디가 달라졌는지"가 눈에 안 들어온다. 한 장에 겹쳐
그리고 **추가분만 다른 색**으로 빼면 차이가 바로 보인다.

| 레이어 | 색 | 내용 |
|---|---|---|
| 공통 구간 | PROPOSED 코랄 | 97구간 — 어느 예산이든 깐다 |
| 추가 구간 | ADDED 청록 | 15구간 — 예산 +30% 일 때 더 깐다 |
| 현행 열선 | CURRENT 파랑 | 34구간 |
| 현행 경계쌍 | ▼ 파랑 | 17개 |
| 공통 구간이 남긴 경계쌍 | ★ 적색 채움 | 10개 |
| 증액으로 해소되는 경계쌍 | ☆ 적색 속빈 | 3개 |
| 추가 구간이 새로 만든 경계쌍 | ★ 청록 채움 | 1개 |

경계쌍 표시 색은 그 경계쌍을 만든 **선 색과 짝**이다. 범례에는 색 이름을 쓰지 않고
무엇인지로 적는다.

기호 크기는 고정하고 **하류 노출은 색 농도**로 읽는다. 크기로 나타내면 큰
표식이 주변 도로를 가려서 정작 배치가 안 보인다.

수치 표는 지도에 얹지 않는다. PPT 에서 따로 배치할 수 있게
`표_p12_시나리오.tsv` 와 `표_p12_시나리오.png` 로 분리했다
(`scripts/build_scenario_table.py`).

시나리오 1에만 있고 2에서 빠지는 구간이 1개 있다. 97+1 = 98이 시나리오 1,
97+15 = 112가 시나리오 2다. 1개는 지도에 따로 그리지 않고 범례 각주로 둔다.

실행: cd streamlit_app && python3 viz/maps/fig_p12_combined_map.py
산출: viz/out/p12_combined_map.png · p12_combined_map_bare.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

VIZ = ROOT / "viz" / "data"
DATA = ROOT / "data"
OUT = ROOT / "viz" / "out"

M_LAT, M_LON = 111320.0, 88322.0
DONGS = ("정릉2동", "정릉3동", "성북동", "돈암2동")


def _geom() -> dict:
    """road_graph 의 link_id(= road_lid) → (xs, ys)."""
    g = pd.read_parquet(DATA / "seoul_link_geom.parquet",
                        columns=["road_lid", "xs_full", "ys_full"])
    return {str(r.road_lid): (r.xs_full, r.ys_full) for r in g.itertuples()
            if isinstance(r.xs_full, (list, np.ndarray))}


def _basemap(box) -> list:
    x0, x1, y0, y1 = box
    l = pd.read_parquet(DATA / "seoul_links.parquet",
                        columns=["road_lid", "lon", "lat"])
    m = l[l.lon.between(x0, x1) & l.lat.between(y0, y1)]
    g = pd.read_parquet(DATA / "seoul_link_geom.parquet",
                        columns=["road_lid", "xs", "ys"])
    j = m.assign(road_lid=m.road_lid.astype(str)).merge(
        g.assign(road_lid=g.road_lid.astype(str)), on="road_lid")
    return [np.column_stack((r.xs, r.ys)) for r in j.itertuples()
            if isinstance(r.xs, (list, np.ndarray))]


def _dong_rings() -> list:
    fc = json.load(open(DATA / "seongbuk_dong.geojson", encoding="utf-8"))
    out = []
    for f in fc["features"]:
        if not f["properties"]["adm_nm"].endswith(DONGS):
            continue
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for ring in poly:
                out.append(np.asarray(ring, float))
    return out


def main() -> None:
    plt = T.use_matplotlib_theme()
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    OUT.mkdir(exist_ok=True)

    sel = pd.read_csv(VIZ / "p12_selection.csv")
    sel["link_id"] = sel.link_id.astype(str)
    bd = pd.read_csv(VIZ / "p12_boundary.csv")
    geo = _geom()

    s1, s2 = sel.시나리오1, sel.시나리오2
    ids = {
        "공통": sel.loc[s1 & s2, "link_id"].tolist(),
        "추가": sel.loc[~s1 & s2, "link_id"].tolist(),
        "현행열선": sel.loc[sel.현행열선, "link_id"].tolist(),
        "후보": sel.loc[sel.후보, "link_id"].tolist(),
    }
    print({k: len(v) for k, v in ids.items()})

    lon, lat = sel.lon.dropna(), sel.lat.dropna()
    x0, x1 = lon.min() - 0.004, lon.max() + 0.004
    y0, y1 = lat.min() - 0.003, lat.max() + 0.003
    base = _basemap((x0, x1, y0, y1))
    dong = _dong_rings()

    fig, a = plt.subplots(figsize=(13.0, 10.0))

    a.add_collection(LineCollection(base, colors=T.BG_FAINT,
                                    linewidths=0.8, zorder=1))
    for ring in dong:
        a.plot(ring[:, 0], ring[:, 1], color=T.GU_LINE, lw=1.0,
               ls=(0, (5, 3)), zorder=2, alpha=0.85)

    def lc(key, color, lw, z, halo=True):
        segs = [np.column_stack(geo[i]) for i in ids[key] if i in geo]
        if not segs:
            return
        if halo:
            a.add_collection(LineCollection(segs, colors="white",
                                            linewidths=lw + 2.6, zorder=z - 0.5))
        a.add_collection(LineCollection(segs, colors=color,
                                        linewidths=lw, zorder=z))

    lc("후보", T.UNPROTECTED, 2.0, 3, halo=False)
    lc("현행열선", T.CURRENT, 3.0, 4)
    lc("공통", T.PROPOSED, 4.6, 6)
    lc("추가", T.ADDED, 4.6, 6)

    # ── 경계쌍 세 종류 ─────────────────────────────────────────
    #   ▼ 파랑   현행 열선이 만드는 경계쌍 17개(상류 14 / 하류 12)
    #   ★ 채움   제안에 그대로 남는 경계쌍 11개 (시나리오 2 기준)
    #   ☆ 속빈   재배치에만 남고 +30% 하면 해소되는 경계쌍 3개
    bc = bd[bd.전략 == "현행"]
    key = lambda d: set(zip(d.상류.astype(str), d.하류.astype(str)))
    b1, b2 = bd[bd.전략 == "시나리오1"], bd[bd.전략 == "시나리오2"]
    solved = b1[~b1.apply(lambda r: (str(r.상류), str(r.하류)) in key(b2),
                          axis=1)]

    # 크기는 고정하고 하류 노출은 **색 농도**로 읽는다 (큰 표식이 도로를 가림)
    a.scatter(bc.하류lon, bc.하류lat, s=62, marker="v",
              color=T.boundary_colors(bc.하류노출, "current"),
              edgecolor="white", linewidth=0.8, alpha=0.9, zorder=7)
    a.scatter(solved.하류lon, solved.하류lat, s=300, marker="*",
              facecolor="white", edgecolor=T.BOUNDARY_DARK, linewidth=1.8,
              zorder=8)
    k1 = key(b1)
    both = b2[b2.apply(lambda r: (str(r.상류), str(r.하류)) in k1, axis=1)]
    newly = b2[~b2.apply(lambda r: (str(r.상류), str(r.하류)) in k1, axis=1)]
    a.scatter(both.하류lon, both.하류lat, s=300, marker="*",
              color=T.boundary_colors(both.하류노출, "proposed"),
              edgecolor="white", linewidth=1.2, zorder=9)
    a.scatter(newly.하류lon, newly.하류lat, s=300, marker="*",
              color=T.boundary_colors(newly.하류노출, "added"),
              edgecolor="white", linewidth=1.2, zorder=9)

    a.set_xlim(x0, x1); a.set_ylim(y0, y1)
    a.set_aspect(M_LAT / M_LON)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(T.GU_LINE); sp.set_linewidth(0.8)

    bar = 500 / M_LON
    a.plot([x0 + 0.0025, x0 + 0.0025 + bar], [y0 + 0.0018] * 2,
           color=T.TEXT, lw=2.8, solid_capstyle="butt")
    a.text(x0 + 0.0025 + bar / 2, y0 + 0.0027, "500 m", ha="center",
           fontsize=9.4, color=T.TEXT)

    # 지도 안 수치 상자
    handles = [
        Line2D([], [], color=T.PROPOSED, lw=4.4,
               label=f"공통 구간 — 두 시나리오 모두 선정 ({len(ids['공통'])})"),
        Line2D([], [], color=T.ADDED, lw=4.4,
               label=f"추가 구간 — 예산 +30% 일 때만 선정 ({len(ids['추가'])})"),
        Line2D([], [], color=T.CURRENT, lw=3.0, label="현행 열선 (34구간)"),
        Line2D([], [], color=T.UNPROTECTED, lw=2.4, label="후보 골목길 (미선정)"),
        Line2D([], [], color=T.BG_LEGEND, lw=2.0, label="그 밖의 도로"),
        Line2D([], [], color=T.GU_LINE, lw=1.2, ls=(0, (5, 3)), label="행정동 경계"),
        Line2D([], [], color="#4E7FC4", lw=0, marker="v", ms=9,
               mec="white", mew=0.8, label="현행 열선의 경계쌍 (17개 · 진할수록 노출 큼)"),
        Line2D([], [], color="#B9553F", lw=0, marker="*", ms=17,
               mec="white", mew=1.2, label="공통 구간이 남긴 경계쌍 (10개 · 진할수록 노출 큼)"),
        Line2D([], [], color="white", lw=0, marker="*", ms=17,
               mec=T.BOUNDARY_DARK, mew=1.6,
               label="증액으로 해소되는 경계쌍 (3개)"),
        Line2D([], [], color=T.BOUNDARY_ADDED, lw=0, marker="*", ms=17,
               mec="white", mew=1.2, label="추가 구간이 새로 만든 경계쌍 (1개)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3,
                     frameon=False, fontsize=10.2, bbox_to_anchor=(0.5, 0.012))

    fig.suptitle("같은 돈으로도 경계쌍을 줄이면서 12배를 막는다",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=0.975)
    fig.text(0.5, -0.022,
             T.caption("성북구 4개동(정릉2·정릉3·성북·돈암2)",
                       "후보 골목길 1,190구간",
                       "재배치 18.53억 δ=4 · 효율 최대 24.10억 δ=5",
                       "p12_scenarios.csv"),
             ha="center", fontsize=8.8, color=T.TEXT_MUTED)
    fig.text(0.5, -0.052,
             "공통 구간은 예산과 무관하게 깔아야 할 곳, 추가 구간은 예산을 30% 더 "
             "썼을 때 비로소 들어오는 곳이다. 현행 경계쌍 17개는 제안 배치와 하나도 "
             "겹치지 않는다.\n"
             "재배치 경계쌍 13개 중 10개가 남고 3개는 증액으로 해소되며, 증액 시 "
             "1곳이 새로 생겨 11개가 된다. 기호 색이 진할수록 하류 노출이 크다.",
             ha="center", fontsize=9, color=T.TEXT_MUTED)
    T.save_both(fig, OUT / "p12_combined_map.png")

    # 범례까지 뺀 '지도만' 판본 — 범례를 슬라이드 텍스트로 따로 둘 때 쓴다
    leg.set_visible(False)
    plain = OUT / "p12_combined_map_plain.png"
    fig.savefig(plain, dpi=T.FIG_DPI, bbox_inches="tight")
    print(f"저장: {plain.name}")


if __name__ == "__main__":
    main()
