"""p12 — 예산 시나리오 배치 지도 (성북구 4개동).

δ는 고정 상수가 아니라 **예산마다 다시 고르는 파라미터**다. 예산선이 주어지면
편익–경계노출 평면의 전환점이 δ를 정해 준다(p11).

| 시나리오 | 예산 | δ | 선정 | 커버리지 | 기대방지 | 경계쌍 | 현행 대비 |
|---|---|---|---|---|---|---|---|
| 현행 실제 | 18.53억 | — | 34 | 8.3%* | 233.7 | 17 | 1.0 |
| 1 재배치 | 18.53억 (같은 돈) | 4.0 | 98 | 7.1% | 2,959.1 | **13** | **12.7배** |
| 2 효율 최대 | 24.10억 (+30%) | 5.0 | 112 | 8.0% | 3,160.6 | **11** | **13.5배** |

\* 현행 커버리지 8.3%는 위험군 밖 열선까지 분자에 든 느슨한 값이다.

δ는 시나리오마다 따로 고른다. 두 시나리오 모두 **경계쌍이 현행(17개)보다 적다**.
시나리오 2는 1보다 모든 지표에서 우월하다 — 커버리지·편익은 높고 경계쌍은 적다.

메시지: 같은 돈으로도 경계를 줄이면서 12.7배를 막는다. 30%를 더 쓰면 13.5배.

실행: cd streamlit_app && python3 viz/maps/fig_p12_scenario_map.py
산출: viz/out/p12_scenario_map.png
"""

from __future__ import annotations

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

#: 등장방형 근사 — 서울 위도에서 경도 1도의 실거리
M_LAT, M_LON = 111320.0, 88322.0


def _geom() -> dict:
    """road_graph 의 link_id → (xs, ys).

    주의: road_graph.json 의 `link_id` 는 geojson 의 feature 인덱스가 아니라
    **road_lid** 다. 인덱스로 조인하면 선이 엉뚱한 곳에 그려진다(중점 대비
    중위 2.9 km 어긋남). road_lid 로 조인하면 중위 2.7 m 로 맞는다.
    """
    g = pd.read_parquet(DATA / "seoul_link_geom.parquet",
                        columns=["road_lid", "xs_full", "ys_full"])
    return {str(r.road_lid): (r.xs_full, r.ys_full) for r in g.itertuples()
            if isinstance(r.xs_full, (list, np.ndarray))}


def _basemap(box) -> list:
    """배경 도로망 — 4개동 범위 안의 서울 전역 링크.

    road_graph 는 분석 후보(골목길 1,190구간)만 담아서 그것만 그리면 선이
    공중에 떠 보인다. 같은 범위의 모든 도로를 연회색으로 깔아야 지도가 된다.
    """
    x0, x1, y0, y1 = box
    l = pd.read_parquet(DATA / "seoul_links.parquet",
                        columns=["link_id", "road_lid", "lon", "lat"])
    m = l[(l.lon.between(x0, x1)) & (l.lat.between(y0, y1))]
    g = pd.read_parquet(DATA / "seoul_link_geom.parquet",
                        columns=["road_lid", "xs", "ys"])
    g["road_lid"] = g.road_lid.astype(str)
    j = m.assign(road_lid=m.road_lid.astype(str)).merge(g, on="road_lid")
    return [np.column_stack((r.xs, r.ys)) for r in j.itertuples()
            if isinstance(r.xs, (list, np.ndarray))]


#: 분석 대상 4개동
DONGS = ("정릉2동", "정릉3동", "성북동", "돈암2동")


def _dong_rings() -> list:
    """4개동 경계 — 분석 범위를 눈으로 알 수 있게 점선으로 깐다.

    파일은 성북구 20개 동을 모두 담고 있어, 전부 그리면 선만 많아진다.
    """
    import json
    fc = json.load(open(DATA / "seongbuk_dong.geojson", encoding="utf-8"))
    out = []
    for f in fc["features"]:
        if not f["properties"]["adm_nm"].endswith(DONGS):
            continue
        g = f["geometry"]
        polys = ([g["coordinates"]] if g["type"] == "Polygon"
                 else g["coordinates"])
        for poly in polys:
            for ring in poly:
                out.append(np.asarray(ring, float))
    return out


def _lc(plt, ax, geo, ids, **kw):
    from matplotlib.collections import LineCollection
    segs = [np.column_stack(geo[i]) for i in ids if i in geo]
    if segs:
        ax.add_collection(LineCollection(segs, **kw))
    return len(segs)


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    sel = pd.read_csv(VIZ / "p12_selection.csv")
    sel["link_id"] = sel.link_id.astype(str)
    bd = pd.read_csv(VIZ / "p12_boundary.csv")
    sc = pd.read_csv(VIZ / "p12_scenarios.csv")
    geo = _geom()

    ids = {c: sel.loc[sel[c], "link_id"].tolist()
           for c in ["현행열선", "후보", "시나리오1", "시나리오2"]}
    allid = sel.link_id.tolist()

    panels = [("시나리오1", "(a) 재배치 — 현행과 같은 예산 18.53억 · δ = 4",
               "시나리오1"),
              ("시나리오2", "(b) 효율 최대 — +30% 24.10억 · δ = 5",
               "시나리오2")]

    lon, lat = sel.lon.dropna(), sel.lat.dropna()
    px, py = 0.004, 0.003
    x0, x1 = lon.min() - px, lon.max() + px
    y0, y1 = lat.min() - py, lat.max() + py
    base = _basemap((x0, x1, y0, y1))
    dong = _dong_rings()
    print(f"배경 도로 {len(base):,}개 · 행정동 경계 {len(dong)}개 링")

    fig, ax = plt.subplots(1, 2, figsize=(13.4, 6.9))
    from matplotlib.collections import LineCollection
    for a, (tag, title, col) in zip(ax, panels):
        a.add_collection(LineCollection(base, colors=T.BG_FAINT,
                                        linewidths=0.65, zorder=1))
        for ring in dong:
            a.plot(ring[:, 0], ring[:, 1], color=T.GU_LINE, lw=0.9,
                   ls=(0, (5, 3)), zorder=2, alpha=0.85)
        _lc(plt, a, geo, ids["후보"], colors=T.UNPROTECTED,
            linewidths=1.7, zorder=3)
        _lc(plt, a, geo, ids["현행열선"], colors=T.CURRENT,
            linewidths=2.3, zorder=4)
        # 선정 구간은 배경에 묻히지 않게 흰 외곽선을 둘러 띄운다
        import matplotlib.patheffects as pe
        _lc(plt, a, geo, ids[col], colors="white", linewidths=6.4, zorder=5.5)
        _lc(plt, a, geo, ids[col], colors=T.PROPOSED,
            linewidths=4.0, zorder=6)

        # 현행 열선이 만드는 경계쌍 17개 — 비교 기준선으로 함께 깐다
        bc = bd[bd.전략 == "현행"]
        if len(bc):
            a.scatter(bc.하류lon, bc.하류lat, s=62, marker="v",
                      color=T.boundary_colors(bc.하류노출, "current"),
                      edgecolor="white", linewidth=0.8, alpha=0.9, zorder=7)

        b = bd[bd.전략 == tag]
        if len(b):
            s = 160 + 460 * (b.하류노출 / bd.하류노출.max())
            # 선정(코랄)과 섞이지 않게 더 어두운 적갈색 + 흰 테두리
            a.scatter(b.하류lon, b.하류lat, s=s, marker="*",
                      color=T.BOUNDARY_DARK, edgecolor="white", linewidth=1.3,
                      zorder=8)

        r = sc[sc.시나리오 == tag].iloc[0]
        a.text(0.5, 1.032,
               f"선정 {int(r.선정)}구간 · 커버리지 {r.커버리지:.1f}% · "
               f"기대방지 {r.기대방지:,.0f} · 잔여 경계쌍 {int(r.경계수)}개",
               transform=a.transAxes, ha="center", fontsize=10.0,
               color=T.ACCENT if tag == "시나리오2" else T.TEXT,
               fontweight="bold")
        a.set_title(title, fontsize=T.TS_PANEL, color=T.ACCENT,
                    fontweight="bold", pad=34)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(T.GU_LINE); sp.set_linewidth(0.8)

    # 공통 범위 · 축척
    for a in ax:
        a.set_xlim(x0, x1); a.set_ylim(y0, y1)
        a.set_aspect(M_LAT / M_LON)
    bar = 500 / M_LON
    ax[0].plot([x0 + 0.0025, x0 + 0.0025 + bar], [y0 + 0.0018] * 2,
               color=T.TEXT, lw=2.4, solid_capstyle="butt")
    ax[0].text(x0 + 0.0025 + bar / 2, y0 + 0.0026, "500 m", ha="center",
               fontsize=8.6, color=T.TEXT)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=T.CURRENT, lw=2.6, label="현행 열선 (34구간)"),
        Line2D([], [], color=T.PROPOSED, lw=3.4, label="(a) 재배치 — 같은 예산 선정"),
        Line2D([], [], color=T.ADDED, lw=3.4, label="(b) 효율 최대 — +30% 선정"),
        Line2D([], [], color=T.UNPROTECTED, lw=2.4, label="후보 골목길 (미보호)"),
        Line2D([], [], color=T.BG_LEGEND, lw=2.0, label="그 밖의 도로"),
        Line2D([], [], color=T.GU_LINE, lw=1.2, ls=(0, (5, 3)), label="행정동 경계"),
        Line2D([], [], color=T.BOUNDARY_CURRENT, lw=0, marker="v", ms=9,
               mec="white", mew=0.8, label="현행 열선의 경계쌍 (17개)"),
        Line2D([], [], color=T.BOUNDARY_DARK, lw=0, marker="*", ms=16,
               mec="white", mew=1.2,
               label="각 배치가 남긴 경계쌍 ((a) 13개 / (b) 11개)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.6, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("같은 돈으로도 경계쌍을 줄이면서 12배를 막는다",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.0)
    fig.text(0.5, -0.072,
             T.caption("성북구 4개동(정릉2·정릉3·성북·돈암2)",
                       "후보 골목길 1,190구간",
                       "δ는 시나리오마다 따로 고름",
                       "p12_scenarios.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.125,
             "현행 열선은 34구간으로 경계쌍 17개 · 기대방지 233.7이다. "
             "두 시나리오 모두 경계쌍을 현행보다 줄이면서 12배 이상을 막는다.\n"
             "(a) 같은 돈 · 경계쌍 17 → 13 · 기대방지 12.7배    "
             "(b) +30% · 경계쌍 17 → 11 · 기대방지 13.5배 · 커버리지 8.0%",
             ha="center", fontsize=9, color=T.BOUNDARY)
    fig.tight_layout()
    T.save_both(fig, OUT / "p12_scenario_map.png")


if __name__ == "__main__":
    main()
