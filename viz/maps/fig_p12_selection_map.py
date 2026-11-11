"""p12 — 최종 배치 지도 (성북구 4개동, δ=0 vs 제안 δ=2).

메시지: 같은 예산·같은 연장인데, 경계를 벌점으로 넣으면 배치가 이어 붙는다.
       잔여 경계(빨간 원)가 38개 → 19개로 준다.

실행: cd streamlit_app && python3 viz/maps/fig_p12_selection_map.py
산출: viz/out/p12_selection_map.png
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
    sm = pd.read_csv(VIZ / "p12_summary.csv")
    geo = _geom()

    ids = {c: sel.loc[sel[c], "link_id"].tolist()
           for c in ["현행열선", "후보", "선정_delta0", "선정_제안"]}
    allid = sel.link_id.tolist()

    panels = [("delta0", "(a) 경계를 무시하면 — δ = 0", "선정_delta0"),
              ("제안", "(b) 경계를 벌점으로 넣으면 — δ = 2", "선정_제안")]

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
        _lc(plt, a, geo, ids[col], colors=T.PROPOSED,
            linewidths=3.6, zorder=6)

        b = bd[bd.전략 == tag]
        if len(b):
            s = 22 + 130 * (b.하류노출 / b.하류노출.max())
            a.scatter(b.하류lon, b.하류lat, s=s, facecolor="none",
                      edgecolor=T.BOUNDARY, linewidth=1.15, alpha=0.9,
                      zorder=7)

        r = sm[sm.전략.str.startswith("경계 미고려" if tag == "delta0" else "제안")].iloc[0]
        a.text(0.5, 1.032,
               f"잔여 경계 {int(r.경계수)}개  ·  경계노출 {r.경계노출:.0f}  "
               f"({r.노출편익비:.1f}% of 편익)",
               transform=a.transAxes, ha="center", fontsize=10.6,
               color=T.BOUNDARY if tag == "delta0" else T.ACCENT,
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
        Line2D([], [], color=T.PROPOSED, lw=3.4, label="제안 선정"),
        Line2D([], [], color=T.UNPROTECTED, lw=2.4, label="후보 골목길 (미보호)"),
        Line2D([], [], color=T.BG_LEGEND, lw=2.0, label="그 밖의 도로"),
        Line2D([], [], color=T.GU_LINE, lw=1.2, ls=(0, (5, 3)), label="행정동 경계"),
        Line2D([], [], color=T.BOUNDARY, lw=0, marker="o", mfc="none",
               mew=1.5, ms=9, label="잔여 경계 (원 크기 = 하류 노출)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=9.6, bbox_to_anchor=(0.5, -0.035))

    fig.suptitle("같은 예산 2,100 m — 경계를 벌점에 넣으면 배치가 이어 붙는다",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.0)
    fig.text(0.5, -0.082,
             T.caption("성북구 4개동(정릉2·정릉3·성북·돈암2)",
                       "후보 골목길 1,190구간", "예산 18.53억 = 현행 투입액",
                       "p12_selection.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.118,
             "선정 100 → 97구간, 기대방지 3,185.9 → 3,060.3 (−3.9%). "
             "그 대가로 경계노출 432.6 → 183.7 (−57.5%). 교체된 구간은 13개뿐이다.",
             ha="center", fontsize=9, color=T.BOUNDARY)
    fig.tight_layout()
    T.save_both(fig, OUT / "p12_selection_map.png")


if __name__ == "__main__":
    main()
