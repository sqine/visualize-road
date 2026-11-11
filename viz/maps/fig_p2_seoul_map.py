"""p2 — 서울 전역 지도: 위험군 골목길과 현행 열선.

메시지: 위험군 골목길은 서울 전역에 1,391 km 깔려 있는데, 열선은 그 위에
       점처럼 찍혀 있다. 성북구는 위험군이 가장 많은 자치구다.

실행: cd streamlit_app && python3 viz/maps/fig_p2_seoul_map.py
산출: viz/out/p2_seoul_map.png
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
HL = "성북구"


def main() -> None:
    plt = T.use_matplotlib_theme()
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    OUT.mkdir(exist_ok=True)

    l = pd.read_parquet(DATA / "seoul_links.parquet")
    g = pd.read_parquet(DATA / "seoul_link_geom.parquet",
                        columns=["road_lid", "xs", "ys"])
    g["road_lid"] = g.road_lid.astype(str)
    d = l.assign(road_lid=l.road_lid.astype(str)).merge(g, on="road_lid")

    risk = (d.road_type == "골목길") & (d.slope_pct >= T.RISK_SLOPE_PCT)
    heat = d.heat.fillna(False).astype(bool)
    print(f"링크 {len(d):,} · 위험군 골목길 {risk.sum():,} · 열선 {heat.sum():,}")

    def segs(mask):
        return [np.column_stack((r.xs, r.ys)) for r in d[mask].itertuples()
                if isinstance(r.xs, (list, np.ndarray))]

    fig, ax = plt.subplots(figsize=(11.0, 9.4))
    ax.add_collection(LineCollection(segs(~risk & ~heat), colors=T.BG_FAINT,
                                     linewidths=0.32, zorder=1))
    ax.add_collection(LineCollection(segs(risk & ~heat), colors=T.UNPROTECTED,
                                     linewidths=0.62, zorder=2))
    ax.add_collection(LineCollection(segs(heat), colors=T.CURRENT,
                                     linewidths=1.5, zorder=4))

    # 자치구 경계 + 성북구 강조
    gu = json.load(open(DATA / "seoul_gu.geojson", encoding="utf-8"))
    for f in gu["features"]:
        nm = next((v for v in f["properties"].values()
                   if isinstance(v, str) and v.endswith("구")), "")
        geom = f["geometry"]
        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        hot = nm == HL
        for poly in polys:
            r = np.asarray(poly[0], float)
            ax.plot(r[:, 0], r[:, 1],
                    color=T.ACCENT if hot else T.GU_LINE,
                    lw=2.0 if hot else 0.6, zorder=5 if hot else 3,
                    alpha=1.0 if hot else 0.7)
            if hot:
                hl_xy = (float(r[:, 0].mean()), float(r[:, 1].mean()))

    # 성북구 설명 상자 — 지도를 가리지 않게 우상단 여백에 두고 선으로 잇는다
    hl_box = ax.annotate("성북구\n위험군 골목길 138.8 km (25개구 중 1위)\n"
                "그런데 커버리지는 8.90% (19위)",
                xy=hl_xy, xycoords="data",
                xytext=(0.985, 0.955), textcoords="axes fraction",
                ha="right", va="top", fontsize=10.6, color=T.ACCENT,
                fontweight="bold", linespacing=1.6, zorder=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="white",
                          ec=T.ACCENT, lw=1.2, alpha=0.96),
                arrowprops=dict(arrowstyle="-", lw=1.2, color=T.ACCENT,
                                shrinkB=0))

    ax.set_xlim(d.lon.min() - 0.004, d.lon.max() + 0.004)
    ax.set_ylim(d.lat.min() - 0.004, d.lat.max() + 0.012)
    ax.set_aspect(M_LAT / M_LON)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(T.GU_LINE); sp.set_linewidth(0.8)

    x0, y0 = ax.get_xlim()[0], ax.get_ylim()[0]
    bar = 5000 / M_LON
    ax.plot([x0 + 0.01, x0 + 0.01 + bar], [y0 + 0.006] * 2, color=T.TEXT,
            lw=2.6, solid_capstyle="butt")
    ax.text(x0 + 0.01 + bar / 2, y0 + 0.009, "5 km", ha="center",
            fontsize=9, color=T.TEXT)

    ax.legend(handles=[
        Line2D([], [], color=T.UNPROTECTED, lw=2.4,
               label="위험군 골목길 (경사 5.56%↑) · 1,391 km"),
        Line2D([], [], color=T.CURRENT, lw=2.4, label="현행 열선 · 148 km"),
        Line2D([], [], color=T.BG_LEGEND, lw=2.0, label="그 밖의 도로"),
    ], loc="lower right", fontsize=9.6, framealpha=0.93)

    ax.set_title("서울 전역 — 위험군 골목길 위에 열선은 점처럼 찍혀 있다",
                 fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=14)
    fig.text(0.5, 0.012,
             T.caption("서울 25개 자치구", f"링크 {len(d):,}개",
                       "위험군 = 골목길 ∩ 경사 5.56% 이상",
                       "seoul_links.parquet"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    T.save_both(fig, OUT / "p2_seoul_map.png", hide_in_bare=[hl_box])


if __name__ == "__main__":
    main()
