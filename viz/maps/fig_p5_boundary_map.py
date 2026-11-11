"""p5 — 재결빙 경계 지도 (현행 열선이 만드는 경계, 성북구 4개동).

메시지: 경계는 추상이 아니다. 지금 깔린 열선 34구간이 경계쌍 17개(상류 14 / 하류 12)를 만들고,
       그 하류가 떠안는 위험이 현행 편익의 74.1%에 이른다.

4개동 전체 한 장 — 현행 열선과 그 하류 경계쌍.
경계쌍은 하류 지점에 **붉은 점 + 반투명 붉은 원**으로 찍는다. 원 면적이 하류 노출이다.

실행: cd streamlit_app && python3 viz/maps/fig_p5_boundary_map.py
산출: viz/out/p5_boundary_map.png
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


def _draw(ax, LineCollection, base, dong, geo, sel, b, lw=1.0, expo_max=None):
    # 도로는 한 가지 색으로 통합한다 — 이 그림의 주제는 '열선과 그 하류 경계쌍'
    # 이지 도로 분류가 아니다. 후보/비후보를 나누면 눈이 분류를 먼저 읽는다.
    ax.add_collection(LineCollection(base, colors=T.UNPROTECTED,
                                     linewidths=0.75 * lw, alpha=0.62, zorder=1))
    for ring in dong:
        ax.plot(ring[:, 0], ring[:, 1], color=T.GU_LINE, lw=0.9,
                ls=(0, (5, 3)), zorder=2, alpha=0.85)

    def lc(ids, **kw):
        segs = [np.column_stack(geo[i]) for i in ids if i in geo]
        if segs:
            ax.add_collection(LineCollection(segs, **kw))

    lc(sel.loc[sel.현행열선, "link_id"], colors=T.CURRENT,
       linewidths=2.6 * lw, zorder=5)

    # 붉은 반투명 원 — 경계쌍이 하류에 남기는 노출을 면적으로 읽힌다.
    # 두 패널의 크기 기준을 맞추려고 전체 최댓값(expo_max)으로 정규화한다.
    expo = b.하류노출.to_numpy(float)
    emax = float(expo_max if expo_max else (expo.max() if len(expo) else 1.0))
    ax.scatter(b.하류lon, b.하류lat,
               s=(240 + 820 * expo / max(emax, 1e-9)) * (1 + (lw - 1) * 0.7),
               marker="o", color=T.BOUNDARY, alpha=0.20, linewidths=0,
               zorder=6)

    # 원 가운데를 찍는 붉은 점 — 경계쌍의 정확한 위치
    ax.scatter(b.하류lon, b.하류lat, s=44 * (1 + (lw - 1) * 0.5), marker="o",
               color=T.BOUNDARY, edgecolor="white", linewidth=0.5, zorder=8)


def main() -> None:
    plt = T.use_matplotlib_theme()
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    OUT.mkdir(exist_ok=True)

    sel = pd.read_csv(VIZ / "p12_selection.csv")
    sel["link_id"] = sel.link_id.astype(str)
    bd = pd.read_csv(VIZ / "p12_boundary.csv")
    b = bd[bd.전략 == "현행"].copy()
    geo = _geom()

    # 창은 '현행 열선 + 경계쌍'이 차지하는 범위에 맞춘다. 4개동 전체로 잡으면
    # 서남쪽 빈 들판까지 들어와 정작 봐야 할 열선·경계쌍이 작게 그려진다.
    h = sel[sel.현행열선]
    lon = pd.concat([h.lon, b.상류lon, b.하류lon]).dropna()
    lat = pd.concat([h.lat, b.상류lat, b.하류lat]).dropna()
    mx, my = 180 / M_LON, 180 / M_LAT          # 사방 180 m 여백
    x0, x1 = lon.min() - mx, lon.max() + mx
    y0, y1 = lat.min() - my, lat.max() + my
    base = _basemap((x0, x1, y0, y1))
    dong = _dong_rings()
    print(f"현행 경계쌍 {len(b)}개 · 배경 도로 {len(base):,}개")

    fig, a = plt.subplots(figsize=(7.4, 6.4))

    emax = float(b.하류노출.max())
    _draw(a, LineCollection, base, dong, geo, sel, b, expo_max=emax)
    a.set_xlim(x0, x1); a.set_ylim(y0, y1)
    a.set_title("지금 깔린 열선 34구간이 만드는 경계쌍 17개 (상류 14 / 하류 12)",
                fontsize=12.2, color=T.ACCENT, fontweight="bold", pad=10)
    bar = 500 / M_LON
    a.plot([x0 + 0.0025, x0 + 0.0025 + bar], [y0 + 0.0018] * 2,
           color=T.TEXT, lw=2.4, solid_capstyle="butt")
    a.text(x0 + 0.0025 + bar / 2, y0 + 0.0026, "500 m", ha="center",
           fontsize=8.6, color=T.TEXT)

    a.set_xticks([]); a.set_yticks([])
    a.set_aspect(M_LAT / M_LON)
    for sp in a.spines.values():
        sp.set_color(T.GU_LINE); sp.set_linewidth(0.8)

    handles = [
        Line2D([], [], color=T.CURRENT, lw=2.8, label="현행 열선"),
        Line2D([], [], color=T.UNPROTECTED, lw=2.4, label="도로"),
        Line2D([], [], color=T.BOUNDARY, lw=0, marker="o", ms=6,
               mec="white", mew=0.8, label="현행 열선의 경계쌍 (하류 지점)"),
        Line2D([], [], color=T.BOUNDARY, lw=0, marker="o", ms=15, alpha=0.20,
               mec="none", label="원 크기 = 하류 노출"),
    ]
    leg = fig.legend(handles=handles, loc="center left", ncol=1, frameon=False,
                     fontsize=10.4, bbox_to_anchor=(0.625, 0.5),
                     labelspacing=1.05, handlelength=2.4)

    fig.text(0.5, -0.105,
             T.caption("성북구 4개동",
                       "현행 열선 34구간 · 경계쌍 17개(상류 14 / 하류 12)",
                       "경계쌍 = 보호 구간의 낙차 2 m 이상 하류가 미보호",
                       "p12_boundary.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.145,
             "현행 열선의 기대방지는 233.7인데 그 경계가 하류에 남기는 노출은 173.3 — "
             "**편익의 74.1%**가 경계 너머로 새어 나간다.".replace("**", ""),
             ha="center", fontsize=9, color=T.BOUNDARY)
    fig.tight_layout()
    T.save_both(fig, OUT / "p5_boundary_map.png", hide_in_bare=[leg])


if __name__ == "__main__":
    main()
