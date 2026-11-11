"""p5b — 경계쌍 지도 2패널: 서울 전역 + 성북구 4개동 확대.

p5(성북구 4개동)는 '경계가 어떻게 생기는지'를 보여 주는 확대판이고,
이 그림은 같은 현상이 **서울 전역에서 1,691쌍** 일어난다는 규모를 보여 준다.
슬라이드 좌측 불릿(열선 2,223곳 → 하류 골목 1,590곳 → 하류 미설치 1,257곳,
75.8%)이 말로만 있던 것을 지도로 받친다.

(a) 서울 전역 — 점 하나가 경계쌍 하류 한 곳. 북부·서남부 구릉지에 몰린다.
(b) 성북구 4개동 확대 — 같은 현상을 가까이서. 원 크기가 하류 노출이다.

- 회색   도로 전체
- 파랑   현행 열선
- 붉은 점 경계쌍의 하류 지점 (열선 위 · 미설치 아래, 고도차 2 m 이상)

범례는 그림에 넣지 않는다(슬라이드에서 Canva 텍스트로 단다).

실행: cd streamlit_app && python3 viz/maps/fig_p5b_seoul_boundary_map.py
산출: viz/out/p5b_seoul_boundary_map.png (+ _bare)
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

DATA = ROOT / "data"
OUT = ROOT / "viz" / "out"
M_LAT, M_LON = 111320.0, 88322.0
HL = "성북구"


DONGS = ("정릉2동", "정릉3동", "성북동", "돈암2동")


def _gu_rings(DATA):
    gu = json.load(open(DATA / "seoul_gu.geojson", encoding="utf-8"))
    out = []
    for f in gu["features"]:
        nm = next((v for v in f["properties"].values()
                   if isinstance(v, str) and v.endswith("구")), "")
        geom = f["geometry"]
        polys = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for poly in polys:
            out.append((nm, np.asarray(poly[0], float)))
    return out


def _dong_rings(DATA):
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
    OUT.mkdir(exist_ok=True)

    l = pd.read_parquet(DATA / "seoul_links.parquet")
    g = pd.read_parquet(DATA / "seoul_link_geom.parquet",
                        columns=["road_lid", "xs", "ys"])
    g["road_lid"] = g.road_lid.astype(str)
    d = l.assign(road_lid=l.road_lid.astype(str)).merge(g, on="road_lid")
    heat = d.heat.fillna(False).astype(bool)

    # 경계쌍 정의를 분모·분자가 같은 모집단에서 나오도록 다시 센다.
    #   모집단 : 상·하류가 모두 골목길이고 낙차 2 m 이상이며 상류에 열선이 있는 인접쌍
    #   경계쌍 : 그중 하류에 열선이 없는 쌍
    # (seoul_boundary_pairs.parquet 은 분자만 담고 있어 비율을 못 만든다)
    e = pd.read_parquet(DATA / "seoul_edges.parquet")
    z = l.z.to_numpy(); rt = l.road_type.to_numpy()
    ht = l.heat.fillna(False).to_numpy()
    sl = l.slope_pct.to_numpy(); dm = l.dist_m.fillna(0).to_numpy()
    ok = ~pd.isna(z) & ~pd.isna(sl) & (dm > 0)
    uu = np.concatenate([e.u.values, e.v.values])
    vv = np.concatenate([e.v.values, e.u.values])
    m = ok[uu] & ok[vv] & (z[uu] - z[vv] >= 2.0)
    uu, vv = uu[m], vv[m]
    pop = (rt[uu] == "골목길") & (rt[vv] == "골목길") & ht[uu]   # 분모
    bnd = pop & ~ht[vv]                                          # 분자
    n_pop, n_bnd = int(pop.sum()), int(bnd.sum())
    down_ids = np.unique(vv[bnd])
    golmok = l.iloc[down_ids][["lon", "lat"]]
    print(f"모집단 {n_pop:,}쌍 · 경계쌍 {n_bnd:,}쌍({n_bnd/n_pop*100:.1f}%) "
          f"· 하류 고유 {len(down_ids):,}곳")

    # (b) 확대 창 — 4개동의 현행 열선 + 경계쌍 범위
    sel = pd.read_csv(ROOT / "viz" / "data" / "p12_selection.csv")
    bd = pd.read_csv(ROOT / "viz" / "data" / "p12_boundary.csv")
    b4 = bd[bd.전략 == "현행"].copy()
    h4 = sel[sel.현행열선]
    zlon = pd.concat([h4.lon, b4.상류lon, b4.하류lon]).dropna()
    zlat = pd.concat([h4.lat, b4.상류lat, b4.하류lat]).dropna()
    mx, my = 180 / M_LON, 180 / M_LAT
    zx0, zx1 = zlon.min() - mx, zlon.max() + mx
    zy0, zy1 = zlat.min() - my, zlat.max() + my

    def segs(mask, sub=None):
        f = d[mask] if sub is None else d[mask & sub]
        return [np.column_stack((r.xs, r.ys)) for r in f.itertuples()
                if isinstance(r.xs, (list, np.ndarray))]

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 6.0),
                           gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.02})

    # ── (a) 서울 전역 ────────────────────────────────────────
    a = ax[0]
    a.add_collection(LineCollection(segs(~heat), colors=T.UNPROTECTED,
                                    linewidths=0.28, alpha=0.45, zorder=1))
    a.add_collection(LineCollection(segs(heat), colors=T.CURRENT,
                                    linewidths=1.1, zorder=3))
    for nm, r in _gu_rings(DATA):
        hot = nm == HL
        a.plot(r[:, 0], r[:, 1], color=T.ACCENT if hot else T.GU_LINE,
               lw=1.6 if hot else 0.55, zorder=6 if hot else 2,
               alpha=1.0 if hot else 0.6)
    a.scatter(golmok.lon, golmok.lat, s=5.5, marker="o", color=T.BOUNDARY,
              alpha=0.8, linewidths=0, zorder=5)
    a.set_xlim(d.lon.min() - 0.004, d.lon.max() + 0.004)
    a.set_ylim(d.lat.min() - 0.004, d.lat.max() + 0.004)
    a.set_title(f"(a) 서울 전역 — 경계쌍 {n_bnd:,}개 (하류 {len(down_ids):,}곳)",
                fontsize=12.2, color=T.ACCENT, fontweight="bold", pad=9)
    x0, y0 = a.get_xlim()[0], a.get_ylim()[0]
    bar = 5000 / M_LON
    a.plot([x0 + 0.012, x0 + 0.012 + bar], [y0 + 0.007] * 2, color=T.TEXT,
           lw=2.4, solid_capstyle="butt")
    a.text(x0 + 0.012 + bar / 2, y0 + 0.011, "5 km", ha="center",
           fontsize=9, color=T.TEXT)

    # ── (b) 성북구 4개동 확대 ───────────────────────────────
    c = ax[1]
    box = d.lon.between(zx0, zx1) & d.lat.between(zy0, zy1)
    c.add_collection(LineCollection(segs(~heat, box), colors=T.UNPROTECTED,
                                    linewidths=0.75, alpha=0.62, zorder=1))
    for ring in _dong_rings(DATA):
        c.plot(ring[:, 0], ring[:, 1], color=T.GU_LINE, lw=0.9,
               ls=(0, (5, 3)), zorder=2, alpha=0.85)
    c.add_collection(LineCollection(segs(heat, box), colors=T.CURRENT,
                                    linewidths=2.6, zorder=4))
    expo = b4.하류노출.to_numpy(float)
    c.scatter(b4.하류lon, b4.하류lat,
              s=240 + 820 * expo / max(expo.max(), 1e-9),
              marker="o", color=T.BOUNDARY, alpha=0.20, linewidths=0, zorder=5)
    c.scatter(b4.하류lon, b4.하류lat, s=44, marker="o", color=T.BOUNDARY,
              edgecolor="white", linewidth=0.5, zorder=6)
    c.set_xlim(zx0, zx1); c.set_ylim(zy0, zy1)
    c.set_title("(b) 성북구 4개동 확대 — 경계쌍 17개",
                fontsize=12.2, color=T.ACCENT, fontweight="bold", pad=9)
    bar = 500 / M_LON
    c.plot([zx0 + 0.0012, zx0 + 0.0012 + bar], [zy0 + 0.0009] * 2,
           color=T.TEXT, lw=2.2, solid_capstyle="butt")
    c.text(zx0 + 0.0012 + bar / 2, zy0 + 0.0014, "500 m", ha="center",
           fontsize=8.6, color=T.TEXT)

    for axx in ax:
        axx.set_xticks([]); axx.set_yticks([])
        axx.set_aspect(M_LAT / M_LON)
        for sp in axx.spines.values():
            sp.set_color(T.GU_LINE); sp.set_linewidth(0.8)

    fig.text(0.5, -0.04,
             T.caption("서울 전역 / 성북구 4개동",
                       f"상·하류 골목길 + 상류 열선 {n_pop:,}쌍 중 경계쌍 "
                       f"{n_bnd:,}쌍 ({n_bnd/n_pop*100:.1f}%)",
                       "경계쌍 = 낙차 2 m 이상 하류에 열선이 없는 쌍",
                       "seoul_links · seoul_edges.parquet · p12_boundary.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p5b_seoul_boundary_map.png")


if __name__ == "__main__":
    main()
