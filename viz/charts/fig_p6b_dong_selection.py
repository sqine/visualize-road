"""p6b — 왜 이 4개 행정동인가 (정릉2·정릉3·성북·돈암2동).

근거 세 가지
------------
(a) **구릉지다.** 20개 행정동을 "골목길 중 급경사 비율 × 위험군 골목길 수"로
    놓으면 4개동이 두 축에서 동시에 상위다. 비율만 보면 돈암1동, 규모만 보면
    장위1동이 뽑히는데 장위1동은 정릉권과 떨어져 있어 (b)가 깨진다.

(b) **이어져 있다.** 정릉2동을 허브로 하는 사슬형. 정릉3동-돈암2동은 직접
    엣지가 0개지만 하나의 연결 성분이므로 상류→하류 경로가 끊기지 않는다.
    E-I index −0.9425로 무작위 800조합 대비 상위 1.4%.

(c) **지정에서 소외됐다.** 위험군 골목길의 38.9%를 안고 결빙취약구간 지정은
    30건 중 6건(20.0%)뿐이다. 정릉2동은 207개를 안고도 0건이다.

주의: 유일 최적해가 아니다. 정릉권 1~4동이 응집도는 더 높고(상위 0.6%) 대상
4개동이 위험군 골목길은 39.6% 많다. 트레이드오프의 균형점이다.

설계: viz/docs/p6b_설계.md
실행: cd streamlit_app && python3 viz/charts/fig_p6b_dong_selection.py
산출: viz/out/p6b_dong_selection.png
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

FOUR = ["정릉2동", "정릉3동", "성북동", "돈암2동"]
#: 규모는 크나 정릉권과 떨어져 있어 탈락한 동 — (b)의 반례
FAR = "장위1동"

#: 동간 직접 연결 엣지 수 (M4 §4-10 한계4)
EDGES = {
    ("정릉2동", "돈암2동"): 36,
    ("정릉2동", "정릉3동"): 25,
    ("성북동", "돈암2동"): 13,
    ("정릉2동", "성북동"): 7,
    ("정릉3동", "성북동"): 6,
    ("정릉3동", "돈암2동"): 0,
}

#: E-I index — 4개동 집합 평가 (M4 §4-8). 음수가 응집
EI = [
    ("정릉권 1~4동", -0.9544, 0.6, False),
    ("대상 4개동", -0.9425, 1.4, True),
    ("위험규모 top4", -0.9315, 5.0, False),
    ("무작위 800조합 평균", -0.8830, 50.0, False),
]
#: 위험군 골목길 수 — 응집도 트레이드오프 상대편
N_STEEP_4DONG = 1233
N_STEEP_JEONGNEUNG = 883


def dong_centroids() -> dict[str, tuple[float, float]]:
    """행정동 경계 폴리곤 좌표 평균. 가장 큰 링만 쓴다."""
    g = json.loads((DATA / "seongbuk_dong.geojson").read_text(encoding="utf-8"))
    out = {}
    for f in g["features"]:
        name = f["properties"]["adm_nm"].split()[-1]
        geom = f["geometry"]
        if geom["type"] == "Polygon":
            ring = geom["coordinates"][0]
        else:
            ring = max(geom["coordinates"], key=lambda p: len(p[0]))[0]
        a = np.asarray(ring, float)
        out[name] = (a[:, 0].mean(), a[:, 1].mean())
    return out


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    s = pd.read_csv(DATA / "seongbuk_dong_stats.csv")
    s.columns = [c.lstrip("﻿") for c in s.columns]
    s["대상"] = s.행정동.isin(FOUR)

    f = s[s.대상]
    tot_steep = int(s.위험군골목길.sum())
    tot_ice = int(s.결빙지정.sum())
    tot_link = int(s.링크수.sum())
    share_steep = f.위험군골목길.sum() / tot_steep * 100
    share_ice = f.결빙지정.sum() / tot_ice * 100
    share_link = f.링크수.sum() / tot_link * 100

    fig = plt.figure(figsize=(15.0, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.92], wspace=0.30)

    # ── (a) 구릉지다 — 비율 × 규모 ─────────────────────────────
    a = fig.add_subplot(gs[0, 0])
    mx, my = s.위험군비율.median(), s.위험군골목길.median()
    a.axvline(mx, color=T.GU_LINE, ls=":", lw=1.0, zorder=1)
    a.axhline(my, color=T.GU_LINE, ls=":", lw=1.0, zorder=1)
    a.text(mx + 0.7, a.get_ylim()[1], "", fontsize=1)

    rest = s[~s.대상]
    size = lambda n: 24 + np.asarray(n, float) / 1270.0 * 300.0  # noqa: E731
    a.scatter(rest.위험군비율, rest.위험군골목길, s=size(rest.링크수),
              color=T.UNPROTECTED, alpha=0.60, lw=0, zorder=2)
    a.scatter(f.위험군비율, f.위험군골목길, s=size(f.링크수),
              color=T.PROPOSED, alpha=0.95, lw=1.1,
              edgecolors="white", zorder=4)

    far = s[s.행정동 == FAR].iloc[0]
    a.scatter([far.위험군비율], [far.위험군골목길], s=size(far.링크수),
              facecolors="none", edgecolors=T.BOUNDARY, lw=1.9, zorder=5)

    lab_off = {"성북동": (0, 16), "정릉3동": (-7.0, 15), "정릉2동": (2.0, 12),
               "돈암2동": (12, -3), FAR: (-13, 1)}
    lab_ha = {"돈암2동": "left", FAR: "right"}
    for _, r in pd.concat([f, far.to_frame().T]).iterrows():
        dx, dy = lab_off[r.행정동]
        col = T.BOUNDARY if r.행정동 == FAR else T.PROPOSED
        a.annotate(r.행정동, (r.위험군비율, r.위험군골목길),
                   textcoords="offset points", xytext=(dx, dy), fontsize=9.6,
                   color=col, fontweight="bold",
                   ha=lab_ha.get(r.행정동, "center"), va="center")
    a.annotate("규모는 3위인데\n정릉권과 떨어져 있다", (far.위험군비율, far.위험군골목길),
               textcoords="offset points", xytext=(-13, -22),
               fontsize=8.6, color=T.BOUNDARY, ha="right", va="top",
               linespacing=1.4)

    # 돈암1동(62.6% · 62개)은 돈암2동(62.5% · 60개)과 거의 완전히 겹친다.
    # 비율 1위가 규모로는 밀린다는 (a)의 논지가 이 둘이라 숨기지 않고 적는다.
    d1 = s[s.행정동 == "돈암1동"].iloc[0]
    a.annotate(f"돈암1동 — 비율 1위 {d1.위험군비율:.1f}%인데 {int(d1.위험군골목길)}개뿐.\n"
               "돈암2동과 거의 겹친다 (62.5% · 60개)",
               (d1.위험군비율, d1.위험군골목길), xytext=(69, -46),
               textcoords="data", fontsize=8.4, color=T.TEXT_MUTED,
               ha="right", va="top", linespacing=1.5,
               arrowprops=dict(arrowstyle="-", lw=0.9, color=T.TEXT_MUTED,
                               connectionstyle="arc3,rad=-0.25"))

    a.set_xlabel("골목길 중 위험군 비율 (%)  — 구릉지 정도", fontsize=9.8)
    a.set_ylabel("위험군 골목길 수 (개)  — 위험 규모", fontsize=9.8)
    a.set_xlim(0, 71); a.set_ylim(-118, 700)
    a.text(mx + 1.0, 672, f"중앙값 {mx:.1f}%", fontsize=8.4, color=T.TEXT_MUTED)
    a.text(70.0, my + 12, f"중앙값 {my:.0f}개", fontsize=8.4,
           color=T.TEXT_MUTED, ha="right")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.grid(alpha=0.16, lw=0.7)
    a.set_title("(a) 구릉지다 — 두 축에서 동시에 상위",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=26)
    hl_a = a.text(0.5, 1.025, "점 크기 ∝ 동 규모(링크수) · 회색 16개동",
                  transform=a.transAxes, ha="center", fontsize=8.8,
                  color=T.TEXT_MUTED)

    # ── (b) 이어져 있다 — 인접 그래프 ──────────────────────────
    b = fig.add_subplot(gs[0, 1])
    cen = dong_centroids()
    pos = {k: cen[k] for k in FOUR}
    for (u, v), w in EDGES.items():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        if w == 0:
            b.plot([x1, x2], [y1, y2], color=T.BOUNDARY, lw=1.2,
                   ls=(0, (2, 3)), alpha=0.75, zorder=2)
            b.text((x1 + x2) / 2 - 0.0012, (y1 + y2) / 2, "0",
                   fontsize=9.2, color=T.BOUNDARY, fontweight="bold",
                   ha="center", va="center",
                   bbox=dict(fc="white", ec=T.BOUNDARY, lw=0.8, pad=1.6))
            continue
        b.plot([x1, x2], [y1, y2], color=T.PROPOSED,
               lw=1.1 + w / 36.0 * 5.4, alpha=0.52, solid_capstyle="round",
               zorder=2)
        b.text((x1 + x2) / 2, (y1 + y2) / 2, str(w), fontsize=8.8,
               color=T.BOUNDARY if w >= 25 else T.TEXT_MUTED,
               fontweight="bold", ha="center", va="center",
               bbox=dict(fc="white", ec="none", pad=1.2))

    nlab = {"정릉3동": (0, 20), "정릉2동": (0, 21), "성북동": (0, -26),
            "돈암2동": (0, -26)}
    for nm, (x, y) in pos.items():
        hub = nm == "정릉2동"
        b.scatter([x], [y], s=430 if hub else 300,
                  color=T.ACCENT if hub else T.PROPOSED,
                  edgecolors="white", lw=1.8, zorder=5)
        n = int(s.loc[s.행정동 == nm, "위험군골목길"].iloc[0])
        b.text(x, y, f"{n}", fontsize=8.6, color="white", fontweight="bold",
               ha="center", va="center", zorder=6)
        dx, dy = nlab[nm]
        b.annotate(nm + ("  (허브)" if hub else ""), (x, y),
                   textcoords="offset points", xytext=(dx, dy), fontsize=9.6,
                   color=T.ACCENT if hub else T.PROPOSED,
                   fontweight="bold", ha="center")

    b.set_xlim(126.9895, 127.0175); b.set_ylim(37.5885, 37.6195)
    b.set_aspect("equal", adjustable="box")
    b.set_xticks([]); b.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        b.spines[sp].set_visible(False)
    b.set_title("(b) 하나의 망이다 — 정릉2동 허브 사슬형",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=26)
    hl_b = b.text(0.5, 1.025, "선 굵기 ∝ 동간 직접 연결 엣지 수 · 원 안 = 위험군 골목길 수",
                  transform=b.transAxes, ha="center", fontsize=8.8,
                  color=T.TEXT_MUTED)
    # 응집도 표 — 좌(라벨) / 우(수치) 두 개의 text 로 열을 맞춘다.
    # monospace 폰트에 한글 글리프가 없어 한 덩어리로는 정렬이 안 된다.
    # 응집도 표 — 축 **바깥 아래**에 둔다. 축 안에 넣으면 성북동 노드를 가린다.
    # 좌(라벨)/우(수치) 두 열로 나누는 이유: monospace 폰트에 한글 글리프가
    # 없어 한 덩어리 문자열로는 열이 맞지 않는다.
    from matplotlib.patches import FancyBboxPatch
    panel = FancyBboxPatch((0.00, -0.285), 1.00, 0.255,
                           boxstyle="round,pad=0.008,rounding_size=0.014",
                           transform=b.transAxes, fc=T.BG_BOX, ec=T.GU_LINE,
                           lw=0.7, zorder=7, clip_on=False)
    b.add_patch(panel)
    box_lab = b.text(0.022, -0.042, "E-I index  (음수 = 응집)",
                     transform=b.transAxes, fontsize=8.4, color=T.TEXT_MUTED,
                     va="top", ha="left", zorder=8, clip_on=False)
    box_r = b.text(0.978, -0.042, "무작위 800조합 대비",
                   transform=b.transAxes, fontsize=8.4, color=T.TEXT_MUTED,
                   va="top", ha="right", zorder=8, clip_on=False)
    rows_b = []
    for i, (nm, v, p, sel) in enumerate(EI):
        y = -0.094 - i * 0.046
        w = "bold" if sel else "normal"
        col = T.ACCENT if sel else T.TEXT
        tail = f"상위 {p:.1f}%" if i < len(EI) - 1 else "(sd 0.0341)"
        for x, txt, ha in ((0.022, nm + ("  ← 채택" if sel else ""), "left"),
                           (0.978, f"{v:.4f}      {tail}", "right")):
            rows_b.append(b.text(x, y, txt, transform=b.transAxes,
                                 fontsize=8.4, color=col, fontweight=w,
                                 va="center", ha=ha, zorder=8, clip_on=False))
    box_b = [panel, box_lab, box_r, *rows_b]

    # ── (c) 지정에서 소외됐다 ──────────────────────────────────
    c = fig.add_subplot(gs[0, 2])
    rows = [
        ("위험군 골목길", share_steep, f"{int(f.위험군골목길.sum()):,} / {tot_steep:,}개",
         T.PROPOSED),
        ("(참고) 링크수\n= 면적 대리", share_link,
         f"{int(f.링크수.sum()):,} / {tot_link:,}개", T.UNPROTECTED),
        ("결빙취약구간 지정", share_ice, f"{int(f.결빙지정.sum())} / {tot_ice}건",
         T.BOUNDARY),
    ]
    for i, (lab, v, sub, col) in enumerate(rows):
        c.barh(i, 100, height=0.44, color=T.BG_FAINT, zorder=1)
        c.barh(i, v, height=0.44, color=col, zorder=2)
        c.text(v + 2.4, i, f"{v:.1f}%", va="center", fontsize=13.0,
               color=col, fontweight="bold")
        c.text(2.4, i + 0.40, sub, va="center", ha="left", fontsize=8.4,
               color=T.TEXT_MUTED)
    c.set_yticks(range(len(rows)))
    c.set_yticklabels([r[0] for r in rows], fontsize=9.8, linespacing=1.45)
    c.invert_yaxis(); c.set_ylim(2.96, -0.56)
    c.set_xlim(0, 100); c.set_xticks([0, 50, 100])
    c.set_xlabel("성북구 20개 행정동 합계 중 4개동 몫 (%)", fontsize=9.8)
    for sp in ("top", "right", "left"):
        c.spines[sp].set_visible(False)
    c.annotate("", xy=(share_ice, 2.64), xytext=(share_steep, 2.64),
               arrowprops=dict(arrowstyle="-|>", lw=1.4, color=T.BOUNDARY))
    c.text((share_steep + share_ice) / 2, 2.86,
           f"위험은 {share_steep:.1f}%, 지정은 {share_ice:.1f}%",
           ha="center", fontsize=9.2, color=T.BOUNDARY, fontweight="bold")
    c.set_title("(c) 지정에서 가장 소외됐다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=26)
    hl_c = c.text(0.5, 1.025, "정릉2동은 위험군 골목길 207개(5위)를 안고도 지정 0건",
                  transform=c.transAxes, ha="center", fontsize=8.8,
                  color=T.BOUNDARY, fontweight="bold")

    fig.suptitle("정릉2·정릉3·성북·돈암2동 — 위험이 모이고, 하나로 이어지고, "
                 "지정에서 빠진 곳",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.045)
    fig.text(0.5, -0.095,
             T.caption("성북구 20개 행정동",
                       f"위험군 골목길 {tot_steep:,}개 · 지정 {tot_ice}건",
                       "위험군 = 골목길 ∩ 경사 5.56% 이상",
                       "seongbuk_dong_stats.csv · M4 §4-8"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.215,
             "유일 최적해가 아니다 — 정릉권 1~4동은 응집도가 더 높으나"
             f"(상위 0.6%) 위험군 골목길이 {N_STEEP_JEONGNEUNG:,}개로 "
             f"{(N_STEEP_4DONG / N_STEEP_JEONGNEUNG - 1) * 100:.0f}% 적다. "
             "응집도↔위험규모 트레이드오프이고 대상 4개동은 그 균형점이다.\n"
             "돈암2동은 위험군 골목길 수로는 18위(60개)다. 비율 2위(62.5%)이면서 "
             "정릉2동·성북동에 직접 연결(36·13 엣지)되어 들어왔다 — 규모 기준만으로는 뽑히지 않는다.",
             ha="center", fontsize=9, color=T.TEXT, linespacing=1.7)

    T.save_both(fig, OUT / "p6b_dong_selection.png",
                hide_in_bare=[hl_a, hl_b, hl_c])

    print(f"  4개동 위험군 골목길 몫 {share_steep:.1f}% · 지정 몫 {share_ice:.1f}% "
          f"· 링크수 몫 {share_link:.1f}% · 집중도 {share_steep / share_link:.2f}배")


if __name__ == "__main__":
    main()
