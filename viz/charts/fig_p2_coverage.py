"""p2 우측 — 골목길 커버리지가 왜 부족한가 (서울 전역).

두 방향에서 같은 결론으로 조인다.

(a) **쓴 돈이 어디로 갔나** — 열선 117.6 km 중 위험군 골목에 간 것은 48.0%.
    나머지는 완경사 골목(34.7%)·대형도로·미분류다.
(b) **수요는 얼마나 메워졌나** — 위험군 골목 1,395.6 km 중 보호는 4.05%.

경사 구간별 커버리지(2.5 → 6.1%)는 쓰지 않는다. "급한 곳은 그래도 더
깔았다"로 읽혀 메시지와 반대 방향이 되고, 위험군 안의 우선순위 문제는
p7 이 따로 다룬다.

자치구별 비교는 p6 이 맡는다.

실행: cd streamlit_app && python3 viz/charts/fig_p2_coverage.py
산출: viz/out/p2_coverage.png (+ _bare)
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
OUT = ROOT / "viz" / "out"

TOTAL_KM, COVER_KM = 1395.6, 56.5

#: (a) 열선 행선지 — build_p2_destination.py 산출
ORDER = ["골목길 · 위험군(5.56%↑)", "골목길 · 완경사", "미분류",
         "대형도로", "애매", "골목길 · 경사 불명"]
COLOR = {"골목길 · 위험군(5.56%↑)": T.CURRENT,
         "골목길 · 완경사": "#9DBDEB",
         "미분류": "#D5DAE1",
         "대형도로": T.UNPROTECTED,
         "애매": "#C6CDD6",
         "골목길 · 경사 불명": "#E4E8ED"}
SHORT = {"골목길 · 위험군(5.56%↑)": "위험군 골목길",
         "골목길 · 완경사": "완경사 골목길",
         "미분류": "미분류", "대형도로": "대형도로",
         "애매": "애매", "골목길 · 경사 불명": "경사 불명"}


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(VIZ / "p2_heat_destination.csv").set_index("구분")
    tot = d.km.sum()

    fig, ax = plt.subplots(2, 1, figsize=(11.6, 5.4),
                           gridspec_kw={"height_ratios": [1, 1],
                                        "hspace": 0.85})

    # ── (a) 쓴 돈이 어디로 갔나 ───────────────────────────────
    a = ax[0]
    left = 0.0
    for k in ORDER:
        v = 100 * float(d.at[k, "km"]) / tot
        a.barh(0, v, left=left, height=0.5, color=COLOR[k],
               edgecolor="white", linewidth=1.3)
        if v >= 6:
            a.text(left + v / 2, 0, f"{SHORT[k]}\n{v:.1f}%", ha="center",
                   va="center", fontsize=9.6, linespacing=1.35,
                   color="white" if k == ORDER[0] else T.TEXT,
                   fontweight="bold" if k == ORDER[0] else "normal")
        elif v >= 3:
            a.annotate(f"{SHORT[k]} {v:.1f}%", xy=(left + v / 2, 0.26),
                       xytext=(left + v / 2, 0.52), ha="center",
                       va="bottom", fontsize=8.8, color=T.TEXT_MUTED,
                       arrowprops=dict(arrowstyle="-", lw=0.8,
                                       color=T.GU_LINE))
        left += v
    a.set_xlim(0, 100); a.set_ylim(-0.6, 0.95)
    a.set_yticks([]); a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel(f"깔아 놓은 열선 {tot:.1f} km 의 행선지 (%)", fontsize=9.8)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.set_title("(a) 쓴 돈의 절반만 위험군 골목으로 갔다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=14)

    # ── (b) 수요는 얼마나 메워졌나 ────────────────────────────
    b = ax[1]
    b.barh(0, TOTAL_KM, height=0.5, color=T.UNPROTECTED, zorder=2)
    b.barh(0, COVER_KM, height=0.5, color=T.CURRENT, zorder=3)
    b.text(TOTAL_KM * 0.55, 0, f"비보호 {TOTAL_KM - COVER_KM:,.1f} km",
           ha="center", va="center", fontsize=11.5, color=T.TEXT,
           fontweight="bold")
    b.annotate(f"보호 {COVER_KM:.1f} km  ({100*COVER_KM/TOTAL_KM:.1f}%)",
               xy=(COVER_KM, 0.22), xytext=(TOTAL_KM * 0.1, 0.56),
               fontsize=11, color=T.CURRENT, fontweight="bold",
               arrowprops=dict(arrowstyle="-", lw=1.1, color=T.CURRENT))
    b.set_xlim(0, TOTAL_KM * 1.02); b.set_ylim(-0.6, 0.95)
    b.set_yticks([]); b.set_xticks([])
    for s in b.spines.values():
        s.set_visible(False)
    b.set_xlabel(f"서울 위험군 골목길 {TOTAL_KM:,.1f} km", fontsize=9.8)
    b.set_title("(b) 그 돈으로 메운 수요는 4.1%",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=14)

    fig.suptitle("많이 깔았다 — 그런데 절반은 엉뚱한 곳에, 나머지로는 4%를 메웠다",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.04)
    fig.text(0.5, -0.06,
             T.caption("서울 25개 자치구",
                       "링크에 매칭된 열선 2,288구간 117.6 km",
                       "위험군 = 골목길 ∩ 경사 5.56% 이상",
                       "p2_heat_destination.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.115,
             "신고 연장은 148.4 km인데 도로망에 붙은 것은 117.6 km다(라우팅 실패분 제외). "
             "위험 가중으로 보면 더 낮아진다 — 위험군 골목의 risk×연장 중 보호된 몫은 4.4%.",
             ha="center", fontsize=9, color=T.BOUNDARY)
    T.save_both(fig, OUT / "p2_coverage.png")


if __name__ == "__main__":
    main()
