"""p3 — 결빙취약구간 지정의 도로유형 분포 (100% 누적막대 2줄).

메시지: 지정은 대형도로에 4.0배 과대 집중된다.

실행: cd streamlit_app && python3 viz/charts/fig_p3_icing_share.py
산출: viz/out/p3_icing_share.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

VIZ = ROOT / "viz" / "data"
OUT = ROOT / "viz" / "out"

ORDER = ["골목길", "애매", "대형도로", "unknown"]
LABEL = {"골목길": "골목길", "애매": "애매", "대형도로": "대형도로",
         "unknown": "미분류"}
COLOR = {"골목길": "#8FBF9F", "애매": "#E8D08A",
         "대형도로": T.CURRENT, "unknown": "#D5DAE1"}


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(VIZ / "p3_icing_roadtype_share.csv")
    d = d[d.도로유형 != "합계"].set_index("도로유형")

    rows = [("서울 전체 도로", "서울전체_비율", "서울전체_구간"),
            ("결빙취약구간 지정", "지정구간_비율", "지정구간_건")]

    fig, ax = plt.subplots(figsize=(10.6, 3.1))
    for yi, (name, pcol, ncol) in enumerate(rows):
        left = 0.0
        small: list = []
        for k in ORDER:
            v = float(d.at[k, pcol])
            ax.barh(yi, v, left=left, height=0.52, color=COLOR[k],
                    edgecolor="white", linewidth=1.2,
                    label=LABEL[k] if yi == 0 else None)
            big = k == "대형도로"
            if v >= 11:
                ax.text(left + v / 2, yi, f"{LABEL[k]}\n{v:.1f}%",
                        ha="center", va="center", fontsize=9.2,
                        color="white" if big else T.TEXT,
                        fontweight="bold" if big else "normal",
                        linespacing=1.3)
            elif v >= 1.5:
                small.append((left + v / 2, k, v))
            left += v
        # 좁은 구간 라벨 — 막대 바깥에 단을 나눠 배치 (겹침 방지)
        sgn = 1 if yi == 1 else -1
        small.sort(key=lambda t: t[0])
        for j, (cx, k, v) in enumerate(small):
            tier = 0.34 + 0.26 * (len(small) - 1 - j)
            y = yi + sgn * tier
            ax.annotate(f"{LABEL[k]} {v:.1f}%", xy=(cx, yi + sgn * 0.26),
                        xytext=(cx, y), ha="center",
                        va="bottom" if sgn > 0 else "top", fontsize=8.8,
                        color=T.CURRENT if k == "대형도로" else T.TEXT_MUTED,
                        fontweight="bold" if k == "대형도로" else "normal",
                        arrowprops=dict(arrowstyle="-", lw=0.8,
                                        color=T.GU_LINE, shrinkA=1, shrinkB=0))
        n = int(d[ncol].sum())
        ax.text(-1.5, yi, f"{name}\n(n={n:,})", ha="right", va="center",
                fontsize=10, color=T.TEXT, fontweight="bold", linespacing=1.4)

    ax.set_xlim(0, 100); ax.set_ylim(-1.15, 1.72)
    ax.set_yticks([]); ax.set_xlabel("구간 수 비율 (%)", fontsize=9.6)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("결빙취약구간 지정은 대형도로에 쏠려 있다",
                 fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=24)
    ax.text(0.5, 1.10, "대형도로 7.0% → 28.2%   ·   4.0배 과대 대표",
            transform=ax.transAxes, ha="center", fontsize=11,
            color=T.BOUNDARY, fontweight="bold")
    fig.text(0.5, -0.09,
             T.caption("서울 전역", "지정 347구간 / 전체 168,976구간",
                       f"χ² = 320.3, {T.pval(4.03e-69)}",
                       "p3_icing_roadtype_share.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p3_icing_share.png")


if __name__ == "__main__":
    main()
