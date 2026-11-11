"""p2 보조 — 연도별 열선 설치 추이.

"많이 깔긴 했다"의 근거. 자치구별 비교는 p6 이 맡는다.

실행: cd streamlit_app && python3 viz/charts/fig_p2_install_trend.py
산출: viz/out/p2_install_trend.png (+ _bare)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "viz" / "out"


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    h = pd.read_csv(DATA / "heating_2026_normalized.csv", encoding="utf-8-sig")
    g = (h.groupby("설치연")
           .agg(건수=("설치연", "size"), 연장=("설치연장_m", "sum"))
           .reset_index())
    g = g[g.설치연 >= 2018]

    fig, b = plt.subplots(figsize=(7.6, 4.6))
    cols = [T.UNPROTECTED if y >= 2026 else T.CURRENT for y in g.설치연]
    b.bar(g.설치연, g.연장 / 1000, color=cols, width=0.68)
    for r in g.itertuples():
        b.text(r.설치연, r.연장 / 1000 + 0.9, f"{r.건수}", ha="center",
               fontsize=9, color=T.TEXT)
    if (g.설치연 >= 2026).any():
        b.text(2026, 0.6, "2026년은\n연중 집계", ha="center", va="bottom",
               fontsize=8.4, color=T.TEXT_MUTED, linespacing=1.3)

    b.set_ylabel("설치 연장 (km)", fontsize=10, labelpad=8)
    b.set_xlabel("설치 연도  (막대 위 숫자 = 건수)", fontsize=9.8)
    b.set_xticks(g.설치연)
    b.set_ylim(0, 44)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.grid(axis="y", alpha=0.22, lw=0.7)
    b.set_title("설치는 빠르게 늘었다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)
    note = b.text(0.5, 0.93,
           "2020년 29건 → 2025년 255건 (8.8배)\n누적 148,433 m (2018년 이후)",
           transform=b.transAxes, ha="center", fontsize=9.8,
           color=T.TEXT, linespacing=1.6,
           bbox=dict(boxstyle="round,pad=0.5", fc=T.BG_BOX,
                     ec=T.GU_LINE, lw=0.8))

    fig.text(0.5, -0.07,
             T.caption("서울 25개 자치구", "열선 993건",
                       "설치시기 정규화 후 집계",
                       "heating_2026_normalized.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p2_install_trend.png", hide_in_bare=[note])


if __name__ == "__main__":
    main()
