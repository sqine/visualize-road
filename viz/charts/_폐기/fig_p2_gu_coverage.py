"""p2 우측 — 자치구별 위험군 골목길 커버리지 + 연도별 설치 추이.

메시지: 많이 깔긴 했다. 그런데 위험군 대비로는 거의 비어 있다.

실행: cd streamlit_app && python3 viz/charts/fig_p2_gu_coverage.py
산출: viz/out/p2_gu_coverage.png
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

VIZ = ROOT / "viz" / "data"
DATA = ROOT / "data"
OUT = ROOT / "viz" / "out"
HL = "성북구"


def _find(name: str) -> Path:
    tgt = unicodedata.normalize("NFC", name)
    for f in VIZ.iterdir():
        if unicodedata.normalize("NFC", f.name) == tgt:
            return f
    raise FileNotFoundError(name)


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(_find("위험군_5.56pct_25gu.csv")).sort_values("커버리지")
    h = pd.read_csv(DATA / "heating_2026_normalized.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.9),
                           gridspec_kw={"width_ratios": [1.45, 1]})

    # ── 좌: 자치구별 커버리지 ────────────────────────────────
    a = ax[0]
    cols = [T.ACCENT if g == HL else T.UNPROTECTED for g in d.자치구]
    a.barh(d.자치구, d.커버리지, color=cols, height=0.72)
    med = d.커버리지.median()
    a.axvline(med, color=T.BOUNDARY, ls="--", lw=1.3, zorder=3)
    a.text(med + 0.6, 0.2, f"중위 {med:.2f}%", fontsize=9,
           color=T.BOUNDARY, fontweight="bold")
    for i, r in enumerate(d.itertuples()):
        if r.자치구 == HL:
            a.text(r.커버리지 + 0.6, i, f"{r.커버리지:.2f}%  (19위)",
                   va="center", fontsize=9.4, color=T.ACCENT, fontweight="bold")
    a.set_xlabel("위험군 골목길 대비 열선 커버리지 (%)", fontsize=10)
    a.set_xlim(0, 34)
    a.tick_params(axis="y", labelsize=9)
    for lab in a.get_yticklabels():
        if lab.get_text() == HL:
            lab.set_fontweight("bold"); lab.set_color(T.ACCENT)
    a.set_title("위험군 대비로 보면 거의 비어 있다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=10)
    a.grid(axis="x", alpha=0.22, lw=0.7)

    # ── 우: 연도별 설치 추이 ─────────────────────────────────
    b = ax[1]
    g = (h.groupby("설치연")
           .agg(건수=("설치연", "size"), 연장=("설치연장_m", "sum"))
           .reset_index())
    g = g[g.설치연 >= 2018]
    cols_y = [T.UNPROTECTED if y >= 2026 else T.CURRENT for y in g.설치연]
    b.bar(g.설치연, g.연장 / 1000, color=cols_y, width=0.68)
    if (g.설치연 >= 2026).any():
        b.text(2026, 0.6, "2026년은\n연중 집계", ha="center", va="bottom",
               fontsize=8.2, color=T.TEXT_MUTED, linespacing=1.3)
    for r in g.itertuples():
        b.text(r.설치연, r.연장 / 1000 + 0.9, f"{r.건수}", ha="center",
               fontsize=8.8, color=T.TEXT)
    b.set_ylabel("설치 연장 (km)", fontsize=10, labelpad=8)
    b.set_xlabel("설치 연도  (막대 위 숫자 = 건수)", fontsize=9.6)
    b.set_xticks(g.설치연)
    b.set_ylim(0, 44)
    b.set_title("설치는 빠르게 늘었다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=10)
    b.grid(axis="y", alpha=0.22, lw=0.7)
    b.text(0.5, 0.93,
           "2020년 29건 → 2025년 255건 (8.8배)\n누적 148,433 m (2018년 이후)",
           transform=b.transAxes, ha="center", fontsize=9.6,
           color=T.TEXT, linespacing=1.6,
           bbox=dict(boxstyle="round,pad=0.5", fc=T.BG_BOX,
                     ec=T.GU_LINE, lw=0.8))

    for axx in ax:
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)
    ax[0].spines["left"].set_visible(False)

    fig.suptitle("많이 깔았지만, 위험이 큰 곳일수록 덜 덮였다",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.0)
    fig.text(0.5, -0.045,
             T.caption("서울 25개 자치구", "위험군 골목길 1,391 km",
                       "커버리지 = 신고 열선 연장 ÷ 위험군 연장",
                       "위험군_5.56pct_25gu.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.085,
             "투입량과 커버리지는 무상관(ρ=0.119, p=0.570)인 반면, "
             "위험군 연장과 커버리지는 ρ=-0.719 (p=0.0001)",
             ha="center", fontsize=8.6, color=T.BOUNDARY)
    fig.tight_layout()
    T.save_both(fig, OUT / "p2_gu_coverage.png")


if __name__ == "__main__":
    main()
