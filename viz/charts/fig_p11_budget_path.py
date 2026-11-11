"""p11 — 예산을 늘리면 무엇이 달라지는가 (δ=2 고정). 두 장으로 분리.

왜 δ를 고정하나
---------------
예산선마다의 전환점 δ는 0.02~9.87로 튄다. δ를 같이 움직이면 "예산 효과"와
"δ 효과"가 섞인다. δ=2 로 고정해야 예산만의 효과가 분리된다.
(δ=2 의 근거는 같은 쪽 히트맵·파레토 그림이 맡는다)

    p11a_budget_path.png  예산 vs 기대방지 — 경계수는 점 위 숫자로
    p11b_marginal.png     한계효율 — 증가 구간 없이 단조 감소

경계수를 막대로 그리면 가로 폭이 예산 간격을 가린다. 값 자체가 중요하지
크기 비교가 중요하지 않으므로 숫자 라벨이 낫다.

실행: cd streamlit_app && python3 viz/charts/fig_p11_budget_path.py
산출: viz/out/p11a_budget_path.png · p11b_marginal.png (각 _bare)
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

CUR = 18.53          # 현행 예산(억)
PLUS = 23.17         # +25% 지점


def path_chart(plt, d) -> None:
    """예산 vs 기대방지. 경계쌍 수는 점 위 숫자로 적는다."""
    fig, a = plt.subplots(figsize=(7.8, 5.0))
    a.plot(d.예산억, d.기대방지, color=T.PROPOSED, lw=2.8, marker="o",
           ms=6.5, mfc="white", mew=1.8, zorder=4)
    for x in d.itertuples():
        a.annotate(f"{int(x.경계수)}", (x.예산억, x.기대방지),
                   textcoords="offset points", xytext=(0, 13), ha="center",
                   fontsize=9.4, color=T.BOUNDARY, fontweight="bold")
        if x.배수 in (0.25, 1.0, 2.0, 4.0):
            a.annotate(f"{x.위험군커버리지:.1f}%", (x.예산억, x.기대방지),
                       textcoords="offset points", xytext=(0, -20),
                       ha="center", fontsize=8.8, color=T.TEXT_MUTED)

    a.axvline(CUR, color=T.CURRENT, ls="--", lw=1.5, zorder=2)
    a.text(CUR - 0.9, 4760, "현행 예산\n18.53억", ha="right", fontsize=9.2,
           color=T.CURRENT, fontweight="bold", linespacing=1.4)
    r = d[np.isclose(d.예산억, PLUS)].iloc[0]
    a.scatter([r.예산억], [r.기대방지], s=230, marker="*", color=T.ACCENT,
              edgecolor="white", linewidth=1.3, zorder=6)
    a.annotate("+25% 지점\n경계쌍 19 → 18", xy=(r.예산억, r.기대방지),
               textcoords="offset points", xytext=(34, -46), fontsize=9.4,
               color=T.ACCENT, fontweight="bold", linespacing=1.5,
               arrowprops=dict(arrowstyle="-|>", lw=1.2, color=T.ACCENT))

    a.set_xlabel("예산 (억원)", fontsize=10)
    a.set_ylabel("기대방지 (risk × flow)", fontsize=10)
    a.set_ylim(1200, 5100)
    a.set_xlim(0, 79)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.2, lw=0.7)
    a.set_title("예산을 늘리면 더 막지만, 경계쌍도 같이 는다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)
    a.text(0.985, 0.045,
           "점 위 숫자 = 잔여 경계쌍 수   ·   점 아래 = 위험군 커버리지",
           transform=a.transAxes, ha="right", fontsize=8.8,
           color=T.TEXT_MUTED)

    fig.text(0.5, -0.065,
             T.caption("성북구 4개동", "예산 0.25~4배 (4.6~74.1억)",
                       "δ = 2 고정 · 커버리지는 위험군 골목 22.5 km 대비",
                       "p11_budget_path_delta2.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.115,
             "노출/편익비는 예산과 무관하게 4.9~6.0%로 평평하다 — "
             "δ=2가 예산이 바뀌어도 경계쌍을 같은 수준으로 눌러 준다.",
             ha="center", fontsize=9, color=T.BOUNDARY)
    fig.tight_layout()
    T.save_both(fig, OUT / "p11a_budget_path.png")


def marginal_chart(plt) -> None:
    """한계효율 — 격자 99점. 증가 구간 없이 단조 감소."""
    fig, b = plt.subplots(figsize=(7.8, 5.0))
    g = pd.read_csv(VIZ / "p11_grid_hires.csv")
    dl = g.delta.unique()
    d2 = dl[np.abs(dl - 2.0).argmin()]
    q = g[g.delta == d2].sort_values("예산억").reset_index(drop=True)
    q["한계효율"] = q.기대방지.diff() / q.예산억.diff()
    m = q.dropna(subset=["한계효율"])
    sm = m.한계효율.rolling(7, center=True, min_periods=3).median()

    b.scatter(m.예산억, m.한계효율, s=15, color=T.UNPROTECTED, zorder=2,
              label="예산 격자 99점")
    b.plot(m.예산억, sm, color=T.PROPOSED, lw=2.8, zorder=4,
           label="평활 추세 (이동중앙값)")
    b.axvline(CUR, color=T.CURRENT, ls="--", lw=1.4, zorder=3)
    b.text(CUR + 1.4, 408, "현행 예산", fontsize=9.2, color=T.CURRENT,
           fontweight="bold")

    b.set_xscale("log")
    b.set_xlabel("예산 (억원, 로그)", fontsize=10)
    b.set_ylabel("억원당 추가 기대방지", fontsize=10)
    b.set_ylim(0, 460)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.grid(alpha=0.2, lw=0.7)
    b.legend(fontsize=9, framealpha=0.93, loc="lower left",
             bbox_to_anchor=(0.0, 0.2))
    b.set_title("쓸수록 효율은 떨어지기만 한다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)
    note = b.text(0.03, 0.02,
           "올라갔다 내려오는 구간이 없다 — risk × flow 상위부터\n"
           "담으므로 처음부터 수확체감이다. 272 → 11까지 단조 감소.",
           transform=b.transAxes, ha="left", fontsize=8.8, color=T.TEXT,
           linespacing=1.5,
           bbox=dict(boxstyle="round,pad=0.42", fc=T.BG_BOX,
                     ec=T.GU_LINE, lw=0.8))

    fig.text(0.5, -0.065,
             T.caption("성북구 4개동", "예산 격자 99점", "δ = 2 고정",
                       "p11_grid_hires.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p11b_marginal.png", hide_in_bare=[note])


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(VIZ / "p11_budget_path_delta2.csv")
    path_chart(plt, d)
    marginal_chart(plt)


if __name__ == "__main__":
    main()
