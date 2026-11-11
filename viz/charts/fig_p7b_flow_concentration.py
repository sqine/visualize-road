"""p7b — 유동인구 집중도 (Canva 7쪽 오른쪽 「유동인구 · 피해 인원」).

주장
----
**통행은 골목에 고루 퍼져 있지 않다.** 4개동 후보 847구간을 통행량 순으로
세우면 상위 10%가 전체 유동인구의 **48.9%**를 가진다. 지니계수 0.658.

그래서 위험(risk)만으로 줄을 세우면 사람이 없는 급경사 막다른 골목을 고르고,
통행(flow)만 보면 평지 큰길을 고른다. **곱해야** 둘 다 반영된다 —
`기대 미끄러짐ᵢ = riskᵢ × flowᵢ`.

왼쪽 칸(`p7a_risk_curve`)과 짝이다. 왼쪽이 risk, 오른쪽이 flow.

실행: cd streamlit_app && python3 viz/charts/fig_p7b_flow_concentration.py
산출: viz/out/p7b_flow_concentration.png (+ _bare)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402
from viz import theme as T  # noqa: E402

OUT = ROOT / "viz" / "out"
#: 강조할 상위 분위
TOP = 0.10


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    G = M.load_graph()
    cand = M.candidates(G)
    f = np.sort(np.array([(G.nodes[n].get("flow_sum") or 0.0) for n in cand]))[::-1]
    n = len(f)
    x = np.arange(1, n + 1) / n * 100
    y = np.cumsum(f) / f.sum() * 100

    k = int(round(TOP * n))
    y_top = float(y[k - 1])
    # 지니 — 오름차순 누적곡선과 대각선 사이 면적의 2배
    gini = float(1 - 2 * np.trapezoid(np.cumsum(f[::-1]) / f.sum(), dx=1 / n))

    fig, a = plt.subplots(figsize=(7.2, 5.0))

    a.plot([0, 100], [0, 100], color=T.UNPROTECTED, ls="--", lw=1.4, zorder=2)
    a.text(78, 71, "고르게 퍼져 있다면", fontsize=9, color=T.TEXT_MUTED,
           rotation=36, rotation_mode="anchor", ha="center", va="bottom")

    a.plot(x, y, color=T.PROPOSED, lw=2.8, zorder=4)
    a.fill_between(x, x, y, color=T.PROPOSED, alpha=0.11, zorder=1)

    # 상위 10% 지점 — 가로·세로 보조선과 값
    a.plot([TOP * 100, TOP * 100], [0, y_top], color=T.ACCENT, ls=":", lw=1.4,
           zorder=3)
    a.plot([0, TOP * 100], [y_top, y_top], color=T.ACCENT, ls=":", lw=1.4,
           zorder=3)
    a.scatter([TOP * 100], [y_top], s=70, color=T.ACCENT, edgecolor="white",
              linewidth=1.6, zorder=6)
    a.annotate(f"상위 10%가\n통행의 {y_top:.0f}%",
               (TOP * 100, y_top), textcoords="offset points",
               xytext=(14, -6), fontsize=11.5, color=T.ACCENT,
               fontweight="bold", linespacing=1.4, va="top")

    # 한글은 mathtext 안에서 깨진다 — 기대 미끄러짐은 E_i 로 두고 아래 줄에서 푼다.
    a.text(0.975, 0.055, r"$E_i = \mathrm{risk}_i \times \mathrm{flow}_i$",
           transform=a.transAxes, ha="right", va="bottom",
           fontsize=15.0, color=T.ACCENT, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.42", fc=T.BG_BOX,
                     ec=T.GU_LINE, lw=0.8))
    a.text(0.975, 0.215,
           f"E = 구간별 기대 미끄러짐 건수 · 지니계수 {gini:.3f}",
           transform=a.transAxes, ha="right", va="bottom",
           fontsize=8.8, color=T.TEXT_MUTED)

    a.set_xlabel("통행량 상위 구간 비율 (%)", fontsize=10)
    a.set_ylabel("누적 유동인구 비중 (%)", fontsize=10)
    a.set_xlim(0, 100); a.set_ylim(0, 102)
    a.set_xticks([0, 10, 25, 50, 75, 100])
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.22, lw=0.7)
    a.set_title("통행은 몇몇 길목에 쏠려 있다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=10)

    # 출처는 Canva 슬라이드에 14pt 텍스트로 따로 올린다 (p7a 와 같은 이유).
    src = fig.text(0.5, -0.06,
                   f"자료: 서울시 10m 도로구간 추정 유동인구(B030, 2016.12) "
                   f"— 4개동 후보 골목길 {n}구간",
                   ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p7b_flow_concentration.png", hide_in_bare=[src])

    print(f"  후보 {n}구간 · 상위 10% = {y_top:.1f}% · 지니 {gini:.3f}")


if __name__ == "__main__":
    main()
