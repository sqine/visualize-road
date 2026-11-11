"""p7 — 현행 열선 진단 (성북구 4개동 + 서울 전역).

주장 세 가지
------------
1. **많이 깔긴 했다. 그런데 턱없이 모자라다.**
   성북구는 열선 연장 12.3 km로 25개구 1위인데, 4개동 위험군 골목
   52.2 km 중 2.7 km(5.2%)만 덮는다. 서울 전역은 4.05%, 성북구는 5.20%다.

2. **위험군 안에서 우선순위를 못 매긴다.**
   위험군을 risk×flow 5분위로 나누면 커버리지가 8.7 / 8.9 / 3.6 / 14.3 / 5.0%
   — 추세가 없다(χ² = 3.44, p = 0.488). 가장 급한 상위 20%가 오히려 5.0%다.
   risk×flow 상위 34개 중 현행 열선은 **0개**다.
   그 결과 같은 연장으로 얻을 수 있었던 편익의 **8.0%**만 얻는다.

3. **깔아 놓은 것의 절반이 경계를 만든다.**
   서울 열선 2,283구간 중 52.8%가 바로 아래에 미보호 하류를 둔다.

주의: "위험군을 못 고른다"는 뜻이 아니다. 열선이 경사 골목을 잡는 비율은
기대의 2.31배(z = 11.88)로 잘 집는다. 못 하는 것은 **그 안에서의 우선순위**다.

실행: cd streamlit_app && python3 viz/charts/fig_apx_current_diagnosis.py
산출: viz/out/apx_current_diagnosis.png
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
HL = "성북구"

#: 서울 전역 경계 보유율(%) — build_current_diagnosis.py 출력
BOUNDARY_ALL = 52.8


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    q = pd.read_csv(VIZ / "p7_quintile_coverage.csv")
    bd = pd.read_csv(VIZ / "p7_boundary_rate_by_gu.csv")

    fig = plt.figure(figsize=(14.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.18, 1.0], wspace=0.34)

    # ── (a) 많이 깔았지만 턱없이 모자라다 ─────────────────────
    a = fig.add_subplot(gs[0, 0])
    # 커버리지는 전역 데이터(경사는 전 구간 매칭)로, 달성률은 모형이 푸는
    # 범위(유동인구 측정 구간)로 잰다. 분모가 다르므로 라벨에 적는다.
    rows = [("위험군 골목\n커버리지", 5.2, "2.7 / 52.2 km · 전역 데이터", T.CURRENT),
            ("경사만 기준으로\n잘 골랐다면", 58.9, "1,182 / 2,007 · 측정 구간", T.ACCENT),
            ("통행까지 보면", 8.0, "233.7 / 2,929.1 · 측정 구간", T.BOUNDARY)]
    for i, (lab, v, sub, col) in enumerate(rows):
        a.barh(i, 100, height=0.40, color=T.BG_FAINT, zorder=1)
        a.barh(i, v, height=0.40, color=col, zorder=2)
        a.text(v + 3, i, f"{v:.1f}%", va="center", fontsize=13.5,
               color=col, fontweight="bold")
        a.text(2.5, i + 0.37, sub, va="center", ha="left", fontsize=8.4,
               color=T.TEXT_MUTED)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r[0] for r in rows], fontsize=10, linespacing=1.5)
    a.invert_yaxis(); a.set_ylim(2.62, -0.55)
    a.set_xlim(0, 100); a.set_xticks([0, 50, 100])
    a.set_xlabel("달성 비율 (%)  — 달성률은 같은 연장 2,101 m 기준", fontsize=9.6)
    for sp in ("top", "right", "left"):
        a.spines[sp].set_visible(False)
    a.set_title("(a) 많이 깔았지만 턱없이 모자라다",
                fontsize=12, color=T.ACCENT, fontweight="bold", pad=30)
    hl = a.text(0.5, 1.02, "성북구 열선 연장 12.3 km — 25개구 1위",
           transform=a.transAxes, ha="center", fontsize=9.6,
           color=T.ACCENT, fontweight="bold")

    # ── (b) 위험군 안에서 우선순위가 없다 ─────────────────────
    b = fig.add_subplot(gs[0, 1])
    A = q[q.기준 == "위험군 안 · 경사 단독"].reset_index(drop=True)
    B = q[q.기준 == "위험군 안 · risk×flow"].reset_index(drop=True)
    x = np.arange(5)
    w = 0.38
    b.bar(x - w / 2 - 0.012, A.커버리지_연장, width=w, color=T.UNPROTECTED,
          label=f"경사 단독으로 줄 세워도  (p = {A.p.iloc[0]:.3f})")
    b.bar(x + w / 2 + 0.012, B.커버리지_연장, width=w, color=T.CURRENT,
          label=f"risk × flow 로 줄 세워도  (p = {B.p.iloc[0]:.3f})")
    for i in range(5):
        b.text(i - w / 2 - 0.012, A.커버리지_연장[i] + 0.35,
               f"{A.커버리지_연장[i]:.1f}", ha="center", fontsize=8.6,
               color=T.TEXT_MUTED)
        b.text(i + w / 2 + 0.012, B.커버리지_연장[i] + 0.35,
               f"{B.커버리지_연장[i]:.1f}", ha="center", fontsize=8.6,
               color=T.CURRENT, fontweight="bold")
    mean = 8.3
    b.axhline(mean, color=T.BOUNDARY, ls="--", lw=1.3, zorder=3)
    b.text(2.0, mean + 0.45, f"위험군 평균 {mean:.1f}%",
           ha="center", fontsize=9, color=T.BOUNDARY, fontweight="bold")
    b.annotate("우선순위가 반영됐다면\n여기가 가장 높아야 한다",
               xy=(4.2, B.커버리지_연장.iloc[-1] + 0.4), xytext=(3.76, 12.2),
               fontsize=9.4, color=T.BOUNDARY, ha="center", linespacing=1.5,
               arrowprops=dict(arrowstyle="-|>", lw=1.3, color=T.BOUNDARY,
                               connectionstyle="arc3,rad=-0.3"))
    b.set_xticks(x)
    b.set_xticklabels(["Q1\n(가장 덜 급함)", "Q2", "Q3", "Q4",
                       "Q5\n(가장 급함)"], fontsize=9.4, linespacing=1.4)
    b.set_ylabel("열선 커버리지 (연장 기준, %)", fontsize=9.8)
    b.set_ylim(0, 18.5)
    b.set_xlabel("위험군 골목길 443구간을 5등분", fontsize=9.8)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    b.grid(axis="y", alpha=0.2, lw=0.7)
    b.legend(loc="upper left", fontsize=8.6, framealpha=0.93)
    b.set_title("(b) 급한 곳부터 깔지 않는다",
                fontsize=12, color=T.ACCENT, fontweight="bold", pad=30)

    # ── (c) 깔린 것의 절반이 경계를 만든다 ────────────────────
    c = fig.add_subplot(gs[0, 2])
    d = bd.sort_values("경계보유율")
    c.barh(d.자치구, d.경계보유율, height=0.74,
           color=[T.ACCENT if x == HL else T.BOUNDARY for x in d.자치구])
    for bar, nm in zip(c.patches, d.자치구):
        bar.set_alpha(1.0 if nm == HL else 0.48)
    c.axvline(BOUNDARY_ALL, color=T.TEXT, ls="--", lw=1.3, zorder=3)
    c.text(BOUNDARY_ALL + 1.6, 0.3, f"서울 전역\n{BOUNDARY_ALL:.1f}%",
           fontsize=8.8, color=T.TEXT, fontweight="bold", linespacing=1.3)
    r = d[d.자치구 == HL].iloc[0]
    c.text(r.경계보유율 + 1.6, list(d.자치구).index(HL), f"{r.경계보유율:.1f}%",
           va="center", fontsize=9.6, color=T.ACCENT, fontweight="bold")
    c.set_xlabel("열선 구간 중 바로 아래가 미보호인 비율 (%)", fontsize=9.8)
    c.set_xlim(0, 100)
    c.tick_params(axis="y", labelsize=8.2)
    for lab in c.get_yticklabels():
        if lab.get_text() == HL:
            lab.set_fontweight("bold"); lab.set_color(T.ACCENT)
    for s in ("top", "right", "left"):
        c.spines[s].set_visible(False)
    c.grid(axis="x", alpha=0.2, lw=0.7)
    c.set_title("(c) 깔린 것의 절반이 경계쌍을 만든다",
                fontsize=12, color=T.ACCENT, fontweight="bold", pad=30)

    fig.suptitle("성북구는 가장 많이 깔았다 — 그런데 어디에 깔지는 고르지 못했다",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.04)
    fig.text(0.5, -0.095,
             T.caption("(a) 전역 데이터 4개동 위험군 1,268구간 · (b) 측정 구간 443구간",
                       "(c) 서울 열선 2,283구간",
                       "위험군 = 골목길 ∩ 경사 5.56%↑ · 경계쌍 = 낙차 2 m 이상 하류가 미보호",
                       "p7_quintile_coverage.csv · p7_boundary_rate_by_gu.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.20,
             "위험군에 들어가는 것까지는 경사를 본다 — 골목길 전체를 경사로 5등분하면 "
             "커버리지가 0.2 → 10.7%로 오른다(p = 0.004).\n"
             "그러나 위험군 안에서는 경사로 줄을 세워도 추세가 없고(p = 0.766), "
             "경사만 기준으로 봐도 효율은 58.9%에 그친다. "
             "성북구 전역으로 넓혀도 45.5%다.",
             ha="center", fontsize=9, color=T.BOUNDARY, linespacing=1.7)
    T.save_both(fig, OUT / "apx_current_diagnosis.png", hide_in_bare=[hl])


if __name__ == "__main__":
    main()
