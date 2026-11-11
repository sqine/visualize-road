"""p10c — 선택 기준 4종 + 현행을 편익–경계 평면에 세워 균형점을 보인다.

같은 예산(18.53억) · 같은 후보군 · **같은 솔버(MILP)** 로 풀고 기준만 바꿨다.
그래서 차이는 오직 '무엇을 보고 골랐는가'에서 온다.

  x = 경계노출 (경계 하류가 떠안는 risk × flow) — 작을수록 좋다
  y = 기대방지 (risk × flow)                     — 클수록 좋다
  → 왼쪽 위가 좋은 쪽이다.

① 경사만·② 통행만은 두 지표 모두에서 제안에 진다(열등). 열등하지 않은 해는
현행·④ 제안·③ 위험×통행 셋이고, **제안이 그 가운데**다. 현행의 노출이 작은
것은 잘 골라서가 아니라 깐 양 자체가 적어서다 — 편익이 제안의 1/13이다.

실행: cd streamlit_app && python3 viz/charts/fig_p10c_balance.py
산출: viz/out/p10c_balance.png (+ _bare)
데이터: scripts/build_strategy_balance.py → p10_balance.csv
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

#: 전략 → (라벨 offset, 열등하지 않은가)
LAB = {
    "현행 실제":              ((16, 12), False),
    "① 경사(위험)만":         ((16, -30), False),
    "② 통행만":               ((16, -30), False),
    "③ 위험 × 통행":          ((-14, 18), True),
    "⑤ 제안 채택안 (δ=4)":     ((14, -40), True),
}
FRONT = ["⑤ 제안 채택안 (δ=4)", "③ 위험 × 통행"]


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(VIZ / "p10_balance.csv").set_index("전략")

    fig, a = plt.subplots(figsize=(8.6, 5.6))

    f = d.loc[FRONT].sort_values("경계노출")
    a.plot(f.경계노출, f.기대방지, color=T.PROPOSED, lw=2.2, alpha=0.5,
           zorder=2, label="열등하지 않은 해 (파레토 프런트)")

    for name, (off, front) in LAB.items():
        r = d.loc[name]
        prop = name.startswith("⑤")
        ours = name.startswith(("④", "⑤"))
        ref = name.startswith("④")
        short = name.replace("⑤ 제안 채택안 (δ=4)", "④ 제안 — 경계쌍까지 (δ=4)")
        a.scatter([r.경계노출], [r.기대방지],
                  s=460 if prop else (110 if ref else 150),
                  marker="*" if prop else "o",
                  alpha=0.55 if ref else 1.0,
                  color=T.PROPOSED if ours else (T.ACCENT if front else T.UNPROTECTED),
                  edgecolor="white", linewidth=1.5 if prop else 1.0,
                  zorder=6 if prop else 4)
        a.annotate(f"{short}\n경계쌍 {int(r.경계수)}개 · 편익 {r.기대방지:,.0f}",
                   (r.경계노출, r.기대방지), textcoords="offset points",
                   xytext=off, fontsize=9.4, linespacing=1.45,
                   fontweight="bold" if ours else "normal",
                   color=T.PROPOSED if ours else (T.ACCENT if front else T.TEXT_MUTED))

    a.set_xlabel("하류 위험전이량  (경계쌍 하류가 떠안는 risk × flow) — 작을수록 좋다",
                 fontsize=10)
    a.set_ylabel("기대방지  (risk × flow) — 클수록 좋다", fontsize=10)
    a.set_xlim(60, 540)
    a.set_ylim(0, 3750)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.2, lw=0.7)
    a.legend(loc="lower right", fontsize=9, framealpha=0.93)
    a.set_title("한 기준만 보면 한쪽으로 쏠린다 — 균형은 프런트 위에만 있다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)

    note = a.text(0.028, 0.46,
                  "열등하지 않은 해는 제안과 ③ 둘뿐이다.\n"
                  "제안은 ③ 대비 편익 7.1% 를 내주고 노출을 71.3% 줄인다.\n"
                  "현행·① 경사만·② 통행만은 두 지표 모두에서 진다(열등).",
                  transform=a.transAxes, va="top", fontsize=9, color=T.TEXT,
                  linespacing=1.5,
                  bbox=dict(boxstyle="round,pad=0.4", fc=T.BG_BOX,
                            ec=T.GU_LINE, lw=0.8))

    fig.suptitle("같은 예산·같은 솔버에서 보는 기준만 바꿔 보면",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.0)
    fig.text(0.5, -0.07,
             T.caption("성북구 4개동", "예산 18.53억 · 후보 골목길 847구간",
                       "다섯 전략 모두 MILP · 현행은 상류 제한 없는 방향집합으로 평가",
                       "p10_balance.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p10c_balance.png", hide_in_bare=[note])


if __name__ == "__main__":
    main()
