"""p10 — 전략 비교 2패널 (평가 방법론).

(a) 경사–통행 평면에서 각 전략이 어디에 서는가
(b) 목적함수 분해 — 기대방지 = 순편익 + 경계 벌점

전략 6종 전체 수치는 `표_p10_전략비교`(PNG·TSV)로 따로 뺐다. 슬라이드에서
표만 키우거나 옮길 수 있어야 하기 때문이다.

메시지: 하나만 보면 고장난다. 둘 다 봐야 하고, 경계까지 넣어야 순편익이 산다.

읽는 포인트
-----------
④(편익만 최대화한 MILP)는 기대방지가 가장 큰데 **순편익은 ③보다 낮다**.
편익만 쫓으면 경계 비용에서 손해 보는 구간이 있다는 뜻이다.

실행: cd streamlit_app && python3 viz/charts/fig_p10_strategy.py
산출: viz/out/p10_strategy.png
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

#: 전략 표시 순서와 짧은 라벨 (p10_strategy_v3.csv 의 '전략' 열 기준)
ORDER = [
    ("① 경사도만 (greedy)", "① 경사도만"),
    ("② 유동인구만 (greedy)", "② 유동인구만"),
    ("③ 위험×유동 (greedy)", "③ 위험×유동"),
    ("④ MILP δ=0 (경계쌍 미고려)", "④ 편익만\n최대화"),
    ("⑤ 제안 MILP δ=2", "⑤ 제안\n(δ=2)"),
    ("현행 실제", "현행 실제"),
]
#: (a) 산점도에 올릴 전략 — ④는 ⑤와 거의 겹쳐 생략
SCATTER = ["① 경사도만 (greedy)", "② 유동인구만 (greedy)",
           "③ 위험×유동 (greedy)", "⑤ 제안 MILP δ=2", "현행 실제"]


def color_of(name: str) -> str:
    if name.startswith("⑤"):
        return T.PROPOSED
    if name.startswith("현행"):
        return T.CURRENT
    return T.UNPROTECTED


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(VIZ / "p10_strategy_v3.csv")
    d["라벨"] = d.전략.map(dict(ORDER))
    d["_o"] = d.전략.map({k: i for i, (k, _) in enumerate(ORDER)})
    d = d.sort_values("_o").reset_index(drop=True)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8),
                           gridspec_kw={"width_ratios": [1, 1.2]})

    # ── (a) 경사–통행 평면 ────────────────────────────────────
    a = ax[0]
    s = d[d.전략.isin(SCATTER)]
    for _, r in s.iterrows():
        c = color_of(r.전략)
        a.scatter(r.평균경사, r.평균통행, s=30 + r.기대방지 / 3.2, color=c,
                  edgecolor="white", linewidth=1.4, zorder=3)
        off = {"①": (0, -22), "②": (0, 17), "③": (36, 10),
               "⑤": (-34, -10), "현": (2, -22)}.get(r.전략[0], (0, -16))
        a.annotate(r.라벨.replace("\n", " "), (r.평균경사, r.평균통행),
                   textcoords="offset points", xytext=off, ha="center",
                   fontsize=9.2, color=T.TEXT,
                   fontweight="bold" if c == T.PROPOSED else "normal")
    a.set_xlabel("평균 경사 (%)", fontsize=10)
    a.set_ylabel("평균 통행량", fontsize=10)
    a.set_title("(a) 한쪽만 보면 한쪽으로 쏠린다",
                fontsize=11.8, color=T.ACCENT, fontweight="bold", pad=10)
    a.set_xlim(1, 29); a.set_ylim(-26, 136)
    a.grid(alpha=0.22, lw=0.7)
    a.text(27.5, 128, "가파른데 사람 없음", fontsize=8.4, color=T.TEXT_MUTED,
           ha="right", style="italic")
    a.text(2.2, 128, "사람 많은데 평지", fontsize=8.4, color=T.TEXT_MUTED,
           ha="left", style="italic")
    a.text(0.98, 0.03, "원 크기 ∝ 기대방지", transform=a.transAxes,
           ha="right", fontsize=8.2, color=T.TEXT_MUTED)

    # ── (b) 목적함수 분해 ─────────────────────────────────────
    b = ax[1]
    x = np.arange(len(d))
    net = d.순편익.values
    pen = d.벌점.fillna(0).values
    cols = [color_of(t) for t in d.전략]
    b.bar(x, net, 0.58, color=cols, zorder=2, label="순편익 (목적함수 값)")
    b.bar(x, pen, 0.58, bottom=net, facecolor="white", edgecolor=T.BOUNDARY,
          linewidth=1.4, hatch="///", zorder=2, label="경계쌍 벌점 δ·Σw·b")

    for i, r in d.iterrows():
        tot = r.순편익 + (r.벌점 if not np.isnan(r.벌점) else 0)
        b.text(i, tot + 70, f"{r.기대방지:,.0f}", ha="center", fontsize=9,
               color=T.TEXT_MUTED)
        b.text(i, r.순편익 / 2, f"{r.순편익:,.0f}", ha="center", va="center",
               fontsize=9.6, color="white" if r.순편익 > 700 else T.TEXT,
               fontweight="bold")
        b.text(i, -250, f"경계쌍 {int(r.경계수)}", ha="center", fontsize=8.8,
               color=T.BOUNDARY if r.경계수 >= 23 else T.TEXT)
    b.text(-0.72, -250, "잔여", fontsize=8.6, color=T.TEXT_MUTED,
           ha="right", va="center")
    b.annotate("편익만 쫓으면\n순편익은 오히려 낮다",
               xy=(3, d.순편익[3] + d.벌점[3] * 0.55), xytext=(2.05, 3880),
               fontsize=9.2, color=T.BOUNDARY, ha="center", linespacing=1.5,
               arrowprops=dict(arrowstyle="-|>", lw=1.2, color=T.BOUNDARY,
                               connectionstyle="arc3,rad=0.25"))

    b.set_xticks(x); b.set_xticklabels(d.라벨, fontsize=8.8, linespacing=1.3)
    b.set_ylabel("기대 미끄러짐 건수", fontsize=10)
    b.set_ylim(-390, 4320)
    b.axhline(0, color=T.GU_LINE, lw=0.8)
    b.legend(fontsize=8.8, frameon=False, loc="upper left")
    b.grid(axis="y", alpha=0.22, lw=0.7)
    b.set_title("(b) 기대방지 = 순편익 + 경계쌍 벌점",
                fontsize=11.8, color=T.ACCENT, fontweight="bold", pad=10)

    for sp in ("top", "right"):
        a.spines[sp].set_visible(False); b.spines[sp].set_visible(False)

    fig.suptitle("같은 예산 18.53억 — 현행 열선 길이 2,100 m 를 처음부터 깐다면",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.02)
    fig.text(0.5, -0.045,
             T.caption("성북구 4개동", "후보 골목길 847곳",
                       "①②③ greedy · ④⑤ MILP · 막대 위 숫자 = 기대방지 "
                       "· 전체 표는 표_p10_전략비교",
                       "p10_strategy_v3.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.085,
             "현행의 벌점은 비워 둔다 — w는 후보 기준으로 정의되는데 현행 열선은 "
             "후보에서 빠져 같은 잣대로 값을 매길 수 없다. "
             "그래서 (b)의 현행 막대는 순편익 = 기대방지다.",
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p10_strategy.png")
    print(d[["라벨", "기대방지", "벌점", "순편익", "경계수"]].to_string(index=False))


if __name__ == "__main__":
    main()
