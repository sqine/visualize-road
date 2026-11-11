"""p11 — 예산(B) × 경계가중(δ) 100×100 격자.

(a) 히트맵 — δ를 올리면 경계노출이 얼마나 줄어드는가 (δ=0 대비 %)
(b) 파레토 평면 — 편익 vs 경계노출. 예산선마다 곡선 하나, 전환점 표시

메시지: δ는 임의로 고른 값이 아니다. 어느 예산에서 보든 파레토 전환점은 같은 자리에 있다.

용어: 파레토 프런트는 곡선 전체, 그 위에서 꺾이는 한 점이 **파레토 전환점**이다.
      (프런트 양 끝을 잇는 현에서 가장 먼 점 — Kneedle)

실행: cd streamlit_app && python3 viz/charts/fig_p11_budget_delta.py
산출: viz/out/p11_budget_delta.png
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

#: 현행 열선 예산(억) — 비교 기준선
CUR_BUDGET = 18.58
#: 채택 운영점
OP_DELTA = 2.06


def main() -> None:
    plt = T.use_matplotlib_theme()
    from matplotlib.colors import LinearSegmentedColormap
    OUT.mkdir(exist_ok=True)

    g = pd.read_csv(VIZ / "p11_grid_hires.csv")
    k = pd.read_csv(VIZ / "p11_knee_by_budget.csv")

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.4),
                           gridspec_kw={"width_ratios": [1.12, 1]})

    # ── (a) 히트맵 : 노출감소 % ────────────────────────────────
    a = ax[0]
    piv = g.pivot_table(index="예산억", columns="delta", values="노출감소")
    piv = piv.sort_index()
    cmap = LinearSegmentedColormap.from_list(
        "exp", ["#FFFFFF", "#FBD9CE", "#E8907A", T.BOUNDARY, "#6E1B10"])
    im = a.pcolormesh(piv.columns.values, piv.index.values, piv.values,
                      cmap=cmap, vmin=0, vmax=100, shading="nearest")
    a.set_xscale("symlog", linthresh=0.05)
    a.set_yscale("log")
    a.set_xlabel("경계쌍 가중 δ  (0 = 경계쌍 무시)", fontsize=10)
    a.set_ylabel("예산 (억원, 로그)", fontsize=10)
    a.set_xlim(0, 12)
    cb = fig.colorbar(im, ax=a, pad=0.02)
    cb.set_label("δ=0 대비 하류 위험전이 감소 (%)", fontsize=9.4)

    a.axhline(CUR_BUDGET, color=T.CURRENT, lw=1.6, ls="--")
    a.text(0.012, CUR_BUDGET * 1.12, "현행 예산 18.6억", ha="left",
           fontsize=9, color=T.CURRENT, fontweight="bold")
    a.plot(k.knee_delta, k.예산억, color=T.ACCENT, lw=0, marker="o",
           ms=3.1, mfc="white", mew=1.0, label="예산선별 파레토 전환점 δ (100개)")
    # 채택 운영점은 두 패널의 결론이다. 코랄(PROPOSED)은 히트맵의 붉은 계열,
    # 그리고 (b)의 전환점 궤적과 같은 색군이라 그 위에서 묻힌다.
    # 흰 후광을 깔고 고채도 보라(ADOPTED)를 얹어 어느 배경에서도 뜨게 한다.
    a.scatter([OP_DELTA], [CUR_BUDGET], s=430, marker="*",
              color="white", edgecolor="white", linewidth=0, zorder=5)
    a.scatter([OP_DELTA], [CUR_BUDGET], s=300, marker="*",
              color=T.ADOPTED, edgecolor="white", linewidth=1.4, zorder=6,
              label=f"채택 운영점 δ={OP_DELTA:.1f}")
    a.legend(loc="lower left", fontsize=10, framealpha=0.92)
    a.set_title("(a) 색이 진할수록 하류 위험전이가 많이 줄어든 조합",
                fontsize=12, color=T.ACCENT, fontweight="bold", pad=10)

    # ── (b) 파레토 평면 ───────────────────────────────────────
    b = ax[1]
    budgets = sorted(g.예산억.unique())
    show = [budgets[i] for i in (14, 32, 49, 66, 83, 99)]
    cur = min(budgets, key=lambda x: abs(x - CUR_BUDGET))
    if cur not in show:
        show.append(cur)
    show = sorted(set(show))
    # 슬라이드에서 이 패널은 가로 538px 로 줄어든다. Blues 를 0.32 부터 쓰면
    # 아래쪽 예산선(3·7·13억)이 흰 배경에 묻혀 사라지므로 램프 바닥을 올린다.
    greys = plt.cm.Blues(np.linspace(0.55, 0.95, len(show)))

    for c, bg in zip(greys, show):
        d = g[g.예산억 == bg].sort_values("경계노출")
        hot = abs(bg - cur) < 1e-9
        b.plot(d.경계노출, d.기대방지,
               color=T.ACCENT if hot else c,
               lw=3.0 if hot else 2.0, zorder=4 if hot else 2,
               alpha=1.0)
        b.text(d.경계노출.iloc[-1] + 6, d.기대방지.iloc[-1],
               f"{bg:.0f}억", fontsize=10.5, va="center",
               color=T.ACCENT if hot else c,
               fontweight="bold")

    kk = k[np.isclose(k.예산억, cur)]
    b.scatter(k.경계노출, k.기대방지, s=20, color=T.BOUNDARY,
              alpha=0.6, linewidth=0, zorder=3,
              label="전환점 궤적 (예산선 100개)")
    if len(kk):
        b.scatter(kk.경계노출, kk.기대방지, s=620, marker="*",
                  color="white", edgecolor="white", linewidth=0, zorder=6)
        b.scatter(kk.경계노출, kk.기대방지, s=440, marker="*",
                  color=T.ADOPTED, edgecolor="white", linewidth=1.6,
                  zorder=7, label="채택 운영점")
        b.annotate(f"δ={OP_DELTA:.1f}\n편익 −3.9% · 노출 −54%",
                   (float(kk.경계노출.iloc[0]), float(kk.기대방지.iloc[0])),
                   textcoords="offset points", xytext=(18, -42),
                   fontsize=11, color=T.ADOPTED, fontweight="bold",
                   linespacing=1.4, zorder=7,
                   arrowprops=dict(arrowstyle="-", lw=1.2, color=T.ADOPTED))

    b.set_xlabel("하류 위험전이량  (경계쌍 하류가 떠안는 risk×flow)", fontsize=10)
    b.set_ylabel("기대방지  (risk×flow)", fontsize=10)
    b.set_xlim(0, 700)
    b.legend(loc="lower right", fontsize=10, framealpha=0.92)
    b.grid(alpha=0.22, lw=0.7)
    b.set_title("(b) 같은 예산에서 편익을 거의 잃지 않고 노출을 줄인다",
                fontsize=12, color=T.ACCENT, fontweight="bold", pad=10)

    for axx in ax:
        for s in ("top", "right"):
            axx.spines[s].set_visible(False)

    fig.suptitle("δ는 임의의 값이 아니다 — 100×100 격자 전역 탐색",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.01)
    fig.text(0.5, -0.045,
             T.caption("성북구 4개동", "예산 100 × δ 100 = 10,000점",
                       "전환점 = 파레토 프런트 양 끝을 잇는 현에서 가장 먼 점",
                       "p11_grid_hires.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.088,
             "예산선별 파레토 전환점 δ 중위 2.20 (사분위 1.16–3.96) — 현행 예산선의 전환점 1.69와 "
             "δ=2.06은 동일 해. 예산을 바꿔도 전환점 위치가 크게 흔들리지 않는다.",
             ha="center", fontsize=8.6, color=T.BOUNDARY)
    fig.tight_layout()
    T.save_both(fig, OUT / "p11_budget_delta.png")


if __name__ == "__main__":
    main()
