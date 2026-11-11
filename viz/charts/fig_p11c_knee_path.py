"""p11c — 예산별 전환점 해 경로와 허용 구간.

δ 를 고정하지 않는다. 예산 r 을 정하면 그 예산선 위에서 편익(risk×flow)과
경계벌점(경계수×하류가중)의 파레토 전환점이 하나 잡히고, 그 전환점이 그 예산의
적정 δ 다. 그 해들을 예산 축으로 이은 것이 윗선이다.

아랫선은 δ 를 더 세게(=5) 걸었을 때의 경로다. 두 선 사이가 **허용 구간**으로,
편익을 조금 내주고 경계를 더 줄이고 싶을 때 고를 수 있는 폭이다.
최종 시나리오 두 개(δ = 4.0 · 5.0)는 이 구간 안에 있다.

실행: cd streamlit_app && python3 viz/charts/fig_p11c_knee_path.py
산출: viz/out/p11c_knee_path.png (+ _bare)
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

CUR = 18.53
#: 현행 열선 34개 구간의 기대 미끄러짐 방지 — p10 전략비교표 · p12_scenarios.csv
CUR_BENEFIT = 233.7
#: (예산, 전환점 해 편익·δ·경계, 시나리오 편익·δ·경계, 라벨 위치)
#: p12_budget_knee.csv · p12_scenarios.csv
SCEN = [(18.53, 3060.2, 2.1, 19, 2959.1, 4.0, 13, "S1 현행 예산", (-142, -176)),
        (24.10, 3332.4, 2.7, 17, 3160.6, 5.0, 11, "S2 +30%", (18, -96))]


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    k = pd.read_csv(VIZ / "p11_knee_by_budget.csv").sort_values("예산억")
    g = pd.read_csv(VIZ / "p11_grid_hires.csv")
    dl = g.delta.unique()
    hard = g[g.delta == dl[np.abs(dl - 5.0).argmin()]].sort_values("예산억")
    # δ=0 — 하류 전이를 전혀 보지 않고 편익만 최대화한 해. "최대 편익의 93%"의
    # 분모가 어디인지 그림 안에서 보여야 한다.
    free = g[g.delta == dl[np.abs(dl - 0.0).argmin()]].sort_values("예산억")
    lab = pd.read_csv(VIZ / "p12_budget_knee.csv")

    fig, a = plt.subplots(figsize=(7.4, 5.2))

    # 허용 구간 — 전환점 해(편익 상한)와 δ=5 해(경계 하한) 사이
    hi = np.interp(hard.예산억, k.예산억, k.기대방지)
    a.fill_between(hard.예산억, hard.기대방지, hi, color=T.PROPOSED,
                   alpha=0.13, lw=0, zorder=1)

    a.plot(free.예산억, free.기대방지, color=T.TEXT_MUTED, lw=1.5,
           ls=(0, (1.6, 1.8)), zorder=2,
           label="편익만 최대화한 해 (δ = 0) — 편익 상한")
    a.plot(k.예산억, k.기대방지, color=T.PROPOSED, lw=2.6, zorder=4,
           label="예산별 파레토 전환점 해")
    a.plot(hard.예산억, hard.기대방지, color=T.BOUNDARY, lw=1.8, ls=(0, (5, 2)),
           zorder=3, label="경계쌍을 더 줄인 해 (δ = 5)")

    # 파레토 전환점 δ 라벨 — 눈금이 성기게
    for r in lab.itertuples():
        if not 6 < r.예산억 < 40:
            continue
        y = float(np.interp(r.예산억, k.예산억, k.기대방지))
        a.annotate(f"δ {r.delta:.1f}", (r.예산억, y), textcoords="offset points",
                   xytext=(0, 11), ha="center", fontsize=8.6,
                   color=T.TEXT_MUTED)

    a.axvline(CUR, color=T.CURRENT, ls="--", lw=1.4, zorder=2)
    a.text(CUR + 0.6, 3870, "현행 예산 18.53억", ha="left", fontsize=9,
           color=T.CURRENT, fontweight="bold")

    # 현행 실제 배치 — 같은 예산에서 어디에 서 있는지가 이 그림의 기준점이다.
    a.scatter([CUR], [CUR_BENEFIT], s=150, marker="X", color=T.WARN,
              edgecolor="white", linewidth=1.6, zorder=9)
    a.annotate(f"현행 실제 {CUR_BENEFIT:,.1f}", (CUR, CUR_BENEFIT),
               textcoords="offset points", xytext=(14, 2), fontsize=9.6,
               color=T.WARN, fontweight="bold", zorder=9)

    blocks = []
    for x, yk, dk, bk, ys, ds, bs, name, off in SCEN:
        a.annotate("", xy=(x, yk), xytext=(x, ys),
                   arrowprops=dict(arrowstyle="<->", lw=1.3, color=T.ACCENT),
                   zorder=7)
        # 채택안 별은 이 그림의 결론이다. 슬라이드에서 601px 로 줄어들므로
        # 흰 후광(바깥 별)을 깔고 그 위에 본 별을 얹어 선·띠 위에서도 뜨게 한다.
        a.scatter([x], [ys], s=760, marker="*", color="white",
                  edgecolor="white", linewidth=0, zorder=8)
        a.scatter([x], [ys], s=560, marker="*", color=T.ADOPTED,
                  edgecolor="white", linewidth=1.8, zorder=9)
        # 별 옆에는 이름표만 — 설명은 Canva 말풍선이 맡는다
        a.annotate(name.split()[0], (x, ys), textcoords="offset points",
                   xytext=(17, -22), fontsize=13, color=T.ADOPTED,
                   fontweight="bold", zorder=9)
        # 기준은 δ=0 상한 하나로 통일한다. 전환점 대비로 쓰면 분모가 이미
        # 한 번 양보한 값이라 손실이 작아 보인다.
        share = 100 * ys / float(np.interp(x, free.예산억, free.기대방지))
        blocks.append(a.annotate(
            f"{name} {x:.2f}억\n전환점 δ {dk:.1f} → 채택 δ {ds:.1f}\n"
            f"편익 상한의 {share:.0f}% · 경계쌍 {bk} → {bs}개",
            (x, ys), textcoords="offset points", xytext=off,
            fontsize=8.8, color=T.ACCENT, fontweight="bold", linespacing=1.5,
            arrowprops=dict(arrowstyle="-", lw=0.9, color=T.ACCENT,
                            alpha=0.55, shrinkA=4, shrinkB=10)))

    a.set_xlabel("예산 (억원)", fontsize=10)
    a.set_ylabel("막아내는 미끄러짐 (risk × flow)", fontsize=10)
    a.set_xlim(0, 42)
    a.set_ylim(0, 4350)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=0.2, lw=0.7)
    # 범례는 왼쪽 위로 — 아래 빈 공간은 Canva 말풍선 자리로 비워 둔다
    a.legend(loc="upper left", fontsize=8.8, framealpha=0.93,
             bbox_to_anchor=(0.0, 0.99))
    a.set_title("예산을 정하면 적정 δ 와 해가 따라온다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)
    note = a.text(0.028, 0.80,
                  "띠 = δ 를 전환점에서 5 까지 올렸을 때의 폭.\n"
                  "띠가 얇다 = 편익을 거의 안 내주고 경계쌍을 줄일 수 있다.",
                  transform=a.transAxes, va="top", fontsize=8.6, color=T.TEXT,
                  linespacing=1.5,
                  bbox=dict(boxstyle="round,pad=0.38", fc=T.BG_BOX,
                            ec=T.GU_LINE, lw=0.8))

    fig.text(0.5, -0.06,
             T.caption("성북구 4개동", "예산선 100개의 전환점 해",
                       "전환점 = 파레토 프런트 양 끝을 잇는 현에서 가장 먼 점",
                       "p11_knee_by_budget.csv · p12_scenarios.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p11c_knee_path.png",
                hide_in_bare=[note, *blocks])

    # Canva 말풍선을 어디에 붙일지 — 별의 위치를 그림 비율로 출력
    for x, yk, dk, bk, ys, ds, bs, name, off in SCEN:
        px, py = a.transData.transform((x, ys))
        fx, fy = px / fig.get_size_inches()[0] / fig.dpi, \
                 py / fig.get_size_inches()[1] / fig.dpi
        print(f"{name}: 그림 내 비율 x={fx:.4f} y(아래0)={fy:.4f}")


if __name__ == "__main__":
    main()
