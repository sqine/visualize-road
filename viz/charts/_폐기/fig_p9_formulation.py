"""p9 — 경계 발생 선형화 info 카드 (Canva 「3. 최적화 모델」 슬라이드 보조).

Canva 덱의 해당 쪽은 Parameters / Sets & Variables / Mathematical Model
3단으로 **수식을 이미 전부 담고 있다.** 그래서 이 그림은 수식을 다시 그리지
않는다. 슬라이드에 없는 것만 세 칸으로 더한다.

    ① 경계쌍 D 가 실제로 몇 개인가        — 규모 감각
    ② b_ij 가 1이 되는 경우는 하나뿐       — 4가지 경우 진리표
    ③ 왜 이진변수로 선언하지 않았는가      — 선형 완화가 공짜인 이유

상·하류 융설수 모식도는 **넣지 않는다.** 앞쪽 「경사길 융설수 재결빙 위험」
쪽의 3단계 도식과 중복이다.

색은 viz/theme.py 가 아니라 **Canva 덱 디자인 시스템**을 따른다. 이 그림만
슬라이드 안에 직접 얹히기 때문이다 (아래 CANVA 딕셔너리).

설계: viz/docs/p9_설계.md
실행: cd streamlit_app && python3 viz/charts/fig_p9_formulation.py
산출: viz/out/p9_formulation.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

OUT = ROOT / "viz" / "out"

#: Canva 덱 디자인 시스템 (덱 마지막 쪽 "디자인 시스템 · 복사해서 쓰는 템플릿")
CANVA = {
    "PRIMARY": "#38669F",
    "PLAIN": "#25364A",
    "EMPHASIS": "#098E91",
    "BG1": "#FFFFFF",
    "BG2": "#EAF2FA",
    "BG3": "#E6EBF0",
    "BORDER1": "#1F4E78",
    "BORDER2": "#5D7A99",
    "BORDER3": "#B8C7D6",
}

#: 경계 판정 고도차(m) — utils/milp.H_MIN
H_MIN = 2.0

#: 4개동 그래프에서 센 값. 재현:
#:   from utils import milp; G = milp.load_graph(); C = set(milp.candidates(G))
#:   D = [(i, j) for u, v in G.edges() for i, j in ((u, v), (v, u))
#:        if G.nodes[i]["z"] - G.nodes[j]["z"] >= milp.H_MIN]
SCALE = [
    ("후보 구간  I", "847", "골목길 ∩ 열선 미설치 · 경사 하한 없음"),
    (f"낙차 {H_MIN:.0f} m 이상 유향쌍", "718", "1,190구간 · 1,775인접 중"),
    ("그중 상류가 후보  D", "508", "하류도 후보 415 · 하류 고정 93"),
]

#: (x_i, x_j, 하한, 최적 b, 설명, 강조)
TRUTH = [
    ("1", "0", "1", "1", "상류만 깐다 → 경계 발생", True),
    ("1", "1", "0", "0", "둘 다 깐다 → 하류가 녹인다", False),
    ("0", "0", "0", "0", "둘 다 안 깐다 → 융설수 없음", False),
    ("0", "1", "−1", "0", "하류만 깐다 → 내려올 물 없음", False),
]


def main() -> None:
    plt = T.use_matplotlib_theme()
    from matplotlib.patches import FancyBboxPatch, Rectangle
    OUT.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14.0, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.86, 1.18, 0.96], wspace=0.085)

    def card(ax, title):
        """Canva 컨테이너 — BG2 채움 · BORDER2 실선 2px · 그림자 없음."""
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (1.5, 2.0), 97.0, 87.0,
            boxstyle="round,pad=0.5,rounding_size=1.8",
            fc=CANVA["BG2"], ec=CANVA["BORDER2"], lw=1.6, zorder=1))
        ax.text(5.5, 94.5, title, fontsize=11.6, color=CANVA["BORDER1"],
                fontweight="bold", va="center", ha="left")

    # ── ① 경계쌍 D 의 규모 ─────────────────────────────────────
    a = fig.add_subplot(gs[0, 0])
    card(a, "① 경계쌍 D 는 몇 개인가")
    # 구분선은 두지 않는다 — 세 항목이 "전체 → 조건 → 그중"으로 좁혀지는
    # 한 흐름이라 선을 그으면 오히려 끊어 읽힌다.
    for k, (lab, val, sub) in enumerate(SCALE):
        y = 68.0 - k * 26.0
        hi = k == len(SCALE) - 1
        col = CANVA["EMPHASIS"] if hi else CANVA["PLAIN"]
        a.text(6.5, y + 8.5, lab, fontsize=9.4, color=CANVA["BORDER2"],
               va="center")
        a.text(6.5, y - 1.0, val, fontsize=20.0, color=col,
               fontweight="bold", va="center")
        a.text(31.0, y - 2.5, "쌍" if k else "구간", fontsize=9.6,
               color=CANVA["BORDER2"], va="center")
        a.text(6.5, y - 11.5, sub, fontsize=8.4, color=CANVA["BORDER2"],
               va="center")

    # ── ② 진리표 ───────────────────────────────────────────────
    b = fig.add_subplot(gs[0, 1])
    card(b, "② b_ij 가 1이 되는 경우는 하나뿐")
    cols = [(9.5, "x_i", "상류"), (20.5, "x_j", "하류"),
            (34.5, "x_i − x_j", "하한"), (47.5, "b_ij", "최적")]
    for x, lab, sub in cols:
        b.text(x, 79.0, lab, fontsize=9.8, color=CANVA["PLAIN"],
               fontweight="bold", ha="center", va="center")
        b.text(x, 71.0, sub, fontsize=8.2, color=CANVA["BORDER2"],
               ha="center", va="center")
    b.plot([5.0, 95.0], [65.0] * 2, color=CANVA["BORDER2"], lw=1.4)

    for k, (xi, xj, lo, bv, desc, hi) in enumerate(TRUTH):
        y = 55.0 - k * 13.0
        if hi:
            b.add_patch(Rectangle((5.0, y - 6.0), 90.0, 12.0,
                                  fc=CANVA["EMPHASIS"], alpha=0.14,
                                  ec="none", zorder=2))
        col = CANVA["EMPHASIS"] if hi else CANVA["BORDER2"]
        w = "bold" if hi else "normal"
        for x, v in zip([c[0] for c in cols], [xi, xj, lo, bv]):
            b.text(x, y, v, fontsize=11.4, color=col, fontweight=w,
                   ha="center", va="center", zorder=3)
        b.text(56.0, y, desc, fontsize=9.0, color=col, fontweight=w,
               va="center", zorder=3)

    # ── ③ 왜 이진변수가 아닌가 ─────────────────────────────────
    c = fig.add_subplot(gs[0, 2])
    card(c, "③ 왜 이진변수로 선언하지 않았나")
    c.add_patch(FancyBboxPatch(
        (6.0, 60.0), 88.0, 18.0,
        boxstyle="round,pad=0.4,rounding_size=1.4",
        fc=CANVA["BG1"], ec=CANVA["BORDER3"], lw=1.3, zorder=2))
    c.text(50.0, 69.0, "b_ij*  =  max(0,  x_i − x_j)", fontsize=14.0,
           color=CANVA["EMPHASIS"], fontweight="bold",
           ha="center", va="center", zorder=3)
    c.text(6.5, 54.0,
           "b_ij 는 목적함수에서 빼지는 항이다.\n"
           "최대화 과정에서 하한까지 내려가므로\n"
           "제약 하나로 max(0, ·) 가 재현된다.",
           fontsize=9.4, color=CANVA["PLAIN"], va="top", linespacing=1.68,
           zorder=3)
    c.plot([6.0, 94.0], [29.0] * 2, color=CANVA["BORDER3"], lw=1.0)
    c.text(6.5, 24.0,
           "x 가 0/1 이면 이 값도 0/1 이다 — 그래서\n"
           "b_ij 를 [0,1] 연속변수로 두어도 정수해가\n"
           "나오고, 이진변수 508개를 늘리지 않는다.",
           fontsize=9.4, color=CANVA["EMPHASIS"], va="top",
           linespacing=1.68, zorder=3)

    fig.text(0.5, -0.025,
             T.caption("성북구 4개동", "후보 847구간 · 경계쌍 D 508",
                       f"D = {{(i,j) : i~j 인접, z_i − z_j ≥ {H_MIN:.0f} m}}",
                       "utils/milp.py"),
             ha="center", fontsize=8.4, color=CANVA["BORDER2"])

    T.save_both(fig, OUT / "p9_formulation.png", keep_panel_title=True)


if __name__ == "__main__":
    main()
