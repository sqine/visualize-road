"""p8 — 노드(도로구간)와 엣지(경계) 모형 (Canva 8쪽 오른쪽아래 「가중치 — 엣지전류」).

주장
----
도로구간을 **노드**로 두면 경계는 **엣지**가 된다. 경계 가중치 wᵢⱼ 는 그 엣지를
지나는 통행량, 곧 current-flow 전파모형의 **엣지 전류**다. 선 굵기가 전류다.

막다른 골목은 지나는 사람이 없어 전류가 0에 가깝고, 핵심 길목은 굵다.
그래서 엣지 기준으로 재면 "경사는 급하지만 아무도 안 다니는 골목"에 감점을
낭비하지 않는다.

크기 주의
---------
슬라이드에서 **295 × 168 px** 자리에 들어간다(모형 설명 칸의 오른쪽 여백).
figsize 를 그 인치 환산값으로 잡아야 글자 크기가 슬라이드에서 맞는다 —
figsize 를 키우고 글자를 키우는 것과 같지 않다(축소되면 같이 작아진다).

실행: cd streamlit_app && python3 viz/charts/fig_p8_node_edge.py
산출: viz/out/p8_node_edge.png (+ _bare)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

OUT = ROOT / "viz" / "out"

#: 슬라이드에 놓이는 크기(px) → 인치 (100 px/inch 로 환산)
FIG_W, FIG_H = 2.95, 1.68

#: 노드 좌표 (축 0~10 × 0~6)
P = {
    "u1": (0.9, 5.0), "u2": (0.9, 2.9),
    "i": (3.5, 4.0), "j": (6.5, 2.5),
    "d1": (9.1, 3.6), "d2": (9.1, 1.1),
}
#: (출발, 도착, 전류) — 전류가 선 굵기이자 w 다
EDGES = [("u1", "i", 0.45), ("u2", "i", 0.35),
         ("i", "j", 1.00),
         ("j", "d1", 0.60), ("j", "d2", 0.40)]


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    fig, a = plt.subplots(figsize=(FIG_W, FIG_H))
    a.set_xlim(0, 10.2); a.set_ylim(0, 6)
    a.axis("off")

    for s, e, cur in EDGES:
        x0, y0 = P[s]; x1, y1 = P[e]
        boundary = (s, e) == ("i", "j")
        col = T.BOUNDARY if boundary else T.TEXT_MUTED
        a.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle="-|>",
                                   shrinkA=16, shrinkB=16,
                                   lw=1.1 + 4.2 * cur, color=col,
                                   alpha=1.0 if boundary else 0.55,
                                   mutation_scale=7 + 4 * cur))

    # 경계 엣지 라벨 — 유일하게 키워야 하는 글자
    a.text(5.0, 3.72, r"$w_{ij}$", fontsize=13, color=T.BOUNDARY,
           fontweight="bold", ha="center", va="bottom")

    for key, (fc, lab) in {
        "i": (T.CURRENT, "i"), "j": (T.UNPROTECTED, "j"),
    }.items():
        x, y = P[key]
        a.scatter([x], [y], s=620, color=fc, edgecolor="white",
                  linewidth=1.8, zorder=5)
        a.text(x, y, lab, fontsize=12, color="white" if key == "i" else T.TEXT,
               fontweight="bold", ha="center", va="center", zorder=6)
    for key in ("u1", "u2", "d1", "d2"):
        x, y = P[key]
        a.scatter([x], [y], s=190, color=T.BG_LEGEND, edgecolor="white",
                  linewidth=1.4, zorder=4)

    a.text(3.5, 5.25, "상류 · 설치", fontsize=10, color=T.CURRENT,
           fontweight="bold", ha="center")
    a.text(6.5, 1.25, "하류 · 미설치", fontsize=10, color=T.TEXT_MUTED,
           fontweight="bold", ha="center", va="top")

    fig.tight_layout(pad=0.1)
    T.save_both(fig, OUT / "p8_node_edge.png")
    print("  노드 6 · 엣지 5 · 경계 엣지 (i,j) 굵기 = wᵢⱼ")


if __name__ == "__main__":
    main()
