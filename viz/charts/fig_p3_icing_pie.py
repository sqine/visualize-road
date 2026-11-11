"""p3 — 결빙취약구간 지정은 대형도로에 쏠려 있다 (파이 + 비교 한 줄).

메시지: 서울 도로에서 대형도로는 7.0%뿐인데, 결빙취약구간 지정에서는
        28.2% — 4.0배로 과대대표된다. 기존 제도의 눈이 차량 쪽에 있다.

색은 테마 의미색을 따른다.
    대형도로  CURRENT 파랑   — 현행 제도가 보고 있는 곳
    골목길    UNPROTECTED    — 지정은 되지만 제도의 초점에서 비켜 있는 곳
    기타      BG_LEGEND

실행: cd streamlit_app && python3 viz/charts/fig_p3_icing_pie.py
산출: viz/out/p3_icing_pie.png (+ _bare)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

VIZ = ROOT / "viz" / "data"
OUT = ROOT / "viz" / "out"


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(VIZ / "p3_icing_roadtype_share.csv")
    d = d[d.도로유형 != "합계"].set_index("도로유형")

    etc = float(d.at["애매", "지정구간_비율"]) + float(d.at["unknown", "지정구간_비율"])
    parts = [("골목길", float(d.at["골목길", "지정구간_비율"]), T.UNPROTECTED),
             ("대형도로", float(d.at["대형도로", "지정구간_비율"]), T.CURRENT),
             ("기타", etc, T.BG_LEGEND)]

    fig, a = plt.subplots(figsize=(6.6, 4.9))
    w, txt, pct = a.pie(
        [v for _, v, _ in parts],
        colors=[c for _, _, c in parts],
        labels=[n for n, _, _ in parts],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90, counterclock=False,
        explode=(0, 0.06, 0),
        wedgeprops=dict(edgecolor="white", linewidth=1.8),
        textprops=dict(fontsize=11.5, color=T.TEXT),
        pctdistance=0.66, labeldistance=1.1)
    for i, t in enumerate(pct):
        t.set_color("white" if i == 1 else T.TEXT)
        t.set_fontweight("bold")
    txt[1].set_color(T.CURRENT)
    txt[1].set_fontweight("bold")

    a.set_title("결빙취약구간 지정 347건의 도로유형",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=10)

    # 비교 한 줄은 축 안에 둔다 — 발표용(_bare)에서도 남아야 하는 메시지다
    a.text(0.5, -0.09, "서울 전체 도로에서 대형도로는 7.0% — 지정에서는 28.2%로 4.0배",
           transform=a.transAxes, ha="center", fontsize=11.5,
           color=T.CURRENT, fontweight="bold")

    fig.text(0.5, -0.015,
             T.caption("서울 전역", "결빙취약구간 지정 347건 · 도로망 168,976구간",
                       "도로유형은 TBGIS·OSM 공간조인 분류",
                       "p3_icing_roadtype_share.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    T.save_both(fig, OUT / "p3_icing_pie.png")


if __name__ == "__main__":
    main()
