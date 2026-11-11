"""p6 좌단 — 자치구별 위험군 골목길 연장 순위 (세로 막대 1장).

`fig_p6_gu_risk_bar.py` 의 (a) 패널을 **슬라이드 좌단 빈칸(600×290px, 2:1)** 에
맞춰 세로형으로 다시 그린 것이다. 가로 막대 25행은 그 높이에서 행당 11px라
자치구명이 읽히지 않는다. 세로 막대는 25개를 폭으로 흘려 보낼 수 있다.

메시지: 성북구가 138.8 km로 1위 — 중위(48.7 km)의 2.9배.

실행: cd streamlit_app && python3 viz/charts/fig_p6_gu_risk_rank.py
산출: viz/out/p6_gu_risk_rank.png (+ _bare ← 슬라이드용)
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

VIZ = ROOT / "viz" / "data"
OUT = ROOT / "viz" / "out"
HL = "성북구"


def _find(name: str) -> Path:
    tgt = unicodedata.normalize("NFC", name)
    for f in VIZ.iterdir():
        if unicodedata.normalize("NFC", f.name) == tgt:
            return f
    raise FileNotFoundError(name)


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    d = pd.read_csv(_find("p6_gu_risk_length.csv"), encoding="utf-8-sig")
    d = d.sort_values("위험군연장km", ascending=False).reset_index(drop=True)
    med = float(d.위험군연장km.median())
    hl = d.index[d.자치구 == HL][0]

    fig, a = plt.subplots(figsize=(10.4, 4.9))

    colors = [T.ACCENT if g == HL else T.UNPROTECTED for g in d.자치구]
    a.bar(range(len(d)), d.위험군연장km, color=colors, width=0.74, zorder=3)

    # 중위 기준선 — "중위의 2.9배"를 눈으로 받쳐 준다
    a.axhline(med, color=T.BOUNDARY, ls="--", lw=1.3, zorder=4)
    a.text(len(d) - 0.4, med + 3, f"중위 {med:.1f} km", ha="right",
           fontsize=10.6, color=T.BOUNDARY, fontweight="bold")

    # 값 라벨은 1위·2위·최하위만 (전부 달면 25개가 서로 가린다)
    for i in (0, 1, len(d) - 1):
        a.text(i, d.위험군연장km[i] + 3, f"{d.위험군연장km[i]:.1f}",
               ha="center", fontsize=10.4,
               fontweight="bold" if i == hl else "normal",
               color=T.ACCENT if i == hl else T.TEXT_MUTED)
    a.annotate(f"중위의 {d.위험군연장km[hl] / med:.1f}배", (hl, d.위험군연장km[hl]),
               textcoords="offset points", xytext=(34, -16), fontsize=10.6,
               color=T.ACCENT, fontweight="bold")

    a.set_xticks(range(len(d)))
    a.set_xticklabels(d.자치구, rotation=52, ha="right", fontsize=10.2)
    for t, g in zip(a.get_xticklabels(), d.자치구):
        if g == HL:
            t.set_color(T.ACCENT); t.set_fontweight("bold")
    a.set_ylabel("위험군 골목길 연장 (km)", fontsize=11)
    a.set_ylim(0, float(d.위험군연장km.max()) * 1.16)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", alpha=0.22, lw=0.7)
    a.set_title("위험군 골목길은 성북구가 가장 길다 — 25개 자치구",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=10)

    fig.text(0.5, -0.16,
             T.caption("서울 25개 자치구", "위험군 골목길 1,391 km",
                       "위험군 = 골목길 ∩ 경사 5.56% 이상",
                       "p6_gu_risk_length.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p6_gu_risk_rank.png")


if __name__ == "__main__":
    main()
