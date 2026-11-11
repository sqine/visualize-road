"""p6 — 자치구별 위험군 골목길 연장과 커버리지 (2패널).

자치구 단위 비교는 전부 여기로 모은다. p2 는 전역 한 덩어리만 말한다.

(a) 위험군 골목길 연장 — 성북구 138.8 km로 1위
(b) 위험군 대비 커버리지 — 성북구 8.90%로 19위

메시지: 위험은 가장 많은데 보호는 하위권이다.

설명 문구는 그림에 넣지 않는다(`07_발표자료_시각자료_전체목록.md` 참조).
발표용(_bare)에서 제목·캡션이 빠져도 축과 수치만으로 읽히게 한다.

실행: cd streamlit_app && python3 viz/charts/fig_p6_gu_risk_bar.py
산출: viz/out/p6_gu_risk_bar.png (+ _bare)
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
    r = pd.read_csv(VIZ / "p6_gu_risk_length.csv")
    c = pd.read_csv(_find("위험군_5.56pct_25gu.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 6.6))

    # ── (a) 위험군 연장 ───────────────────────────────────────
    a = ax[0]
    d = r.sort_values("위험군연장km")
    a.barh(d.자치구, d.위험군연장km, height=0.72,
           color=[T.ACCENT if g == HL else T.UNPROTECTED for g in d.자치구])
    med = d.위험군연장km.median()
    a.axvline(med, color=T.BOUNDARY, ls="--", lw=1.3, zorder=3)
    a.text(med + 2.5, 0.2, f"중위 {med:.1f} km", fontsize=8.8,
           color=T.BOUNDARY, fontweight="bold")
    for i, x in enumerate(d.itertuples()):
        if x.자치구 in (HL, "서대문구"):
            a.text(x.위험군연장km + 2.5, i, f"{x.위험군연장km:.1f} km",
                   va="center", fontsize=9.4,
                   color=T.ACCENT if x.자치구 == HL else T.TEXT,
                   fontweight="bold" if x.자치구 == HL else "normal")
    a.set_xlabel("위험군 골목길 연장 (km)", fontsize=10)
    a.set_xlim(0, 168)
    a.set_title("(a) 위험은 성북구가 가장 많다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)

    # ── (b) 커버리지 ──────────────────────────────────────────
    b = ax[1]
    e = c.sort_values("커버리지")
    b.barh(e.자치구, e.커버리지, height=0.72,
           color=[T.ACCENT if g == HL else T.UNPROTECTED for g in e.자치구])
    med2 = e.커버리지.median()
    b.axvline(med2, color=T.BOUNDARY, ls="--", lw=1.3, zorder=3)
    b.text(med2 + 0.6, 0.2, f"중위 {med2:.2f}%", fontsize=8.8,
           color=T.BOUNDARY, fontweight="bold")
    rank = int((e.커버리지 > e.loc[e.자치구 == HL, "커버리지"].iloc[0]).sum()) + 1
    for i, x in enumerate(e.itertuples()):
        if x.자치구 == HL:
            b.text(x.커버리지 + 0.6, i, f"{x.커버리지:.2f}%  ({rank}위)",
                   va="center", fontsize=9.4, color=T.ACCENT,
                   fontweight="bold")
    b.set_xlabel("위험군 골목길 대비 열선 커버리지 (%)", fontsize=10)
    b.set_xlim(0, 34)
    b.set_title("(b) 그런데 보호는 하위권이다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)

    for axx in ax:
        axx.tick_params(axis="y", labelsize=9)
        for lab in axx.get_yticklabels():
            if lab.get_text() == HL:
                lab.set_fontweight("bold"); lab.set_color(T.ACCENT)
        for s in ("top", "right", "left"):
            axx.spines[s].set_visible(False)
        axx.grid(axis="x", alpha=0.22, lw=0.7)

    fig.suptitle("성북구 — 위험군 1위, 커버리지 19위",
                 fontsize=T.TS_MAIN, color=T.ACCENT, fontweight="bold", y=1.0)
    fig.text(0.5, -0.045,
             T.caption("서울 25개 자치구", "위험군 골목길 1,391 km",
                       "위험군 = 골목길 ∩ 경사 5.56% 이상 · "
                       "커버리지 = 신고 열선 연장 ÷ 위험군 연장",
                       "p6_gu_risk_length.csv · 위험군_5.56pct_25gu.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.085,
             "서울 전역 위험군 1,391 km 중 성북구가 10.0% — 2위 서대문구의 "
             "1.16배, 중위의 2.9배다. 투입량과 커버리지는 무상관(ρ=0.119, "
             "p=0.570)인 반면 위험군 연장과 커버리지는 ρ=−0.719 (p=0.0001)로, "
             "위험한 구일수록 오히려 덜 덮였다.",
             ha="center", fontsize=9, color=T.BOUNDARY)
    fig.tight_layout()
    T.save_both(fig, OUT / "p6_gu_risk_bar.png")


if __name__ == "__main__":
    main()
