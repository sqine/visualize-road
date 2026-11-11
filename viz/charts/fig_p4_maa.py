"""p4 — 보행 미끄러짐 위험 (Observed MAA EDF). 두 장으로 분리.

한 장에 둘을 붙이면 슬라이드에서 작아진다. 각각 따로 쓸 수 있게 나눈다.

    p4_maa_edf.png     MAA 경험분포 — 사람마다 버티는 한계가 다르다
    p7a_risk_curve.png  경사 → P(MAA<θ) 계단곡선 — 1° 양자화라 step 이 맞다

실행: cd streamlit_app && python3 viz/charts/fig_p4_maa.py
산출: viz/out/p4_maa_edf.png · p7a_risk_curve.png (+ 각 _bare)
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

#: risk(θ)=P(MAA<θ) 가 정의임을 밝히는 각주 — 두 그림이 공유한다
NOTE = ("risk(θ)=P(MAA<θ)는 가정이 아니라 정의다 — 경사 θ에서 미끄러진다는 것과 "
        "MAA<θ는 같은 사건이다. 1° 단위 기록이라 계단함수다.")


def edf(plt) -> None:
    """(a) MAA 경험분포."""
    e = pd.read_csv(VIZ / "p4_maa_edf.csv")
    fig, a = plt.subplots(figsize=(7.2, 4.4))

    a.bar(e["MAA_deg"], e.비율 * 100, color=T.CURRENT, width=0.78)
    med = float(e.loc[(e["누적비율"] - 0.5).abs().idxmin(), "MAA_deg"])
    a.text(0.985, 0.95, "보행자 58세션 · 토론토재활원 WinterLab",
           transform=a.transAxes, ha="right", fontsize=8.8, color=T.TEXT_MUTED)
    a.text(0.985, 0.88, f"n=58 · 중앙 {med:.0f}°", transform=a.transAxes,
           ha="right", fontsize=9.2, color=T.TEXT_MUTED)

    a.set_xticks(range(0, 13))
    a.set_xlabel("Observed MAA (°) — 미끄러지지 않고 오르내릴 수 있는 최대 경사",
                 fontsize=9.6)
    a.set_ylabel("세션 비율 (%)", fontsize=10)
    a.set_ylim(0, 17.5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", alpha=0.22, lw=0.7)
    a.set_title("사람마다 버티는 한계가 다르다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=10)

    fig.text(0.5, -0.07,
             T.caption("Dadkhah et al.(2026) Fig.1", "세션 58건(원자료 74)",
                       "Observed MAA 경험분포", "p4_maa_edf.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p4_maa_edf.png")


def curve(plt) -> None:
    """(b) 경사 → 미끄러질 확률 계단곡선."""
    c = pd.read_csv(VIZ / "p4_risk_curve.csv")
    fig, b = plt.subplots(figsize=(7.2, 4.4))

    m = c.slope_pct <= 25
    b.step(c.slope_pct[m], c.risk[m], where="post",
           color=T.PROPOSED, lw=2.8, zorder=3)
    b.fill_between(c.slope_pct[m], 0, c.risk[m], step="post",
                   color=T.PROPOSED, alpha=0.10)
    b.axvspan(T.SATURATION_SLOPE_PCT, 25, color=T.UNPROTECTED,
              alpha=0.28, zorder=0)
    b.text(23.2, 0.40, "21.3% 이상\n전부 1.0", fontsize=8.8,
           color=T.TEXT_MUTED, ha="center", linespacing=1.3)

    b.axvline(T.RISK_SLOPE_PCT, color=T.ACCENT, ls="--", lw=1.3, zorder=2)
    b.text(T.RISK_SLOPE_PCT + 0.45, 0.93, "위험군 기준\n5.56%", fontsize=9,
           color=T.ACCENT, linespacing=1.3, va="top")

    for s, lab in [(5.56, "0.414"), (10, "0.534"), (15, "0.741"),
                   (20, "0.931")]:
        v = float(c.loc[(c.slope_pct - s).abs().idxmin(), "risk"])
        b.scatter([s], [v], s=50, color=T.PROPOSED, edgecolor="white",
                  linewidth=1.5, zorder=4)
        b.annotate(lab, (s, v), textcoords="offset points", xytext=(6, -14),
                   fontsize=9.4, color=T.ACCENT, fontweight="bold")

    b.set_xlabel("도로 경사 (%)", fontsize=10)
    b.set_ylabel("미끄러질 확률  P(MAA < θ)", fontsize=10)
    b.set_xlim(0, 25); b.set_ylim(0, 1.06)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.grid(alpha=0.22, lw=0.7)
    b.set_title("경사가 커질수록 미끄러질 확률이 오른다",
                fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=10)

    # 정의를 글로 풀어 쓰는 대신 **식 하나**로 보인다. 아래 각주 두 줄은
    # 같은 말을 반복하던 것이라 걷어냈다(2026-10-11) — 남기는 것은 출처뿐.
    # 아래첨자 ᵢ(U+1D62)는 Noto Sans CJK 에 없다 — mathtext 로 쓴다.
    b.text(0.975, 0.055, r"$\mathrm{risk}_i = P(\mathrm{MAA} < \theta_i)$",
           transform=b.transAxes, ha="right", va="bottom",
           fontsize=15.0, color=T.ACCENT, fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.42", fc=T.BG_BOX,
                     ec=T.GU_LINE, lw=0.8))
    b.text(0.975, 0.215,
           "MAA = 빙판에서 미끄러지지 않는 최대 경사각 (실측)",
           transform=b.transAxes, ha="right", va="bottom",
           fontsize=8.8, color=T.TEXT_MUTED)

    # 출처는 Canva 슬라이드에 14pt 텍스트로 따로 올린다. PNG 안에 8.4pt 로
    # 구우면 슬라이드 크기에서 읽히지 않는다 → _bare 에서는 숨긴다.
    src = fig.text(0.5, -0.07,
                   "자료: Dadkhah et al.(2026) Applied Ergonomics 131:104678, "
                   "Fig.1 — 토론토재활원 WinterLab 빙판 보행 58세션",
                   ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.tight_layout()
    T.save_both(fig, OUT / "p7a_risk_curve.png", hide_in_bare=[src])


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)
    edf(plt)
    curve(plt)


if __name__ == "__main__":
    main()
