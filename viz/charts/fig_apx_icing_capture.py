"""p7 — 결빙취약구간 지정의 경사 골목 포착률 (관측 vs 기대).

메시지: 성북구는 위험군 연장도 1위, 깔아 놓은 열선 연장도 1위다.
       그런데 결빙 지정이 경사 골목을 잡는 비율은 기대의 0.48배 —
       지형으로는 설명되지 않는다. 계획 자체를 다시 세울 여지가 있다.

표본 주의
---------
자치구별 매칭 건수가 1~51건으로 들쭉날쭉하다. 매칭 2~7건짜리 구는 배수가
크게 흔들리므로 **매칭 20건 이상**인 구만 비교한다(6개 구, 전체 매칭의 56%).

실행: cd streamlit_app && python3 viz/charts/fig_apx_icing_capture.py
산출: viz/out/apx_icing_capture.png
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

HL = "성북구"
#: 배수가 안정적으로 읽히는 최소 매칭 건수
MIN_N = 20


def main() -> None:
    plt = T.use_matplotlib_theme()
    OUT.mkdir(exist_ok=True)

    d = pd.read_csv(VIZ / "icing_capture_by_gu_v2.csv", encoding="utf-8-sig")
    r = pd.read_csv(VIZ / "p6_gu_risk_length.csv")
    m = d.merge(r[["자치구", "위험군연장km", "신고열선m", "커버리지"]], on="자치구")
    m = m[m.매칭 >= MIN_N].sort_values("ratio556").reset_index(drop=True)
    print(f"비교 대상 {len(m)}개 구 · 매칭 {m.매칭.sum()}건 "
          f"(전체 {d.매칭.sum()}건의 {100*m.매칭.sum()/d.매칭.sum():.0f}%)")

    fig, ax = plt.subplots(figsize=(10.4, 4.5))
    y = np.arange(len(m))
    h = 0.34

    for i, row in m.iterrows():
        hot = row.자치구 == HL
        ax.barh(i + h / 2 + 0.02, row.obs556 * 100, height=h,
                color=T.ACCENT if hot else T.CURRENT,
                label="관측 (지정구간 중 경사 골목 비율)" if i == 0 else None)
        ax.barh(i - h / 2 - 0.02, row.base556 * 100, height=h,
                color=T.BG_LEGEND,
                label="무작위 지정 시 기대값" if i == 0 else None)
        ax.text(row.obs556 * 100 + 1.2, i + h / 2 + 0.02,
                f"{row.obs556*100:.1f}%", va="center", fontsize=9.4,
                color=T.ACCENT if hot else T.CURRENT, fontweight="bold")
        ax.text(row.base556 * 100 + 1.2, i - h / 2 - 0.02,
                f"{row.base556*100:.1f}%", va="center", fontsize=9.2,
                color=T.TEXT_MUTED)
        sig = "" if abs(row.z556) < 1.96 else " *"
        ax.text(101, i, f"×{row.ratio556:.2f}{sig}", va="center",
                ha="left", fontsize=11,
                color=T.BOUNDARY if row.ratio556 < 1 else T.TEXT,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}\n(지정 {b}건)" for a, b in zip(m.자치구, m.매칭)],
                       fontsize=9.6, linespacing=1.4)
    for lab, nm in zip(ax.get_yticklabels(), m.자치구):
        if nm == HL:
            lab.set_fontweight("bold"); lab.set_color(T.ACCENT)
    ax.set_xlim(0, 100)
    ax.set_xlabel("지정구간 중 경사 골목(골목길 ∩ 경사 5.56%↑) 비율 (%)",
                  fontsize=9.8)
    ax.axvline(0, color=T.GU_LINE, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.2, lw=0.7)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.93)

    ax.set_title("결빙 지정의 경사 골목 포착률 — 성북구만 기대를 밑돈다",
                 fontsize=T.TS_PANEL, color=T.ACCENT, fontweight="bold", pad=12)

    fig.text(0.5, -0.045,
             T.caption("매칭 20건 이상 6개 구", "지정 193건 (전체 341건의 57%)",
                       "기대값 = 구내 전체 링크의 경사 골목 비율",
                       "icing_capture_by_gu_v2.csv"),
             ha="center", fontsize=8.4, color=T.TEXT_MUTED)
    fig.text(0.5, -0.095,
             "성북구는 위험군 골목 138.8 km(1위)에 열선 12.3 km(1위)를 깔아 놓고도 "
             "지정은 경사 골목을 기대의 0.48배만 집는다.\n"
             "경사 2% 기준으로 보면 0.45배(z = −2.72)로 유의하다. "
             "지형 조건으로 설명되지 않으므로, 설치 계획 자체를 다시 세울 여지가 있다.",
             ha="center", fontsize=9, color=T.BOUNDARY, linespacing=1.6)
    fig.text(0.5, -0.158, "* |z| > 1.96", ha="center", fontsize=8.2,
             color=T.TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "apx_icing_capture.png", dpi=T.FIG_DPI, bbox_inches="tight")
    print("저장: viz/out/apx_icing_capture.png")

    m.to_csv(VIZ / "apx_icing_capture.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
