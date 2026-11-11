"""PPT 삽입용 표 — 그림과 분리해서 따로 만든다.

지도 안에 표를 얹으면 지도가 가려지고, 슬라이드에서 표만 키우거나 옮길 수도
없다. 그래서 두 형식으로 따로 뽑는다.

    *.tsv   PPT·엑셀에 붙여넣어 **편집 가능한 표**로 쓴다
    *.png   그대로 올려 쓰는 **이미지 표** (테마 색 적용)

표를 그림 안에 넣으면 슬라이드에서 표만 키우거나 옮길 수 없고, 지도의 경우
지도를 가린다. 그래서 항상 분리한다.

실행: cd streamlit_app && python3 scripts/build_scenario_table.py
산출: viz/out/표_p10_전략비교.{tsv,png} · 표_p12_시나리오.{tsv,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viz import theme as T  # noqa: E402

VIZ = ROOT / "viz" / "data"
OUT = ROOT / "viz" / "out"

COLS = ["구분", "예산", "δ", "선정", "커버리지", "기대방지", "경계", "노출",
        "노출/편익", "현행 대비"]


def rows() -> list[list[str]]:
    sc = pd.read_csv(VIZ / "p12_scenarios.csv")
    label = {"현행 실제": "현행 실제",
             "시나리오1": "재배치 · 같은 돈",
             "시나리오2": "효율 최대 · +30%"}
    out = []
    for key in ["현행 실제", "시나리오1", "시나리오2"]:
        r = sc[sc.시나리오 == key].iloc[0]
        out.append([
            label[key],
            f"{r.예산억:.2f}억",
            "—" if pd.isna(r.delta) else f"{r.delta:.0f}",
            f"{int(r.선정)}구간",
            "—" if key == "현행 실제" else f"{r.커버리지:.1f}%",
            f"{r.기대방지:,.1f}",
            f"{int(r.경계수)}개",
            f"{r.경계노출:.1f}",
            f"{r.노출편익비:.1f}%",
            f"{r.현행대비:.1f}배",
        ])
    return out


def render(name, cols, body, widths, figsize, style):
    """공통 표 렌더러 — TSV 와 PNG 를 함께 떨군다.

    style(ri) → (배경색, 글자색, 굵게) 를 돌려주는 함수. ri=0 은 헤더.
    """
    (OUT / f"{name}.tsv").write_text(
        "\n".join(["\t".join(cols)] + ["\t".join(r) for r in body]),
        encoding="utf-8-sig")

    plt = T.use_matplotlib_theme()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tab = ax.table(cellText=body, colLabels=cols, cellLoc="center",
                   loc="center", colWidths=widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(11.2)
    tab.scale(1, 2.25)
    for (ri, ci), c in tab.get_celld().items():
        c.set_edgecolor(T.GU_LINE); c.set_linewidth(0.7)
        bg, fg, bold = style(ri, ci)
        c.set_facecolor(bg)
        c.set_text_props(color=fg, fontweight="bold" if bold else "normal")
    fig.savefig(OUT / f"{name}.png", dpi=T.FIG_DPI, bbox_inches="tight",
                facecolor="white")
    print(f"저장: {name}.tsv · {name}.png")


# ── p10 전략 비교 ────────────────────────────────────────────────
# 다섯 전략을 같은 예산·같은 후보군·같은 MILP 로 다시 푼 결과를 쓴다
# (scripts/build_strategy_balance.py → p10_balance.csv).
P10_COLS = ["선택 기준", "선정", "기대방지", "경계", "경계노출", "노출/편익",
            "평균경사", "평균통행"]
#: 전환점 δ=2 행은 슬라이드에서 뺀다 — 채택안 하나만 세워야 메시지가 선다
P10_ORDER = ["현행 실제", "① 경사(위험)만", "② 통행만", "③ 위험 × 통행",
             "⑤ 제안 채택안 (δ=4)"]
P10_LABEL = {"⑤ 제안 채택안 (δ=4)": "④ 제안 — 경계까지 (δ=4)"}


def p10_table() -> None:
    d = pd.read_csv(VIZ / "p10_balance.csv")
    d = d[d.전략.isin(P10_ORDER)].copy()
    d["_o"] = d.전략.map({k: i for i, k in enumerate(P10_ORDER)})
    d = d.sort_values("_o").reset_index(drop=True)
    body = [[P10_LABEL.get(r.전략, r.전략), f"{int(r.선정수)}",
             f"{r.기대방지:,.1f}", f"{int(r.경계수)}개", f"{r.경계노출:,.1f}",
             f"{r.노출편익비:.1f}%", f"{r.평균경사:.2f}%", f"{r.평균통행:.1f}"]
            for r in d.itertuples()]

    def style(ri, ci):
        if ri == 0:
            return "#EDF0F3", T.ACCENT, True
        nm = d.전략.iloc[ri - 1]
        if nm.startswith("⑤"):
            return "#FDEDE8", T.PROPOSED if ci in (0, 3, 4) else T.TEXT, True
        if nm.startswith("현행"):
            return "#EDF3FD", T.TEXT_MUTED, False
        return "white", T.TEXT, False

    render("표_p10_전략비교", P10_COLS, body,
           [.235, .075, .115, .085, .115, .105, .105, .105],
           (12.0, 2.8), style)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    p10_table()

    body = rows()

    def style(ri, ci):
        if ri == 0:
            return "#EDF0F3", T.ACCENT, True
        if ri == 1:                                   # 현행 실제
            return "#F7F8FA", T.TEXT_MUTED, False
        if ri == 2:                                   # 재배치
            fg = T.PROPOSED if ci == 0 else (
                T.BOUNDARY_DARK if ci == 6 else T.TEXT)
            return "#FDEDE8", fg, True
        fg = T.ADDED if ci == 0 else (                # 효율 최대
            T.BOUNDARY_DARK if ci == 6 else T.TEXT)
        return "#E4F2F2", fg, True

    render("표_p12_시나리오", COLS, body,
           [.185, .085, .045, .082, .092, .105, .068, .082, .098, .095],
           (11.6, 1.95), style)


if __name__ == "__main__":
    main()
