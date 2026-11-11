"""예산 × δ 이중 탐색 그림 — 앱(pages/optimize.py)과 스크립트(run_sensitivity_grid.py)가 함께 쓴다.

세 그림
-------
1. 격자 히트맵    예산(행) × δ(열) 칸에 지표 값. 예산선마다 전환점(하한)과 상한을 테두리로 표시.
2. 파레토 프런트  예산선마다 경사위험 방지(편익) vs 하류 전이위험. 예산이 클수록 진한 선.
3. 권장 δ 범위   예산별 하한(전환점)~상한(편익 5% 손실) 띠 — "예산을 정하면 δ가 따라온다".
"""

from __future__ import annotations

import altair as alt
import pandas as pd

INK = "#2B3348"
MUTED = "#9AA5B4"
KNEE = "#C23B22"
ACCENT = "#E0674A"
SEQ = "oranges"          # 단일 색상 순차 램프 (크기)

METRICS = {
    "편익손실%": "편익 손실 (%)",
    "감소%_경계통행": "전이위험 감소 (%, 경계 통행)",
    "감소%_하류통행": "전이위험 감소 (%, 하류 통행)",
    "경사위험방지": "경사위험 방지 (건)",
    "위험경계": "위험경계 (곳)",
    "전이위험_경계통행": "하류 전이위험 (경계 통행)",
    "전이위험_하류통행": "하류 전이위험 (하류 통행)",
}
TIP = ["예산억", "δ", "선정", "경사위험방지", "편익손실%", "위험경계",
       "전이위험_경계통행", "감소%_경계통행", "전이위험_하류통행", "감소%_하류통행"]


def _tips(df: pd.DataFrame) -> list:
    fmt = {"예산억": ",.2f", "δ": "~g", "편익손실%": ".1f", "감소%_경계통행": ".1f",
           "감소%_하류통행": ".1f", "경사위험방지": ",.1f"}
    return [alt.Tooltip(f"{c}:Q", format=fmt.get(c, ",.1f") if c not in ("선정", "위험경계") else "d")
            for c in TIP if c in df.columns]


def _marks(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """각 칸에 '전환점'/'상한' 표시 열을 붙인다."""
    s = summary.set_index("_B")
    d = df.drop(columns=["선정링크", "_ben"], errors="ignore").copy()   # 그림 데이터는 가볍게
    d["전환점"] = [s.at[b, "하한δ"] == v if b in s.index else False for b, v in zip(d["_B"], d["δ"])]
    d["상한"] = [s.at[b, "상한δ"] == v if b in s.index else False for b, v in zip(d["_B"], d["δ"])]
    return d


def heatmap(df: pd.DataFrame, summary: pd.DataFrame, metric: str = "감소%_경계통행",
            sel: tuple[float, float] | None = None, height: int | None = None) -> alt.Chart:
    d = _marks(df, summary)
    d["δ표시"] = d["δ"].map(lambda v: f"{v:g}")
    d["예산표시"] = d["예산억"].map(lambda v: f"{v:,.2f}억")
    dorder = [f"{v:g}" for v in sorted(d["δ"].unique())]
    border = [f"{v:,.2f}억" for v in sorted(d["예산억"].unique(), reverse=True)]
    h = height or max(220, 22 * len(border))
    x = alt.X("δ표시:O", sort=dorder, title="δ (전이위험 가중치)", axis=alt.Axis(labelAngle=0))
    y = alt.Y("예산표시:O", sort=border, title="예산")
    base = alt.Chart(d).encode(x=x, y=y)
    cells = base.mark_rect(stroke="white", strokeWidth=2, cornerRadius=2).encode(
        color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme=SEQ),
                        legend=alt.Legend(title=METRICS.get(metric, metric), orient="right",
                                          gradientLength=max(120, h - 60))),
        tooltip=_tips(d))
    layers = [cells,
              base.transform_filter(alt.datum.상한).mark_rect(
                  fill=None, stroke=INK, strokeWidth=1.5, strokeDash=[3, 2]),
              base.transform_filter(alt.datum.전환점).mark_rect(
                  fill=None, stroke=INK, strokeWidth=2.5)]
    if sel is not None:
        B, dv = sel
        layers.append(alt.Chart(d[(d["_B"] == B) & (d["δ"] == dv)]).encode(x=x, y=y)
                      .mark_point(shape="circle", size=70, filled=True, color="white",
                                  stroke=INK, strokeWidth=2))
    return alt.layer(*layers).properties(height=h)


def fronts(df: pd.DataFrame, summary: pd.DataFrame, ycol: str = "전이위험_경계통행",
           sel_B: float | None = None, height: int = 340) -> alt.Chart:
    """예산선별 파레토 프런트. x=편익, y=하류 전이위험(작을수록 좋음)."""
    d = _marks(df, summary).sort_values(["_B", "δ"])
    color = alt.Color("예산억:Q", scale=alt.Scale(scheme={"name": SEQ, "extent": [0.35, 1]}),
                      legend=alt.Legend(title="예산 (억)", format=",.1f"))
    base = alt.Chart(d).encode(
        x=alt.X("경사위험방지:Q", title="경사위험 방지 (건) →", scale=alt.Scale(zero=False)),
        y=alt.Y(f"{ycol}:Q", title=METRICS.get(ycol, ycol) + " ↓"),
        detail="예산억:Q", order="δ:Q")
    lines = base.mark_line(strokeWidth=2).encode(color=color)
    pts = base.mark_point(filled=True, size=40, stroke="white", strokeWidth=1).encode(
        color=color, tooltip=_tips(d))
    knee = base.transform_filter(alt.datum.전환점).mark_point(
        shape="diamond", filled=True, size=120, color=KNEE, stroke="white", strokeWidth=1.5)
    layers = [lines, pts, knee]
    if sel_B is not None:
        layers.insert(0, alt.Chart(d[d["_B"] == sel_B]).mark_line(
            strokeWidth=7, color=INK, opacity=0.3).encode(
            x="경사위험방지:Q", y=f"{ycol}:Q", order="δ:Q"))
    return alt.layer(*layers).properties(height=height)


def band(summary: pd.DataFrame, cur_budget_eok: float | None = None, height: int = 260) -> alt.Chart:
    """예산별 권장 δ 범위: 하한(전환점) ~ 상한(편익 5% 손실 이내 최대 δ)."""
    s = summary.dropna(subset=["하한δ"]).copy()
    s["상한표시"] = s["상한δ"].fillna(s["하한δ"])
    x = alt.X("예산억:Q", title="예산 (억)")
    area = alt.Chart(s).mark_area(opacity=0.22, color=ACCENT).encode(
        x=x, y=alt.Y("하한δ:Q", title="δ"), y2="상한표시:Q")
    up = alt.Chart(s).mark_line(color=MUTED, strokeDash=[4, 3], strokeWidth=2).encode(
        x=x, y="상한표시:Q")
    lo = alt.Chart(s).mark_line(color=KNEE, strokeWidth=2).encode(x=x, y="하한δ:Q")
    pts = alt.Chart(s).mark_point(color=KNEE, filled=True, size=50).encode(
        x=x, y="하한δ:Q",
        tooltip=[alt.Tooltip("예산억:Q", format=",.2f"), alt.Tooltip("하한δ:Q", title="하한 δ(전환점)", format="~g"),
                 alt.Tooltip("상한δ:Q", title="상한 δ(5%)", format="~g"),
                 alt.Tooltip("전환점_편익손실%:Q", title="전환점 편익손실%", format=".1f"),
                 alt.Tooltip("전환점_감소%:Q", title="전환점 전이위험 감소%", format=".1f")])
    layers = [area, up, lo, pts]
    if cur_budget_eok:
        layers.append(alt.Chart(pd.DataFrame({"x": [cur_budget_eok]})).mark_rule(
            color=INK, strokeDash=[2, 2]).encode(x="x:Q"))
    return alt.layer(*layers).properties(height=height)


def report_html(df: pd.DataFrame, summary: pd.DataFrame, meta: dict) -> str:
    """스크립트용 단독 HTML 보고서 (인터넷 연결 시 vega CDN으로 그린다)."""
    cur = meta.get("current_budget_eok")
    charts = alt.vconcat(
        heatmap(df, summary, "감소%_경계통행").properties(
            title="예산 × δ — 하류 전이위험 감소 (%, 경계 통행)  ·  진한 테두리 = 전환점(하한), 점선 = 상한(편익 5% 손실)",
            width=520),
        heatmap(df, summary, "편익손실%").properties(title="예산 × δ — 편익 손실 (%)", width=520),
        fronts(df, summary).properties(title="예산선별 파레토 프런트 (◆ = 전환점)", width=520),
        band(summary, cur).properties(title="예산별 권장 δ 범위", width=520),
        spacing=36,
    ).resolve_scale(color="independent").configure_axis(labelColor=INK, titleColor=INK, gridColor="#E6E9EE", domainColor=MUTED
    ).configure_view(stroke=None).configure_title(color=INK, fontSize=13, anchor="start")
    head = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in meta.items() if k != "columns")
    tbl = summary.drop(columns=["_B"]).to_html(index=False, float_format=lambda v: f"{v:,.2f}", na_rep="-")
    chart_html = charts.to_html()
    body = chart_html.split("<body>", 1)[1].rsplit("</body>", 1)[0] if "<body>" in chart_html else chart_html
    headpart = chart_html.split("<head>", 1)[1].split("</head>", 1)[0] if "<head>" in chart_html else ""
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>예산 × δ 이중 탐색 결과</title>{headpart}
<style>body{{font-family:-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;color:{INK};max-width:980px;margin:32px auto;padding:0 20px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0 24px}}th,td{{border-bottom:1px solid #E6E9EE;padding:4px 10px;text-align:right}}
th{{text-align:left;color:#5B6576;font-weight:600}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}</style></head><body>
<h1>예산 × δ 이중 탐색 결과</h1>
<p>이 파일은 <code>scripts/run_sensitivity_grid.py</code>가 만든 보고서입니다. 같은 폴더의 <code>sensitivity_grid.parquet</code>를
앱의 최적배치 → MILP → 이중 탐색 탭에 올리면 칸을 골라 지도에서 배치를 볼 수 있습니다.</p>
<h2>계산 조건</h2><table>{head}</table>
{body}
<h2>예산별 권장 δ</h2>{tbl}
</body></html>"""
