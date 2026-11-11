"""δ 실험실 — 경계 패널티 가중치의 트레이드오프를 보는 Streamlit 섹션.

질문: 경계를 하나 줄이려면 편익을 얼마나 포기해야 하는가?

`scripts/run_delta_experiment.py`가 만든 data/exp_delta.csv를 읽어 렌더하고,
필요하면 같은 함수(utils.milp.solve_msa)로 즉석 재실행한다. 배치와 앱이
같은 코드를 쓰므로 발표 숫자와 화면이 어긋나지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"
EXP_CSV = DATA / "exp_delta.csv"
SEL_CSV = DATA / "exp_selection.csv"

# 의미색 — 지도·차트·표 전부 이 네 가지만 쓴다
C_CURRENT = "#1F6FEB"   # 현행
C_PROPOSED = "#E0674A"  # 제안
C_BOUNDARY = "#C23B22"  # 경계
C_MUTED = "#B9C2CE"     # 배경

MODE_LABEL = {
    "current": "현행 재배치 (18.53억 · 현행 열선 2,101 m 재투입)",
    "model": "모형 기준 (8.81억 · 보고서 본문 기준선)",
}


@st.cache_data(show_spinner=False)
def _load_exp() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not EXP_CSV.exists():
        return None, None
    d = pd.read_csv(EXP_CSV)
    s = pd.read_csv(SEL_CSV) if SEL_CSV.exists() else None
    return d, s


def _knee(df: pd.DataFrame) -> float:
    """트레이드오프 곡선의 파레토 전환점 — 양 끝점을 잇는 현에서 가장 먼 점.

    '단위 손실당 효율 최대'는 오목한 곡선에서 언제나 첫 점을 고르므로
    쓰지 않는다. 여기서는 곡선이 꺾이는 지점, 즉 그 뒤로는 손실만
    늘고 경계 감소가 둔해지는 δ를 찾는다 (Kneedle 방식).
    """
    base = df.loc[df.delta == 0, "경계노출"].iloc[0]
    x = df["편익손실%"].to_numpy(float)
    y = (100 * (1 - df["경계노출"] / base)).to_numpy(float)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    norm = (dx * dx + dy * dy) ** 0.5
    if norm == 0:
        return 1.0
    # 현(chord)으로부터의 수직거리
    dist = abs(dy * (x - x[0]) - dx * (y - y[0])) / norm
    return float(df["delta"].iloc[int(dist.argmax())])


def _tradeoff_chart(df: pd.DataFrame, knee: float) -> alt.Chart:
    base = df.loc[df.delta == 0, "경계노출"].iloc[0]
    d = df.copy()
    d["경계노출감소%"] = 100 * (1 - d["경계노출"] / base)
    d["권장"] = d.delta == knee

    line = (
        alt.Chart(d)
        .mark_line(color=C_PROPOSED, strokeWidth=2.5, point=False)
        .encode(
            x=alt.X("편익손실%:Q", title="편익 손실 (%)"),
            y=alt.Y("경계노출감소%:Q", title="경계 노출 감소 (%)"),
        )
    )
    pts = (
        alt.Chart(d)
        .mark_point(filled=True, size=130)
        .encode(
            x="편익손실%:Q",
            y="경계노출감소%:Q",
            color=alt.condition(
                alt.datum.권장, alt.value(C_BOUNDARY), alt.value(C_PROPOSED)
            ),
            tooltip=[
                alt.Tooltip("delta:Q", title="δ"),
                alt.Tooltip("선정수:Q", title="선정 구간"),
                alt.Tooltip("기대방지:Q", title="기대 방지 건수", format=",.1f"),
                alt.Tooltip("편익손실%:Q", title="편익 손실(%)", format=".2f"),
                alt.Tooltip("잔여경계:Q", title="잔여 경계"),
                alt.Tooltip("경계노출:Q", title="경계 노출", format=",.1f"),
            ],
        )
    )
    labels = (
        alt.Chart(d)
        .mark_text(dy=-14, fontSize=11, color="#2B3348")
        .encode(x="편익손실%:Q", y="경계노출감소%:Q",
                text=alt.Text("delta:Q", format="g"))
    )
    return (line + pts + labels).properties(height=360).configure_view(stroke=None)


def _selection_map(sel: pd.DataFrame) -> pdk.Deck:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=sel,
        get_position="[lon, lat]",
        get_radius="3 + 18 * risk",
        radius_units="meters",
        get_fill_color=[224, 103, 74, 200],
        pickable=True,
    )
    view = pdk.ViewState(
        latitude=float(sel.lat.mean()), longitude=float(sel.lon.mean()),
        zoom=13.2, pitch=0,
    )
    return pdk.Deck(
        layers=[layer], initial_view_state=view,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip={"text": "{dong}\n경사 {slope_pct}% · 통행 {flow}\n기대 {기대미끄러짐}건"},
    )


def render_delta_lab() -> None:
    st.subheader("δ 실험실 — 경계를 줄이는 비용")
    st.caption(
        "경계 패널티 가중치 δ를 올리면 재결빙 경계는 줄지만 편익은 깎인다. "
        "그 교환비를 본다. 위험도는 Observed MAA EDF, 경계가중치는 "
        "current-flow 엣지전류(MSA 고정점), h_min은 2 m 고정."
    )

    df, sel = _load_exp()
    if df is None:
        st.info(
            "실험 결과가 없습니다. 터미널에서 다음을 실행하세요 (약 6초).\n\n"
            "```bash\npython3 scripts/run_delta_experiment.py\n```"
        )
        return

    mode = st.radio(
        "예산 모드", list(MODE_LABEL), format_func=MODE_LABEL.get,
        horizontal=True, key="delta_mode",
    )
    d = df[df.예산모드 == mode].sort_values("delta").reset_index(drop=True)
    knee = _knee(d)
    base = d.iloc[0]
    rec = d[d.delta == knee].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("권장 δ", f"{knee:g}")
    c2.metric("선정 구간", f"{int(rec.선정수)}곳",
              delta=f"{int(rec.선정수 - base.선정수)}곳 vs δ=0")
    c3.metric("잔여 경계", f"{int(rec.잔여경계)}곳",
              delta=f"-{rec['경계감소%']:.0f}%", delta_color="inverse")
    c4.metric("편익 손실", f"{rec['편익손실%']:.2f}%",
              delta=f"{rec.기대방지:,.1f}건 방지")

    st.markdown(
        f"**δ={knee:g}에서 재결빙 경계를 {int(base.잔여경계)}곳 → "
        f"{int(rec.잔여경계)}곳({rec['경계감소%']:.0f}% 감소)으로 줄이는 데 "
        f"편익은 {rec['편익손실%']:.2f}%만 포기한다.**"
    )

    st.altair_chart(_tradeoff_chart(d, knee), use_container_width=True)
    st.caption(
        "경계 '수'는 정수라 계단형으로 움직이므로, 세로축은 연속량인 "
        "**경계 노출**(경계 하류의 risk×flow 합)을 썼다. 붉은 점이 "
        "단위 편익 손실당 노출 감소가 가장 큰 지점이다."
    )

    st.markdown("##### δ별 결과")
    show = ["delta", "선정수", "선정연장m", "기대방지", "편익손실%",
            "잔여경계", "경계감소%", "경계노출", "평균경사", "평균통행", "건당만원"]
    st.dataframe(
        d[[c for c in show if c in d.columns]].rename(columns={"delta": "δ"}),
        use_container_width=True, hide_index=True,
    )

    if sel is not None:
        st.markdown("##### 선정 구간 지도")
        pick = st.select_slider(
            "δ", options=sorted(d.delta.unique()), value=knee, key="delta_map"
        )
        sub = sel[(sel.예산모드 == mode) & (sel.delta == pick)].dropna(
            subset=["lat", "lon"]
        )
        if len(sub):
            st.pydeck_chart(_selection_map(sub), use_container_width=True)
            st.caption(
                f"성북구 4개동 · 선정 {len(sub)}곳 · 원 크기 = 미끄러짐 위험 "
                f"P(MAA<θ) · exp_selection.csv"
            )

    with st.expander("재실행 (결과를 다시 계산)", expanded=False):
        st.caption(
            "`utils.milp.solve_msa`를 그대로 호출합니다. δ 11점 × 예산 2종이 "
            "약 6초 걸립니다."
        )
        if st.button("실험 재실행"):
            import subprocess, sys
            root = Path(__file__).resolve().parent.parent
            with st.spinner("MILP 재실행 중..."):
                p = subprocess.run(
                    [sys.executable, "scripts/run_delta_experiment.py"],
                    cwd=root, capture_output=True, text=True,
                )
            st.code(p.stdout[-2000:] or p.stderr[-2000:])
            _load_exp.clear()
            st.rerun()
