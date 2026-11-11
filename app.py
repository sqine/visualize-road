"""진입점 — 열선 취약구역 확인 및 배치 시스템.

    streamlit run app.py

페이지
    pages/visualize.py  시각화 (서울 전역 열선 현황 · 골목길 미끄러짐 위험 · 기대 미끄러짐)
    pages/optimize.py   최적배치 (지역 데이터 점검 · 위험구간 · MILP 예산/δ 탐색)
    pages/guide.py      설명 (README.md 렌더링 + GitHub 링크)
"""

import streamlit as st

TITLE = "열선 취약구역 확인 및 배치 시스템"

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

nav = st.navigation({
    TITLE: [
        st.Page("pages/visualize.py", title="시각화", url_path="visualize", default=True),
        st.Page("pages/optimize.py", title="최적배치", url_path="optimize"),
        st.Page("pages/guide.py", title="설명", url_path="guide"),
    ]
})
nav.run()
