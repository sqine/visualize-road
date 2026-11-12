"""진입점 — 열선 취약구역 확인 및 배치 시스템.

    streamlit run app.py

페이지
    pages/visualize.py  시각화 (서울 전역 열선 현황 · 골목길 미끄러짐 위험 · 기대 미끄러짐)
    pages/optimize.py   최적배치 (지역 데이터 점검 · 위험구간 · MILP 예산/δ 탐색)
    pages/guide.py      설명 (README.md 렌더링 + GitHub 링크)
"""

import importlib
import os
import sys

import streamlit as st

TITLE = "열선 취약구역 확인 및 배치 시스템"

st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")


def _reload_changed_utils() -> None:
    """배포 서버는 git push 뒤 페이지 파일만 다시 실행하고, 이미 불러온 utils 모듈은
    옛 버전을 그대로 쓴다(새 함수를 찾지 못해 AttributeError). 파일이 바뀐 utils 모듈이
    있으면 utils 전체를 다시 불러온다. 서로 참조하므로 두 번 돌려 옛 참조를 없앤다."""
    mods = {n: m for n, m in sys.modules.items()
            if (n == "utils" or n.startswith("utils.")) and getattr(m, "__file__", None)}
    changed = False
    for m in mods.values():
        try:
            mt = os.path.getmtime(m.__file__)
        except OSError:
            continue
        if getattr(m, "_src_mtime", mt) != mt:
            changed = True
        m._src_mtime = mt
    if changed:
        for _ in range(2):
            for n in sorted(mods):
                importlib.reload(sys.modules[n])
        for m in mods.values():
            m._src_mtime = os.path.getmtime(m.__file__)
        st.cache_resource.clear()   # 옛 클래스로 만든 객체(그래프·최적화 문맥) 폐기
        st.cache_data.clear()


_reload_changed_utils()

nav = st.navigation({
    TITLE: [
        st.Page("pages/visualize.py", title="시각화", url_path="visualize", default=True),
        st.Page("pages/optimize.py", title="최적배치", url_path="optimize"),
        st.Page("pages/guide.py", title="설명", url_path="guide"),
    ]
})
nav.run()
