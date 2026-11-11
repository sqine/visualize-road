"""설명 — 저장소 README.md 를 그대로 보여준다.

README 의 상대경로 이미지(docs/images/…)는 로컬 파일이 있으면 그 자리에 st.image 로,
표 안의 이미지는 GitHub 원본 주소로 바꿔 표시한다.
"""

import re
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent.parent
REPO = "https://github.com/sqine/visualize-road"
RAW = "https://raw.githubusercontent.com/sqine/visualize-road/main/"
IMG_LINE = re.compile(r"^!\[([^\]]*)\]\((docs/images/[^)]+)\)\s*$")

c = st.columns([3, 1])
c[0].title("설명")
c[1].link_button("GitHub에서 보기", f"{REPO}#readme", use_container_width=True)

readme = APP / "README.md"
if not readme.exists():
    st.info(f"README.md 가 없습니다. [GitHub]({REPO}#readme)에서 확인하세요.")
    st.stop()

text = readme.read_text(encoding="utf-8")
buf: list[str] = []


def flush() -> None:
    if buf:
        md = re.sub(r"\]\((docs/images/[^)]+)\)", lambda m: f"]({RAW}{m.group(1)})", "\n".join(buf))
        st.markdown(md, unsafe_allow_html=False)
        buf.clear()


for line in text.splitlines():
    m = IMG_LINE.match(line.strip())
    if m and (APP / m.group(2)).exists():
        flush()
        st.image(str(APP / m.group(2)), caption=m.group(1) or None)
    else:
        buf.append(line)
flush()
