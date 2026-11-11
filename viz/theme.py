"""시각화 테마 — 색·폰트·캡션의 단일 출처.

지도(viz/maps), 차트(viz/charts), Streamlit 앱, 발표자료 PNG가 **전부 여기만**
참조한다. 색을 바꾸려면 이 파일만 고친다.

규칙 5가지
----------
1. 파랑은 언제나 **현행/보호**, 코랄은 언제나 **제안**.
   지도·차트·표 강조가 모두 동일해야 한 장면에서 다음 장면으로 눈이 이어진다.
2. 경사 농도가 필요하면 SLOPE 4등급만 쓴다. 연속 그라데이션은 등급 비교를
   흐리므로 금지.
3. 캡션은 `구분 · 모집단 · 기준 · 출처파일` 4요소 고정 (caption() 사용).
4. 비율을 쓸 때는 **분모를 캡션에 명시**한다.
   예: 75.8%는 "하류 이웃 보유 열선 1,590 기준".
5. 위첨자 글리프가 Noto Sans CJK에 없으므로 p값은 `p = 6.9e-135` 표기로 통일.
"""

from __future__ import annotations

# ── 의미색 ───────────────────────────────────────────────────────
#: 현행 열선 / 보호된 구간
CURRENT = "#1F6FEB"
#: 제안 모형이 선정한 구간 (MILP 해)
PROPOSED = "#E0674A"
#: 재결빙 경계 · 하류 위험 구간
BOUNDARY = "#C23B22"
#: 위험군이지만 보호되지 않은 구간
UNPROTECTED = "#B9C2CE"

#: 지도 배경 — 위험군이 아닌 일반 도로
ROAD_BG = "#E8EBEF"
#: 자치구 경계선
GU_LINE = "#9AA5B1"
#: 강조(성북구 등) 테두리·라벨
ACCENT = "#1F3A5F"
#: 본문 텍스트
TEXT = "#2B3348"
#: 보조 텍스트·캡션
TEXT_MUTED = "#6B7685"

SEMANTIC = {
    "현행": CURRENT, "보호": CURRENT,
    "제안": PROPOSED,
    "경계": BOUNDARY, "하류": BOUNDARY,
    "위험군": UNPROTECTED, "미보호": UNPROTECTED,
}

# ── 경사 4등급 (utils/geo.py SLOPE_COLORS 와 동일) ───────────────
SLOPE_HEX = {
    "0_3": "#B4B4B4",
    "3_6": "#FFD250",
    "6_10": "#F08228",
    "10_plus": "#DC2828",
    "unknown": "#C8C8C8",
}
SLOPE_RGBA = {
    "0_3": [180, 180, 180, 110],
    "3_6": [255, 210, 80, 200],
    "6_10": [240, 130, 40, 220],
    "10_plus": [220, 40, 40, 240],
    "unknown": [200, 200, 200, 80],
}

# ── 도로유형 (utils/geo.py ROAD_TYPE_COLORS 와 동일) ─────────────
ROAD_TYPE_RGBA = {
    "골목길": [70, 170, 100, 190],
    "애매": [230, 195, 60, 190],
    "대형도로": [70, 110, 210, 170],
    "unknown": [150, 150, 150, 80],
}

# ── 기준값 ───────────────────────────────────────────────────────
#: 위험군 경사 기준(%) — 장애인등편의법 접근로 1/18
RISK_SLOPE_PCT = 100 / 18
#: 급경사 기준(%)
STEEP_SLOPE_PCT = 10.0
#: risk 가 1.0 으로 포화되는 경사(%) — MAA 12°
SATURATION_SLOPE_PCT = 100.0 * __import__("math").tan(__import__("math").radians(12.0))

# ── 폰트 ─────────────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_NAME = "Noto Sans CJK KR"

#: 발표자료 PNG 기본 크기 (16:9)
FIG_SIZE_16_9 = (12.8, 7.2)
FIG_DPI = 300


def use_matplotlib_theme():
    """matplotlib 전역 설정 — 한글 폰트 + 기본 스타일.

    각 그림 스크립트 맨 위에서 한 번 호출한다.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    try:
        fm.fontManager.addfont(FONT_PATH)
        name = fm.FontProperties(fname=FONT_PATH).get_name()
    except Exception:
        name = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": name,
        "axes.unicode_minus": False,
        "axes.edgecolor": GU_LINE,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "figure.dpi": 110,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
    })
    return plt


# ── 중립 배경색 (의미 없음. 눈에 띄면 안 되는 것들) ─────────────
#: 지도 배경 도로 · 100% 막대의 빈 부분
BG_FAINT = "#EDEFF2"
#: 범례에서 BG_FAINT 를 보여줄 때 (가는 선이라 조금 진하게)
BG_LEGEND = "#DDE1E6"
#: 설명 상자 배경
BG_BOX = "#F4F6F8"

# ── 글자 크기 ────────────────────────────────────────────────────
#: 그림 전체 제목
TS_MAIN = 13.5
#: 패널 제목 (a)(b)
TS_PANEL = 12.2

#: 예산을 더 썼을 때 추가로 들어오는 구간 (제안 코랄과 확실히 구분)
ADDED = "#0E8F8F"

#: 채택안 표식(★) — 그림의 결론. 코랄 선·적색 점선·파랑 기준선 어느 것과도
#: 겹치지 않는 고채도 보라. ACCENT(남색)는 축·라벨 색이라 선 위에서 묻혔다.
ADOPTED = "#098E91"

#: 주의·문제 강조 — 디자인 시스템(p16) TEXT COLORS
WARN = "#C74646"

# ── 경계 기호 규칙 (지도 전부 동일) ─────────────────────────────
#   ▼ 파랑 채움      현행 열선이 만드는 경계
#   ★ 적색 채움      제안 배치에 그대로 남는 경계
#   ☆ 적색 속빈      예산을 늘리면 해소되는 경계
#   크기 ∝ 하류 노출 (risk × flow). 기준은 그림 안에서 통일한다.
#: 현행 열선이 만드는 경계
BOUNDARY_CURRENT = "#1B4F9C"
#: 제안 배치가 남기는 경계 (코랄 선과 짝)
BOUNDARY_DARK = "#7E1F10"
#: 증액으로 추가된 구간이 남기는 경계 (청록 선과 짝)
BOUNDARY_ADDED = "#0A5C5C"


def boundary_colors(exposure, which="proposed"):
    """하류 노출을 **색 농도**로 바꾼다.

    크기로 나타내면 큰 표식이 주변 도로를 가린다. 기호 크기는 고정하고
    노출은 색의 진하기로 읽게 한다(연할수록 작고, 진할수록 크다).
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    pair = {"proposed": ("#F0BFB4", BOUNDARY_DARK),
            "added": ("#9ED4D4", BOUNDARY_ADDED),
            "current": ("#A8C2E8", BOUNDARY_CURRENT)}[which]
    cmap = LinearSegmentedColormap.from_list("bd", list(pair))
    v = np.asarray(exposure, float)
    if v.size == 0:
        return v
    lo, hi = float(v.min()), float(v.max())
    t = (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.7)
    return cmap(0.25 + 0.75 * t)


def save_both(fig, path, keep_panel_title: bool = False, hide_in_bare=()):
    """설명본과 발표본을 함께 저장한다.

    - `path`          : 제목·캡션·각주가 모두 있는 **설명본**
    - `path`_bare.png : 그것들을 뺀 **발표자료용**. 축 라벨·범례·수치 주석은
                        남긴다(그래야 슬라이드에서도 읽힌다).

    발표자료에서는 제목을 슬라이드 헤드라인이 대신하고 출처는 별도 표기하므로,
    그림 안에 같은 말이 두 번 들어가지 않게 한다.

    `hide_in_bare` 에는 **축 안에 그린 설명 상자**처럼 발표본에서 빼야 하는
    artist 를 넘긴다. fig.suptitle / fig.text 는 자동으로 빠지지만 ax.text 나
    ax.annotate 는 그림의 일부라 자동 판별이 안 된다.
    """
    from pathlib import Path as _P
    path = _P(path)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")

    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    for t in list(fig.texts):
        t.set_visible(False)
    for art in hide_in_bare:
        if art is not None:
            art.set_visible(False)
    if not keep_panel_title:
        for ax in fig.axes:
            ax.set_title("")
    bare = path.with_name(path.stem + "_bare" + path.suffix)
    fig.savefig(bare, dpi=FIG_DPI, bbox_inches="tight")
    print(f"저장: {path.parent.name}/{path.name}  ·  {bare.name}")


def caption(구분: str, 모집단: str, 기준: str, 출처: str) -> str:
    """캡션 4요소 고정 포맷.

    >>> caption("서울 전역", "열선 2,243구간", "경사 5.56% 이상",
    ...         "seoul_links.parquet")
    '서울 전역 · 열선 2,243구간 · 경사 5.56% 이상 · seoul_links.parquet'
    """
    return " · ".join([구분, 모집단, 기준, 출처])


def pval(p: float) -> str:
    """p값 표기 통일 — 위첨자 대신 지수 표기.

    >>> pval(6.9e-135)
    'p = 6.9e-135'
    """
    return f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
