"""유동인구 CSV 로딩 + 정규화 + 집계.

빅데이터캠퍼스 표준 형식 (도로구간별 추정 유동인구):
    기준_년월_코드, 도로링크_ID, 시군구코드/명, 행정동코드/명,
    요일코드, 연령대코드, 시간대코드, 유동인구_수

요일코드:   1=평일, 2=주말
연령대코드: 0=전체/미상, 10/20/30/40/50/60(=60대 이상)
시간대코드: 0~6 (캠퍼스 메타에 따라 다를 수 있음. 기본 가정은 아래)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# 기본 시간대 매핑 (3시간 단위 가정 — 캠퍼스 메타에 따라 다를 수 있음)
# 사용자가 실데이터로 수정 가능하도록 외부에서 override 가능
DEFAULT_TIME_LABELS = {
    0: "00–06 (심야)",
    1: "06–09 (출근)",
    2: "09–12 (오전)",
    3: "12–15 (오후)",
    4: "15–18 (저녁)",
    5: "18–21 (밤)",
    6: "21–24 (심야)",
}

DAY_LABELS = {1: "평일", 2: "주말"}

AGE_LABELS = {
    0: "전체/미상",
    10: "10대",
    20: "20대",
    30: "30대",
    40: "40대",
    50: "50대",
    60: "60대 이상",
}

# CSV 원본 컬럼 → 영문 정규화
# 두 가지 형식 모두 지원:
#   (A) 한글+영문 (샘플 데이터):  "시군구명(SIGNGU_NM)" 등
#   (B) 영문 소문자 (캠퍼스 풀 데이터):  "signgu_nm" 등
_RENAME = {
    # 형식 A
    "기준_년월_코드(STDR_YM_CD)": "ym",
    "도로링크_ID(RD_LINK_ID)": "link_id",
    "시군구코드(SIGNGU_CD)": "gu_code",
    "시군구명(SIGNGU_NM)": "gu",
    "행정동코드(ADSTRD_CD)": "dong_code",
    "행정동명(ADSTRD_NM)": "dong",
    "요일코드(DAYWEEK_CD)": "day_code",
    "연령대코드(AGRDE_CD)": "age_code",
    "시간대코드(TMZON_CD)": "time_code",
    "유동인구_수(FLPOP_CO)": "flow",
    # 형식 B (영문 소문자)
    "date": "ym",
    "rd_link_cd": "link_id",
    "rd_link_id": "link_id",
    "signgu_cd": "gu_code",
    "signgu_nm": "gu",
    "adstrd_cd": "dong_code",
    "adstrd_nm": "dong",
    "dayweek_cd": "day_code",
    "agrde_cd": "age_code",
    "tmzon_cd": "time_code",
    "flpop_co": "flow",
}


def load_foot_traffic_csv(
    src: str | Path | "io_bytes_like",  # type: ignore[name-defined]
    time_labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """유동인구 CSV 로드 + 정규화.

    src 는 파일 경로 또는 file-like (Streamlit UploadedFile).
    인코딩은 CP949 / UTF-8(BOM) 모두 시도.
    """
    df = None
    last_err: Exception | None = None
    encodings = ("cp949", "utf-8-sig", "utf-8")

    if hasattr(src, "read"):
        # file-like (Streamlit UploadedFile): 첫 번째로 buffer 읽고 디코드 시도
        raw = src.read() if not hasattr(src, "getvalue") else src.getvalue()
        if hasattr(src, "seek"):
            try:
                src.seek(0)
            except Exception:
                pass
        for enc in encodings:
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(raw.decode(enc)))
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_err = e
    else:
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(p)
        for enc in encodings:
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_err = e

    if df is None:
        raise last_err or ValueError("CSV 로드 실패")

    # 첫 컬럼이 unnamed index (예: pandas to_csv 결과)이면 제거
    if df.columns[0] in ("", "Unnamed: 0") or str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # 컬럼명 정규화 (한글/영문 → 영문 표준)
    df = df.rename(columns=_RENAME)

    # 타입 정규화
    for c in ("day_code", "age_code", "time_code"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "flow" in df.columns:
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce")

    # 라벨 컬럼 추가 (보기 좋게)
    tl = time_labels or DEFAULT_TIME_LABELS
    if "day_code" in df.columns:
        df["day"] = df["day_code"].map(DAY_LABELS).astype("string")
    if "age_code" in df.columns:
        df["age"] = df["age_code"].map(AGE_LABELS).astype("string")
    if "time_code" in df.columns:
        df["time"] = df["time_code"].map(tl).astype("string")

    return df


# ──────────────────────────────────────────────────────────────────────
# 필터링 + 피봇
# ──────────────────────────────────────────────────────────────────────

def filter_foot_traffic(
    df: pd.DataFrame,
    gus: Iterable[str] | None = None,
    dongs: Iterable[str] | None = None,
    day_codes: Iterable[int] | None = None,
    age_codes: Iterable[int] | None = None,
    time_codes: Iterable[int] | None = None,
) -> pd.DataFrame:
    """다차원 필터. None 또는 빈 리스트는 해당 차원 미적용."""
    out = df
    if gus:
        out = out[out["gu"].isin(list(gus))]
    if dongs:
        out = out[out["dong"].isin(list(dongs))]
    if day_codes:
        out = out[out["day_code"].isin(list(day_codes))]
    if age_codes:
        out = out[out["age_code"].isin(list(age_codes))]
    if time_codes:
        out = out[out["time_code"].isin(list(time_codes))]
    return out


def aggregate_by_link(df: pd.DataFrame) -> pd.DataFrame:
    """필터 적용된 flpop을 도로링크 단위로 합계 집계.

    반환 컬럼: link_id, flow_sum, dong (첫 행 기준), gu
    """
    out = (
        df.groupby("link_id", as_index=False)
        .agg(
            flow_sum=("flow", "sum"),
            dong=("dong", "first"),
            gu=("gu", "first"),
        )
        .sort_values("flow_sum", ascending=False)
        .reset_index(drop=True)
    )
    return out


def pivot_foot_traffic(
    df: pd.DataFrame,
    row: str,
    col: str | None = None,
    agg: str = "sum",
) -> pd.DataFrame:
    """피봇 테이블 생성.

    row/col 차원: 'gu' / 'dong' / 'day' / 'age' / 'time' 중 선택
    agg: 'sum' / 'mean' / 'count'
    col=None 이면 단순 1차원 groupby
    """
    if col is None or col == "(없음)":
        g = df.groupby(row, as_index=False)["flow"].agg(agg)
        g = g.sort_values("flow", ascending=False).reset_index(drop=True)
        return g

    pv = df.pivot_table(
        index=row, columns=col, values="flow", aggfunc=agg, fill_value=0
    )
    # 합계 컬럼 추가
    pv["합계"] = pv.sum(axis=1)
    pv = pv.sort_values("합계", ascending=False)
    return pv.reset_index()
