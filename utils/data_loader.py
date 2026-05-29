"""CSV 로딩 및 전처리.

- 인코딩 자동 감지(UTF-8 BOM / CP949)
- 설치구간 컬럼에서 도로명 추출
- 자치구·연도·연장 정규화
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# "도로명(주소범위)" 또는 "도로명" 패턴에서 도로명 후보를 뽑는 정규식.
# 예: "명륜길(명륜길 74 ~ 명륜9길 1)"  -> 명륜길
#     "퇴계로6가길"                     -> 퇴계로6가길
#     "퇴계로6가길32 ~ 소월로 39-1(퇴계로2길, 퇴계로6가길)"
#     -> 괄호 안의 도로명들이 더 정확하므로 괄호 안을 우선
_PAREN = re.compile(r"\(([^)]+)\)")
# 도로명 토큰: '~숫자/-/공백/한글' 노이즈를 제외하고 도로명만 추출
_ROAD_TOKEN = re.compile(r"([가-힣A-Za-z0-9]+(?:로|길))")


def _extract_road_names(text: str) -> list[str]:
    """설치구간 문자열에서 가능한 도로명 목록을 반환.

    우선순위:
        1. 괄호 밖 첫 도로명 (대표 도로명일 가능성이 가장 큼)
        2. 괄호 안 도로명 (콤마로 나열된 보조 도로명)
        3. 괄호 밖 나머지 도로명
    """
    if not isinstance(text, str) or not text.strip():
        return []

    candidates: list[str] = []

    # 1순위: 괄호 밖 텍스트에서 도로명 추출 (메인 도로명일 가능성)
    outside = _PAREN.sub("", text)
    outside_tokens = [
        tok for tok in _ROAD_TOKEN.findall(outside) if tok.endswith(("로", "길"))
    ]
    candidates.extend(outside_tokens)

    # 2순위: 괄호 안 도로명 (보조)
    paren_match = _PAREN.search(text)
    if paren_match:
        inside = paren_match.group(1)
        for part in inside.split(","):
            for tok in _ROAD_TOKEN.findall(part):
                if tok.endswith(("로", "길")):
                    candidates.append(tok)

    # 중복 제거(순서 보존)
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def load_heating_csv(path: str | Path) -> pd.DataFrame:
    """열선 설치 CSV 로드 + 정규화.

    반환 컬럼:
        gu          : 자치구 (예: 종로구)
        year        : 설치연도 (int)
        section     : 원본 설치구간 텍스트
        length_m    : 연장(m), int
        road_names  : 추출된 도로명 리스트(list[str])
        primary_road: 대표 도로명(첫 번째)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # UTF-8 BOM, CP949 순서로 시도
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raise last_err  # type: ignore[misc]

    # 컬럼명 정규화 (공백 제거)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]

    rename_map = {
        "연번": "row_id",
        "관리기관": "gu",
        "설치연도": "year",
        "설치구간": "section",
        "연장(m)": "length_m",
    }
    df = df.rename(columns=rename_map)

    # 타입 정리
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["length_m"] = pd.to_numeric(df["length_m"], errors="coerce").fillna(0).astype(int)
    df["gu"] = df["gu"].astype(str).str.strip()
    df["section"] = df["section"].astype(str).str.strip()

    # 도로명 추출
    df["road_names"] = df["section"].apply(_extract_road_names)
    df["primary_road"] = df["road_names"].apply(lambda xs: xs[0] if xs else "")

    return df


def aggregate_by_gu(df: pd.DataFrame) -> pd.DataFrame:
    """자치구별 집계."""
    g = df.groupby("gu", as_index=False).agg(
        road_count=("section", "count"),
        total_length_m=("length_m", "sum"),
        avg_length_m=("length_m", "mean"),
    )
    g["avg_length_m"] = g["avg_length_m"].round(1)
    g = g.sort_values("total_length_m", ascending=False).reset_index(drop=True)
    return g


def aggregate_by_road(df: pd.DataFrame) -> pd.DataFrame:
    """도로명 단위 집계 (primary_road 기준)."""
    sub = df[df["primary_road"] != ""]
    g = sub.groupby(["gu", "primary_road"], as_index=False).agg(
        segment_count=("section", "count"),
        total_length_m=("length_m", "sum"),
        years=("year", lambda s: sorted({int(x) for x in s.dropna()})),
    )
    g = g.sort_values("total_length_m", ascending=False).reset_index(drop=True)
    return g


def load_icing_csv(path: str | Path) -> pd.DataFrame:
    """행정안전부 상습결빙구간 CSV 로드.

    원본 컬럼:
        구간번호, 관리청, 도로분류, 대표지역, 도로(노선)명, 총길이(km),
        기점 위도/경도, 종점 위도/경도, (방향 각도들)

    반환 컬럼:
        seg_id, agency, road_class, region, road_name, length_km,
        lat_start, lon_start, lat_end, lon_end
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"icing CSV not found: {path}")

    last_err: Exception | None = None
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raise last_err  # type: ignore[misc]

    df.columns = [c.strip() for c in df.columns]
    rename = {
        "구간번호": "seg_id",
        "관리청": "agency",
        "도로분류": "road_class",
        "대표지역": "region",
        "도로(노선)명": "road_name",
        "총길이(km)": "length_km",
        "기점 위도(WGS84(4326))": "lat_start",
        "기점 경도(WGS84(4326))": "lon_start",
        "종점 위도(WGS84(4326))": "lat_end",
        "종점 경도(WGS84(4326))": "lon_end",
    }
    df = df.rename(columns=rename)

    for col in ("lat_start", "lon_start", "lat_end", "lon_end", "length_km"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 좌표 결측 행 제외
    df = df.dropna(subset=["lat_start", "lon_start", "lat_end", "lon_end"]).copy()
    for c in ("seg_id", "agency", "road_class", "region", "road_name"):
        df[c] = df[c].astype(str).str.strip()

    return df.reset_index(drop=True)


def compute_risk_score(df_gu: pd.DataFrame) -> pd.DataFrame:
    """자치구별 임시 위험도 점수.

    실제 경사도/결빙위험 데이터가 들어오기 전까지의 placeholder.
    설치 길이에 음의 상관(설치가 적을수록 위험 높음)이라고 가정한
    상대 점수를 0~100으로 환산.
    """
    out = df_gu.copy()
    if out.empty:
        out["risk_score"] = []
        return out
    max_len = out["total_length_m"].max() or 1
    # 설치 길이가 짧을수록 점수가 높게 — 보수적인 placeholder
    out["risk_score"] = (
        (1 - out["total_length_m"] / max_len) * 100
    ).round(1)
    return out
