"""열선 설치시기 표기 정규화.

배경
----
heating_2026.csv의 `설치연도` 컬럼은 자유 텍스트다. 120종 이상의 표기가 섞여
있어 집계마다 다른 숫자가 나올 위험이 있다.

    "2014.11."   "2016. 11."  "2017.11"   "2018. 12."
    "21.12.07."  "22.11.30."  "25.9.30."  "2023.3"
    "2025.05"    "2026.6."    "2024.01."

원본은 건드리지 않고, 파싱 결과를 덧붙인 파생 파일을 만든다.

출력 컬럼
---------
    설치연도_원문   원본 그대로
    설치연         4자리 연도 (int)
    설치월         1~12 (없으면 NaN)
    설치일         1~31 (없으면 NaN)
    설치시기       YYYY-MM 정규형 (월 없으면 YYYY)
    파싱패턴       YYYY.M.D / YYYY.M / YY.M.D / YYYY / 실패

실행
----
    cd streamlit_app
    python3 scripts/normalize_heating_dates.py

산출
----
    data/heating_2026_normalized.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = DATA / "heating_2026.csv"
OUT = DATA / "heating_2026_normalized.csv"

# 2자리 연도 해석 기준 — 자료 범위가 2014~2026이므로 모두 20xx
_YY_BASE = 2000


def parse(raw) -> dict:
    """설치연도 문자열 → (연, 월, 일, 패턴)."""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return dict(설치연=None, 설치월=None, 설치일=None, 파싱패턴="실패")

    # 공백 제거 후 숫자 토큰만 추출 ("2016. 11." → ["2016","11"])
    nums = re.findall(r"\d+", s)
    if not nums:
        return dict(설치연=None, 설치월=None, 설치일=None, 파싱패턴="실패")

    head = nums[0]
    if len(head) == 4:                       # YYYY...
        year, rest = int(head), nums[1:]
        pat = "YYYY"
    elif len(head) == 2:                     # YY.M.D
        year, rest = _YY_BASE + int(head), nums[1:]
        pat = "YY"
    else:
        return dict(설치연=None, 설치월=None, 설치일=None, 파싱패턴="실패")

    month = day = None
    if rest:
        m = int(rest[0])
        if 1 <= m <= 12:
            month = m
            pat += ".M"
            if len(rest) > 1:
                d = int(rest[1])
                if 1 <= d <= 31:
                    day = d
                    pat += ".D"
    return dict(설치연=year, 설치월=month, 설치일=day, 파싱패턴=pat)


def main() -> None:
    if not SRC.exists():
        print(f"❌ 원본 없음: {SRC}")
        sys.exit(1)

    df = pd.read_csv(SRC, encoding="utf-8-sig")
    ycol = next(c for c in df.columns if "연도" in c)
    lcol = next(c for c in df.columns if "연장" in c)

    parsed = pd.DataFrame([parse(v) for v in df[ycol]])
    out = df.copy()
    out = out.rename(columns={ycol: "설치연도_원문"})
    out["설치연"] = parsed["설치연"].astype("Int64")
    out["설치월"] = parsed["설치월"].astype("Int64")
    out["설치일"] = parsed["설치일"].astype("Int64")
    out["파싱패턴"] = parsed["파싱패턴"]
    out["설치시기"] = [
        f"{y}-{m:02d}" if pd.notna(y) and pd.notna(m)
        else (f"{y}" if pd.notna(y) else "")
        for y, m in zip(out["설치연"], out["설치월"])
    ]
    out["설치연장_m"] = pd.to_numeric(
        out[lcol].astype(str).str.replace(",", ""), errors="coerce"
    )

    out.to_csv(OUT, index=False, encoding="utf-8-sig")

    # ── 품질 보고 ─────────────────────────────────────────────────
    n = len(out)
    print(f"원본 {n}행 → {OUT.name}")
    print(f"원문 표기 고유값 {df[ycol].nunique()}종 → 정규형 "
          f"{out['설치시기'].nunique()}종")
    print("\n[파싱 패턴]")
    print(out["파싱패턴"].value_counts().to_string())
    print(f"\n연 파싱 성공 {out['설치연'].notna().sum()}/{n} "
          f"({100*out['설치연'].notna().mean():.1f}%)")
    print(f"월 파싱 성공 {out['설치월'].notna().sum()}/{n} "
          f"({100*out['설치월'].notna().mean():.1f}%)")
    print(f"일 파싱 성공 {out['설치일'].notna().sum()}/{n} "
          f"({100*out['설치일'].notna().mean():.1f}%)")

    fail = out[out["파싱패턴"] == "실패"]
    if len(fail):
        print(f"\n[파싱 실패 {len(fail)}건]")
        print(fail["설치연도_원문"].value_counts().head(10).to_string())

    print("\n[연도별 건수·연장]")
    g = out.groupby("설치연").agg(건수=("설치연", "size"),
                                  연장m=("설치연장_m", "sum"))
    g["연장m"] = g["연장m"].round().astype("Int64")
    print(g.to_string())

    print("\n[월별 분포 — 동절기 집중 확인]")
    mv = out["설치월"].value_counts().sort_index()
    for m, c in mv.items():
        print(f"  {m:2d}월 {c:4d}  {'█' * (c // 8)}")
    win = out["설치월"].isin([10, 11, 12, 1, 2]).sum()
    print(f"  10~2월 {win}건 = 월 파싱분의 "
          f"{100*win/out['설치월'].notna().sum():.1f}%")

    tot = out["설치연장_m"].sum()
    print(f"\n총 연장 {tot:,.0f} m (연장 결측 "
          f"{out['설치연장_m'].isna().sum()}행)")
    print(f"2026년분 {int((out['설치연'] == 2026).sum())}건 / "
          f"{out.loc[out['설치연'] == 2026, '설치연장_m'].sum():,.0f} m  "
          f"— 공표 자료이므로 포함(현행 유지)")


if __name__ == "__main__":
    main()
