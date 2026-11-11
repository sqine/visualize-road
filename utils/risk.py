"""보행자 미끄러짐 위험도 — Observed MAA 경험분포.

채택 근거
---------
Dadkhah, Ghomashchi & Dutta (2026), *Applied Ergonomics* 131:104678.
WinterLab에서 참가자 27명이 신발 3켤레씩 착용하고 빙판 경사를 1°씩 올리며,
"미끄러지지 않고 **오르내릴 수 있는** 가장 가파른 경사"(Observed MAA)를 기록.

MAA의 정의가 "그 사람이 버틸 수 있는 경사 한계"이므로
    risk(θ) = P(MAA < θ)
는 가정이 아니라 **정의상 관계**다. 필요한 것은 사람의 한계 분포이지
신발 등급 분포가 아니다.

자료 복원
---------
논문 Fig.1의 세션별 회색 산점을 500dpi로 추출해 x좌표(=MAA)를 읽었다.
막대 중심(MAA 0/1.5/3.5/5.5/7.5°)으로 축 보정(1° = 100.9px).
재현 스크립트: outputs/digitize_fig1.py

한계
----
- 겹친 점이 병합되어 74세션 중 **58개만 복원** → 추정치
- 구간 경계 ±0.5° 오차, 1° 양자화로 risk는 **계단함수**
- 12°(경사 21.3%) 이상은 모두 1.0으로 **포화**
- 27명이 3세션씩 수행 → 세션 간 **독립성 불완전**. 점추정으로만 사용하고
  신뢰구간은 제시하지 않는다.
- 참가자 평균 30.3세(20–58), **고령자 미포함**. 참가자가 빙판을 인지한
  상태였으므로 논문 스스로 과소추정이라 명시 → 추정 방향은 보수적.

기각한 대안
-----------
- `edf_2026` (논문 Table 1, 신발 등급 가중): 평지 0.378. 논문 내 최하위
  신발 등급의 걸음당 미끄러짐률(36.8%)을 전 보행자에 부여하는 꼴 → 기각
- `strict_2016` (6값 가정 분포): 평지 0.041, 10% 0.889로 실측과 괴리 → 기각
두 대안은 `outputs/maa_risk.py`에 재현용으로 보존.
"""

from __future__ import annotations

import math

import numpy as np

# Observed MAA(도) → 세션 수. 합계 58 (원자료 74세션 중 복원)
MAA_OBS_COUNTS: dict[int, int] = {
    0: 9, 1: 5, 2: 6, 3: 4, 4: 4, 5: 3, 6: 3,
    7: 4, 8: 5, 9: 2, 10: 4, 11: 5, 12: 4,
}

_EDF_OBS = np.sort(
    np.repeat(list(MAA_OBS_COUNTS), list(MAA_OBS_COUNTS.values())).astype(float)
)

#: 경사 결측 시 대입값(%). 후보 평균 risk(0.423)보다 낮아 보수적으로 작용.
DEFAULT_SLOPE_PCT = 5.0

#: risk가 1.0으로 포화되는 경사(%). tan(12°) × 100
SATURATION_SLOPE_PCT = 100.0 * math.tan(math.radians(12.0))


def slope_pct_to_deg(slope_pct: float) -> float:
    """경사(%) → 각도(°)."""
    return math.degrees(math.atan(slope_pct / 100.0))


def obs_risk(slope_pct: float | None, default_slope: float = DEFAULT_SLOPE_PCT) -> float:
    """P(MAA < θ) — 경사 θ를 지나는 보행자 중 미끄러지는 비율.

    단위는 **통행(traversal)당 확률**이다. 논문이 보고한 평지 11.75%는
    걸음(step)당 확률이므로 서로 다른 양이며, 직접 비교할 수 없다.
    모형이 `risk × flow`(유동인구, 사람 수)를 쓰므로 통행 단위가 맞다.
    """
    if slope_pct is None or (isinstance(slope_pct, float) and math.isnan(slope_pct)):
        slope_pct = default_slope
    theta = slope_pct_to_deg(float(slope_pct))
    return float((_EDF_OBS < theta).mean())


def obs_risk_inclusive(slope_pct: float | None,
                       default_slope: float = DEFAULT_SLOPE_PCT) -> float:
    """P(MAA ≤ θ) — 경계조건 대안.

    엄격 부등호를 쓰면 경사 0%에서 risk가 정확히 0이 된다. 평지 빙판에서도
    MAA 0° 세션이 15.5% 존재하므로 이 함수는 그 값을 반환한다.
    전체 기대건수 차이는 -0.2%에 불과해 결론에 영향이 없다(검증 완료).
    """
    if slope_pct is None or (isinstance(slope_pct, float) and math.isnan(slope_pct)):
        slope_pct = default_slope
    theta = slope_pct_to_deg(float(slope_pct))
    return float((_EDF_OBS <= theta).mean())


def edf_table() -> list[dict]:
    """발표·보고서용 EDF 표 (MAA도 / 세션수 / 비율 / 누적)."""
    n = sum(MAA_OBS_COUNTS.values())
    rows, cum = [], 0
    for deg in sorted(MAA_OBS_COUNTS):
        cnt = MAA_OBS_COUNTS[deg]
        cum += cnt
        rows.append({
            "MAA(°)": deg,
            "세션수": cnt,
            "비율": cnt / n,
            "누적 P(MAA≤θ)": cum / n,
        })
    return rows
