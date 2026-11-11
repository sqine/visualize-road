# viz — 시각화

> **상태: 최신 (2026-10-11)**
> 기준 세대 G4 — Observed MAA EDF · `seoul_links.parquet` · heating_2026.csv(993행)
> 이 폴더가 시각화의 **단일 출처**다. 여기 밖의 그림 파일은 전부 구세대로 간주한다.

---

## 구조

```
viz/
├── theme.py          색·폰트·캡션 단일 출처 ★ 전부 여기만 참조
├── data/             그림 입력 전용 집계표 (목록·목적은 data/README.md)
├── maps/             도로망 표시 (공간 데이터, matplotlib LineCollection)
├── charts/           차트 (집계 데이터, matplotlib / altair)
├── docs/             설계 문서 (쪽별)
└── out/              PNG 산출물
```

**`data/` 분리 기준**: "이 파일을 읽는 것이 그림 스크립트뿐인가?" → 그렇다면 `viz/data/`.
원자료·파이프라인 산출물(`seoul_links.parquet`, `icing_zones.csv` 등)은
`streamlit_app/data/`에 그대로 두고 여기로 옮기지 않는다.

**지도와 차트를 나눈 이유**: 입력이 다르다. 지도는 링크 단위 공간 데이터
(`seoul_links.parquet` + 좌표열)를, 차트는 자치구·연도 단위 집계 CSV를 쓴다.
라이브러리도 matplotlib `LineCollection` vs `altair`로 갈린다.

---

## 쪽번호 = Canva 쪽번호

**파일명의 `pN` 은 Canva 덱 `DAHXjhhvWoQ` 의 N쪽을 가리킨다.** 2026-10-11 이전에는
논리 순서로 번호를 매겨 6~8쪽 구간이 덱과 어긋나 있었다. 바로잡은 내역:

| 옛 이름 | 새 이름 | 이유 |
|---|---|---|
| `p4a_maa_edf` | **`p4_maa_edf`** | Canva 4쪽 왼쪽 |
| `p4b_risk_curve` | **`p7a_risk_curve`** | Canva **7쪽** 왼쪽 — 같은 스크립트가 두 쪽에 걸친다 |
| `fig_p8_dong_selection` | **`fig_p6b_dong_selection`** | Canva **6쪽** 오른쪽(행정동 선정) |
| `fig_p7_current_diagnosis` | **`fig_apx_current_diagnosis`** | 본편 1~15쪽에 자리 없음 — 부록/근거용 |
| `fig_p7_icing_capture` | **`fig_apx_icing_capture`** | 위와 같음 |

`apx_` = **덱에 배치되지 않은 근거 그림.** 쪽을 받으면 그때 `pN` 로 바꾼다.
Canva 16쪽 이후는 구버전·템플릿 보관용이므로 번호 대상이 아니다.

---

## 산출물 현황

`out/` 실물 기준 (2026-10-11). 각 PNG는 설명본과 발표본(`_bare`) 두 장이다.

| Canva 쪽 | 쪽 제목 | 그림 | 모듈 | 상태 |
|---|---|---|---|---|
| 1 | 표지 | 그림 없음 | — | — |
| 2 | 서론·01 열선 설치 현황 | 서울 전역 열선 vs 위험군 지도 | `maps/fig_p2_seoul_map.py` | ✅ `p2_seoul_map` |
| 2 | ″ | 자치구별 위험군 커버리지 | `charts/fig_p2_coverage.py` | ✅ `p2_coverage` |
| 2 | ″ | 연도별 설치 추이 | `charts/fig_p2_install_trend.py` | ✅ `p2_install_trend` |
| 3 | 서론·02 기존 연구와 차별점 | 결빙지정 도로유형 | `charts/fig_p3_icing_share.py` | ✅ `p3_icing_share` · `p3_icing_pie` |
| 4 | 서론·03 보행 위험과 설계기준 | MAA 경험분포 | `charts/fig_p4_maa.py` | ✅ `p4_maa_edf` |
| 5 | 분석·01 경계효과와 실태 | 경계쌍 지도 (4개동 / 서울 전역) | `maps/fig_p5_boundary_map.py` · `fig_p5b_seoul_boundary_map.py` | ✅ `p5_boundary_map` · `p5b_seoul_boundary_map` |
| 6 | 분석·02 대상지 선정 (왼쪽) | 자치구별 위험군 연장·순위 | `charts/fig_p6_gu_risk_bar.py` · `fig_p6_gu_risk_rank.py` | ✅ `p6_gu_risk_bar` · `p6_gu_risk_rank` |
| **6** | ″ (오른쪽) | **4개동 선정 근거 3패널** | `charts/fig_p6b_dong_selection.py` | ✅ `p6b_dong_selection` — **덱 미삽입** |
| 7 | 분석·03 도로 선정 요인 (왼쪽) | risk 계단곡선 P(MAA<θ) | `charts/fig_p4_maa.py` | ✅ `p7a_risk_curve` — **Canva 반영** |
| 7 | ″ (오른쪽) | 유동인구 집중도 곡선 | `charts/fig_p7b_flow_concentration.py` | ✅ `p7b_flow_concentration` — **Canva 반영** (신규) |
| 8 | 모형·01 파라미터 산출 | ④ 가중치 칸 — 노드(도로구간)·엣지(경계) 도식 | `charts/fig_p8_node_edge.py` | ✅ `p8_node_edge` — **Canva 반영** (신규) |
| 9 | 모형·02 최적화 모델 | 그림 없음 — 4블록 본문 | — | ✅ `docs/p9_설계.md` |
| 10 | 평가 | 전략 비교 · 균형 | `charts/fig_p10_strategy.py` · `fig_p10c_balance.py` | ✅ `p10_strategy` · `p10c_balance` · 표 `표_p10_전략비교` |
| 11 | 민감도 분석 | 예산 경로 · 한계효율 · 파레토 전환점 | `charts/fig_p11_budget_path.py` · `fig_p11c_knee_path.py` · `fig_p11_budget_delta.py` | ✅ `p11a_budget_path` · `p11b_marginal` · `p11c_knee_path` · `p11_budget_delta` |
| 12 | 시나리오 | 채택 배치 · 시나리오 지도 | `maps/fig_p12_combined_map.py` · `fig_p12_scenario_map.py` · `fig_p12_selection_map.py` | ✅ `p12_combined_map` · `p12_scenario_map` · `p12_selection_map` · 표 `표_p12_시나리오` |
| 13~15 | 결론·한계·참고문헌 | 그림 없음 | — | ✅ Canva 본문 |
| — | (미배치) | 현행 열선 진단 3패널 | `charts/fig_apx_current_diagnosis.py` | ✅ `apx_current_diagnosis` |
| — | (미배치) | 결빙지정 포착력 | `charts/fig_apx_icing_capture.py` | ⬜ `out/` 미산출 — 필요 시 실행 |

### 남은 일

| 할 일 | 비고 |
|---|---|
| 보고서 §9-4·§9-5 수치 정정 | 현세대 재산출과 어긋난다. `docs/p9_설계.md` §6에 판정표·재현 명령 |
| 보고서 경사보정 절(§11-1) 수치 정정 | 구세대 "43곳 중 13곳 · 11→8곳 · 공통 30곳" → 현세대 **98곳 중 21곳 · 13→10곳 · 공통 77곳**. 근거 `viz/data/p14_slope_correction.csv` |
| `발표자료_수치표.txt` 통합본 | 쪽별 수치를 한 파일로 |
| p7 `icing_capture` 재실행 | 모듈은 있으나 `out/` 에 PNG 없음 |

## 글자 크기

`theme.py`가 가진 크기 상수는 `TS_MAIN`(13.5) · `TS_PANEL`(12.2) **둘뿐이다.**
본문·주석 크기는 쪽별 스크립트가 각자 적는다. 기준값과 하한(8.4pt)은
`docs/p6b_설계.md` §4-1 표를 따른다.

---

## 색 규칙 (theme.py)

| 역할 | 색 | 상수 |
|---|---|---|
| 현행 열선 / 보호됨 | `#1F6FEB` 파랑 | `CURRENT` |
| 제안 선정 (MILP 해) | `#E0674A` 코랄 | `PROPOSED` |
| 경계 · 하류 위험 | `#C23B22` 진한 적 | `BOUNDARY` |
| 위험군 미보호 | `#B9C2CE` 회색 | `UNPROTECTED` |
| 일반 도로 배경 | `#E8EBEF` | `ROAD_BG` |
| 자치구 경계 | `#9AA5B1` | `GU_LINE` |
| 강조(성북구) | `#1F3A5F` 네이비 | `ACCENT` |

1. 파랑 = 언제나 현행/보호, 코랄 = 언제나 제안
2. 경사 농도는 `SLOPE_HEX` 4등급만 (연속 그라데이션 금지)
3. 캡션은 `theme.caption(구분, 모집단, 기준, 출처)` — 4요소 고정.
   단, **정의·해석을 글로 반복하지 않는다.** 식으로 보일 수 있으면 그림 안에
   식을 넣고 캡션은 출처 한 줄만 남긴다 (`p7a_risk_curve` 가 그 예)
4. 비율은 **분모를 캡션에 명시**
5. p값은 `theme.pval()` — 위첨자 글리프가 폰트에 없으므로 `p = 6.9e-135` 표기

---

## 라이브러리

| 용도 | 선택 | 이유 |
|---|---|---|
| 발표자료 PNG | matplotlib `LineCollection` | 180k 선을 한 번에. 굵기·색·알파를 배열로 일괄 지정. 300 dpi |
| Streamlit 지도 | `pydeck` `PathLayer` | 앱에 이미 사용 중, 줌·툴팁 |
| 차트 | `altair` | `utils/delta_lab.py`와 일관 |
| 좌표 투영 | 등장방형 근사 | `x = lon × 88322`, `y = lat × 111320`. 서울 범위에서 충분하고 기존 파이프라인과 동일 상수 |

**geopandas·folium은 쓰지 않는다.** GDAL 의존이 무겁고, 타일 서버를 타면 PNG
품질이 들쭉날쭉하다. 필요한 것은 배경 지도 없는 도로망 실루엣이다.

---

## 데이터 의존

| 파일 | 쓰는 곳 | 상태 |
|---|---|---|
| `data/seoul_links.parquet` | 모든 지도 | ✅ |
| `data/seoul_boundary_pairs.parquet` | p5 | ✅ |
| `data/seoul_gu.geojson` | 모든 지도 | ✅ |
| **`data/seoul_link_geom.parquet`** | **모든 지도 (선 형상)** | ⬜ **선행 작업** |
| `viz/data/*` | 그림 입력 집계표 5종 | ✅ `viz/data/README.md` |
| `data/heating_2026_normalized.csv` | p2 연도별 | ✅ |

| `data/exp_delta.csv` | p12 | ✅ |
| `data/final_selection.csv` | p11 | ⬜ δ 확정 후 재생성 |

`seoul_links.parquet`에는 **링크 중점만** 있다. 선을 그리려면 좌표열이 필요한데
원본 `road_links.geojson`은 92 MB라 매번 읽기 무겁다. 좌표점은 총 805,614개
(링크당 평균 4.5점, 32%는 2점)뿐이므로 5~8 MB 파켓으로 뽑는다.
→ `scripts/build_seoul_dataset.py`에 산출 단계 추가

---

## 작업 순서

1. `seoul_link_geom.parquet` 산출 추가 ← **선행**
2. `maps/fig_p2_coverage_map.py` — p2 좌측 지도
3. `charts/fig_p2_gu_coverage.py` + `fig_p2_timeline.py` — p2 우측
4. `charts/fig_p3_icing_share.py` — p3 누적막대 (지정 도로유형 분포)
5. `maps/fig_p5_boundary_map.py` — p5 경계 지도
6. δ 확정 후 `maps/fig_p11_selection_map.py` + `charts/fig_p12_delta_curve.py`

---

## 폐기된 그림 — 쓰지 말 것

`05_폐기/` 로 이동 완료. 파일명이 현행처럼 보여 가장 위험한 항목이다.

- `fig_greedy_vs_milp_map.png` · `fig_tier_intersection_map.png`
- `fig_steep_golmok_map.png` · `fig_path_improvement.png`
- `fig_p9_formulation.py` / `p9_formulation.png` — p9 info 카드. Canva 본문과
  중복이라 폐기(2026-10-11). `charts/_폐기/` · `out/_폐기/`

모두 2025-09-15 생성(G1 세대)으로 경사 보정 전·MAA 전 결과다. 선정 구간도
경사 등급도 현재와 다르다.

---

## 설계 문서

쪽 번호는 Canva 쪽 번호다.

- `docs/p2_설계.md` — 2쪽 좌/우 구성, 레이어 순서, 수치
- `docs/p3_설계.md` — 3쪽 누적막대, 소절별 수치, 사고데이터 한계와 인용 가능 출처
- `docs/p4_설계.md` — **4쪽** MAA 분포 + **7쪽** risk 곡선 (한 스크립트, 두 쪽)
- `docs/p5_설계.md` — 5쪽 경계 지도 레이어, 인셋 후보 비교, 17쌍 전체 표
- `docs/p6_설계.md` — 6쪽 왼쪽: 자치구 막대, 수치 검증 결과
- `docs/p6b_설계.md` — 6쪽 오른쪽: 4개동 선정 3패널, 글자 크기 기준(§4-1)
- `docs/p7b_설계.md` — 7쪽 오른쪽: 유동인구 집중도, Canva 이미지 교체 절차
- `docs/p9_설계.md` — 9쪽 MILP 정식화 (그림 없음 · Canva 본문)
- `docs/p10_p12_설계.md` — 10쪽 전략 비교 · 11쪽 예산×δ 민감도 · 12쪽 배치 지도
