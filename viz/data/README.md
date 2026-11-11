# viz/data — 시각화 입력 데이터

> **상태: 최신 (2026-10-09)**
> 여기 있는 파일은 **그림을 그리기 위해 만든 집계표**다.
> 원자료·파이프라인 산출물은 `streamlit_app/data/`에 있고 여기로 오지 않는다.

## 분류 원칙

| | `streamlit_app/data/` | `streamlit_app/viz/data/` |
|---|---|---|
| 성격 | 원자료 · 파이프라인 산출물 | **그림 입력 전용 집계표** |
| 크기 | 수십 KB ~ 수 MB | 보통 1~2 KB |
| 수명 | 파이프라인 재실행 시 갱신 | 그림 스펙이 바뀌면 갱신 |
| 예 | `seoul_links.parquet`, `icing_zones.csv` | `p3_icing_roadtype_share.csv` |

판단 기준 한 줄: **"이 파일을 읽는 것이 그림 스크립트뿐인가?"** 그렇다면 여기.

---

## 파일 목록

| 파일 | 목적 | 원천 데이터 | 산출 | 쓰는 곳 |
|---|---|---|---|---|
| `p3_icing_roadtype_share.csv` | 결빙취약구간 지정이 대형도로에 4.0배 과대 집중됨을 보이는 **100% 누적막대** | `data/seoul_links.parquet` (도로유형) × `data/icing_zones.csv` (지정 좌표) | `viz/docs/p3_설계.md` §4 재현 명령 | **p3** `charts/fig_p3_icing_share.py` |
| `위험군_5.56pct_25gu.csv` | 자치구별 위험군 연장 대비 열선 커버리지 **수평막대 25개** (중앙값 10.29%, 성북 8.90% 19위) | `data/seoul_links.parquet` + `data/heating_2026.csv` | `scripts/recompute_gu_stats.py` 계열 | **p2 우측** `charts/fig_p2_gu_coverage.py` |
| `golmok_slope_class_25gu_v2.csv` | 자치구별 골목길·위험군 수·연장·순위 + **위험군 중 골목길 비중** (성북구 표3 근거). **전부 5.56% 단일 기준**, 급경사 10% 열 제거 | `data/seoul_links.parquet` | `scripts/recompute_gu_stats.py` | **p6** 표3 |
| `icing_capture_by_gu_v2.csv` | 결빙지정의 급경사 골목 **포착률** obs/base (성북 z=-1.75 최하위) | `data/seoul_links.parquet` | `scripts/recompute_gu_stats.py` | **p6** 포착률 비교 |
| `p4_maa_edf.csv` | MAA 경험분포 **막대** (n=58, 0° 15.5%) | Dadkhah+2026 Fig.1 디지타이즈 → `utils/risk.MAA_OBS_COUNTS` | `viz/docs/p4_설계.md` §7 | **p4 좌** `charts/fig_p4_maa.py` |
| `p4_risk_curve.csv` | 경사(%)→`P(MAA<θ)` **계단곡선** (0~30%, 0.25% 간격) | `utils/risk.obs_risk` | 동일 | **p4 우** `charts/fig_p4_maa.py` |
| `p6_gu_risk_length.csv` | 자치구별 위험군 연장 **수평막대 25개** (성북 138.8 km 1위) + 커버리지 병기 | `golmok_slope_class_25gu_v2.csv` + `위험군_5.56pct_25gu.csv` | `viz/docs/p6_설계.md` §5 | **p6** `charts/fig_p6_gu_risk_bar.py` |
| `p10_strategy_compare.csv` | 전략 6종 비교 — **현행 예산 18.53억 기준**. 경사만/통행만/위험×통행/제안/현행 | `data/road_graph.json` + `utils/milp` | `viz/docs/p10_p12_설계.md` | **p10** `charts/fig_p10_strategy.py` |
| `p11_budget_delta_grid.csv` | 예산 11점 × δ 8점 = 88행 격자 | 동일 | `scripts/run_budget_delta_grid.py` | **p11** `charts/fig_p11_sensitivity.py` |
| `p11_budget_frontier.csv` | 예산별 **권장 δ**와 그 해, 한계효율 | 위 격자 | 동일 | **p11·p12** |
| `p5_boundary_4dong.csv` | 4개동 경계쌍 **17개**(상류 14 / 하류 12) 좌표·낙차·경사·위험·통행 — **지도 입력** | `data/road_graph.json` (4개동, z·heat) | `viz/docs/p5_설계.md` §6 재현 명령 | **p5** `maps/fig_p5_boundary_map.py` |
| `tile_coverage_by_gu.csv` | 수치지형도 도엽 **커버리지 검증** 결과 (전역 99.6%) | `서울시 수치지형도 파일/` zip 2,489개 shapefile 헤더 | `scripts/verify_tile_coverage.py` | **p14** 한계·신뢰성 근거 (그림 아님, 수치 인용) |

---

## 기준 통일

모든 파일이 아래 기준을 쓴다. **섞어 쓰지 않는다.**

- **위험군** = 골목길 ∩ 경사 ≥ **5.56%** (장애인등편의법 접근로 1/18)
- **커버리지(느슨)** = 신고 열선 연장 ÷ 위험군 연장 — `위험군_5.56pct_25gu.csv`,
  중앙값 10.29%가 이 기준. **지도·막대는 전부 이 기준**
- **커버리지(엄격)** = (열선 ∩ 위험군) 연장 ÷ 위험군 연장 = 4.05% — **본문 수치로만**
- `unknown`(OSM 매칭 실패 22,729건)은 **분석에서 제외**. 포함 시 위험군 1,395.6 →
  1,669.7 km, 커버리지 4.05 → 3.65%로 **주장이 강해지는 방향**이므로 제외가 보수적

---

## 아직 없는 것

| 필요 파일 | 쓸 곳 | 선행 작업 |
|---|---|---|
| `p2_timeline.csv` | p2 연도별 누적 연장 | `data/heating_2026_normalized.csv`에서 집계 |
| `p12_delta_curve.csv` | p12 δ 트레이드오프 | `data/exp_delta.csv` 가공 (δ 확정 후) |

---

## 갱신 절차

`streamlit_app/data/` 쪽 원자료가 바뀌면 → 산출 스크립트 재실행 → 여기 파일이 갱신된다.
역방향은 없다. **여기 파일을 손으로 고치지 않는다.**

```bash
cd streamlit_app
python3 scripts/build_seoul_dataset.py   # 선행: seoul_links.parquet 등
python3 scripts/recompute_gu_stats.py    # 자치구 집계 3종
python3 scripts/verify_tile_coverage.py  # 도엽 커버리지
```
