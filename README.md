# 서울시 도로 열선 설치 현황 시각화

Streamlit 기반 인터랙티브 대시보드. 2026년 자치구별 도로열선 설치 데이터를
지도와 표로 분석합니다.

## 🧭 해석 가이드 (지도 색상 한눈에)

### 도로 색상이 의미하는 것

| 색상 | 의미 | 어디서? |
|---|---|---|
| 🟪 **마젠타** (분홍) | 열선이 이미 설치된 도로 | "열선 설치 도로" 레이어 |
| 🔵 **파랑** | 상습 결빙구간 (행정안전부 지정) | "상습 결빙구간" 레이어 |
| 🔴 **빨강** | 경사도 ≥ 10% (매우 가파름) | "경사도 도로" / "유동인구×경사도" 레이어 |
| 🟠 **주황** | 경사도 6~10% (가파름) | 동일 |
| 🟡 **노랑** | 경사도 3~6% (보통) | 동일 |
| ⚪ **회색** | 평탄 또는 일반 도로 (배경 컨텍스트) | "전체 도로 배경" / 경사도 평탄 |

### 도로 굵기·투명도

- **굵기**: 경사도 위험 등급이 높을수록 굵게
- **투명도** (유동인구 × 경사도 레이어만): 옅음 = 사람 적음 / 진함 = 사람 많음

### 자치구 색상

자치구 경계 폴리곤은 사이드바 "자치구 색상 기준 지표"에 따라 색이 바뀝니다:
- **푸른 톤** = 값이 낮음
- **빨간 톤** = 값이 높음

### ⭐ 최우선 열선 설치 후보 찾는 법

**빨강이면서 진하고 굵은 도로** 를 지도에서 찾으세요. 이 도로가:
- 경사도 ≥ 10% (결빙 시 매우 위험)
- 유동인구 상위 분위 (사고 발생 시 피해자 많음)

여기에 추가로 다른 레이어를 같이 켜면:
- **+ 상습 결빙구간(파랑)** 이 겹치면: 행정안전부 인증된 위험 도로
- **+ 위성영상** 으로 실제 골목 모습 확인
- **- 열선 설치(마젠타)** 가 안 깔린 곳 = 시급한 신규 설치 대상

### 자주 쓰는 시나리오

| 분석 목표 | 켜야 할 레이어 | 사이드바 설정 |
|---|---|---|
| 열선 미설치 위험 도로 찾기 | 경사도, 결빙구간 | 경사도 ≥6% 선택 |
| 보행자 많은 위험 골목 | 유동인구×경사도, 위성영상 | 유동인구 CSV 업로드 후 평일·취약시간 필터 |
| 자치구별 우선순위 | 자치구 색상 | 색상 기준 = 위험도 점수 |
| 행정 보고서용 검증 | 위성영상 + 모든 레이어 | 자치구 1~2개로 좁히기 |

---

## 폴더 구조

```
streamlit_app/
├── app.py                      # 메인 Streamlit 엔트리
├── requirements.txt            # 의존성
├── data/
│   ├── heating_2026.csv        # 열선 설치 현황 CSV
│   ├── icing_zones.csv         # 행정안전부 상습 결빙구간 (전국)
│   ├── seoul_gu.geojson        # 자치구 경계 (25개)
│   ├── seoul_roads_raw.geojson # OSM Overpass 원본 (선택)
│   ├── seoul_roads.geojson     # 전처리된 도로 라인 (slope_pct 포함)
│   └── slope_raw/              # 서울시 표고/등고선 SHP (해체된 zip)
├── scripts/
│   ├── preprocess_roads.py     # OSM raw → 정제
│   └── preprocess_slope.py     # 표고 SHP → 도로별 경사도 추가
└── utils/
    ├── constants.py            # 자치구 중심좌표 등
    ├── data_loader.py          # CSV 로딩 + 도로명 추출
    ├── geo.py                  # GeoJSON 로딩 + 매칭 + 공간조인 + 슬로프 등급
    └── proj.py                 # EPSG:5174 → WGS84 좌표 변환
```

## 실행 방법

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

기본 브라우저로 `http://localhost:8501` 이 자동으로 열립니다.

## 첫 실행 시 데이터 준비

앱은 다음 4개 파일이 `data/` 폴더에 있어야 동작합니다:

- `heating_2026.csv` (작음, repo에 commit)
- `icing_zones.csv` (작음, repo에 commit)
- `seoul_gu.geojson` (작음, repo에 commit)
- `seoul_roads.geojson` (~18MB, `.gitignore`로 제외 — 별도 다운로드 필요)

`seoul_roads.geojson`이 없으면 앱이 자동으로 **데이터 준비 화면**을 띄워줍니다. 두 가지 경로 중 하나를 선택:

### A. Google Drive 다운로드 (온라인 환경)

기본 URL이 이미 코드에 박혀있어 **부트스트랩 화면에서 "다운로드 시작" 버튼만 누르면 끝**입니다.

```
파일: road_data.zip (≈30 MB 압축)
내용: seoul_roads.geojson (18MB) + road_links.geojson (55MB)
URL : https://drive.google.com/file/d/1Mn8bvZF5UIpCqg1WYBGekx1Gkdc1NX3h/view
```

별도 URL로 운영하려면 `.streamlit/secrets.toml` 에서 덮어쓸 수 있습니다:
```toml
DATA_DRIVE_URL = "https://drive.google.com/file/d/YOUR_FILE_ID/view"
```

### B. 로컬 zip/파일 업로드 (폐쇄망 환경)

USB 등으로 받아온 zip을 그대로 브라우저에 업로드하면 자동 압축 해제·배치됩니다. 단일 GeoJSON·CSV 파일도 업로드 가능. zip 안에 폴더 구조가 있어도 파일 이름만 인식하므로 신경 안 써도 됩니다.

## 공개 배포 (Streamlit Community Cloud)

1. **GitHub 새 repo 생성** (public 또는 private 모두 가능)
2. **로컬에서 push** (raw 원본 데이터는 `.gitignore`로 제외됨):
   ```bash
   cd streamlit_app
   git init
   git add .
   git commit -m "initial deploy"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
3. **Streamlit Community Cloud 가입**: `https://share.streamlit.io` → GitHub 로그인
4. **New app** → repo · branch (`main`) · 메인 파일 (`app.py`) 지정 → Deploy
5. 1~2분 후 `https://<app-name>.streamlit.app` 공개 URL 생성

### 파일 분류 (배포 가능성 + 런타임 필요성)

#### 🟢 GitHub에 commit (가벼움, 항상 함께)

| 파일 | 크기 | 용도 |
|---|---|---|
| 코드 전체 (`app.py`, `utils/`, `scripts/`) | ~80KB | 핵심 로직 |
| `data/heating_2026.csv` | 50KB | 열선 설치 현황 |
| `data/icing_zones.csv` | 564KB | 상습 결빙구간 |
| `data/seoul_gu.geojson` | 36KB | 자치구 경계 |
| `data/flpop.csv` | ~750KB | 유동인구 기본 샘플 |
| **소계** | **~1.5MB** | |

#### 🟡 별도 배포 (Drive zip 또는 USB)

런타임에 필요하지만 큰 파일들.

| 파일 | 크기 | 필수? | 기능 |
|---|---|---|---|
| `data/seoul_roads.geojson` | 18 MB | ✅ 필수 | 도로 시각화 + 경사도 |
| `data/road_links.geojson` | 55 MB | ⚙️ 선택 | 유동인구 × 경사도 도로 시각화 |

**필수 파일**이 없으면 → 앱 진입 시 부트스트랩 화면 자동 표시 → Drive 다운로드 / USB 업로드 안내

**선택 파일**이 없으면 → 앱은 정상 동작, 메인 화면 상단 배너로 안내 + 사이드바 "데이터 상태" 에서 별도 업로드 가능. 도로링크 시각화 기능만 비활성.

#### ⛔ 100MB 초과 — GitHub 절대 불가 (`.gitignore` 자동 차단)

| 파일 | 크기 | 비고 |
|---|---|---|
| `data/slope_raw/등고선 5000/N3L_F001.shp` | **169 MB** | 사용 안 함 (표고만 씀) |
| `data/서울시 경사도.zip` | 100 MB | 압축 해제 결과만 쓰임 |

#### 🟠 전처리 입력 파일 (런타임 불필요)

협업자가 재전처리할 때만 받으면 됩니다. **일반 사용자는 받을 필요 없음**.

| 파일 | 크기 |
|---|---|
| `data/seoul_roads_raw.geojson` | 42 MB |
| `data/shp/TBGIS_ROAD_LINK_FRM.*` | 29 MB |
| `data/slope_raw/표고 5000/` | 11 MB |

#### 🔒 개인/캠퍼스 데이터 (`.gitignore`)

| 파일 | 비고 |
|---|---|
| `data/flpop.csv` | 캠퍼스에서 추출한 유동인구. 개인 데이터로 보호 |

### 정리: 사용자가 받아야 할 것

**일반 사용자** (앱만 쓰는 사람):
- GitHub repo clone → `seoul_roads.geojson` + `road_links.geojson` zip 다운로드 → 끝

**협업자/개발자** (재전처리 필요한 사람):
- 위 + 표고 SHP + OSM raw GeoJSON
- 자세한 출처는 "데이터 출처" 섹션 참고

## 외부 데이터 추가 (라인/경계 정확도 향상)

기본 상태에서도 동작하지만, 아래 두 파일을 `data/` 폴더에 두면 더 정확한
시각화가 가능합니다.

### 1) 자치구 경계 GeoJSON

파일명: `data/seoul_gu.geojson`
- 출처 예시: GitHub `southkorea/seoul-maps`
  (kostat/2013/json/seoul_municipalities_geo_simple.json)
- 또는 서울 열린데이터광장의 행정동/자치구 경계 데이터를 GeoJSON으로 변환

자치구명이 담긴 properties 키는 다음 중 어느 것이라도 자동 인식:
`SIG_KOR_NM`, `sig_kor_nm`, `name`, `자치구`, `SGG_NM`

### 2) 도로명 라인 GeoJSON (이미 포함)

파일명: `data/seoul_roads.geojson` (14.5 MB, 약 52,000 라인)

이미 OSM Overpass에서 받아 전처리한 결과가 포함되어 있습니다. 새로 갱신하려면:

```
1. https://overpass-turbo.eu/ 접속
2. 쿼리 실행:
   [out:json][timeout:300];
   area["name"="서울특별시"]["admin_level"=4]->.seoul;
   (way["highway"]
     ["highway"!~"^(footway|cycleway|path|steps|track|pedestrian|service|construction|proposed|raceway|bus_guideway)$"]
     (area.seoul););
   out geom;
3. Export → GeoJSON 다운로드
4. data/seoul_roads_raw.geojson 으로 저장
5. python3 scripts/preprocess_roads.py 실행
```

`preprocess_roads.py` 가 자동으로:
- Polygon geometry 제외 (휴게소 등)
- 각 도로 centroid를 자치구 경계에 ray-casting 공간조인 → `gu` 부여
- 도로명 정규화 (`name:ko` > `name`)
- 좌표 5자리 정밀도로 압축 (43MB → 14.5MB)

## 기능

- **사이드바**
  - 배경 지도: 위성영상(Esri) 토글 + 투명도 슬라이더
  - 레이어 토글: 자치구 색상 / 전체 도로 배경 / 열선 설치 하이라이트 / 상습 결빙구간 / 경사도 등급별
  - 경사도 등급 멀티셀렉트 (0~3%, 3~6%, 6~10%, ≥10%, 미상)
  - 자치구 멀티셀렉트 (선택된 자치구만 도로·결빙 lazy 로드)
  - 설치연도 범위 슬라이더
  - 자치구 색상 기준 지표 선택(총 연장 / 도로 수 / 위험도)
- **지도** (pydeck)
  - 위성영상 배경 (Esri World Imagery, TileLayer, 선택)
  - 자치구 choropleth (PolygonLayer, 위성영상 켜졌을 때 자동 반투명)
  - 전체 도로 배경 (회색, lazy)
  - 열선 설치 도로 하이라이트 (빨강, OSM 매칭)
  - 상습 결빙구간 (파랑, OSM 도로 라인에 스냅)
  - 경사도 등급별 도로 (회색→노랑→주황→빨강 그라데이션)
- **분석 섹션** (expander)
  - 자치구별 집계 (열선)
  - 도로별 집계 (도로명 단위)
  - 원본 데이터 (필터 적용 후)
  - **경사도 분석** (자치구별 ≥6% / ≥10% 비율)
  - **경사도 × 결빙구간 교차 분석** (결빙 위험의 경사도 분포)
  - 상습 결빙구간 분석 (자치구·도로분류별)
  - **🚶 유동인구 분석** (피봇 테이블, CSV 업로드)
  - 도로 매칭 통계 (OSM ↔ CSV)
  - 위험도 점수 (placeholder)

### 유동인구 데이터 (캠퍼스 추출 CSV) + 도로링크 시각화

빅데이터캠퍼스에서 받은 *도로구간별 추정 유동인구* CSV를 사이드바
**🚶 유동인구 데이터** 에 업로드하면 활성화됩니다.

**🗺️ 도로 단위 시각화** 도 가능. flpop의 `rd_link_cd` ↔ TBGIS 도로링크 SHP
의 `ROAD_LID` 로 정확 매칭됩니다.

전처리:
```bash
# 도로링크 SHP 한 번 변환 (EPSG:5181 → WGS84)
python3 scripts/preprocess_road_links.py
# → data/road_links.geojson 생성 (~40 MB, 180K 도로링크)
```

사이드바 **유동인구 × 경사도 (TBGIS 도로링크)** 토글을 켜면 필터·시간대·연령대를
바꿀 때마다 도로링크별 합계가 색·두께로 즉시 갱신됩니다.

**색상 (hue) = 경사도 등급**:
- 🔴 빨강: ≥10% (매우 위험)
- 🟠 주황: 6-10% (위험)
- 🟡 노랑: 3-6% (보통)
- ⚪ 회색: 평탄

**투명도 (alpha) = 유동인구 분위수**: 적음(20%) → 많음(100%) 5단계.

**빨강 + 진함 = 가파르고 유동인구 많은 최우선 열선 설치 대상.**

**flpop CSV 컬럼 형식 두 가지 모두 지원**:
- (A) 한글+영문: `시군구명(SIGNGU_NM)` 등 (샘플 데이터 형식)
- (B) 영문 소문자: `signgu_nm, adstrd_nm, dayweek_cd, agrde_cd, tmzon_cd, flpop_co` 등 (캠퍼스 풀 데이터 형식)

기능:
- **필터**: 자치구·행정동·요일(평일/주말)·시간대(7구간)·연령대(7구간) 멀티셀렉트
- **빠른 프리셋 버튼**:
  - `🌅 출근시간(7-9시)` — 결빙 취약 시간대
  - `👴 60대 이상 평일` — 낙상 위험층
- **피봇 테이블**: 행/열 차원 자유 선택 (자치구·행정동·요일·연령·시간), 합계/평균/건수 집계
- **CSV 다운로드**: 필터·피봇 적용 결과 그대로 내보내기

기대 컬럼 (캠퍼스 원본):
`기준_년월_코드, 도로링크_ID, 시군구코드/명, 행정동코드/명, 요일코드, 연령대코드, 시간대코드, 유동인구_수`

인코딩은 CP949 / UTF-8 자동 감지. 시간대 코드 매핑은 `utils/foot_traffic.py`
의 `DEFAULT_TIME_LABELS` 에서 조정 가능.

## 경사도 계산

`scripts/preprocess_slope.py` 가 도로 GeoJSON 의 각 라인에 다음을 추가:

| 컬럼 | 의미 |
|---|---|
| `elev_start_m` | 시작점 최근접 표고 점의 높이(m) |
| `elev_end_m` | 끝점 최근접 표고 점의 높이(m) |
| `dist_m` | 시작-끝 직선거리(m) |
| `slope_pct` | `|Δh|/dist × 100` (%) |
| `slope_level` | `0_3` / `3_6` / `6_10` / `10_plus` / `unknown` |

원본은 EPSG:5174 평면좌표라 `utils/proj.py` 의 자체 구현 TM 역변환 +
Helmert datum shift로 WGS84로 변환한 뒤 100m 그리드 인덱스로 lookup.

서울 전체 분포: 0~3% **69.8%** · 3~6% **12.0%** · 6~10% **7.2%** · ≥10% **10.1%** · 미상 **0.9%**

### 핵심 발견

**상습 결빙구간 paths의 83.7%가 경사도 ≥6%** (≥10% 만 70.6%). 경사도 6%
이상이 결빙 위험의 강력한 신호임을 데이터로 확인.

## 도로 매칭 규칙

CSV의 `설치구간`에서 추출한 도로명을 OSM 도로 라인과 매칭:

1. **정확 매칭**: `(자치구, 도로명)` 키가 일치
2. **이름 fallback**: 자치구가 다르더라도 같은 도로명이 OSM에 있으면 매칭
   - CSV의 자치구는 *관리주체*, OSM의 자치구는 *실제 위치*. 둘이 다른 경우가 종종 있음
   - 툴팁에 `※ 위치 자치구: XX (관리주체와 다름)` 으로 표시

## 향후 확장 포인트

- `compute_risk_score`를 실제 도로 경사도/결빙위험 데이터 결합으로 교체
- 도로 라인 매칭률 향상: 도로명 normalize, alias 사전, district 보정
- 추가 레이어: 사고 다발구역, 교통량, 결빙 발생 이력 등
