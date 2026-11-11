"""서울시 도로 열선 설치 현황 시각화 & 분석 (Streamlit).

실행:
    cd streamlit_app
    pip install -r requirements.txt
    streamlit run app.py

데이터 폴더에 둘 수 있는 추가 파일(있으면 자동 인식):
    data/seoul_gu.geojson      - 서울 25개 자치구 경계
    data/seoul_roads.geojson   - 도로명 라인 (LineString)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.constants import (
    DATA_DIR,
    GU_GEOJSON,
    HEATING_CSV,
    ICING_CSV,
    ROAD_GRAPH_NODES_CSV,
    ROAD_LINKS_GEOJSON,
    ROAD_SHAPEFILE,
    SEOUL_CENTER,
    SEOUL_GU_CENTROIDS,
)
from utils.data_bootstrap import (
    DEFAULT_DRIVE_URL,
    FILE_INFO,
    OPTIONAL_FILES,
    REQUIRED_FILES,
    download_from_drive,
    extract_archive_to,
    missing_files,
    missing_optional,
    save_uploaded_to,
)
from utils.foot_traffic import (
    AGE_LABELS,
    DAY_LABELS,
    DEFAULT_TIME_LABELS,
    aggregate_by_link,
    filter_foot_traffic,
    load_foot_traffic_csv,
    pivot_foot_traffic,
)
from utils.delta_lab import render_delta_lab
from utils.risk import obs_risk
from utils.data_loader import (
    aggregate_by_gu,
    aggregate_by_road,
    compute_risk_score,
    load_heating_csv,
    load_icing_csv,
)
from utils.geo import (
    ROAD_TYPE_COLORS,
    ROAD_TYPE_LABELS,
    ROAD_TYPE_LEVELS,
    SLOPE_COLORS,
    SLOPE_LABELS,
    build_gu_polygon_index,
    fallback_points,
    filter_icing_to_seoul,
    filter_roads_by_road_type,
    filter_roads_by_slope,
    highway_to_road_type,
    load_gu_geojson,
    load_roads_geojson,
    match_roads,
    snap_icing_to_osm_paths,
)


# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="서울시 도로 열선 분석",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / DATA_DIR

#: 위험군 경사(%) — 「장애인등편의법 시행규칙」 별표1 접근로 기울기 1/18 (발표자료 2쪽)
STEEP_PCT = 100 / 18


def _risk_rgba(p: float | None, alpha: int = 220) -> list[int]:
    """미끄러짐 확률 P(MAA<θ) 0~1 → 연노랑→진빨강. 미상은 회색."""
    if p is None or p != p:
        return [160, 160, 160, 90]
    p = max(0.0, min(1.0, float(p)))
    stops = [(0.0, (253, 231, 179)), (0.4, (247, 160, 90)),
             (0.7, (220, 75, 50)), (1.0, (130, 20, 30))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if p <= b:
            t = (p - a) / (b - a)
            return [int(ca[k] + t * (cb[k] - ca[k])) for k in range(3)] + [alpha]
    return list(stops[-1][1]) + [alpha]


@st.cache_data(show_spinner="1:1,000 경사 링크 통계 계산 중...")
def _load_link_risk_stats() -> pd.DataFrame | None:
    """seoul_links.parquet(1:1,000 경사·열선·결빙) → 골목길 링크별 위험 지표.

    발표자료와 같은 정의: 위험군 = 경사 ≥5.56%, 위험도 = P(MAA<θ)(WinterLab EDF).
    """
    p = DATA_PATH / "seoul_links.parquet"
    if not p.exists():
        return None
    L = pd.read_parquet(p, columns=["gu", "slope_pct", "road_type", "dist_m", "heat", "icing"])
    L = L[L.gu.notna()].copy()
    L["len"] = L.dist_m.fillna(0.0)
    L["risk"] = [obs_risk(v) if v == v else None for v in L.slope_pct]
    L["steep"] = L.slope_pct >= STEEP_PCT
    return L


# ──────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_csv() -> pd.DataFrame:
    return load_heating_csv(DATA_PATH / HEATING_CSV)


@st.cache_data(show_spinner=False)
def _load_gu_geojson() -> dict | None:
    return load_gu_geojson(DATA_PATH / GU_GEOJSON)


@st.cache_data(show_spinner="도로 라인 데이터 로딩 중...")
def _load_road_features() -> list[dict] | None:
    return load_roads_geojson(DATA_PATH / ROAD_SHAPEFILE)


def _path_length_m(coords: list[list[float]]) -> float:
    """LineString 좌표열 → 총 길이(m). 위경도 → 미터 근사."""
    from utils.geo import _euclid_m  # 서울 위도 가정 근사
    if not coords or len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(coords)):
        a, b = coords[i - 1], coords[i]
        total += _euclid_m(a[0], a[1], b[0], b[1])
    return total


@st.cache_data(show_spinner="도로링크(TBGIS) 로드 중...")
def _load_road_link_data() -> dict[str, dict]:
    """road_links.geojson 을 {road_lid: {coords, slope_level, slope_pct, length_m}} 로 로드.

    파일이 없으면 빈 dict (유동인구 시각화 비활성).
    """
    p = DATA_PATH / ROAD_LINKS_GEOJSON
    if not p.exists():
        return {}
    import json
    with p.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    out: dict[str, dict] = {}
    for feat in gj.get("features", []):
        if feat.get("geometry", {}).get("type") != "LineString":
            continue
        p_ = feat.get("properties", {})
        lid = p_.get("road_lid", "")
        if not lid:
            continue
        coords = feat["geometry"]["coordinates"]
        out[lid] = {
            "coords": coords,
            "slope_level": p_.get("slope_level", "unknown"),
            "slope_pct": p_.get("slope_pct"),
            "length_m": round(_path_length_m(coords), 1),
            # scripts/classify_road_type.py 가 부여 (없으면 미분류로 취급)
            "road_type": p_.get("road_type", "unknown"),
            "road_type_highway": p_.get("road_type_highway", ""),
            "road_type_match_dist_m": p_.get("road_type_match_dist_m"),
            # 2026-09-13 1:1,000 정밀 경사도 확보 여부 (preprocess_slope_1000*.py)
            "source_1000": bool(p_.get("source_1000")),
            "slope1000_method": p_.get("slope1000_method", ""),
        }
    return out


@st.cache_data(show_spinner="골목길 경사도 데이터 로드 중...")
def _load_golmok_paths() -> list[dict]:
    """road_links.geojson 중 road_type == '골목길' 인 링크만 지도용 path로 변환.

    _load_road_link_data() 는 link_id 키 dict라 지도 레이어 구성 시 매번 순회해야
    하므로, 골목길만 뽑은 얕은 리스트를 따로 캐싱 — '골목길 경사도' /
    '경사도 정밀도 갱신 현황' 두 레이어가 공유해서 씀.
    각 항목: road_lid, path, slope_level, slope_pct, source_1000(1:1,000 확보 여부),
    slope1000_method(TIN_local/IDW_fallback/혼합), elev_start_m, elev_end_m.
    """
    p = DATA_PATH / ROAD_LINKS_GEOJSON
    if not p.exists():
        return []
    import json
    with p.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    out: list[dict] = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        if props.get("road_type") != "골목길":
            continue
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        out.append(
            {
                "road_lid": props.get("road_lid", ""),
                "path": coords,
                "slope_level": props.get("slope_level", "unknown"),
                "slope_pct": props.get("slope_pct"),
                "source_1000": bool(props.get("source_1000")),
                "slope1000_method": props.get("slope1000_method", ""),
                "elev_start_m": props.get("elev_start_m"),
                "elev_end_m": props.get("elev_end_m"),
            }
        )
    return out


@st.cache_data(show_spinner="4개동 후보 링크 로드 중...")
def _load_tier_risk_nodes() -> pd.DataFrame:
    """road_graph.json(성북구 4개동 1,190개) — 열선 사각지대 레이어용.

    위험군 = 경사 ≥5.56%(발표자료 기준). 지표온도(LST) 기반 지형/노출위험
    분류는 발표 모형에서 제외(7쪽: 기온·일사량은 골목 간 차이를 설명 못함)되어
    레이어에서도 뺐다.
    """
    import json
    p = DATA_PATH / "road_graph.json"     # 정밀 열선 매칭(heating_installed_precise) 포함
    if not p.exists():
        return pd.DataFrame()
    df = pd.DataFrame(json.load(open(p, encoding="utf-8"))["nodes"])
    df = df.rename(columns={"id": "_id"})
    df["flow_sum"] = df["flow_sum"].fillna(0.0)
    valid = df[df["slope_pct"].notna()].copy()
    if valid.empty:
        return valid
    valid["risk"] = valid["slope_pct"].map(obs_risk)
    valid["steep_golmok_no_heat"] = (
        (valid["road_type"] == "골목길")
        & (valid["slope_pct"] >= STEEP_PCT)
        & (valid["heating_installed_precise"] == False)  # noqa: E712
    )
    return valid


@st.cache_data(show_spinner="결빙구간 OSM 스냅 중...")
def _load_icing_snapped() -> list[dict]:
    """결빙구간 → OSM 도로 라인에 스냅된 path 리스트.

    결빙구간 도로명이 OSM 도로와 매칭되고 거리 임계값 안에 있으면 OSM의
    실제 좌표열을 사용 (곡선 도로도 정확히 표현). 매칭 실패시 기점-종점 직선.
    """
    df_icing = _load_icing_seoul()
    if df_icing.empty:
        return []
    feats = _load_road_features()
    return snap_icing_to_osm_paths(df_icing, feats, max_dist_m=600.0)


@st.cache_data(show_spinner="결빙구간 데이터 처리 중...")
def _load_icing_seoul() -> pd.DataFrame:
    """결빙구간 CSV + 서울 자치구 공간조인 결과를 캐시."""
    csv_path = DATA_PATH / ICING_CSV
    if not csv_path.exists():
        return pd.DataFrame()
    df_raw = load_icing_csv(csv_path)
    gj = _load_gu_geojson()
    if not gj:
        # 자치구 GeoJSON 없으면 bbox 폴백 (서울 대략 경계)
        bbox_filt = (
            df_raw["lat_start"].between(37.42, 37.71)
            & df_raw["lon_start"].between(126.76, 127.20)
        )
        out = df_raw[bbox_filt].copy()
        out["gu_start"] = ""
        out["gu_end"] = ""
        out["gu"] = ""
        return out.reset_index(drop=True)
    gu_idx = build_gu_polygon_index(gj)
    return filter_icing_to_seoul(df_raw, gu_idx)


@st.cache_data(show_spinner=False)
def _split_roads_by_gu(_features: list[dict] | None) -> dict[str, list[dict]]:
    """전체 도로 features를 OSM 자치구 키로 미리 분할 (lazy 렌더링용).

    `_features` 앞 언더스코어는 streamlit 캐시 해싱 제외 (큰 객체 hash 회피).
    """
    if not _features:
        return {}
    buckets: dict[str, list[dict]] = {}
    for feat in _features:
        gu = feat.get("properties", {}).get("gu", "")
        if gu:
            buckets.setdefault(gu, []).append(feat)
    return buckets


# ──────────────────────────────────────────────────────────────────────
# Layer builders
# ──────────────────────────────────────────────────────────────────────

def _fmt_int(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return "-"


ESRI_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def build_satellite_layer(opacity: float = 0.85) -> pdk.Layer:
    """Esri World Imagery 타일 레이어 (무료, API 키 불필요).

    pydeck TileLayer가 XYZ 타일 URL 템플릿을 받아 자동으로 로드/캐시.
    """
    return pdk.Layer(
        "TileLayer",
        ESRI_IMAGERY_URL,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        opacity=opacity,
        pickable=False,
    )


def build_gu_choropleth_layer(gj: dict, gu_metrics: pd.DataFrame, metric_col: str) -> pdk.Layer:
    """자치구 색상 레이어. metric_col 값 비례로 채색.

    GeoJsonLayer 대신 PolygonLayer로 평탄한 구조의 row 리스트를 사용 →
    tooltip이 다른 레이어와 동일한 `{description}` 문법으로 동작.
    """
    metrics_by_gu = gu_metrics.set_index("gu").to_dict(orient="index")
    NAME_KEYS = ("SIG_KOR_NM", "sig_kor_nm", "name", "자치구", "SGG_NM", "gu")

    vmax = (gu_metrics[metric_col].max() or 1) if metric_col in gu_metrics else 1

    def _color(v: float) -> list[int]:
        ratio = (v / vmax) if vmax else 0
        r = int(40 + 200 * ratio)
        g = int(80 + 60 * (1 - ratio))
        b = int(180 - 140 * ratio)
        return [r, g, b, 140]

    rows: list[dict] = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        gu_name = next(
            (props[k] for k in NAME_KEYS if k in props and isinstance(props[k], str)),
            "?",
        )
        m = metrics_by_gu.get(gu_name, {})
        v = m.get(metric_col, 0) or 0

        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        if gtype == "Polygon":
            polys = [geom["coordinates"]]
        elif gtype == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue

        desc = (
            f"<b>{gu_name}</b><br/>"
            f"도로 수: {_fmt_int(m.get('road_count'))} 건<br/>"
            f"총 연장: {_fmt_int(m.get('total_length_m'))} m<br/>"
            f"구간 평균: {_fmt_int(m.get('avg_length_m'))} m<br/>"
            f"위험군 골목길: {m.get('steep_km', 0) or 0:,.1f} km · "
            f"열선 커버리지 {m.get('steep_cov_pct', 0) or 0:.1f}%"
        )

        for poly_rings in polys:
            outer = poly_rings[0] if poly_rings else []
            if not outer:
                continue
            rows.append(
                {
                    "polygon": outer,
                    "gu": gu_name,
                    "fill_color": _color(v),
                    "description": desc,
                }
            )

    return pdk.Layer(
        "PolygonLayer",
        rows,
        pickable=True,
        stroked=True,
        filled=True,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[60, 60, 60],
        line_width_min_pixels=1,
    )


def build_gu_circle_layer(gu_metrics: pd.DataFrame, metric_col: str) -> pdk.Layer:
    """자치구 GeoJSON이 없을 때의 fallback. 중심점에 metric 비례 원."""
    rows = []
    vmax = (gu_metrics[metric_col].max() or 1)
    for _, r in gu_metrics.iterrows():
        c = SEOUL_GU_CENTROIDS.get(r["gu"])
        if not c:
            continue
        lat, lon = c
        ratio = (r[metric_col] / vmax) if vmax else 0
        rows.append(
            {
                "gu": r["gu"],
                "lat": lat,
                "lon": lon,
                "value": float(r[metric_col]),
                "radius": 200 + ratio * 1500,
                "color": [
                    int(40 + 200 * ratio),
                    int(80 + 60 * (1 - ratio)),
                    int(180 - 140 * ratio),
                    180,
                ],
                "description": (
                    f"<b>{r['gu']}</b><br/>"
                    f"도로 수: {_fmt_int(r.get('road_count'))} 건<br/>"
                    f"총 연장: {_fmt_int(r.get('total_length_m'))} m<br/>"
                    f"구간 평균: {_fmt_int(r.get('avg_length_m'))} m"
                ),
            }
        )
    return pdk.Layer(
        "ScatterplotLayer",
        pd.DataFrame(rows),
        pickable=True,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        stroked=True,
        get_line_color=[40, 40, 40],
        line_width_min_pixels=1,
    )


def build_road_bg_layer(features_by_gu: dict, selected_gus: list[str]) -> pdk.Layer | None:
    """선택된 자치구의 전체 도로를 회색 배경으로 렌더링 (lazy).

    선택된 자치구만 features 추출 → 5만개 전부 그리지 않고 부분만.
    """
    paths: list[dict] = []
    for gu in selected_gus:
        for feat in features_by_gu.get(gu, []):
            coords = feat.get("geometry", {}).get("coordinates", [])
            if not coords:
                continue
            paths.append({
                "path": coords,
                "description": (
                    f"<b>{feat['properties'].get('road_name','(이름 없음)')}</b>"
                    f"<br/>자치구: {gu}"
                    f"<br/>등급: {feat['properties'].get('highway','')}"
                ),
            })
    if not paths:
        return None
    return pdk.Layer(
        "PathLayer",
        paths,
        pickable=True,
        get_path="path",
        get_color=[170, 170, 170, 110],
        width_scale=1,
        width_min_pixels=1,
    )


def build_road_line_layer(road_lines) -> pdk.Layer:
    """매칭된 열선 설치 도로 라인 PathLayer (마젠타 하이라이트).

    색상 마젠타 (#DC32B4) — 경사도 색상(빨강·주황 계열)과 결빙(파랑)과
    모두 구분되도록 선택.
    """
    paths = []
    for rl in road_lines:
        diff_note = (
            f"<br/><small>※ 위치 자치구: {rl.osm_gu} (관리주체와 다름)</small>"
            if rl.osm_gu and rl.osm_gu != rl.gu
            else ""
        )
        paths.append({
            "path": rl.coords,
            "gu": rl.gu,
            "road_name": rl.road_name,
            "description": (
                f"<b>{rl.road_name}</b>"
                f"<br/>관리: {rl.gu}"
                f"{diff_note}"
            ),
        })
    return pdk.Layer(
        "PathLayer",
        paths,
        pickable=True,
        get_path="path",
        get_color=[220, 50, 180, 235],  # 마젠타 (경사도/결빙과 구분)
        width_scale=1,
        width_min_pixels=4,
    )


def build_road_type_layer(
    features_by_gu: dict[str, list[dict]],
    selected_gus: list[str],
    selected_types: list[str],
) -> pdk.Layer | None:
    """도로유형(골목길/애매/대형도로/미분류)별 색상 도로 레이어.

    OSM highway 태그를 기준으로 분류(utils.geo.highway_to_road_type).
    시급도 후보 산출 시 '대형도로'는 기본 제외되므로, 이 레이어로 제외 대상을
    지도에서 미리 눈으로 확인할 수 있다.
    """
    feats = filter_roads_by_road_type(features_by_gu, selected_gus, selected_types)
    if not feats:
        return None

    paths = []
    for feat in feats:
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue
        hw = props.get("highway", "")
        rt = highway_to_road_type(hw)
        paths.append(
            {
                "path": coords,
                "color": ROAD_TYPE_COLORS.get(rt, ROAD_TYPE_COLORS["unknown"]),
                "description": (
                    f"<b>{props.get('road_name','(이름 없음)')}</b>"
                    f"<br/>자치구: {props.get('gu','')}"
                    f"<br/>도로유형: {ROAD_TYPE_LABELS.get(rt, rt)}"
                    f"<br/>OSM highway: {hw or '(태그 없음)'}"
                ),
            }
        )

    if not paths:
        return None
    return pdk.Layer(
        "PathLayer",
        paths,
        pickable=True,
        get_path="path",
        get_color="color",
        width_scale=1,
        width_min_pixels=2,
    )


def build_golmok_slope_layer(
    golmok_paths: list[dict],
    steep_only: bool = True,
) -> pdk.Layer | None:
    """골목길(TBGIS road_links.geojson) 미끄러짐 위험 레이어.

    경사는 수치지형도 1:1,000 기반. 색은 경사 등급이 아니라 발표 모형의
    위험도 P(MAA<θ)(WinterLab 빙판 실측 MAA 경험분포)로 칠한다.
    steep_only=True 면 위험군(경사 ≥5.56%)만 그린다(서울 전역 14만 링크 부담 완화).
    """
    paths = []
    for item in golmok_paths:
        slope = item["slope_pct"]
        if not isinstance(slope, (int, float)):
            continue
        if steep_only and slope < STEEP_PCT:
            continue
        r = obs_risk(slope)
        es, ee = item.get("elev_start_m"), item.get("elev_end_m")
        elev_text = f"{es} m → {ee} m" if es is not None and ee is not None else "-"
        paths.append(
            {
                "path": item["path"],
                "color": _risk_rgba(r),
                "description": (
                    f"<b>골목길 링크 {item['road_lid']}</b>"
                    f"<br/>경사도: {slope:.1f}%{' (위험군)' if slope >= STEEP_PCT else ''}"
                    f"<br/>미끄러짐 확률 P(MAA<θ): {r:.2f}"
                    f"<br/>표고: {elev_text}"
                ),
            }
        )
    if not paths:
        return None
    return pdk.Layer(
        "PathLayer",
        paths,
        pickable=True,
        get_path="path",
        get_color="color",
        width_scale=1,
        width_min_pixels=2,
    )


# 최적화/시급도 후보에서 기본 제외할 도로유형 — 대형도로는 열선 설치 대상이 아님
DEFAULT_EXCLUDED_ROAD_TYPES = frozenset({"대형도로"})


def compute_urgency_scores(
    agg_df: pd.DataFrame,
    link_data: dict[str, dict],
    exclude_road_types: frozenset[str] | set[str] = DEFAULT_EXCLUDED_ROAD_TYPES,
) -> pd.DataFrame:
    """도로링크별 기대 미끄러짐 건수 = P(MAA<θ) × 유동인구 (발표자료 7~8쪽).

    경사를 등급 점수로 바꿔 유동인구 백분위와 섞던 이전 방식(임의 가중치)을
    버리고, 실측 MAA 경험분포에서 나온 확률을 그대로 곱한다. 단위가 '건'이라
    링크 간·지역 간 합산과 비교가 가능하다. urgency 컬럼 = expected_slips.
    """
    cols = ["slope_pct", "length_m", "road_type", "risk", "expected_slips", "urgency"]
    if agg_df.empty:
        return agg_df.assign(**{c: [] for c in cols})

    out = agg_df.copy()
    out["link_id"] = out["link_id"].astype(str)
    out["slope_pct"] = out["link_id"].map(lambda k: link_data.get(k, {}).get("slope_pct"))
    out["length_m"] = out["link_id"].map(lambda k: link_data.get(k, {}).get("length_m", 0.0))
    out["road_type"] = out["link_id"].map(
        lambda k: link_data.get(k, {}).get("road_type", "unknown"))

    if exclude_road_types:
        out = out[~out["road_type"].isin(exclude_road_types)].reset_index(drop=True)
    if out.empty:
        return out.assign(risk=[], expected_slips=[], urgency=[])

    out["risk"] = out["slope_pct"].map(obs_risk).round(3)
    out["expected_slips"] = (out["risk"] * out["flow_sum"]).round(2)
    out["urgency"] = out["expected_slips"]
    return out.sort_values("expected_slips", ascending=False).reset_index(drop=True)


def build_search_highlight_layer(
    lids, link_data: dict[str, dict]
) -> pdk.Layer | None:
    """검색된 link_id 도로(들)를 형광 노랑으로 하이라이트.

    lids: 단일 문자열 또는 리스트 모두 허용.
    """
    if not lids or not link_data:
        return None
    if isinstance(lids, str):
        lid_list = [lids]
    else:
        lid_list = list(lids)
    lid_list = [str(x).strip() for x in lid_list if x]

    paths = []
    for lid in lid_list:
        item = link_data.get(lid)
        if not item:
            continue
        coords = item.get("coords", [])
        if not coords:
            continue
        slope = item.get("slope_pct")
        slope_text = (
            f"{slope:.1f}%" if isinstance(slope, (int, float)) else "미상"
        )
        desc = (
            f"<b>선택 도로</b><br/>link_id: {lid}<br/>"
            f"경사도: {slope_text} ({item.get('slope_level','?')})<br/>"
            f"길이: {item.get('length_m', '-')} m"
        )
        paths.append({"path": coords, "description": desc})

    if not paths:
        return None
    return pdk.Layer(
        "PathLayer",
        paths,
        pickable=True,
        get_path="path",
        get_color=[255, 255, 0, 255],  # 형광 노랑
        width_min_pixels=8,
        width_scale=1,
    )


def compute_view_for_links(
    lids: list[str], link_data: dict[str, dict]
) -> tuple[float, float, float]:
    """선택된 도로들의 bbox 중심 + 자동 zoom 계산.

    반환: (lat, lon, zoom)
    """
    import math
    coords_all: list[list[float]] = []
    for lid in lids:
        item = link_data.get(str(lid).strip())
        if not item:
            continue
        coords_all.extend(item.get("coords", []))
    if not coords_all:
        return (SEOUL_CENTER[0], SEOUL_CENTER[1], 10.4)

    lons = [c[0] for c in coords_all]
    lats = [c[1] for c in coords_all]
    cx, cy = (max(lons) + min(lons)) / 2, (max(lats) + min(lats)) / 2

    # span 큰 쪽 기준 zoom 계산
    span_lon = max(lons) - min(lons)
    span_lat = max(lats) - min(lats)
    span = max(span_lon, span_lat, 1e-5)
    # log2(360/span) - 1 (패딩)
    zoom = math.log2(360.0 / span) - 1.5
    zoom = max(11.0, min(18.0, zoom))
    return (cy, cx, zoom)


def build_foot_traffic_layer(
    agg_df: pd.DataFrame, link_data: dict[str, dict]
) -> pdk.Layer | None:
    """기대 미끄러짐 = P(MAA<θ) × 유동인구 PathLayer.

    색 = 기대 미끄러짐(상위 2% 값으로 정규화한 연노랑→진빨강), 굵기도 같은 값.
    경사 등급 색 × 유동인구 투명도로 두 축을 눈으로 섞던 방식 대신, 발표 모형의
    목적함수 계수(risk_i·flow_i)를 그대로 보여준다.
    """
    if agg_df.empty or not link_data:
        return None
    rows = []
    for _, row in agg_df.iterrows():
        lid = str(row["link_id"])
        link = link_data.get(lid)
        if not link:
            continue
        s = link.get("slope_pct")
        r = obs_risk(s) if isinstance(s, (int, float)) else None
        v = float(row["flow_sum"])
        rows.append((lid, link, s, r, v, (r or 0.0) * v, row))
    if not rows:
        return None
    ev = pd.Series([x[5] for x in rows])
    top = float(ev.quantile(0.98)) or float(ev.max()) or 1.0

    paths = []
    for lid, link, s, r, v, e, row in rows:
        k = min(e / top, 1.0)
        slope_text = f"{s:.1f}%" if isinstance(s, (int, float)) else "미상"
        risk_text = f"{r:.2f}" if r is not None else "-"
        paths.append(
            {
                "path": link["coords"],
                "color": _risk_rgba(k, alpha=int(70 + 185 * k)),
                "width": 2.0 + 4.0 * k,
                "description": (
                    f"<b>기대 미끄러짐: {e:,.1f}건</b>"
                    f"<br/>미끄러짐 확률 P(MAA<θ): {risk_text} (경사 {slope_text})"
                    f"<br/>유동인구: {v:,.1f}"
                    f"<br/>link_id: {lid}"
                    f"<br/>행정동: {row.get('dong','')} ({row.get('gu','')})"
                ),
            }
        )
    return pdk.Layer(
        "PathLayer",
        paths,
        pickable=True,
        get_path="path",
        get_color="color",
        get_width="width",
        width_scale=1,
        width_min_pixels=2,
    )


def build_icing_layer(
    snapped: list[dict], selected_gus: list[str]
) -> pdk.Layer | None:
    """상습 결빙구간 PathLayer.

    snapped: snap_icing_to_osm_paths의 결과. OSM 스냅된 path 또는 직선 fallback.
    선택된 자치구(기점/종점 둘 중 하나)에 속하는 구간만 표시.
    """
    if not snapped:
        return None
    sel = set(selected_gus)

    paths = []
    for it in snapped:
        if not (it["gu"] in sel or it["gu_start"] in sel or it["gu_end"] in sel):
            continue

        cross = ""
        if it["gu_start"] and it["gu_end"] and it["gu_start"] != it["gu_end"]:
            cross = (
                f"<br/><small>※ {it['gu_start']} → {it['gu_end']} 경계 도로</small>"
            )
        elif not it["gu_start"] or not it["gu_end"]:
            cross = "<br/><small>※ 일부 구간이 서울 밖</small>"

        length_text = (
            f"{it['length_km']:.2f} km" if it["length_km"] is not None else "-"
        )
        snap_tag = (
            "OSM 스냅" if it["snap"] == "osm" else "직선 (매칭 실패)"
        )

        paths.append(
            {
                "path": it["coords"],
                "description": (
                    f"<b>{it['road_name'] or '(이름 없음)'}</b>"
                    f"<br/>분류: {it['road_class']}"
                    f"<br/>관리: {it['agency']}"
                    f"<br/>자치구: {it['gu']}"
                    f"<br/>길이: {length_text}"
                    f"<br/><small>표시 방식: {snap_tag}</small>"
                    f"{cross}"
                ),
            }
        )

    if not paths:
        return None
    return pdk.Layer(
        "PathLayer",
        paths,
        pickable=True,
        get_path="path",
        get_color=[60, 130, 230, 230],
        width_scale=1,
        width_min_pixels=4,
    )


def build_road_point_layer(points_df: pd.DataFrame) -> pdk.Layer:
    """도로 라인 매칭 실패시 fallback 점 레이어."""
    df = points_df.copy()
    df["description"] = df.apply(
        lambda r: (
            f"<b>{r['primary_road'] or '도로명 없음'}</b><br/>"
            f"자치구: {r['gu']}<br/>"
            f"설치연도: {int(r['year']) if pd.notna(r['year']) else '-'}<br/>"
            f"연장: {_fmt_int(r['length_m'])} m<br/>"
            f"<small>{r['section'][:60]}</small>"
        ),
        axis=1,
    )
    return pdk.Layer(
        "ScatterplotLayer",
        df,
        pickable=True,
        get_position="[lon, lat]",
        get_radius=80,
        get_fill_color=[220, 50, 180, 215],  # 마젠타 (열선 라인과 통일)
        stroked=False,
    )


def _tier_description(r: pd.Series, tag: str) -> str:
    return (
        f"<b>{tag}</b> — link {r['link_id']} ({r['dong']})"
        f"<br/>경사도: {r['slope_pct']:.2f}% · 미끄러짐 확률 P(MAA<θ) {r['risk']:.2f}"
        f"<br/>유동인구: {r['flow_sum']:.1f} · 기대 미끄러짐 {r['risk'] * r['flow_sum']:.1f}건"
        f"<br/>공식 결빙취약구역: {'예' if r['icing'] else '아니오'}"
        f"<br/>열선설치(정밀): {'예' if r['heating_installed_precise'] else '아니오'}"
    )


def build_heating_gap_layer(nodes: pd.DataFrame) -> pdk.Layer | None:
    """위험군(경사 ≥5.56%) 골목길인데 열선(정밀 클리핑 기준) 미설치인 사각지대.

    검은 테두리 + 진한 빨강 채움 — 3장/그림 3-3에서 강조한, 공식 지정에도
    현재 열선 우선순위에도 거의 포착되지 않는 실질적 표적 구간.
    """
    sub = nodes[nodes["steep_golmok_no_heat"]]
    if sub.empty:
        return None
    rows = [
        {"lat": r["lat"], "lon": r["lon"],
         "description": _tier_description(r, "위험군 골목길·열선 미설치")}
        for _, r in sub.iterrows()
    ]
    return pdk.Layer(
        "ScatterplotLayer",
        pd.DataFrame(rows),
        pickable=True,
        get_position="[lon, lat]",
        get_radius=7,
        radius_units='"pixels"',
        get_fill_color=[165, 15, 21, 230],
        stroked=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1.5,
    )


# ──────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.title("필터 & 레이어")

    st.sidebar.subheader("배경 지도")
    show_satellite = st.sidebar.checkbox(
        "위성영상 (Esri)",
        value=False,
        help="Esri World Imagery 타일을 배경으로 깔아 실제 지형을 확인. "
        "무료, API 키 불필요.",
    )
    satellite_opacity = st.sidebar.slider(
        "위성영상 투명도",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        disabled=not show_satellite,
    )

    st.sidebar.subheader("레이어")
    show_gu_layer = st.sidebar.checkbox("자치구 색상", value=True)
    show_road_bg = st.sidebar.checkbox(
        "전체 도로 (배경, 회색)",
        value=True,
        help="OSM에서 받은 서울 차량 도로 전체. 선택된 자치구만 렌더링(lazy)."
    )
    show_road_layer = st.sidebar.checkbox(
        "열선 설치 도로 (하이라이트, 마젠타)",
        value=True,
    )
    show_icing_layer = st.sidebar.checkbox(
        "상습 결빙구간 (파랑)",
        value=True,
        help="행정안전부 상습 결빙구간 데이터 중 서울 자치구 내부에 있는 구간만 표시.",
    )
    show_heating_gap_layer = st.sidebar.checkbox(
        "위험군 골목길 & 열선 미설치 (사각지대)",
        value=False,
        help="경사 5.56% 이상(위험군) 골목길인데 VWorld 정밀 클리핑 기준 열선이 "
        "설치되지 않은 구간(성북구 4개동). 검은 테두리 진한 빨강.",
    )
    show_foot_traffic_layer = st.sidebar.checkbox(
        "기대 미끄러짐 = P(MAA<θ) × 유동인구",
        value=False,
        help="발표 모형의 목적함수 계수(risk_i·flow_i). 미끄러짐 확률은 WinterLab "
        "빙판 보행 실측 MAA 경험분포. 색·굵기 = 기대 미끄러짐 건수. "
        "유동인구 분석 섹션의 필터(기본: 평일 06~11시)를 따른다.",
    )
    show_road_type_layer = st.sidebar.checkbox(
        "도로유형 (골목길/대형도로, 색상)",
        value=False,
        help="OSM highway 태그 기준 분류. 초록=골목길(residential/living_street/"
        "unclassified) · 노랑=애매(tertiary) · 파랑=대형도로(primary/secondary/"
        "trunk/motorway 등) · 회색=미분류. 기대 미끄러짐 순위에서는 대형도로가 기본 "
        "제외됩니다.",
    )
    show_golmok_slope_layer = st.sidebar.checkbox(
        "골목길 미끄러짐 위험 P(MAA<θ)",
        value=False,
        help="수치지형도 1:1,000 경사를 WinterLab 실측 MAA 경험분포로 미끄러짐 "
        "확률로 바꿔 칠한다(연노랑 0 → 진빨강 1, 경사 21.3% 이상 포화).",
    )
    steep_only = st.sidebar.checkbox(
        "　└ 위험군(경사 ≥5.56%)만",
        value=True,
        disabled=not show_golmok_slope_layer,
        help="끄면 서울 골목길 14만 링크를 모두 그려 느려질 수 있다.",
    )

    st.sidebar.subheader("도로유형 필터")
    selected_road_types = st.sidebar.multiselect(
        "도로유형 (도로유형 레이어용)",
        options=list(ROAD_TYPE_LEVELS),
        default=list(ROAD_TYPE_LEVELS),
        format_func=lambda x: ROAD_TYPE_LABELS.get(x, x),
        help="'도로유형' 레이어를 켰을 때 지도에 표시할 유형만 선택.",
    )

    st.sidebar.subheader("필터")
    all_gus = sorted(df["gu"].unique().tolist())
    selected_gus = st.sidebar.multiselect(
        "자치구",
        options=all_gus,
        default=all_gus,
        help="선택한 자치구만 지도/표에 반영됩니다.",
    )

    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    if years:
        y_min, y_max = st.sidebar.select_slider(
            "설치연도 범위",
            options=years,
            value=(years[0], years[-1]),
        )
    else:
        y_min = y_max = None

    st.sidebar.subheader("색상 기준")
    metric = st.sidebar.radio(
        "자치구 색상 기준 지표",
        options=[
            ("total_length_m", "총 연장(m)"),
            ("road_count", "도로 수"),
            ("steep_km", "위험군 골목길 연장(km)"),
            ("steep_cov_pct", "위험군 열선 커버리지(%)"),
        ],
        format_func=lambda x: x[1],
        horizontal=False,
    )

    # ── 도로링크 검색 ──
    st.sidebar.subheader("도로링크 찾기")
    search_lid = st.sidebar.text_input(
        "link_id 입력 후 Enter",
        value="",
        placeholder="예: 32731",
        help="기대 미끄러짐 순위표에서 본 link_id 를 입력하면 지도에서 해당 도로 위치로 자동 줌인 + 형광 강조 표시됩니다.",
        key="search_lid",
    )

    # ── 유동인구 CSV 업로드 (선택) ──
    st.sidebar.subheader("유동인구 데이터")
    default_flpop_exists = (DATA_PATH / "flpop.csv").exists()
    if default_flpop_exists:
        st.sidebar.caption("기본 `flpop.csv` 자동 로드됨. 다른 파일로 분석하려면 업로드:")
    ft_upload = st.sidebar.file_uploader(
        "도로구간별 추정 유동인구 CSV (선택)" if default_flpop_exists
        else "도로구간별 추정 유동인구 CSV",
        type=["csv"],
        help="빅데이터캠퍼스에서 받은 CSV. CP949/UTF-8 자동 감지. "
        "업로드하면 기본 데이터를 덮어씁니다.",
        key="ft_upload",
        label_visibility="visible",
    )

    return {
        "show_satellite": show_satellite,
        "satellite_opacity": satellite_opacity,
        "ft_upload": ft_upload,
        "show_foot_traffic_layer": show_foot_traffic_layer,
        "search_lid": search_lid.strip() if search_lid else "",
        "show_gu_layer": show_gu_layer,
        "show_road_bg": show_road_bg,
        "show_road_layer": show_road_layer,
        "show_icing_layer": show_icing_layer,
        "show_heating_gap_layer": show_heating_gap_layer,
        "show_road_type_layer": show_road_type_layer,
        "show_golmok_slope_layer": show_golmok_slope_layer,
        "steep_only": steep_only,
        "selected_gus": selected_gus,
        "selected_road_types": selected_road_types,
        "year_range": (y_min, y_max),
        "metric_col": metric[0],
        "metric_label": metric[1],
    }


def render_map(df: pd.DataFrame, opts: dict, gu_metrics: pd.DataFrame) -> None:
    layers: list[pdk.Layer] = []

    # 위성영상 배경 (가장 아래)
    if opts["show_satellite"]:
        layers.append(build_satellite_layer(opacity=opts["satellite_opacity"]))

    # 자치구 레이어 (choropleth or fallback circles)
    if opts["show_gu_layer"]:
        gj = _load_gu_geojson()
        if gj:
            poly_layer = build_gu_choropleth_layer(
                gj, gu_metrics, opts["metric_col"]
            )
            # 위성영상 위에 깔릴 땐 더 투명하게
            if opts["show_satellite"]:
                for row in poly_layer.data:
                    c = row["fill_color"]
                    if len(c) >= 4:
                        c[3] = max(40, int(c[3] * 0.4))
            layers.append(poly_layer)
        else:
            layers.append(build_gu_circle_layer(gu_metrics, opts["metric_col"]))

    # 배경 도로 (선택 자치구 lazy 렌더). 슬로프 레이어와 동시 표시 가능.
    if opts["show_road_bg"]:
        road_features = _load_road_features()
        if road_features:
            by_gu = _split_roads_by_gu(road_features)
            bg = build_road_bg_layer(by_gu, opts["selected_gus"])
            if bg:
                layers.append(bg)

    # 도로유형(골목길/대형도로) 색상 도로
    if opts["show_road_type_layer"]:
        road_features = _load_road_features()
        if road_features:
            by_gu = _split_roads_by_gu(road_features)
            rt = build_road_type_layer(
                by_gu, opts["selected_gus"], opts["selected_road_types"]
            )
            if rt is not None:
                layers.append(rt)

    # 골목길 미끄러짐 위험 P(MAA<θ) — 1:1,000 경사
    if opts["show_golmok_slope_layer"]:
        golmok_paths = _load_golmok_paths()
        if golmok_paths:
            gsl = build_golmok_slope_layer(golmok_paths, opts["steep_only"])
            if gsl is not None:
                layers.append(gsl)

    # 열선 설치 도로 하이라이트
    if opts["show_road_layer"]:
        road_features = _load_road_features()
        road_lines = match_roads(df, road_features)
        if road_lines:
            layers.append(build_road_line_layer(road_lines))
        else:
            # fallback: 자치구 중심 jitter 점 (도로 데이터 자체가 없을 때만 발동)
            pts = fallback_points(df)
            if not pts.empty:
                layers.append(build_road_point_layer(pts))

    # 기대 미끄러짐 = P(MAA<θ) × 유동인구
    if opts.get("show_foot_traffic_layer") and opts.get("ft_link_agg") is not None:
        link_data = _load_road_link_data()
        if link_data:
            ft_layer = build_foot_traffic_layer(opts["ft_link_agg"], link_data)
            if ft_layer is not None:
                layers.append(ft_layer)

    # 상습 결빙구간 (OSM 스냅된 path)
    if opts["show_icing_layer"]:
        snapped = _load_icing_snapped()
        ic = build_icing_layer(snapped, opts["selected_gus"])
        if ic is not None:
            layers.append(ic)

    # 위험군 골목길 & 열선 미설치 사각지대
    if opts.get("show_heating_gap_layer"):
        tier_nodes = _load_tier_risk_nodes()
        if not tier_nodes.empty:
            gap = build_heating_gap_layer(tier_nodes)
            if gap is not None:
                layers.append(gap)

    # 강조할 link_id 들 — 다중 또는 단일 모두 동일 흐름으로 처리
    highlight_lids = opts.get("highlight_lids") or []
    view_lat, view_lon, view_zoom = SEOUL_CENTER[0], SEOUL_CENTER[1], 10.4
    if highlight_lids:
        link_data = _load_road_link_data()
        # 매칭된 lid만 추림
        matched = [l for l in highlight_lids if link_data.get(str(l).strip())]
        if matched:
            view_lat, view_lon, view_zoom = compute_view_for_links(matched, link_data)
            hl = build_search_highlight_layer(matched, link_data)
            if hl is not None:
                layers.append(hl)
            if len(matched) < len(highlight_lids):
                miss = [
                    str(l) for l in highlight_lids
                    if not link_data.get(str(l).strip())
                ]
                st.warning(f"link_id 찾을 수 없음: {', '.join(miss)}")
        else:
            st.warning(
                f"link_id `{', '.join(map(str, highlight_lids))}` 를 찾을 수 없습니다."
            )

    view_state = pdk.ViewState(
        latitude=view_lat,
        longitude=view_lon,
        zoom=view_zoom,
        pitch=0,
        bearing=0,
    )

    # 모든 레이어가 description 필드를 직접 만들어 넣어둠 → 통합 tooltip 한 줄
    tooltip = {
        "html": "{description}",
        "style": {
            "backgroundColor": "rgba(20,20,20,0.88)",
            "color": "white",
            "fontSize": "12px",
            "padding": "8px 10px",
        },
    }

    # 위성영상이 켜져있으면 underlying basemap 제거 (이중 표시 방지)
    map_style = None if opts["show_satellite"] else "light"
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=map_style,
    )
    st.pydeck_chart(deck, use_container_width=True)


def render_kpis(df: pd.DataFrame, gu_metrics: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("대상 자치구", f"{gu_metrics['gu'].nunique()} 개")
    c2.metric("설치 도로 구간", f"{len(df):,} 건")
    c3.metric("총 연장", f"{int(df['length_m'].sum()):,} m")
    avg = df["length_m"].mean() if len(df) else 0
    c4.metric("구간 평균", f"{avg:,.0f} m")


def render_data_status() -> None:
    """사이드바 — 추가 데이터 업로드만 둔다. 데이터 상태는 본문 '데이터 점검' 표 하나로 본다."""
    with st.sidebar.expander("추가 데이터 업로드", expanded=False):
        extra = st.file_uploader(
            "zip 또는 단일 파일",
            type=["zip", "geojson", "csv", "json", "parquet"],
            accept_multiple_files=True,
            key="extra_upload",
            label_visibility="collapsed",
        )
        if extra and st.button("배치", key="extra_install"):
            try:
                placed_all: list[str] = []
                for f in extra:
                    tmp = save_uploaded_to(f, f.name, DATA_PATH)
                    placed = extract_archive_to(tmp, DATA_PATH)
                    placed_all.extend(placed)
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                st.success(f"{len(placed_all)}개 파일 배치: {', '.join(placed_all)}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"실패: {e}")


@st.cache_data(show_spinner=False)
def _flow_file_stats(content: bytes | None, name: str | None) -> dict | None:
    """유동인구 파일 요약 — 업로드 파일이 있으면 그것, 없으면 기본 flpop.csv."""
    try:
        if content is not None:
            d = _parse_foot_traffic(content, name)
            src = f"{name} (업로드)"
        else:
            p = DATA_PATH / "flpop.csv"
            if not p.exists():
                return None
            d = load_foot_traffic_csv(p)
            src = "flpop.csv"
    except Exception:
        return None
    return {"src": src, "rows": len(d), "links": set(d["link_id"].astype(str)),
            "dongs": d["dong"].nunique() if "dong" in d else 0,
            "gus": d["gu"].nunique() if "gu" in d else 0}


def render_data_check(df: pd.DataFrame, opts: dict) -> None:
    """데이터 점검 — 파일 존재·규모·결합 품질을 표 하나로."""
    rows = []

    def row(name, file, ok, size, quality, used):
        rows.append({"데이터": name, "파일": file, "상태": ok, "규모": size,
                     "결합·품질": quality, "쓰이는 곳": used})

    # 열선 + 도로명 매칭
    feats = _load_road_features()
    if feats:
        lines = match_roads(df, feats)
        keys = {(r["gu"], rn) for _, r in df.iterrows() for rn in r["road_names"] if rn}
        matched = {(m.gu, m.road_name) for m in lines}
        q = f"도로명 {len(matched):,}/{len(keys):,} 매칭 ({100*len(matched)/max(len(keys),1):.1f}%)"
    else:
        q = "도로 라인 없음 → 자치구 중심점으로 표시"
    row("열선 설치 현황", HEATING_CSV, "정상", f"{len(df):,}건 · {df['length_m'].sum()/1000:,.1f} km (필터 후)",
        q, "KPI, 열선 레이어, 자치구 집계")

    gj = _load_gu_geojson()
    row("자치구 경계", GU_GEOJSON, "정상" if gj else "없음",
        f"{len(gj.get('features', []))}개" if gj else "-", "-", "자치구 색상")

    row("도로 라인 (OSM)", ROAD_SHAPEFILE, "정상" if feats else "없음",
        f"{len(feats):,}개 라인" if feats else "-", "-", "배경 도로, 도로유형, 열선 매칭")

    ic = _load_icing_seoul()
    if not ic.empty:
        snapped = _load_icing_snapped()
        sr = 100 * len({x["seg_id"] for x in snapped if x["snap"] == "osm"}) / max(len(ic), 1)
        row("상습 결빙구간", ICING_CSV, "정상", f"서울 {len(ic):,}건 · {ic['length_km'].sum():,.1f} km",
            f"도로 라인 스냅 {sr:.1f}%", "결빙 레이어, 위험×결빙 표")
    else:
        row("상습 결빙구간", ICING_CSV, "없음", "-", "-", "결빙 레이어")

    link_data = _load_road_link_data()
    if link_data:
        n = len(link_data)
        n_s = sum(1 for v in link_data.values() if isinstance(v.get("slope_pct"), (int, float)))
        n_g = sum(1 for v in link_data.values() if v.get("road_type") == "골목길")
        row("도로링크 · 경사 (1:1,000)", "road_links.geojson", "정상",
            f"{n:,}개 링크 (골목길 {n_g:,})", f"경사 보유 {100*n_s/n:.1f}%",
            "골목길 위험 레이어, 기대 미끄러짐, 순위")
    else:
        row("도로링크 · 경사 (1:1,000)", "road_links.geojson", "없음", "-",
            "Drive zip 또는 사이드바 업로드로 추가", "골목길 위험 레이어, 기대 미끄러짐, 순위")

    LS = _load_link_risk_stats()
    if LS is not None:
        gm = LS[LS.road_type == "골목길"]
        row("서울 링크 통계", "seoul_links.parquet", "정상",
            f"{len(LS):,}개 링크 · 골목길 {gm.len.sum()/1000:,.0f} km",
            f"경사 보유 {100*LS.slope_pct.notna().mean():.1f}% · 열선 링크 {int(LS.heat.sum()):,}",
            "자치구 위험군·커버리지, 위험×결빙 표, 열선 최적배치 페이지")
    else:
        row("서울 링크 통계", "seoul_links.parquet", "없음", "-", "-", "자치구 위험군·커버리지")

    up = opts.get("ft_upload")
    fs = _flow_file_stats(up.getvalue() if up else None, up.name if up else None)
    if fs:
        hit = len(fs["links"] & set(link_data)) if link_data else 0
        row("유동인구 (B030)", fs["src"], "정상",
            f"{fs['rows']:,}행 · {len(fs['links']):,}개 링크 · {fs['gus']}개 구 {fs['dongs']}개 동",
            f"도로링크 매칭 {100*hit/max(len(fs['links']),1):.1f}%",
            "기대 미끄러짐 레이어·순위, 최적배치(2단계)")
    else:
        row("유동인구 (B030)", "flpop.csv", "없음", "-", "사이드바에서 업로드", "기대 미끄러짐")

    tab = pd.DataFrame(rows)
    n_ok = int((tab["상태"] == "정상").sum())
    with st.expander(f"데이터 점검 — {len(tab)}개 중 {n_ok}개 정상", expanded=n_ok < len(tab)):
        st.dataframe(tab, use_container_width=True, hide_index=True)
        st.caption("없는 데이터가 있어도 앱은 동작하며 해당 기능만 비활성된다. "
                   "추가는 사이드바 '추가 데이터 업로드' 또는 `data/` 폴더에 파일을 두면 된다.")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _render_urgency_ranking(opts: dict) -> None:
    """기대 미끄러짐 순위 분석 UI.

    유동인구 필터 결과(ft_link_agg) × P(MAA<θ) = 기대 미끄러짐 건수로 순위.
    """
    agg = st.session_state.get("ft_link_agg")
    if agg is None or agg.empty:
        st.info(
            "**유동인구 분석** 섹션에서 필터를 적용하면 자동으로 활성화됩니다. "
            "(현재 필터 결과가 비어있음)"
        )
        return
    link_data = _load_road_link_data()
    if not link_data:
        st.warning(
            "`road_links.geojson` 이 없어 경사도·길이 매칭 불가. "
            "사이드바 데이터 상태에서 추가 데이터 업로드해주세요."
        )
        return

    st.caption(
        "순위 = **기대 미끄러짐 건수 = P(MAA<θ) × 유동인구** (발표자료 7~8쪽). "
        "P(MAA<θ)는 WinterLab 빙판 보행 실측 58세션의 MAA 경험분포에서 경사 θ보다 "
        "한계각이 낮은 사람의 비율이다."
    )
    top_n = st.number_input(
        "상위 N개 표시", min_value=10, max_value=200, value=30, step=10
    )

    exclude_large = st.checkbox(
        "대형도로 후보 제외 (추천)",
        value=True,
        help="OSM 도로유형 분류(TBGIS↔OSM 공간매칭, scripts/classify_road_type.py)상 "
        "'대형도로'로 판정된 도로링크를 기대 미끄러짐 순위·후보군에서 제외합니다. "
        "열선 설치 후보는 좁은 골목길이어야 하므로 기본 활성화 권장.",
    )
    exclude_types = {"대형도로"} if exclude_large else set()

    # 도로유형 분포 (제외 적용 전, 필터된 agg 기준) — 제외 효과를 숫자로 확인
    _rt_counts = (
        agg["link_id"].astype(str)
        .map(lambda k: link_data.get(k, {}).get("road_type", "unknown"))
        .value_counts()
    )
    st.caption(
        f"도로유형 분포(현재 필터 내 {len(agg):,}건): "
        f"골목길 {int(_rt_counts.get('골목길', 0)):,} · "
        f"대형도로 {int(_rt_counts.get('대형도로', 0)):,}"
        f"{' → 후보에서 제외됨' if exclude_large else ' (제외 안 함)'} · "
        f"애매(tertiary) {int(_rt_counts.get('애매', 0)):,} · "
        f"미분류 {int(_rt_counts.get('unknown', 0)):,}"
    )

    # 점수 계산
    scored = compute_urgency_scores(agg, link_data, exclude_road_types=exclude_types)

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("대상 도로링크", f"{len(scored):,}")
    k2.metric("기대 미끄러짐 합", f"{scored['expected_slips'].sum():,.1f}건")
    k3.metric(
        "총 길이",
        f"{scored['length_m'].sum()/1000:,.2f} km",
    )
    top10_pct = scored.head(int(len(scored) * 0.1) or 1)
    k4.metric(
        "상위 10% 평균 길이",
        f"{top10_pct['length_m'].mean():.0f} m",
        help="기대 미끄러짐 상위 10% 도로링크의 평균 구간 길이",
    )

    # 표시용 컬럼명/순서 정리
    # ── 빠른 이동 multiselect ──
    st.markdown("**지도로 빠르게 이동 (여러 개 선택 가능)**")
    top_for_jump = scored.head(int(top_n)).reset_index(drop=True)

    def _fmt_jump_option(idx: int) -> str:
        r = top_for_jump.iloc[idx]
        slope = r["slope_pct"]
        slope_str = f"{slope:.1f}%" if isinstance(slope, (int, float)) else "미상"
        return (
            f"#{idx+1}  "
            f"link_id {r['link_id']}  ·  "
            f"{r['dong']}  ·  "
            f"경사 {slope_str}  ·  "
            f"길이 {int(r['length_m'])}m  ·  "
            f"기대 미끄러짐 {r['expected_slips']:.1f}건"
        )

    qb1, qb2, qb3 = st.columns([1, 1, 1])
    if qb1.button("TOP 5 한번에", use_container_width=True):
        st.session_state["urgency_jump_multi"] = list(range(min(5, len(top_for_jump))))
    if qb2.button("TOP 10 한번에", use_container_width=True):
        st.session_state["urgency_jump_multi"] = list(range(min(10, len(top_for_jump))))
    if qb3.button("선택 초기화", use_container_width=True):
        st.session_state["urgency_jump_multi"] = []

    jump_idxs = st.multiselect(
        "여기서 여러 도로를 선택하면 모두 형광 노랑으로 지도에 표시됩니다",
        options=list(range(len(top_for_jump))),
        format_func=_fmt_jump_option,
        key="urgency_jump_multi",
    )
    if jump_idxs:
        selected_lids = [
            str(top_for_jump.iloc[i]["link_id"]) for i in jump_idxs
        ]
        st.session_state["_jumped_lids"] = selected_lids
        # 총 길이·유동 합계 표시
        sub = top_for_jump.iloc[jump_idxs]
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("선택 개수", f"{len(sub)}개")
        sc2.metric("총 길이", f"{sub['length_m'].sum():,.0f} m")
        sc3.metric("유동인구 합", f"{sub['flow_sum'].sum():,.0f}")
    else:
        st.session_state["_jumped_lids"] = []

    st.caption(
        "사이드바 **도로링크 찾기** 에 link_id 를 직접 입력해도 동일하게 작동합니다. "
        "(다중 선택이 비어있을 때만 사이드바 입력값 사용)"
    )
    st.divider()

    show = scored.head(int(top_n)).copy()
    show["slope_pct"] = show["slope_pct"].apply(
        lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "-"
    )
    show["length_m"] = show["length_m"].round(0).astype(int)
    show["flow_sum"] = show["flow_sum"].round(1)
    show_disp = show[
        ["link_id", "gu", "dong", "road_type", "slope_pct", "risk",
         "length_m", "flow_sum", "expected_slips"]
    ].rename(columns={
        "link_id": "도로링크ID",
        "gu": "자치구",
        "dong": "행정동",
        "road_type": "도로유형",
        "slope_pct": "경사도 %",
        "risk": "미끄러짐 확률 P(MAA<θ)",
        "length_m": "길이(m)",
        "flow_sum": "유동인구",
        "expected_slips": "기대 미끄러짐(건)",
    })
    show_disp.insert(0, "순위", range(1, len(show_disp) + 1))
    st.dataframe(show_disp, use_container_width=True, hide_index=True)

    # 자치구·행정동별 시급도 평균 집계
    st.markdown("**행정동별 기대 미끄러짐 집계**")
    gby = (
        scored.groupby(["gu", "dong"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "구간 수": len(g),
                "총 길이(km)": round(g["length_m"].sum() / 1000, 2),
                "기대 미끄러짐 합(건)": round(g["expected_slips"].sum(), 1),
                "평균 미끄러짐 확률": round(g["risk"].mean(), 3),
                "위험군(≥5.56%) 구간 수": int((g["slope_pct"] >= STEEP_PCT).sum()),
            }),
            include_groups=False,
        )
        .reset_index(drop=False)
        .sort_values("기대 미끄러짐 합(건)", ascending=False)
    )
    if "level_0" in gby.columns:
        gby = gby.drop(columns=["level_0"])
    if "level_1" in gby.columns:
        gby = gby.drop(columns=["level_1"])
    gby = gby.rename(columns={"gu": "자치구", "dong": "행정동"})
    st.dataframe(gby, use_container_width=True, hide_index=True)

    # CSV 다운로드
    csv = scored[
        ["link_id", "gu", "dong", "road_type", "slope_pct", "risk",
         "length_m", "flow_sum", "expected_slips"]
    ].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "전체 기대 미끄러짐 순위 CSV 다운로드",
        data=csv,
        file_name="expected_slips_ranking.csv",
        mime="text/csv",
    )


@st.cache_data(show_spinner=False)
def _parse_foot_traffic(content_bytes: bytes, file_name: str) -> pd.DataFrame:
    """업로드된 CSV bytes 를 DataFrame 으로. 캐시되어 동일 파일은 재파싱 안 함."""
    import io
    return load_foot_traffic_csv(io.BytesIO(content_bytes))


# 차원 라벨 (UI 표시용)
_DIM_OPTIONS = {
    "gu": "자치구",
    "dong": "행정동",
    "day": "요일",
    "age": "연령대",
    "time": "시간대",
}


def _render_foot_traffic_section(opts: dict) -> None:
    """유동인구 CSV 필터 + 피봇 분석 UI.

    우선순위:
        1. 사이드바 업로드한 파일 (사용자 분석용)
        2. data/flpop.csv (기본 commit된 샘플)
    """
    upload = opts.get("ft_upload")
    default_path = DATA_PATH / "flpop.csv"

    if upload is not None:
        try:
            df = _parse_foot_traffic(upload.getvalue(), upload.name)
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")
            return
        source_label = f"업로드: `{upload.name}`"
    elif default_path.exists():
        try:
            with default_path.open("rb") as f:
                df = _parse_foot_traffic(f.read(), default_path.name)
        except Exception as e:
            st.error(f"기본 데이터 로드 실패: {e}")
            return
        source_label = f"기본 데이터: `{default_path.name}`"
    else:
        st.info(
            "유동인구 데이터가 없습니다. 사이드바 **유동인구 데이터** 에서 "
            "CSV 파일을 업로드하거나 `data/flpop.csv` 파일을 추가해주세요.\n\n"
            "기대 컬럼: `시군구명, 행정동명, 요일코드, 연령대코드, 시간대코드, 유동인구_수`"
        )
        return

    st.caption(
        f"{source_label} · {len(df):,} 행 · 자치구 {df['gu'].nunique()}개 · "
        f"행정동 {df['dong'].nunique()}개"
    )

    # ── 빠른 프리셋 ──
    preset_cols = st.columns([1, 1, 1, 2])
    preset_dawn = preset_cols[0].button(
        "출근시간(7-9시)",
        help="평일 06-09 시간대만 필터 (결빙 취약 시간)",
        use_container_width=True,
    )
    preset_elderly = preset_cols[1].button(
        "60대 이상 평일",
        help="평일 + 60대 이상 (낙상 위험층)",
        use_container_width=True,
    )
    preset_all = preset_cols[2].button(
        "필터 초기화",
        use_container_width=True,
    )
    if preset_dawn:
        st.session_state["ft_day_codes"] = [1]
        st.session_state["ft_time_codes"] = [1]
    if preset_elderly:
        st.session_state["ft_day_codes"] = [1]
        st.session_state["ft_age_codes"] = [60]
    if preset_all:
        for k in ("ft_gus", "ft_dongs", "ft_day_codes", "ft_age_codes", "ft_time_codes"):
            st.session_state.pop(k, None)

    # ── 필터 ──
    st.markdown("**필터**")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        gus = st.multiselect(
            "자치구",
            options=sorted(df["gu"].dropna().unique().tolist()),
            default=st.session_state.get("ft_gus", []),
            key="ft_gus",
        )
        dong_pool = df[df["gu"].isin(gus)]["dong"] if gus else df["dong"]
        dongs = st.multiselect(
            "행정동",
            options=sorted(dong_pool.dropna().unique().tolist()),
            default=st.session_state.get("ft_dongs", []),
            key="ft_dongs",
        )
    with fcol2:
        day_codes = st.multiselect(
            "요일",
            options=list(DAY_LABELS.keys()),
            default=st.session_state.get("ft_day_codes", [1]),
            format_func=lambda c: f"{c} · {DAY_LABELS[c]}",
            key="ft_day_codes",
        )
        time_codes = st.multiselect(
            "시간대",
            options=list(DEFAULT_TIME_LABELS.keys()),
            default=st.session_state.get("ft_time_codes", [1, 2]),
            format_func=lambda c: f"{c} · {DEFAULT_TIME_LABELS[c]}",
            key="ft_time_codes",
        )
        age_codes = st.multiselect(
            "연령대",
            options=list(AGE_LABELS.keys()),
            default=st.session_state.get("ft_age_codes", []),
            format_func=lambda c: f"{c} · {AGE_LABELS[c]}",
            key="ft_age_codes",
        )

    # 연령 미선택 시 agrde_cd=0('전체') 행만 쓴다. 0 은 10~60 행의 정확한 합이라
    # 전부 합치면 약 2배로 부풀려진다(구 KNOWN BUG). 발표 모형과 같은 처리.
    if not age_codes and "age_code" in df.columns and (df["age_code"] == 0).any():
        age_codes = [0]
    filt = filter_foot_traffic(
        df,
        gus=gus or None,
        dongs=dongs or None,
        day_codes=day_codes or None,
        age_codes=age_codes or None,
        time_codes=time_codes or None,
    )

    # 필터 결과를 session_state 에 저장 → render_map 에서 도로링크 시각화에 사용
    if not filt.empty:
        st.session_state["ft_link_agg"] = aggregate_by_link(filt)
    else:
        st.session_state["ft_link_agg"] = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("필터 적용 행", f"{len(filt):,}")
    c2.metric("유동인구 합계", f"{filt['flow'].sum():,.0f}")
    c3.metric(
        "1행 평균",
        f"{filt['flow'].mean():.2f}" if len(filt) else "-",
    )
    link_n = (
        st.session_state["ft_link_agg"]["link_id"].nunique()
        if st.session_state.get("ft_link_agg") is not None
        else 0
    )
    c4.metric("매칭 도로링크", f"{link_n:,}")
    st.caption(
        "사이드바 **기대 미끄러짐 = P(MAA<θ) × 유동인구** 레이어를 켜면 위 필터 결과가 "
        "도로 색·두께로 지도에 표시됩니다."
    )
    st.divider()

    # ── 피봇 ──
    st.markdown("**피봇 / 집계**")
    pcol1, pcol2, pcol3 = st.columns(3)
    row_dim = pcol1.selectbox(
        "행 차원",
        options=list(_DIM_OPTIONS.keys()),
        index=1,  # 기본 행정동
        format_func=lambda x: _DIM_OPTIONS[x],
        key="ft_row",
    )
    col_dim = pcol2.selectbox(
        "열 차원 (선택)",
        options=["(없음)"] + [x for x in _DIM_OPTIONS.keys() if x != row_dim],
        index=0,
        format_func=lambda x: x if x == "(없음)" else _DIM_OPTIONS[x],
        key="ft_col",
    )
    agg = pcol3.selectbox(
        "집계 방식",
        options=["sum", "mean", "count"],
        format_func=lambda x: {"sum": "합계", "mean": "평균", "count": "건수"}[x],
        key="ft_agg",
    )

    if filt.empty:
        st.warning("필터 적용 결과가 비어있습니다.")
        return

    pivot = pivot_foot_traffic(filt, row=row_dim, col=col_dim, agg=agg)
    st.dataframe(pivot, use_container_width=True, hide_index=True)

    # ── 다운로드 ──
    csv_bytes = pivot.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 피봇 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"foot_traffic_pivot_{row_dim}.csv",
        mime="text/csv",
    )


def _render_data_bootstrap(missing: list[str]) -> None:
    """필수 데이터가 없을 때 다운로드/업로드 UI를 보여주고 st.stop().

    두 가지 경로:
        A. Google Drive 다운로드 (온라인)
        B. 로컬 zip/geojson 업로드 (폐쇄망)
    """
    st.title("초기 데이터 준비")
    st.error(
        f"**앱 실행에 필요한 필수 데이터 {len(missing)}개**가 없습니다. "
        "아래 두 가지 방법 중 하나로 데이터를 채워주세요."
    )

    # 누락된 파일 상세 안내 표
    with st.container(border=True):
        st.markdown("##### 누락된 파일")
        rows = []
        for name in missing:
            info = FILE_INFO.get(name, {})
            rows.append({
                "파일": name,
                "크기": info.get("size", "-"),
                "용도": info.get("desc", "-"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "데이터를 한 zip 파일에 담아서 받으시면 자동으로 풀려서 배치됩니다."
        )

    tab_dl, tab_up = st.tabs(["Google Drive 다운로드", "로컬 파일 업로드"])

    # ── 탭 A: Drive 다운로드 ──
    with tab_dl:
        st.markdown(
            "**온라인 환경에서 사용**\n\n"
            "데이터를 Google Drive에 공개로 올려둔 zip 파일을 자동으로 받아옵니다."
        )
        default_url = ""
        try:
            default_url = st.secrets.get("DATA_DRIVE_URL", DEFAULT_DRIVE_URL)
        except Exception:
            default_url = DEFAULT_DRIVE_URL

        url = st.text_input(
            "Google Drive 링크 또는 file ID",
            value=default_url,
            placeholder="https://drive.google.com/file/d/FILE_ID/view",
            help="공개(링크 있는 모든 사용자) 권한이어야 합니다.",
            key="dl_url",
        )

        if st.button("다운로드 시작", key="dl_btn", type="primary"):
            if not url.strip():
                st.error("URL을 입력해주세요.")
                st.stop()
            bar = st.progress(0.0, text="다운로드 중...")
            try:
                tmp_zip = DATA_PATH / "_bootstrap_download.zip"
                download_from_drive(
                    url,
                    tmp_zip,
                    progress=lambda d, t: bar.progress(
                        min(d / max(t, 1), 1.0),
                        text=f"다운로드 중... {d/1024/1024:.1f} MB",
                    ),
                )
                placed = extract_archive_to(tmp_zip, DATA_PATH)
                try:
                    tmp_zip.unlink()
                except Exception:
                    pass
                bar.progress(1.0, text="완료")
                st.success(
                    f"{len(placed)}개 파일 배치 완료: `{', '.join(placed)}`"
                )
                still_missing = missing_files(DATA_PATH)
                if still_missing:
                    st.error(
                        f"여전히 누락: `{', '.join(still_missing)}`. "
                        "zip 내용에 필수 파일이 포함되어 있는지 확인해주세요."
                    )
                else:
                    st.balloons()
                    st.button("앱 시작 →", on_click=st.rerun)
            except Exception as e:
                st.error(f"다운로드 실패: {e}")

    # ── 탭 B: 로컬 업로드 ──
    with tab_up:
        st.markdown(
            "**폐쇄망(단독망) 환경에서 사용**\n\n"
            "USB 등으로 받아온 zip 또는 geojson 파일을 직접 업로드합니다. "
            "zip이면 자동으로 압축 해제됩니다."
        )
        uploaded = st.file_uploader(
            "zip · geojson · csv 파일을 끌어다 놓거나 선택",
            type=["zip", "geojson", "json", "csv"],
            accept_multiple_files=True,
            key="up_files",
        )
        if uploaded and st.button("업로드한 파일로 설치", key="up_btn", type="primary"):
            placed_all: list[str] = []
            try:
                for f in uploaded:
                    tmp = save_uploaded_to(f, f.name, DATA_PATH)
                    placed = extract_archive_to(tmp, DATA_PATH)
                    placed_all.extend(placed)
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
                st.success(
                    f"{len(placed_all)}개 파일 배치 완료: "
                    f"`{', '.join(placed_all)}`"
                )
                still_missing = missing_files(DATA_PATH)
                if still_missing:
                    st.error(
                        f"여전히 누락: `{', '.join(still_missing)}`. "
                        "필요한 파일을 추가로 업로드해주세요."
                    )
                else:
                    st.balloons()
                    st.button("앱 시작 →", on_click=st.rerun, key="start_after_up")
            except Exception as e:
                st.error(f"설치 실패: {e}")

    with st.expander("필요한 파일 목록 / 안내", expanded=False):
        st.markdown(
            "앱 실행에 필요한 파일:\n"
            + "\n".join(f"- `{name}`" for name in REQUIRED_FILES)
            + "\n\n"
            "협업자에게 받은 zip이면 그대로 업로드하시면 됩니다 (위 4개 파일이 "
            "들어있어야 함). zip 내부 폴더 구조는 무시하고 파일 이름만 인식합니다."
        )

    st.stop()


def _gu_risk_table(LS: pd.DataFrame, gus: list[str] | None = None) -> pd.DataFrame:
    """자치구별 골목길 위험군 현황 (발표자료 6쪽 표와 같은 축)."""
    g = LS[LS.road_type == "골목길"]
    if gus:
        g = g[g.gu.isin(gus)]
    t = g.assign(s_len=g.len.where(g.steep, 0.0),
                 sh_len=g.len.where(g.steep & g.heat, 0.0))
    out = t.groupby("gu").agg(골목길_km=("len", "sum"), 위험군_km=("s_len", "sum"),
                              위험군_열선_km=("sh_len", "sum"),
                              평균_P_MAA=("risk", "mean")).reset_index()
    out[["골목길_km", "위험군_km", "위험군_열선_km"]] /= 1000
    out["위험군비율%"] = 100 * out.위험군_km / out.골목길_km
    out["커버리지%"] = 100 * out.위험군_열선_km / out.위험군_km.replace(0, float("nan"))
    out = out.rename(columns={"gu": "자치구"}).round(
        {"골목길_km": 1, "위험군_km": 1, "위험군_열선_km": 2, "평균_P_MAA": 3,
         "위험군비율%": 1, "커버리지%": 1})
    return out.sort_values("위험군_km", ascending=False)


def _attach_gu_risk(gu_metrics: pd.DataFrame) -> pd.DataFrame:
    """자치구 지표에 위험군 골목길 연장·커버리지를 붙인다 (지도 색상 기준용)."""
    LS = _load_link_risk_stats()
    if LS is None:
        return gu_metrics.assign(steep_km=0.0, steep_cov_pct=0.0)
    t = _gu_risk_table(LS)[["자치구", "위험군_km", "커버리지%"]].rename(
        columns={"자치구": "gu", "위험군_km": "steep_km", "커버리지%": "steep_cov_pct"})
    return gu_metrics.merge(t, on="gu", how="left").fillna({"steep_km": 0.0, "steep_cov_pct": 0.0})


def main() -> None:
    # 초기 데이터 부트스트랩 — 필수 파일이 없으면 다운로드/업로드 UI 표시
    missing = missing_files(DATA_PATH)
    if missing:
        _render_data_bootstrap(missing)
        return  # st.stop() 후에는 도달 안 함

    st.title("서울시 도로 열선 설치 현황")
    st.caption(
        "2026년 자치구별 도로열선 설치현황 데이터 기반. "
        "레이어 토글로 자치구·도로를 비교하고, 아래 분석 섹션에서 표를 확인하세요."
    )

    # 데이터 로드
    try:
        df_raw = _load_csv()
    except FileNotFoundError as e:
        st.error(f"CSV를 찾지 못했습니다: {e}")
        st.stop()

    # 사이드바
    opts = render_sidebar(df_raw)
    render_data_status()

    # 필터 적용
    df = df_raw[df_raw["gu"].isin(opts["selected_gus"])]
    if opts["year_range"][0] is not None:
        y_min, y_max = opts["year_range"]
        df = df[df["year"].between(y_min, y_max)]

    # 집계
    gu_metrics = _attach_gu_risk(compute_risk_score(aggregate_by_gu(df)))
    road_metrics = aggregate_by_road(df)

    # KPI
    render_kpis(df, gu_metrics)
    render_data_check(df, opts)
    st.divider()

    # 지도 — 유동인구 도로링크 시각화용 집계 결과를 opts에 합쳐 전달
    opts["ft_link_agg"] = st.session_state.get("ft_link_agg")

    # 다중 선택 결과(시급도 multiselect) 또는 사이드바 단일 검색 통합
    jumped_lids = st.session_state.get("_jumped_lids", [])
    if jumped_lids:
        opts["highlight_lids"] = jumped_lids
    elif opts.get("search_lid"):
        opts["highlight_lids"] = [opts["search_lid"]]
    else:
        opts["highlight_lids"] = []

    st.subheader("지도")
    render_map(df, opts, gu_metrics)

    st.divider()
    render_delta_lab()

    # 분석 섹션 (접고 펼치기)
    st.subheader("분석")

    with st.expander("자치구별 집계", expanded=True):
        st.caption(
            f"색상 기준 지표: **{opts['metric_label']}**. 표는 도로 수·총연장 기준으로 정렬."
        )
        st.dataframe(
            gu_metrics.rename(
                columns={
                    "gu": "자치구",
                    "road_count": "도로 수",
                    "total_length_m": "총 연장(m)",
                    "avg_length_m": "구간 평균(m)",
                    "steep_km": "위험군 골목길(km)",
                    "steep_cov_pct": "위험군 열선 커버리지(%)",
                }
            ).drop(columns=["risk_score"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("골목길 경사위험 (자치구별, 1:1,000)", expanded=False):
        LS = _load_link_risk_stats()
        if LS is None:
            st.info("`data/seoul_links.parquet` 이 없어 계산할 수 없습니다.")
        else:
            g = LS[(LS.road_type == "골목길") & LS.gu.isin(opts["selected_gus"])]
            gs = g[g.steep]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("골목길 연장", f"{g.len.sum()/1000:,.0f} km")
            c2.metric("위험군(≥5.56%) 연장", f"{gs.len.sum()/1000:,.0f} km",
                      f"{100*gs.len.sum()/max(g.len.sum(),1):.1f}%", delta_color="off")
            c3.metric("위험군 열선 커버리지", f"{100*gs[gs.heat].len.sum()/max(gs.len.sum(),1):.1f}%")
            c4.metric("골목길 평균 P(MAA<θ)", f"{g.risk.mean():.3f}")
            st.dataframe(_gu_risk_table(LS, opts["selected_gus"]),
                         use_container_width=True, hide_index=True)
            st.caption(
                "위험군 = 경사 ≥5.56%(장애인등편의법 접근로 기울기 1/18). P(MAA<θ) = "
                "WinterLab 빙판 보행 실측 MAA 경험분포로 본 미끄러짐 확률. 경사는 "
                "수치지형도 1:1,000 등고선 기반. 커버리지는 링크 매칭된 열선 기준이라 "
                "자치구 신고 연장(대형도로 포함)과 다를 수 있다."
            )

    with st.expander("미끄러짐 위험 × 상습 결빙구간", expanded=False):
        LS = _load_link_risk_stats()
        if LS is None:
            st.info("`data/seoul_links.parquet` 이 없어 계산할 수 없습니다.")
        else:
            g = LS[LS.gu.isin(opts["selected_gus"])]
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
            labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
            tab = (g.assign(구간=pd.cut(g.risk, bins, right=False, labels=labels))
                    .groupby(["구간", "icing"], observed=False).len.sum().unstack(fill_value=0)
                    .rename(columns={False: "일반 km", True: "결빙구간 km"}) / 1000).round(1)
            tab = tab.reset_index().rename(columns={"구간": "P(MAA<θ)"})
            st.dataframe(tab, use_container_width=True, hide_index=True)
            ic = g[g.icing]
            gm = g[g.road_type == "골목길"]
            c1, c2, c3 = st.columns(3)
            c1.metric("결빙구간 링크 평균 P(MAA<θ)", f"{ic.risk.mean():.3f}")
            c2.metric("결빙구간 중 골목길 비율", f"{100*(ic.road_type=='골목길').mean():.1f}%")
            c3.metric("위험군 골목길 중 결빙 지정", f"{100*gm[gm.steep].icing.mean():.1f}%")
            st.caption("발표자료 3·6쪽: 결빙취약구간 지정은 대형도로에 편중되고 경사 골목을 잘 잡지 못한다.")

    with st.expander("유동인구 분석 (피봇 테이블)", expanded=False):
        _render_foot_traffic_section(opts)

    with st.expander("기대 미끄러짐 순위 (P(MAA<θ) × 유동인구)", expanded=False):
        _render_urgency_ranking(opts)

    with st.expander("열선 설치 원자료 (도로별 집계 · 원본)", expanded=False):
        t_road, t_raw = st.tabs(["도로명별 집계", "원본 (필터 적용 후)"])
        with t_road:
            st.caption(
                f"같은 도로명의 여러 설치 구간을 합산. "
                f"도로명을 추출하지 못한 {(df['primary_road']=='').sum()}건은 제외."
            )
            st.dataframe(
                road_metrics.rename(columns={
                    "gu": "자치구", "primary_road": "도로명", "segment_count": "구간 수",
                    "total_length_m": "총 연장(m)", "years": "설치연도"}),
                use_container_width=True, hide_index=True,
            )
        with t_raw:
            st.dataframe(
                df.rename(columns={
                    "gu": "자치구", "year": "설치연도", "section": "설치구간",
                    "length_m": "연장(m)", "primary_road": "대표 도로명",
                    "road_names": "추출 도로명"})[
                    ["자치구", "설치연도", "설치구간", "연장(m)", "대표 도로명", "추출 도로명"]],
                use_container_width=True, hide_index=True,
            )



if __name__ == "__main__":
    main()
