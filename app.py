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
from utils.data_loader import (
    aggregate_by_gu,
    aggregate_by_road,
    compute_risk_score,
    load_heating_csv,
    load_icing_csv,
)
from utils.geo import (
    SLOPE_COLORS,
    SLOPE_LABELS,
    SLOPE_LEVELS,
    build_gu_polygon_index,
    fallback_points,
    filter_icing_to_seoul,
    filter_roads_by_slope,
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
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / DATA_DIR


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
        }
    return out


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
            f"구간 평균: {_fmt_int(m.get('avg_length_m'))} m"
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


def build_slope_layer(
    features_by_gu: dict[str, list[dict]],
    selected_gus: list[str],
    selected_levels: list[str],
) -> pdk.Layer | None:
    """경사도 등급별 색상 도로 레이어.

    각 도로의 slope_level 속성으로 색상 결정. SLOPE_COLORS 매핑 사용.
    """
    feats = filter_roads_by_slope(features_by_gu, selected_gus, selected_levels)
    if not feats:
        return None

    paths = []
    for feat in feats:
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue
        lvl = props.get("slope_level", "unknown")
        slope = props.get("slope_pct")
        slope_text = (
            f"{slope:.1f} %" if isinstance(slope, (int, float)) else "미상"
        )
        es = props.get("elev_start_m")
        ee = props.get("elev_end_m")
        elev_text = (
            f"{es} m → {ee} m"
            if es is not None and ee is not None
            else "표고 매칭 실패"
        )
        paths.append(
            {
                "path": coords,
                "color": SLOPE_COLORS.get(lvl, SLOPE_COLORS["unknown"]),
                "description": (
                    f"<b>{props.get('road_name','(이름 없음)')}</b>"
                    f"<br/>자치구: {props.get('gu','')}"
                    f"<br/>경사도: {slope_text} ({SLOPE_LABELS.get(lvl, lvl)})"
                    f"<br/>표고: {elev_text}"
                    f"<br/>거리: {props.get('dist_m','-')} m"
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


# 경사도 등급별 시급도 점수 (0~100)
_SLOPE_URGENCY_SCORE = {
    "10_plus": 100.0,
    "6_10":    70.0,
    "3_6":     40.0,
    "0_3":     10.0,
    "unknown": 20.0,
}


def compute_urgency_scores(
    agg_df: pd.DataFrame,
    link_data: dict[str, dict],
    method: str = "geometric",
    slope_weight: float = 0.5,
) -> pd.DataFrame:
    """도로링크별 시급도 점수 (0~100) + 길이/경사 컬럼 추가.

    method:
        'geometric'     : √(경사 × 유동)  — 둘 다 높을 때만 큼
        'weighted_mean' : α·경사 + (1-α)·유동
        'product'       : 경사 × 유동 / 100  — 극단적
    slope_weight: weighted_mean 일 때 경사 비중 (0~1)
    """
    if agg_df.empty:
        return agg_df.assign(
            slope_level=[], slope_pct=[], length_m=[],
            slope_score=[], flow_score=[], urgency=[],
        )

    out = agg_df.copy()
    out["link_id"] = out["link_id"].astype(str)
    out["slope_level"] = out["link_id"].map(
        lambda k: link_data.get(k, {}).get("slope_level", "unknown")
    )
    out["slope_pct"] = out["link_id"].map(
        lambda k: link_data.get(k, {}).get("slope_pct")
    )
    out["length_m"] = out["link_id"].map(
        lambda k: link_data.get(k, {}).get("length_m", 0.0)
    )
    out["slope_score"] = out["slope_level"].map(_SLOPE_URGENCY_SCORE).fillna(20.0)

    # 유동인구 백분위(0~100) — 같은 필터 내 상대 점수
    if len(out) > 1:
        out["flow_score"] = out["flow_sum"].rank(pct=True) * 100.0
    else:
        out["flow_score"] = 100.0

    if method == "geometric":
        out["urgency"] = (out["slope_score"] * out["flow_score"]) ** 0.5
    elif method == "product":
        out["urgency"] = out["slope_score"] * out["flow_score"] / 100.0
    else:  # weighted_mean
        w = max(0.0, min(1.0, slope_weight))
        out["urgency"] = out["slope_score"] * w + out["flow_score"] * (1.0 - w)

    out["urgency"] = out["urgency"].round(1)
    out["slope_score"] = out["slope_score"].round(0).astype(int)
    out["flow_score"] = out["flow_score"].round(1)
    return out.sort_values("urgency", ascending=False).reset_index(drop=True)


# 경사도 등급별 base RGB (알파는 유동인구로 결정)
_FT_SLOPE_BASE_RGB = {
    "10_plus": [220, 30, 30],   # 빨강 — 매우 위험
    "6_10":    [240, 130, 40],  # 주황 — 위험
    "3_6":     [255, 210, 80],  # 노랑 — 보통
    "0_3":     [150, 150, 160], # 회색 — 평탄
    "unknown": [140, 140, 140], # 회색
}
# 경사도 등급별 두께 가중치 (위험 등급일수록 굵게)
_FT_SLOPE_WIDTH = {
    "10_plus": 5.0,
    "6_10":    4.0,
    "3_6":     2.8,
    "0_3":     2.0,
    "unknown": 2.0,
}


def build_foot_traffic_layer(
    agg_df: pd.DataFrame, link_data: dict[str, dict]
) -> pdk.Layer | None:
    """유동인구 × 경사도 조합 PathLayer.

    색상 (hue) = 경사도 등급 (≥10% 빨강, 6-10% 주황, 3-6% 노랑, 그외 회색)
    투명도 (alpha) = 유동인구 분위수 (50 → 255)
    두께 = 경사도 등급별 base × 유동인구 분위수 가중치

    이렇게 하면 "빨강+진함" = 가파른 곳 + 유동인구 많음 = 최우선 열선 설치 대상.
    """
    if agg_df.empty or not link_data:
        return None

    vals = agg_df["flow_sum"].astype(float)
    if vals.max() <= 0:
        return None

    # 유동인구 분위수 → alpha (20%~100% 범위)
    qs = vals.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    q25 = qs.get(0.25, 0)
    q50 = qs.get(0.5, 0)
    q75 = qs.get(0.75, 0)
    q90 = qs.get(0.9, 0)

    def alpha_for(v: float) -> int:
        # 20% → 100%
        if v <= q25: return 50    # 20%
        if v <= q50: return 100   # 40%
        if v <= q75: return 160   # 63%
        if v <= q90: return 210   # 82%
        return 255                # 100%

    def width_for(slope_lv: str, v: float) -> float:
        base = _FT_SLOPE_WIDTH.get(slope_lv, 2.0)
        # 유동인구 많을수록 약간 더 굵게 (+0~+1.5)
        bonus = 0.0
        if v > q75: bonus = 0.7
        if v > q90: bonus = 1.5
        return base + bonus

    paths = []
    skipped_no_match = 0
    for _, row in agg_df.iterrows():
        lid = str(row["link_id"])
        link = link_data.get(lid)
        if not link:
            skipped_no_match += 1
            continue
        v = float(row["flow_sum"])
        slope_lv = link.get("slope_level", "unknown") or "unknown"
        rgb = _FT_SLOPE_BASE_RGB.get(slope_lv, _FT_SLOPE_BASE_RGB["unknown"])
        alpha = alpha_for(v)
        slope_pct = link.get("slope_pct")
        slope_text = (
            f"{slope_pct:.1f}%" if isinstance(slope_pct, (int, float)) else "미상"
        )
        paths.append(
            {
                "path": link["coords"],
                "color": [rgb[0], rgb[1], rgb[2], alpha],
                "width": width_for(slope_lv, v),
                "description": (
                    f"<b>유동인구: {v:,.1f}</b>"
                    f"<br/>경사도: {slope_text} ({slope_lv})"
                    f"<br/>link_id: {lid}"
                    f"<br/>행정동: {row.get('dong','')} ({row.get('gu','')})"
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


# ──────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.title("⚙️ 필터 & 레이어")

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
    show_foot_traffic_layer = st.sidebar.checkbox(
        "유동인구 × 경사도 (TBGIS 도로링크)",
        value=False,
        help=(
            "유동인구 CSV 업로드 시 활성. 도로 **색상=경사도** "
            "(≥10% 빨강 / 6-10% 주황 / 3-6% 노랑 / 그외 회색), "
            "**투명도=유동인구 분위수** (적음 옅음 → 많음 진함). "
            "빨강+진함 = 가파르고 유동인구 많은 최우선 열선 설치 대상."
        ),
    )
    show_slope_layer = st.sidebar.checkbox(
        "경사도 도로 (등급별 색상)",
        value=False,
        help="도로 시작/끝점 표고로 산출한 경사도를 등급별 색상으로 표시. "
        "전체 도로 배경 레이어 대신 등급 색상 레이어가 표시됩니다.",
    )

    st.sidebar.subheader("경사도 필터")
    slope_options = list(SLOPE_LEVELS)
    # 기본값: 6% 이상만 켜기 (사용자가 결빙 위험과 연관해 보고싶어 함)
    default_levels = ["6_10", "10_plus"]
    selected_levels = st.sidebar.multiselect(
        "경사도 등급",
        options=slope_options,
        default=default_levels,
        format_func=lambda x: SLOPE_LABELS[x],
        help="6% 이상은 결빙 시 위험이 급증하는 구간으로 알려져 있음.",
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
            ("risk_score", "위험도 점수(임시)"),
        ],
        format_func=lambda x: x[1],
        horizontal=False,
    )

    # ── 유동인구 CSV 업로드 (선택) ──
    st.sidebar.subheader("🚶 유동인구 데이터")
    default_flpop_exists = (DATA_PATH / "flpop.csv").exists()
    if default_flpop_exists:
        st.sidebar.caption("✅ 기본 `flpop.csv` 자동 로드됨. 다른 파일로 분석하려면 업로드:")
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
        "show_gu_layer": show_gu_layer,
        "show_road_bg": show_road_bg,
        "show_road_layer": show_road_layer,
        "show_icing_layer": show_icing_layer,
        "show_slope_layer": show_slope_layer,
        "selected_gus": selected_gus,
        "selected_slope_levels": selected_levels,
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

    # 경사도 등급별 색상 도로
    if opts["show_slope_layer"]:
        road_features = _load_road_features()
        if road_features:
            by_gu = _split_roads_by_gu(road_features)
            sl = build_slope_layer(
                by_gu, opts["selected_gus"], opts["selected_slope_levels"]
            )
            if sl is not None:
                layers.append(sl)

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

    # 유동인구 × 경사도 도로링크 레이어
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

    view_state = pdk.ViewState(
        latitude=SEOUL_CENTER[0],
        longitude=SEOUL_CENTER[1],
        zoom=10.4,
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
    """데이터 파일 상태 표시."""
    with st.sidebar.expander("📁 데이터 상태", expanded=False):
        gj_path = DATA_PATH / GU_GEOJSON
        rd_path = DATA_PATH / ROAD_SHAPEFILE
        gj_ok = gj_path.exists()
        rd_ok = rd_path.exists()

        st.write(f"- 열선 CSV: ✅ `{HEATING_CSV}`")
        st.write(
            f"- 자치구 경계: {'✅' if gj_ok else '⚠️ 없음 (fallback)'} `{GU_GEOJSON}`"
        )
        if rd_ok:
            mb = rd_path.stat().st_size / 1024 / 1024
            st.write(f"- 도로 라인: ✅ `{ROAD_SHAPEFILE}` ({mb:.1f} MB)")
        else:
            st.write(f"- 도로 라인: ⚠️ 없음 (fallback) `{ROAD_SHAPEFILE}`")

        icing_path = DATA_PATH / ICING_CSV
        if icing_path.exists():
            kb = icing_path.stat().st_size / 1024
            st.write(f"- 결빙구간: ✅ `{ICING_CSV}` ({kb:.0f} KB)")
        else:
            st.write(f"- 결빙구간: ⚠️ 없음 `{ICING_CSV}`")

        # 슬로프 처리 여부 (도로 GeoJSON의 첫 feature에 slope_pct 있는지)
        if rd_ok:
            try:
                rd_feats = _load_road_features() or []
                has_slope = any(
                    f.get("properties", {}).get("slope_pct") is not None
                    for f in rd_feats[:200]
                )
            except Exception:
                has_slope = False
            if has_slope:
                st.write("- 경사도: ✅ 도로 GeoJSON에 포함")
            else:
                st.write(
                    "- 경사도: ⚠️ 미처리 — `python3 scripts/preprocess_slope.py` 실행 필요"
                )

        # 도로링크 (선택 데이터)
        rl_path = DATA_PATH / "road_links.geojson"
        if rl_path.exists():
            mb = rl_path.stat().st_size / 1024 / 1024
            st.write(f"- 도로링크: ✅ `road_links.geojson` ({mb:.1f} MB)")
        else:
            st.write(
                "- 도로링크: ⚠️ 없음 (유동인구 도로 시각화 비활성). "
                "아래에서 업로드 가능."
            )

        st.markdown("---")
        st.markdown("**📥 추가 데이터 업로드**")
        extra = st.file_uploader(
            "zip 또는 단일 파일",
            type=["zip", "geojson", "csv", "json"],
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
                st.rerun()
            except Exception as e:
                st.error(f"실패: {e}")

        if not (gj_ok and rd_ok):
            st.caption("README의 데이터 준비 섹션 참고.")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _render_urgency_ranking(opts: dict) -> None:
    """시급도 순위 분석 UI.

    유동인구 필터 결과(ft_link_agg) × 도로링크 경사도/길이로 점수 산출.
    """
    agg = st.session_state.get("ft_link_agg")
    if agg is None or agg.empty:
        st.info(
            "🚶 **유동인구 분석** 섹션에서 필터를 적용하면 자동으로 활성화됩니다. "
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

    # 산식 선택 UI
    ccol1, ccol2, ccol3 = st.columns([2, 2, 2])
    method = ccol1.selectbox(
        "점수 산식",
        options=["geometric", "weighted_mean", "product"],
        index=0,
        format_func=lambda x: {
            "geometric": "기하평균 √(경사 × 유동) ⭐ 추천",
            "weighted_mean": "가중평균",
            "product": "곱셈 (극단적)",
        }[x],
        help=(
            "**기하평균**: 두 차원 모두 높을 때만 큼. "
            "**가중평균**: 한 쪽만 높아도 점수 올라감. "
            "**곱셈**: 한쪽이 낮으면 폭락."
        ),
    )
    slope_weight = 0.5
    if method == "weighted_mean":
        slope_weight = ccol2.slider(
            "경사도 비중",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="0 = 유동인구만, 1 = 경사도만",
        )
    top_n = ccol3.number_input(
        "상위 N개 표시", min_value=10, max_value=200, value=30, step=10
    )

    # 점수 계산
    scored = compute_urgency_scores(agg, link_data, method=method, slope_weight=slope_weight)

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("대상 도로링크", f"{len(scored):,}")
    k2.metric("평균 시급도", f"{scored['urgency'].mean():.1f}")
    k3.metric(
        "총 길이",
        f"{scored['length_m'].sum()/1000:,.2f} km",
    )
    top10_pct = scored.head(int(len(scored) * 0.1) or 1)
    k4.metric(
        "상위 10% 평균 길이",
        f"{top10_pct['length_m'].mean():.0f} m",
        help="시급도 상위 10% 도로링크의 평균 구간 길이",
    )

    # 표시용 컬럼명/순서 정리
    show = scored.head(int(top_n)).copy()
    show["slope_pct"] = show["slope_pct"].apply(
        lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "-"
    )
    show["length_m"] = show["length_m"].round(0).astype(int)
    show["flow_sum"] = show["flow_sum"].round(1)
    show_disp = show[
        ["link_id", "gu", "dong", "slope_level", "slope_pct",
         "length_m", "flow_sum", "slope_score", "flow_score", "urgency"]
    ].rename(columns={
        "link_id": "도로링크ID",
        "gu": "자치구",
        "dong": "행정동",
        "slope_level": "경사 등급",
        "slope_pct": "경사도 %",
        "length_m": "길이(m)",
        "flow_sum": "유동인구 합",
        "slope_score": "경사 점수",
        "flow_score": "유동 점수",
        "urgency": "🚨 시급도",
    })
    show_disp.insert(0, "순위", range(1, len(show_disp) + 1))
    st.dataframe(show_disp, use_container_width=True, hide_index=True)

    # 자치구·행정동별 시급도 평균 집계
    st.markdown("**행정동별 시급도 집계** (구간 길이 가중평균)")
    gby = (
        scored.groupby(["gu", "dong"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "구간 수": len(g),
                "총 길이(km)": round(g["length_m"].sum() / 1000, 2),
                "평균 시급도": round(g["urgency"].mean(), 1),
                "길이가중 시급도": round(
                    (g["urgency"] * g["length_m"]).sum() / max(g["length_m"].sum(), 1),
                    1,
                ),
                "≥10% 구간 수": int((g["slope_level"] == "10_plus").sum()),
            }),
            include_groups=False,
        )
        .reset_index(drop=False)
        .sort_values("길이가중 시급도", ascending=False)
    )
    if "level_0" in gby.columns:
        gby = gby.drop(columns=["level_0"])
    if "level_1" in gby.columns:
        gby = gby.drop(columns=["level_1"])
    gby = gby.rename(columns={"gu": "자치구", "dong": "행정동"})
    st.dataframe(gby, use_container_width=True, hide_index=True)

    # CSV 다운로드
    csv = scored[
        ["link_id", "gu", "dong", "slope_level", "slope_pct",
         "length_m", "flow_sum", "slope_score", "flow_score", "urgency"]
    ].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "전체 시급도 순위 CSV 다운로드",
        data=csv,
        file_name=f"urgency_ranking_{method}.csv",
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
        source_label = f"📄 업로드: `{upload.name}`"
    elif default_path.exists():
        try:
            with default_path.open("rb") as f:
                df = _parse_foot_traffic(f.read(), default_path.name)
        except Exception as e:
            st.error(f"기본 데이터 로드 실패: {e}")
            return
        source_label = f"📦 기본 데이터: `{default_path.name}`"
    else:
        st.info(
            "유동인구 데이터가 없습니다. 사이드바 **🚶 유동인구 데이터** 에서 "
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
        "🌅 출근시간(7-9시)",
        help="평일 06-09 시간대만 필터 (결빙 취약 시간)",
        use_container_width=True,
    )
    preset_elderly = preset_cols[1].button(
        "👴 60대 이상 평일",
        help="평일 + 60대 이상 (낙상 위험층)",
        use_container_width=True,
    )
    preset_all = preset_cols[2].button(
        "🔄 필터 초기화",
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
            default=st.session_state.get("ft_day_codes", []),
            format_func=lambda c: f"{c} · {DAY_LABELS[c]}",
            key="ft_day_codes",
        )
        time_codes = st.multiselect(
            "시간대",
            options=list(DEFAULT_TIME_LABELS.keys()),
            default=st.session_state.get("ft_time_codes", []),
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
        "💡 사이드바 **유동인구 (TBGIS 도로링크)** 토글을 켜면 위 필터 결과가 "
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
    st.title("📦 초기 데이터 준비")
    st.error(
        f"⚠️ **앱 실행에 필요한 필수 데이터 {len(missing)}개**가 없습니다. "
        "아래 두 가지 방법 중 하나로 데이터를 채워주세요."
    )

    # 누락된 파일 상세 안내 표
    with st.container(border=True):
        st.markdown("##### 📋 누락된 파일")
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
            "💡 데이터를 한 zip 파일에 담아서 받으시면 자동으로 풀려서 배치됩니다."
        )

    tab_dl, tab_up = st.tabs(["🌐 Google Drive 다운로드", "📁 로컬 파일 업로드"])

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


def main() -> None:
    # 초기 데이터 부트스트랩 — 필수 파일이 없으면 다운로드/업로드 UI 표시
    missing = missing_files(DATA_PATH)
    if missing:
        _render_data_bootstrap(missing)
        return  # st.stop() 후에는 도달 안 함

    st.title("🛣️ 서울시 도로 열선 설치 현황")
    st.caption(
        "2026년 자치구별 도로열선 설치현황 데이터 기반. "
        "레이어 토글로 자치구·도로를 비교하고, 아래 분석 섹션에서 표를 확인하세요."
    )

    # 선택 데이터 누락 안내 (있어도 앱은 작동, 일부 기능만 비활성)
    missing_opt = missing_optional(DATA_PATH)
    if missing_opt:
        with st.container(border=True):
            st.warning(
                "⚙️ **추가 기능을 위한 데이터가 없습니다.** 아래 파일을 "
                "`data/` 폴더에 추가하면 해당 기능이 활성화됩니다. "
                "(현재 상태로도 다른 기능은 모두 사용 가능)"
            )
            rows = []
            for name in missing_opt:
                info = FILE_INFO.get(name, {})
                rows.append({
                    "파일": name,
                    "크기": info.get("size", "-"),
                    "추가되면 활성화될 기능": info.get("desc", "-"),
                })
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True
            )
            st.caption(
                "💡 `seoul_roads.geojson` 받을 때 zip에 같이 담아 받으셔도 되고, "
                "사이드바 **📁 데이터 상태 → 추가 데이터** 에서 별도 업로드도 가능합니다."
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
    gu_metrics = compute_risk_score(aggregate_by_gu(df))
    road_metrics = aggregate_by_road(df)

    # KPI
    render_kpis(df, gu_metrics)
    st.divider()

    # 지도 — 유동인구 도로링크 시각화용 집계 결과를 opts에 합쳐 전달
    opts["ft_link_agg"] = st.session_state.get("ft_link_agg")
    st.subheader("🗺️ 지도")
    render_map(df, opts, gu_metrics)

    # 분석 섹션 (접고 펼치기)
    st.subheader("📊 분석")

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
                    "risk_score": "위험도 점수(임시)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("도로별 집계 (도로명 단위)", expanded=False):
        st.caption(
            f"같은 도로명에서 발생한 여러 설치 구간을 합산. "
            f"도로명을 추출하지 못한 {(df['primary_road']=='').sum()}건은 제외."
        )
        st.dataframe(
            road_metrics.rename(
                columns={
                    "gu": "자치구",
                    "primary_road": "도로명",
                    "segment_count": "구간 수",
                    "total_length_m": "총 연장(m)",
                    "years": "설치연도",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("원본 데이터 (필터 적용 후)", expanded=False):
        st.caption("CSV 원문에 도로명 추출 결과를 덧붙인 형태.")
        st.dataframe(
            df.rename(
                columns={
                    "gu": "자치구",
                    "year": "설치연도",
                    "section": "설치구간",
                    "length_m": "연장(m)",
                    "primary_road": "대표 도로명",
                    "road_names": "추출 도로명",
                }
            )[
                [
                    "자치구",
                    "설치연도",
                    "설치구간",
                    "연장(m)",
                    "대표 도로명",
                    "추출 도로명",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("경사도 분석 (자치구별 등급 분포)", expanded=False):
        road_features = _load_road_features()
        if not road_features:
            st.info("도로 라인 GeoJSON이 없어 경사도 통계를 계산할 수 없습니다.")
        elif not any(
            f.get("properties", {}).get("slope_pct") is not None
            for f in road_features[:200]
        ):
            st.warning(
                "도로 GeoJSON에 `slope_pct` 컬럼이 없습니다. "
                "`python3 scripts/preprocess_slope.py` 를 먼저 실행해주세요."
            )
        else:
            by_gu = _split_roads_by_gu(road_features)

            # 자치구별 등급 분포 (선택된 자치구만)
            rows = []
            for gu in opts["selected_gus"]:
                feats_g = by_gu.get(gu, [])
                cnt = {lvl: 0 for lvl in SLOPE_LEVELS}
                for f in feats_g:
                    cnt[f["properties"].get("slope_level", "unknown")] += 1
                total = sum(cnt.values()) or 1
                rows.append(
                    {
                        "자치구": gu,
                        "전체": total if total != 1 else len(feats_g),
                        "≥6% 건수": cnt["6_10"] + cnt["10_plus"],
                        "≥6% 비율": f"{(cnt['6_10']+cnt['10_plus'])/total*100:.1f} %",
                        "≥10% 건수": cnt["10_plus"],
                        **{SLOPE_LABELS[lv]: cnt[lv] for lv in SLOPE_LEVELS},
                    }
                )

            # 전체 통계
            total_all = sum(
                1
                for gu in opts["selected_gus"]
                for _ in by_gu.get(gu, [])
            )
            steep_all = sum(
                1
                for gu in opts["selected_gus"]
                for f in by_gu.get(gu, [])
                if f["properties"].get("slope_level") in ("6_10", "10_plus")
            )
            very_steep = sum(
                1
                for gu in opts["selected_gus"]
                for f in by_gu.get(gu, [])
                if f["properties"].get("slope_level") == "10_plus"
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("필터내 도로 수", f"{total_all:,}")
            c2.metric(
                "경사도 ≥ 6%",
                f"{steep_all:,}",
                f"{steep_all/max(total_all,1)*100:.1f} %",
            )
            c3.metric(
                "경사도 ≥ 10%",
                f"{very_steep:,}",
                f"{very_steep/max(total_all,1)*100:.1f} %",
            )

            st.caption("자치구별 경사도 등급 분포 (선택 자치구만)")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("경사도 × 결빙구간 교차 분석", expanded=False):
        st.caption(
            "OSM 도로 라인 매칭이 된 결빙구간의 경사도 분포를 봅니다. "
            "6%↑ 결빙구간이 우선 열선 설치 대상으로 검토되어야 합니다."
        )
        road_features = _load_road_features()
        snapped = _load_icing_snapped()
        if road_features and snapped:
            # 매칭된 OSM segment의 slope_level을 결빙 항목에 입혀 집계
            # snap_icing_to_osm_paths 는 coords만 들고있어 직접 매칭 어려움.
            # 대안: OSM 도로 (road_name, gu) → slope_level lookup 사전을 만들고
            # 결빙의 (road_name, gu_start)로 lookup.
            slope_by_key: dict[tuple[str, str], str] = {}
            for feat in road_features:
                props = feat.get("properties", {})
                rn = props.get("road_name") or ""
                gu = props.get("gu") or ""
                if rn:
                    key = (rn, gu)
                    # 같은 도로명+자치구에 여러 segment 있으면 최댓값 등급으로 단순화
                    cur = slope_by_key.get(key, "unknown")
                    new = props.get("slope_level", "unknown")
                    rank = {"unknown": 0, "0_3": 1, "3_6": 2, "6_10": 3, "10_plus": 4}
                    if rank.get(new, 0) > rank.get(cur, 0):
                        slope_by_key[key] = new

            ic_by_lvl = {lvl: 0 for lvl in SLOPE_LEVELS}
            for it in snapped:
                if it["snap"] != "osm":
                    ic_by_lvl["unknown"] += 1
                    continue
                key = (it["road_name"], it["gu"])
                lvl = slope_by_key.get(key, "unknown")
                ic_by_lvl[lvl] += 1

            total_ic = sum(ic_by_lvl.values()) or 1
            rows = [
                {
                    "경사도 등급": SLOPE_LABELS[lv],
                    "결빙 매칭 건수": ic_by_lvl[lv],
                    "비율": f"{ic_by_lvl[lv]/total_ic*100:.1f} %",
                }
                for lv in SLOPE_LEVELS
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
            steep_ic = ic_by_lvl["6_10"] + ic_by_lvl["10_plus"]
            st.metric(
                "결빙구간 중 경사도 ≥ 6%",
                f"{steep_ic} / {total_ic}",
                f"{steep_ic/total_ic*100:.1f} %",
            )
        else:
            st.info("도로 라인 또는 결빙 데이터가 없습니다.")

    with st.expander("상습 결빙구간 분석", expanded=False):
        df_icing_all = _load_icing_seoul()
        if df_icing_all.empty:
            st.info(f"`data/{ICING_CSV}` 가 없어 결빙구간을 표시할 수 없습니다.")
        else:
            # 자치구 필터 반영
            mask = (
                df_icing_all["gu"].isin(opts["selected_gus"])
                | df_icing_all["gu_start"].isin(opts["selected_gus"])
                | df_icing_all["gu_end"].isin(opts["selected_gus"])
            )
            df_icing = df_icing_all[mask]

            # OSM 스냅 통계
            snapped_all = _load_icing_snapped()
            snapped_ids = {s["seg_id"] for s in snapped_all if s["snap"] == "osm"}
            snap_rate = (
                len(snapped_ids) / len(df_icing_all) * 100
                if len(df_icing_all)
                else 0
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("결빙구간 (서울 전체)", f"{len(df_icing_all):,} 건")
            c2.metric("필터 적용 후", f"{len(df_icing):,} 건")
            c3.metric(
                "OSM 스냅 성공률",
                f"{snap_rate:.1f} %",
                help="결빙 도로명이 OSM에서 매칭되어 실제 도로 형상으로 표시된 비율. "
                "미매칭은 기점-종점 직선으로 fallback.",
            )
            c4.metric(
                "필터 결빙 총 길이",
                f"{df_icing['length_km'].sum():.1f} km",
            )

            # 자치구별 집계
            gu_icing = (
                df_icing.groupby("gu", as_index=False)
                .agg(
                    icing_count=("seg_id", "count"),
                    icing_length_km=("length_km", "sum"),
                )
                .sort_values("icing_count", ascending=False)
            )
            gu_icing["icing_length_km"] = gu_icing["icing_length_km"].round(2)
            st.caption("자치구별 결빙구간 집계 (필터 적용 후)")
            st.dataframe(
                gu_icing.rename(
                    columns={
                        "gu": "자치구",
                        "icing_count": "결빙구간 수",
                        "icing_length_km": "총 길이(km)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # 도로분류별
            cls_icing = (
                df_icing.groupby("road_class", as_index=False)
                .agg(count=("seg_id", "count"), length_km=("length_km", "sum"))
                .sort_values("count", ascending=False)
            )
            cls_icing["length_km"] = cls_icing["length_km"].round(2)
            st.caption("도로분류별 집계")
            st.dataframe(
                cls_icing.rename(
                    columns={
                        "road_class": "도로분류",
                        "count": "건수",
                        "length_km": "총 길이(km)",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("도로 매칭 통계 (OSM ↔ CSV)", expanded=False):
        road_features = _load_road_features()
        if road_features:
            road_lines = match_roads(df, road_features)
            csv_keys = {
                (r["gu"], rn)
                for _, r in df.iterrows()
                for rn in r["road_names"] if rn
            }
            matched_keys = {(m.gu, m.road_name) for m in road_lines}
            exact = sum(1 for m in road_lines if m.match_type == "exact")
            name_only = sum(1 for m in road_lines if m.match_type == "name_only")
            cross_gu = sum(
                1 for m in road_lines
                if m.osm_gu and m.osm_gu != m.gu
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CSV unique 도로명", f"{len(csv_keys):,}")
            c2.metric(
                "매칭 성공",
                f"{len(matched_keys):,}",
                f"{len(matched_keys)/max(len(csv_keys),1)*100:.1f}%",
            )
            c3.metric("정확 매칭 라인", f"{exact:,}")
            c4.metric("자치구 불일치", f"{cross_gu:,}")
            st.caption(
                "자치구 불일치: CSV의 관리주체 자치구와 OSM 상의 도로 위치 자치구가 "
                "다른 경우. 도로 라인 hover시 ※표시로 확인 가능."
            )
        else:
            st.info("도로 라인 GeoJSON이 없어 매칭 통계를 계산할 수 없습니다.")

    with st.expander("🚶 유동인구 분석 (피봇 테이블)", expanded=False):
        _render_foot_traffic_section(opts)

    with st.expander("🚨 시급도 순위 (경사도 × 유동인구)", expanded=False):
        _render_urgency_ranking(opts)

    with st.expander("위험도 점수 (placeholder)", expanded=False):
        st.warning(
            "현재 위험도 점수는 **설치 길이가 짧을수록 위험이 크다**는 가정의 임시 점수입니다. "
            "추후 도로 경사도, 결빙위험 등의 실제 데이터가 들어오면 이 컬럼을 교체할 예정."
        )
        st.dataframe(
            gu_metrics[["gu", "total_length_m", "risk_score"]].rename(
                columns={
                    "gu": "자치구",
                    "total_length_m": "총 연장(m)",
                    "risk_score": "위험도 점수(임시)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
