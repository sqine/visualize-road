"""TBGIS 도로링크 SHP/DBF → GeoJSON 변환 (EPSG:5181 → WGS84).

입력:
    data/shp/TBGIS_ROAD_LINK_FRM.shp  - PolyLine geometry (EPSG:5181)
    data/shp/TBGIS_ROAD_LINK_FRM.dbf  - ROAD_LID, ROAD_CD, STDR_YM_CD

출력:
    data/road_links.geojson  - WGS84 LineString,
                               properties: road_lid, road_cd, ym

이후 flpop.csv 의 rd_link_cd 와 road_lid 로 join 가능.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.proj import tm5181_to_wgs84  # noqa: E402

# 표고/경사도 모듈 (없으면 경사도 계산 스킵)
try:
    from scripts.preprocess_slope import (  # noqa: E402
        GridIndex,
        _euclid_m,
        parse_dbf_height_field,
        parse_shp_points,
        slope_level,
    )
    from utils.proj import tm5174_to_wgs84  # noqa: E402
    _HAS_SLOPE = True
except Exception:
    _HAS_SLOPE = False

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
SHP_DIR = DATA / "shp"
SHP = SHP_DIR / "TBGIS_ROAD_LINK_FRM.shp"
DBF = SHP_DIR / "TBGIS_ROAD_LINK_FRM.dbf"
OUT = DATA / "road_links.geojson"


def parse_dbf(dbf_path: Path) -> list[dict[str, str]]:
    """DBF → list of {column: value} (cp949 디코드, strip)."""
    out: list[dict[str, str]] = []
    with dbf_path.open("rb") as f:
        hdr = f.read(32)
        n_records = struct.unpack("<i", hdr[4:8])[0]
        hdr_len = struct.unpack("<h", hdr[8:10])[0]
        rec_len = struct.unpack("<h", hdr[10:12])[0]

        f.seek(32)
        fields: list[tuple[str, int]] = []
        while True:
            b = f.read(32)
            if not b or b[0] == 0x0D:
                break
            name = b[0:11].split(b"\x00", 1)[0].decode("cp949", errors="replace")
            flen = b[16]
            fields.append((name, flen))

        f.seek(hdr_len)
        for _ in range(n_records):
            rec = f.read(rec_len)
            if len(rec) < rec_len:
                break
            pos = 1
            row: dict[str, str] = {}
            for name, flen in fields:
                row[name] = rec[pos:pos + flen].decode("cp949", errors="replace").strip()
                pos += flen
            out.append(row)
    return out


def parse_shp_polylines(shp_path: Path):
    """SHP PolyLine yield: index, list[ (x, y) ] (parts concatenated)."""
    with shp_path.open("rb") as f:
        f.seek(100)
        idx = 0
        while True:
            rec_hdr = f.read(8)
            if len(rec_hdr) < 8:
                return
            content_len = struct.unpack(">i", rec_hdr[4:8])[0] * 2
            content = f.read(content_len)
            st = struct.unpack("<i", content[0:4])[0]
            if st == 0:
                # null
                yield idx, []
                idx += 1
                continue
            if st not in (3, 13):  # PolyLine, PolyLineZ
                idx += 1
                continue
            n_parts = struct.unpack("<i", content[36:40])[0]
            n_points = struct.unpack("<i", content[40:44])[0]
            # parts offset is 44, parts are n_parts int32
            parts_start = 44
            pts_offset = parts_start + 4 * n_parts
            pts: list[tuple[float, float]] = []
            for i in range(n_points):
                off = pts_offset + i * 16
                x, y = struct.unpack("<2d", content[off:off + 16])
                pts.append((x, y))
            yield idx, pts
            idx += 1


def _round5(coords):
    return [[round(x, 5), round(y, 5)] for x, y in coords]


def _build_elevation_index() -> "GridIndex | None":
    """표고 SHP 가 있으면 GridIndex 빌드, 없으면 None."""
    if not _HAS_SLOPE:
        return None
    elev_shp = DATA / "slope_raw" / "표고 5000" / "N3P_F002.shp"
    elev_dbf = DATA / "slope_raw" / "표고 5000" / "N3P_F002.dbf"
    if not (elev_shp.exists() and elev_dbf.exists()):
        return None

    print("  표고 SHP/DBF 파싱 (EPSG:5174 → WGS84)")
    pts5174 = parse_shp_points(elev_shp)
    heights = parse_dbf_height_field(elev_dbf)
    n = min(len(pts5174), len(heights))
    idx = GridIndex(cell_m=100.0)
    skipped_e = 0
    for i in range(n):
        x, y = pts5174[i]
        h = heights[i]
        try:
            if not (h == h):  # NaN 체크
                skipped_e += 1
                continue
            lat, lon = tm5174_to_wgs84(x, y)
            idx.add(lon, lat, h)
        except Exception:
            skipped_e += 1
    print(f"  표고 점: {len(idx.h):,}  (스킵 {skipped_e})")
    return idx


def main() -> None:
    t0 = time.time()
    print(f"[1/5] DBF 파싱: {DBF.name}")
    rows = parse_dbf(DBF)
    print(f"  rows: {len(rows):,}")

    print(f"[2/5] 표고 인덱스 구축 (도로링크 경사도 계산용)")
    elev_idx = _build_elevation_index()
    if elev_idx is None:
        print("  ⚠️ 표고 데이터 없음 — 경사도 컬럼 생략")

    print(f"[3/5] SHP 파싱 + 좌표 변환 + 경사도 매칭")
    out_feats = []
    skipped = 0
    bbox = [180.0, 90.0, -180.0, -90.0]
    counts = {"0_3": 0, "3_6": 0, "6_10": 0, "10_plus": 0, "unknown": 0}

    for idx, pts in parse_shp_polylines(SHP):
        if not pts or len(pts) < 2 or idx >= len(rows):
            skipped += 1
            continue

        coords_ll: list[list[float]] = []
        for x, y in pts:
            lat, lon = tm5181_to_wgs84(x, y)
            coords_ll.append([lon, lat])
            if lon < bbox[0]: bbox[0] = lon
            if lat < bbox[1]: bbox[1] = lat
            if lon > bbox[2]: bbox[2] = lon
            if lat > bbox[3]: bbox[3] = lat

        # 경사도 lookup
        elev_s = elev_e = slope = None
        slope_lv = "unknown"
        if elev_idx is not None and len(coords_ll) >= 2:
            lon_s, lat_s = coords_ll[0]
            lon_e, lat_e = coords_ll[-1]
            h_s, _ = elev_idx.nearest(lon_s, lat_s, max_m=200.0)
            h_e, _ = elev_idx.nearest(lon_e, lat_e, max_m=200.0)
            dist_m = _euclid_m(lon_s, lat_s, lon_e, lat_e)
            if h_s is not None and h_e is not None and dist_m >= 5.0:
                slope = abs(h_e - h_s) / dist_m * 100.0
                slope_lv = slope_level(slope)
                elev_s = round(h_s, 2)
                elev_e = round(h_e, 2)
        counts[slope_lv] += 1

        row = rows[idx]
        props = {
            "road_lid": row.get("ROAD_LID", ""),
            "road_cd": row.get("ROAD_CD", ""),
            "ym": row.get("STDR_YM_CD", ""),
        }
        if elev_idx is not None:
            props.update(
                elev_start_m=elev_s,
                elev_end_m=elev_e,
                slope_pct=round(slope, 2) if slope is not None else None,
                slope_level=slope_lv,
            )

        out_feats.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": _round5(coords_ll),
                },
            }
        )

        if (idx + 1) % 20000 == 0:
            print(f"  ...{idx+1:,} 처리 ({time.time()-t0:.1f}s)")

    print(f"  변환 완료: {len(out_feats):,} (스킵 {skipped})")
    print(f"  bbox: lon {bbox[0]:.4f}~{bbox[2]:.4f}  lat {bbox[1]:.4f}~{bbox[3]:.4f}")

    if elev_idx is not None:
        print(f"\n  경사도 등급 분포:")
        total = len(out_feats)
        for k in ("0_3", "3_6", "6_10", "10_plus", "unknown"):
            v = counts[k]
            print(f"    {k:>8}: {v:>7,}  ({v/total*100:5.1f}%)")

    print(f"\n[4/5] road_lid 중복 검사")
    lids = [f["properties"]["road_lid"] for f in out_feats]
    unique_lids = set(lids)
    print(f"  unique road_lid: {len(unique_lids):,}")

    print(f"[5/5] 저장: {OUT.name}")
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(
            {"type": "FeatureCollection", "features": out_feats},
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    mb = OUT.stat().st_size / 1024 / 1024
    print(f"  파일 크기: {mb:.1f} MB")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
