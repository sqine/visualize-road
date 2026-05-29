"""seoul_roads.geojson 추가 압축 (배포용).

작업:
    - 좌표 정밀도 5자리 → 4자리 (약 11m 정확도, 도시 도로 시각화엔 충분)
    - 불필요 properties 제거 (has_name)
    - 일부 properties null/0 압축
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "data" / "seoul_roads.geojson"
DST = HERE / "data" / "seoul_roads.geojson"  # 덮어쓰기

KEEP_PROPS = (
    "road_name",
    "gu",
    "highway",
    "elev_start_m",
    "elev_end_m",
    "dist_m",
    "slope_pct",
    "slope_level",
)


def main() -> None:
    print(f"[1/3] 로드: {SRC.name}")
    with SRC.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    print(f"  features: {len(feats):,}")
    before = SRC.stat().st_size

    print("[2/3] 좌표 정밀도 5→4 + properties 정리")
    for feat in feats:
        # properties 정리
        props = feat.get("properties", {})
        feat["properties"] = {k: props.get(k) for k in KEEP_PROPS if k in props}
        # 좌표 round
        geom = feat.get("geometry", {})
        if geom.get("type") == "LineString":
            geom["coordinates"] = [
                [round(c[0], 4), round(c[1], 4)] for c in geom["coordinates"]
            ]

    print(f"[3/3] 저장: {DST.name}")
    with DST.open("w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, separators=(",", ":"))
    after = DST.stat().st_size
    print(f"  {before/1024/1024:.1f} MB → {after/1024/1024:.1f} MB "
          f"({(1-after/before)*100:.0f}% 절감)")


if __name__ == "__main__":
    main()
