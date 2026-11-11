"""도엽 실제 커버리지 검증 — 감사 로그가 아니라 zip 원본에서 직접 확인.

배경
----
`01_데이터/수집이력_도엽/도엽_대조_20260919/summary.json`은 서울 25개구 도엽
확보율을 72.7%(필요 3,075 / 완료 2,237)로 기록한다. 그런데 링크 경사 보유율은
자치구별 92~99%로 균일하고 도엽 완료율과 무상관(ρ=-0.052)이다. 둘 중 하나가
틀렸다.

방법
----
감사 로그를 믿지 않고 zip을 직접 연다. shapefile(.shp)의 **헤더 100바이트**에
bounding box가 들어 있으므로 전체 파싱 없이 도엽 범위를 읽을 수 있다.

    byte 36..68 : Xmin, Ymin, Xmax, Ymax (little-endian float64, EPSG:5186)

도엽 bbox를 WGS84로 변환한 뒤, 각 도로 링크 중점이 어느 도엽에도 들어가지
않으면 '미커버'로 센다. 자치구별 커버리지를 내어 감사 로그와 대조한다.

실행
----
    cd streamlit_app
    python3 scripts/verify_tile_coverage.py

산출
----
    viz/data/tile_coverage_by_gu.csv
"""

from __future__ import annotations

import glob
import json
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT.parent / "서울시 수치지형도 파일" / "수치지형도" / "1_1000"
DATA = ROOT / "data"
VIZ = ROOT / "viz" / "data"
AUDIT = (ROOT.parent / "01_데이터" / "수집이력_도엽" /
         "도엽_대조_20260919" / "summary.json")

_T = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)


def shp_bbox(zpath: str) -> tuple[float, float, float, float] | None:
    """zip 안 .shp 헤더에서 bbox(EPSG:5186)를 읽어 WGS84 (W,S,E,N)로 반환."""
    try:
        with zipfile.ZipFile(zpath) as z:
            shps = [n for n in z.namelist() if n.lower().endswith(".shp")]
            if not shps:
                return None
            xmin = ymin = float("inf")
            xmax = ymax = float("-inf")
            for n in shps:
                with z.open(n) as f:
                    head = f.read(100)
                if len(head) < 100:
                    continue
                a, b, c, d = struct.unpack("<4d", head[36:68])
                if not all(map(np.isfinite, (a, b, c, d))):
                    continue
                if a == 0 and b == 0 and c == 0 and d == 0:
                    continue
                xmin, ymin = min(xmin, a), min(ymin, b)
                xmax, ymax = max(xmax, c), max(ymax, d)
            if xmin == float("inf"):
                return None
            w, s = _T.transform(xmin, ymin)
            e, n = _T.transform(xmax, ymax)
            return w, s, e, n
    except Exception:
        return None


def main() -> None:
    t0 = time.time()
    zips = sorted(glob.glob(str(SRC_DIR / "*.zip")))
    print(f"zip {len(zips)}개 스캔 — {SRC_DIR}")
    if not zips:
        print("❌ 소스 폴더에 zip이 없습니다. 경로를 확인하세요.")
        sys.exit(1)

    boxes, bad = [], 0
    for i, z in enumerate(zips, 1):
        bb = shp_bbox(z)
        if bb is None:
            bad += 1
        else:
            boxes.append(bb)
        if i % 500 == 0:
            print(f"  {i}/{len(zips)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"유효 도엽 {len(boxes)} / 판독실패 {bad}  ({time.time()-t0:.0f}s)")

    B = np.array(boxes)  # (n,4) W,S,E,N

    # 도로 링크 중점
    gj = json.load(open(DATA / "road_links.geojson", encoding="utf-8"))
    feats = gj["features"]
    mid, keep = [], []
    for f in feats:
        g = f["geometry"]
        if g["type"] != "LineString" or len(g["coordinates"]) < 2:
            continue
        c = g["coordinates"][len(g["coordinates"]) // 2]
        mid.append((c[0], c[1]))
        keep.append(f["properties"].get("slope_pct"))
    M = np.array(mid)
    slope = np.array([s is not None for s in keep])
    print(f"도로 링크 {len(M):,}개 (경사 보유 {slope.sum():,})")

    # 자치구 배정
    from matplotlib.path import Path as MPath
    gu = json.load(open(DATA / "seoul_gu.geojson", encoding="utf-8"))
    name = np.array([None] * len(M), dtype=object)
    for f in gu["features"]:
        nm = [v for k, v in f["properties"].items()
              if isinstance(v, str) and v.endswith("구")][0]
        geom = f["geometry"]
        rings = ([geom["coordinates"]] if geom["type"] == "Polygon"
                 else geom["coordinates"])
        for poly in rings:
            un = np.array([name[i] is None for i in range(len(M))])
            if not un.any():
                break
            ins = MPath(np.array(poly[0])[:, :2]).contains_points(M[un])
            name[np.where(un)[0][ins]] = nm

    # 도엽 커버 여부 — bbox 포함 검사 (벡터화)
    covered = np.zeros(len(M), bool)
    lon, lat = M[:, 0], M[:, 1]
    for w, s, e, n in B:
        covered |= (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)

    rows = []
    for g in sorted({x for x in name if x}):
        m = name == g
        rows.append({
            "자치구": g,
            "링크": int(m.sum()),
            "도엽커버": int(covered[m].sum()),
            "도엽커버율": round(100 * covered[m].mean(), 1),
            "경사보유율": round(100 * slope[m].mean(), 1),
        })
    df = pd.DataFrame(rows)

    if AUDIT.exists():
        a = pd.DataFrame(json.load(open(AUDIT, encoding="utf-8")))
        a = a.rename(columns={"구": "자치구"})
        a["감사_완료율"] = (100 * a["완료"] / a["필요"]).round(1)
        df = df.merge(a[["자치구", "필요", "완료", "감사_완료율"]],
                      on="자치구", how="left")
        df["차이"] = (df["도엽커버율"] - df["감사_완료율"]).round(1)

    df = df.sort_values("도엽커버율")
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))
    df.to_csv(VIZ / "tile_coverage_by_gu.csv", index=False,
              encoding="utf-8-sig")

    tot_cov = 100 * covered.mean()
    print(f"\n서울 전역 도엽 커버율 {tot_cov:.1f}%  "
          f"(링크 {covered.sum():,} / {len(M):,})")
    if "감사_완료율" in df:
        import scipy.stats as st
        d = df.dropna(subset=["감사_완료율"])
        r = st.spearmanr(d["감사_완료율"], d["도엽커버율"])
        print(f"감사 완료율 vs 실측 커버율  Spearman ρ={r.statistic:.3f} "
              f"p={r.pvalue:.4f}")
        print("\n판정:")
        if tot_cov > 95 and (d["차이"] > 20).sum() >= 3:
            print("  ✅ 실측 커버리지가 감사 로그보다 훨씬 높다 "
                  "→ 감사 로그의 '필요' 도엽 수가 과다 산정된 것. 로그 폐기 가능")
        elif abs(tot_cov - d["감사_완료율"].mean()) < 10:
            print("  ⚠️ 실측과 감사 로그가 일치 → 도엽 결손이 실재. "
                  "경사도 출처(1:5,000 폴백/보간) 추적 필요")
        else:
            print("  ❓ 판정 보류 — 위 표를 직접 확인할 것")
    print(f"\n저장: viz/data/tile_coverage_by_gu.csv  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
