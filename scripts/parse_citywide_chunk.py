"""체크포인트 방식으로 나눠서 실행하는 파싱 단계 (환경의 실행시간 제한 우회용).
매 호출마다 N개 zip씩 처리해서 /tmp/citywide_parse_state.pkl 에 누적 저장.
"""
import glob
import io
import pickle
import sys
import time
import zipfile
from pathlib import Path

import shapefile
from pyproj import Transformer
from shapely.geometry import Polygon

HERE = Path("/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app")
ROOT = HERE.parent
SRC_DIR = ROOT / "서울시 수치지형도 파일" / "수치지형도" / "1_1000"
STATE_FILE = Path("/tmp/citywide_parse_state_v2.pkl")

_TRANSFORMER = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)

CHUNK_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 150


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return {
        "tile_polys": [],
        "pts_lonlat": [],
        "pts_elev": [],
        "n_lines_total": 0,
        "n_zip_err": 0,
        "done_zips": set(),
        "err_zips": [],
    }


def save_state(state):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)


def main():
    t0 = time.time()
    zips = sorted(glob.glob(str(SRC_DIR / "(B010)수치지도_*.zip")))
    state = load_state()
    remaining = [z for z in zips if z not in state["done_zips"]]
    print(f"전체 {len(zips)}개, 이미 처리 {len(state['done_zips'])}개, 남음 {len(remaining)}개")

    batch = remaining[:CHUNK_SIZE]
    if not batch:
        print("모두 처리 완료.")
        return

    for zpath in batch:
        try:
            with zipfile.ZipFile(zpath) as z:
                shp = io.BytesIO(z.read("N1A_H0010000.shp"))
                dbf = io.BytesIO(z.read("N1A_H0010000.dbf"))
                shx = io.BytesIO(z.read("N1A_H0010000.shx"))
                r = shapefile.Reader(shp=shp, dbf=dbf, shx=shx, encoding="cp949")
                s0 = r.shape(0)
                ring = [_TRANSFORMER.transform(x, y) for x, y in s0.points]
                if len(ring) >= 3:
                    state["tile_polys"].append(Polygon(ring))

                shp2 = io.BytesIO(z.read("N1L_F0010000.shp"))
                dbf2 = io.BytesIO(z.read("N1L_F0010000.dbf"))
                shx2 = io.BytesIO(z.read("N1L_F0010000.shx"))
                r2 = shapefile.Reader(shp=shp2, dbf=dbf2, shx=shx2, encoding="cp949")
                field_names = [f[0] for f in r2.fields[1:]]
                elev_idx = field_names.index("등고수치")
                for rec, s in zip(r2.records(), r2.shapes()):
                    try:
                        elev = float(rec[elev_idx])
                    except (ValueError, TypeError):
                        continue
                    state["n_lines_total"] += 1
                    for x, y in s.points:
                        lon, lat = _TRANSFORMER.transform(x, y)
                        state["pts_lonlat"].append((lon, lat))
                        state["pts_elev"].append(elev)
            state["done_zips"].add(zpath)
        except Exception as e:
            state["n_zip_err"] += 1
            state["err_zips"].append((zpath, str(e)))
            state["done_zips"].add(zpath)  # 에러난 것도 재시도 안 하게 완료 처리

    save_state(state)
    print(f"이번 배치 {len(batch)}개 처리 완료 ({time.time()-t0:.1f}s). "
          f"누적: 도엽 {len(state['tile_polys'])}개, 등고선 {state['n_lines_total']:,}개, "
          f"point cloud {len(state['pts_lonlat']):,}점, 에러 {state['n_zip_err']}개")
    print(f"남은 zip: {len(zips) - len(state['done_zips'])}개")


if __name__ == "__main__":
    main()
