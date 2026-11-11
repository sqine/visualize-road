"""서울 전역 993건 열선설치 주소(1,826개 시점/종점) VWorld 지오코딩.
체크포인트(/tmp/heating_geocode_cache_citywide.pkl)에 100건마다 저장 -- 중간에 멈춰도 안전.
"""
import pickle
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
CACHE = Path("/tmp/heating_geocode_cache_citywide.pkl")
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.env import get_secret  # noqa: E402

KEY = get_secret("VWORLD_API_KEY")  # 환경변수 또는 .env — 코드에 키를 적지 않는다

CHUNK = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def load_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(c):
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(c, f)
    tmp.replace(CACHE)


def geocode(addr_full, session):
    for typ in ("road", "parcel"):
        try:
            r = session.get(
                "https://api.vworld.kr/req/address",
                params={"service": "address", "request": "getcoord", "version": "2.0",
                        "crs": "epsg:4326", "address": addr_full, "refine": "true",
                        "simple": "false", "format": "json", "type": typ, "key": KEY},
                timeout=5,
            )
            j = r.json()
            resp = j.get("response", {})
            if resp.get("status") == "OK":
                pt = resp["result"]["point"]
                return float(pt["x"]), float(pt["y"])
        except Exception:
            pass
    return None


def split_sides(loc):
    parts = re.split(r"[~∼]", str(loc))
    return [p.strip() for p in parts if p.strip()]


def main():
    t0 = time.time()
    cache = load_cache()
    heat = pd.read_csv(f"{DATA}/heating_2026.csv", encoding="utf-8-sig")
    heat["sides"] = heat["설치위치 (시점 ~ 종점)"].apply(split_sides)

    tasks = []
    for _, row in heat.iterrows():
        gu = row["관리기관"]
        for side in row["sides"]:
            full = f"서울특별시 {gu} {side}"
            if full not in cache:
                tasks.append(full)
    print(f"전체 필요: {heat['sides'].apply(len).sum()}건, 캐시됨 {len(cache)}건, 남음 {len(tasks)}건", flush=True)
    if not tasks:
        print("모두 완료.")
        return

    session = requests.Session()
    batch = tasks[:CHUNK]
    n_ok = 0
    for i, full in enumerate(batch):
        pt = geocode(full, session)
        cache[full] = pt
        if pt:
            n_ok += 1
        if (i + 1) % 100 == 0:
            save_cache(cache)
            print(f"  {i+1}/{len(batch)} ({time.time()-t0:.1f}s, 성공 {n_ok}) 저장됨", flush=True)
    save_cache(cache)
    print(f"배치 완료: {len(batch)}건, 성공 {n_ok} ({time.time()-t0:.1f}s)", flush=True)
    print(f"누적 캐시 {len(cache)}, 남은 {len(tasks)-len(batch)}", flush=True)


if __name__ == "__main__":
    main()
