"""8개 정밀구 열선설치(heating_2026.csv) 389건의 시점~종점 주소를 VWorld로 지오코딩.
체크포인트(/tmp/heating_geocode_cache.pkl)에 누적 저장 — 여러 번 호출해도 이어서 진행.
"""
import pickle
import re
import time
from pathlib import Path

import pandas as pd
import requests

DATA = "/sessions/cool-optimistic-edison/mnt/road-analysis/streamlit_app/data"
CACHE = Path("/tmp/heating_geocode_cache.pkl")
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.env import get_secret  # noqa: E402

KEY = get_secret("VWORLD_API_KEY")  # 환경변수 또는 .env — 코드에 키를 적지 않는다
TARGET8 = ["성북구", "중랑구", "광진구", "서대문구", "종로구", "성동구", "동대문구", "중구"]


def load_cache():
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(c):
    with open(CACHE, "wb") as f:
        pickle.dump(c, f)


def geocode(addr_full, gu, session):
    """도로명 우선, 실패시 지번으로 재시도. (lon, lat) or None."""
    for typ in ("road", "parcel"):
        try:
            r = session.get(
                "https://api.vworld.kr/req/address",
                params={
                    "service": "address", "request": "getcoord", "version": "2.0",
                    "crs": "epsg:4326", "address": addr_full, "refine": "true",
                    "simple": "false", "format": "json", "type": typ, "key": KEY,
                },
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
    sub = heat[heat["관리기관"].isin(TARGET8)].copy()
    sub["sides"] = sub["설치위치 (시점 ~ 종점)"].apply(split_sides)

    tasks = []
    for _, row in sub.iterrows():
        gu = row["관리기관"]
        for side in row["sides"]:
            full = f"서울특별시 {gu} {side}"
            if full not in cache:
                tasks.append((full, gu))
    print(f"전체 주소 필요: {sub['sides'].apply(len).sum()}건, 캐시됨: {len(cache)}건, "
          f"남음: {len(tasks)}건", flush=True)

    session = requests.Session()
    CHUNK = 400
    batch = tasks[:CHUNK]
    n_ok = 0
    for i, (full, gu) in enumerate(batch):
        pt = geocode(full, gu, session)
        cache[full] = pt
        if pt:
            n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(batch)} 처리 ({time.time()-t0:.1f}s, 성공 {n_ok})", flush=True)
    save_cache(cache)
    print(f"이번 배치 완료: {len(batch)}건 처리, 성공 {n_ok}건 ({time.time()-t0:.1f}s)", flush=True)
    print(f"누적 캐시: {len(cache)}건, 남은 작업: {len(tasks)-len(batch)}건", flush=True)


if __name__ == "__main__":
    main()
