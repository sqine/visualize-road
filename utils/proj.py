"""EPSG:5174 (Korean 1985 / Modified Korea Central Belt) ↔ WGS84 좌표 변환.

표준 라이브러리만 사용. pyproj/geopandas 등 외부 의존성 없음.

수식:
    1) Transverse Mercator inverse on Bessel 1841 ellipsoid
       (X, Y) → 베셀 위경도 (lat_b, lon_b)
    2) Helmert 3-parameter datum shift Bessel → WGS84
       (한국 국가표준 파라미터 사용)

서울 시내에서 ~수m 정확도. 좌표 변환 + 표고 매칭 용도로는 충분.
"""

from __future__ import annotations

import math

# ── Bessel 1841 ellipsoid ──
_A_B = 6377397.155
_F_B = 1.0 / 299.1528128
_E2_B = 2 * _F_B - _F_B * _F_B

# ── WGS84 ellipsoid ──
_A_W = 6378137.0
_F_W = 1.0 / 298.257223563
_E2_W = 2 * _F_W - _F_W * _F_W

# ── TM projection parameters for EPSG:5174 ──
_LAT0 = math.radians(38.0)
_LON0 = math.radians(127.0028902777778)
_K0 = 1.0
_FE = 200000.0
_FN = 500000.0

# ── Bessel → WGS84 datum shift (한국 NGII 표준 3-parameter) ──
_DX, _DY, _DZ = -145.907, 505.034, 685.756


def _meridional_arc_bessel(lat: float) -> float:
    n = _F_B / (2.0 - _F_B)
    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n
    A = (_A_B / (1.0 + n)) * (1.0 + n2 / 4.0 + n4 / 64.0)
    B = (3.0 * n / 2.0) * (1.0 - n2 / 8.0)
    C = (15.0 * n2 / 16.0) * (1.0 - n2 / 4.0)
    D = 35.0 * n3 / 48.0
    E = 315.0 * n4 / 512.0
    return A * (
        lat
        - B * math.sin(2 * lat)
        + C * math.sin(4 * lat)
        - D * math.sin(6 * lat)
        + E * math.sin(8 * lat)
    )


def _tm_inverse(X: float, Y: float) -> tuple[float, float]:
    x = X - _FE
    y = Y - _FN
    M0 = _meridional_arc_bessel(_LAT0)
    M = y / _K0 + M0

    e1 = (1 - math.sqrt(1 - _E2_B)) / (1 + math.sqrt(1 - _E2_B))
    mu = M / (
        _A_B
        * (1.0 - _E2_B / 4.0 - 3.0 * _E2_B**2 / 64.0 - 5.0 * _E2_B**3 / 256.0)
    )

    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8 * mu)
    )

    sin_p = math.sin(phi1)
    cos_p = math.cos(phi1)
    tan_p = math.tan(phi1)

    N1 = _A_B / math.sqrt(1 - _E2_B * sin_p * sin_p)
    T1 = tan_p * tan_p
    e_p2 = _E2_B / (1 - _E2_B)
    C1 = e_p2 * cos_p * cos_p
    R1 = _A_B * (1 - _E2_B) / (1 - _E2_B * sin_p * sin_p) ** 1.5
    D = x / (N1 * _K0)
    D2 = D * D
    D3 = D2 * D
    D4 = D3 * D
    D5 = D4 * D
    D6 = D5 * D

    lat = phi1 - (N1 * tan_p / R1) * (
        D2 / 2.0
        - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1 * C1 - 9.0 * e_p2) * D4 / 24.0
        + (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1 * T1 - 252.0 * e_p2 - 3.0 * C1 * C1)
        * D6 / 720.0
    )
    lon = _LON0 + (
        D
        - (1.0 + 2.0 * T1 + C1) * D3 / 6.0
        + (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1 * C1 + 8.0 * e_p2 + 24.0 * T1 * T1)
        * D5 / 120.0
    ) / cos_p
    return lat, lon


def _geo_to_ecef(lat: float, lon: float, h: float, a: float, e2: float):
    sin_p = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_p * sin_p)
    X = (N + h) * math.cos(lat) * math.cos(lon)
    Y = (N + h) * math.cos(lat) * math.sin(lon)
    Z = (N * (1 - e2) + h) * sin_p
    return X, Y, Z


def _ecef_to_geo(X: float, Y: float, Z: float, a: float, e2: float):
    p = math.sqrt(X * X + Y * Y)
    lon = math.atan2(Y, X)
    lat = math.atan2(Z, p * (1 - e2))
    for _ in range(5):
        sin_p = math.sin(lat)
        N = a / math.sqrt(1 - e2 * sin_p * sin_p)
        h = p / math.cos(lat) - N
        lat = math.atan2(Z, p * (1 - e2 * N / (N + h)))
    return lat, lon


def tm5174_to_wgs84(X: float, Y: float) -> tuple[float, float]:
    """EPSG:5174 (X, Y) [m] → WGS84 (lat°, lon°). Bessel 1841 + datum shift."""
    lat_b, lon_b = _tm_inverse(X, Y)
    Xe, Ye, Ze = _geo_to_ecef(lat_b, lon_b, 0.0, _A_B, _E2_B)
    Xe += _DX
    Ye += _DY
    Ze += _DZ
    lat_w, lon_w = _ecef_to_geo(Xe, Ye, Ze, _A_W, _E2_W)
    return math.degrees(lat_w), math.degrees(lon_w)


# ──────────────────────────────────────────────────────────────────────
# EPSG:5181 (Korea 2000 / Central Belt, GRS80)
# Datum: Korea 2000 ≈ ITRF2000 ≈ WGS84 (m 단위 차이 무시 가능)
# Projection: TM, central meridian 127.0°, FE=200000, FN=500000, k0=1.0
# ──────────────────────────────────────────────────────────────────────

_LAT0_5181 = math.radians(38.0)
_LON0_5181 = math.radians(127.0)  # 5174와 다름 (5174는 127.00289°)


def _meridional_arc_grs80(lat: float) -> float:
    """GRS80 타원체 자오선 호."""
    n = _F_W / (2.0 - _F_W)
    n2 = n * n
    n3 = n2 * n
    n4 = n3 * n
    A = (_A_W / (1.0 + n)) * (1.0 + n2 / 4.0 + n4 / 64.0)
    B = (3.0 * n / 2.0) * (1.0 - n2 / 8.0)
    C = (15.0 * n2 / 16.0) * (1.0 - n2 / 4.0)
    D = 35.0 * n3 / 48.0
    E = 315.0 * n4 / 512.0
    return A * (
        lat
        - B * math.sin(2 * lat)
        + C * math.sin(4 * lat)
        - D * math.sin(6 * lat)
        + E * math.sin(8 * lat)
    )


def tm5181_to_wgs84(X: float, Y: float) -> tuple[float, float]:
    """EPSG:5181 (X, Y) [m] → WGS84 (lat°, lon°). GRS80, datum shift 무시."""
    x = X - _FE
    y = Y - _FN
    M0 = _meridional_arc_grs80(_LAT0_5181)
    M = y / _K0 + M0

    e1 = (1 - math.sqrt(1 - _E2_W)) / (1 + math.sqrt(1 - _E2_W))
    mu = M / (
        _A_W
        * (1.0 - _E2_W / 4.0 - 3.0 * _E2_W**2 / 64.0 - 5.0 * _E2_W**3 / 256.0)
    )
    phi1 = (
        mu
        + (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8 * mu)
    )
    sin_p = math.sin(phi1)
    cos_p = math.cos(phi1)
    tan_p = math.tan(phi1)

    N1 = _A_W / math.sqrt(1 - _E2_W * sin_p * sin_p)
    T1 = tan_p * tan_p
    e_p2 = _E2_W / (1 - _E2_W)
    C1 = e_p2 * cos_p * cos_p
    R1 = _A_W * (1 - _E2_W) / (1 - _E2_W * sin_p * sin_p) ** 1.5
    D = x / (N1 * _K0)
    D2 = D * D
    D3 = D2 * D
    D4 = D3 * D
    D5 = D4 * D
    D6 = D5 * D

    lat = phi1 - (N1 * tan_p / R1) * (
        D2 / 2.0
        - (5.0 + 3.0 * T1 + 10.0 * C1 - 4.0 * C1 * C1 - 9.0 * e_p2) * D4 / 24.0
        + (61.0 + 90.0 * T1 + 298.0 * C1 + 45.0 * T1 * T1 - 252.0 * e_p2 - 3.0 * C1 * C1)
        * D6 / 720.0
    )
    lon = _LON0_5181 + (
        D
        - (1.0 + 2.0 * T1 + C1) * D3 / 6.0
        + (5.0 - 2.0 * C1 + 28.0 * T1 - 3.0 * C1 * C1 + 8.0 * e_p2 + 24.0 * T1 * T1)
        * D5 / 120.0
    ) / cos_p
    return math.degrees(lat), math.degrees(lon)


if __name__ == "__main__":
    # 검증: 알려진 기준점
    samples = [
        (198091.0, 552762.0, "서울 광화문 부근", 37.5759, 126.9769),
        (197879.5, 551701.0, "서울 시청 부근", 37.5663, 126.9779),
        (202576.0, 544147.0, "강남역 부근", 37.4981, 127.0276),
    ]
    print(f"{'기준':<20} {'예상':>22}  {'결과':>22}  diff(m)")
    for X, Y, name, e_lat, e_lon in samples:
        lat, lon = tm5174_to_wgs84(X, Y)
        d_lat_m = (lat - e_lat) * 111320
        d_lon_m = (lon - e_lon) * 88322
        d = (d_lat_m**2 + d_lon_m**2) ** 0.5
        print(
            f"{name:<20} ({e_lat:.4f}, {e_lon:.4f})  "
            f"({lat:.4f}, {lon:.4f})  {d:6.1f}"
        )
