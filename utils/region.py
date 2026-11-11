"""지역 단위 열선 배치 분석 — 데이터 점검 · 그래프 구성 · 예산 스윕.

흐름
----
1. **데이터 찾기**  앱 data/ → 로컬 폴더(다운로드 받은 자료) 순으로 찾는다.
   - 서울 전역 링크: seoul_links / seoul_edges / seoul_link_geom (.parquet)
     → 열선 설치구간(heat) · 경사도(slope_pct) · 링크 중점 표고(z)
   - 유동인구: B030 형식 CSV(rd_link_cd · flpop_co …) 아무 이름이나
2. **결합 점검**  지역의 링크에 경사·표고·유동인구가 얼마나 붙는지 본다.
3. **단계 판정**
   - 0단계: 경사·표고가 없음 → 열선 위치만
   - 1단계: 경사·표고(고도차)까지 → 위험구간 시각화
   - 2단계: 유동인구까지 → MILP 최적화 (발표자료 9쪽 모형)
4. **예산 스윕**  δ를 고정하고 예산 격자마다 MILP를 풀어 슬라이더로 본다.

MILP 자체는 utils.milp 를 그대로 호출한다(발표 숫자와 같은 코드).
성북구 4개동 프리셋은 road_graph.json(정밀 열선 매칭)을 그대로 써서
발표 수치를 재현하고, 그 밖의 지역은 seoul_*.parquet 에서 그래프를 만든다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from utils import milp as M
from utils.risk import obs_risk

APP_ROOT = Path(__file__).resolve().parent.parent
DATA = APP_ROOT / "data"
PROJECT_ROOT = APP_ROOT.parent          # road-analysis/
FLOW_UPLOAD_DIR = DATA / "flow_uploads"

#: 발표 기준 대상지
PRESET_GU = "성북구"
PRESET_DONGS = ("정릉2동", "정릉3동", "성북동", "돈암2동")
PRESET_LABEL = "성북구 4개동 (발표 기준: 정릉2·정릉3·성북·돈암2)"

#: 위험군 경사 — 「장애인등편의법 시행규칙」 별표1 접근로 기울기 1/18
STEEP_PCT = 100 / 18

SEOUL_FILES = ("seoul_links.parquet", "seoul_edges.parquet", "seoul_link_geom.parquet")

#: 로컬 폴더 스캔에서 건너뛸 디렉터리 (대용량·환경 폴더)
SKIP_DIRS = {
    ".git", "venv", ".venv", "__pycache__", "node_modules", ".ipynb_checkpoints",
    "수치지형도", "서울시 수치지형도 파일", "slope_raw", "shp", "out",
}

#: 단계 판정 기준 — 골목길 링크 중 해당 값이 붙은 비율
LEVEL_MIN_COVERAGE = 0.5

#: 유동인구 필터 (scripts/build_road_graph.py 와 동일)
FLOW_DAY_CODE = 1            # 평일
FLOW_TIME_CODES = (1, 2)     # 06~11시
FLOW_AGE_TOTAL = 0           # '전체' 행 — 10~60 행의 합이라 이것만 쓴다

#: 큰 지역에서 current-flow source 상한 (N × source 행렬 크기 제한)
LARGE_REGION_NODES = 3000
LARGE_REGION_N_SRC = 400


# ══════════════════════════════════════════════════════════════════════
# 1. 로컬 데이터 찾기
# ══════════════════════════════════════════════════════════════════════
def _walk(root: Path, max_depth: int = 4) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except (PermissionError, OSError):
            continue
        for p in entries:
            if p.is_dir():
                if p.name in SKIP_DIRS or p.name.startswith(".") or depth >= max_depth:
                    continue
                stack.append((p, depth + 1))
            elif p.is_file():
                yield p


def find_seoul_files(roots: list[Path]) -> dict[str, Path | None]:
    """seoul_*.parquet 위치. data/ 를 먼저 보고, 없으면 로컬 폴더를 뒤진다."""
    out: dict[str, Path | None] = {n: (DATA / n if (DATA / n).exists() else None)
                                   for n in SEOUL_FILES}
    need = {n for n, p in out.items() if p is None}
    if need:
        for r in roots:
            for p in _walk(r):
                if p.name in need and out[p.name] is None:
                    out[p.name] = p
            if all(out[n] for n in need):
                break
    return out


_LINK_COLS = {"rd_link_cd", "rd_link_id", "도로링크_ID(RD_LINK_ID)"}
_FLOW_COLS = {"flpop_co", "유동인구_수(FLPOP_CO)"}
#: B030 원자료에만 있는 코드 컬럼 — 분석 산출물 CSV(link_id·flow 컬럼)를 걸러낸다
_CODE_COLS = {"dayweek_cd", "tmzon_cd", "agrde_cd",
              "요일코드(DAYWEEK_CD)", "시간대코드(TMZON_CD)", "연령대코드(AGRDE_CD)"}


def sniff_flow_csv(path: Path) -> dict | None:
    """B030 형식 유동인구 CSV면 {path, encoding, size_mb} 반환. 헤더만 읽는다."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            cols = set(pd.read_csv(path, encoding=enc, nrows=0).columns)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        except Exception:
            return None
        if cols & _LINK_COLS and cols & _FLOW_COLS and cols & _CODE_COLS:
            return {"path": path, "encoding": enc,
                    "size_mb": round(path.stat().st_size / 1e6, 2)}
        return None
    return None


def scan_flow_files(roots: list[Path]) -> list[dict]:
    """앱 data/ 와 로컬 폴더에서 유동인구 CSV를 찾는다."""
    seen, out = set(), []
    for r in [DATA, *roots]:
        for p in _walk(r):
            if p.suffix.lower() != ".csv":
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            info = sniff_flow_csv(p)
            if info:
                out.append(info)
    return out


# ══════════════════════════════════════════════════════════════════════
# 2. 유동인구
# ══════════════════════════════════════════════════════════════════════
def load_flow(paths: list[Path | str]) -> tuple[pd.DataFrame, dict]:
    """여러 B030 CSV를 합쳐 링크별 유동인구로 집계.

    반환 df 컬럼: link_id(str) · gu · dong · flow_sum
    필터는 발표 모형과 같다 — 평일 · 06~11시 · 연령 '전체' 행.
    같은 자료가 두 번 들어와도(예: flpop.csv 와 캠퍼스 발췌본) 키 기준으로
    중복을 지워 이중 합산하지 않는다.
    """
    from utils.foot_traffic import load_foot_traffic_csv

    frames, note = [], {"files": [], "filters": []}
    for p in paths:
        try:
            d = load_foot_traffic_csv(Path(p))
        except Exception as e:  # 깨진 파일은 건너뛴다
            note["files"].append(f"{Path(p).name}: 읽기 실패 ({e})")
            continue
        if "link_id" not in d.columns or "flow" not in d.columns:
            note["files"].append(f"{Path(p).name}: 형식 불일치")
            continue
        note["files"].append(f"{Path(p).name}: {len(d):,}행")
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["link_id", "gu", "dong", "flow_sum"]), note

    df = pd.concat(frames, ignore_index=True)
    df["link_id"] = df["link_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    key = [c for c in ("ym", "link_id", "dong", "day_code", "age_code", "time_code")
           if c in df.columns]
    df = df.drop_duplicates(subset=key, keep="first")

    if "day_code" in df.columns and (df.day_code == FLOW_DAY_CODE).any():
        df = df[df.day_code == FLOW_DAY_CODE]
        note["filters"].append("평일")
    if "time_code" in df.columns and df.time_code.isin(FLOW_TIME_CODES).any():
        df = df[df.time_code.isin(FLOW_TIME_CODES)]
        note["filters"].append("06~11시")
    else:
        note["filters"].append("시간대 전체(06~11시 코드 없음)")
    if "age_code" in df.columns:
        if (df.age_code == FLOW_AGE_TOTAL).any():
            df = df[df.age_code == FLOW_AGE_TOTAL]
            note["filters"].append("연령 '전체' 행")
        else:
            df = df[df.age_code != FLOW_AGE_TOTAL]
            note["filters"].append("연령대 합산")

    for c in ("gu", "dong"):
        if c not in df.columns:
            df[c] = None
    g = (df.groupby("link_id", as_index=False)
           .agg(flow_sum=("flow", "sum"), gu=("gu", "first"), dong=("dong", "first")))
    g["flow_sum"] = g.flow_sum.astype(float).round(3)
    return g[["link_id", "gu", "dong", "flow_sum"]], note


# ══════════════════════════════════════════════════════════════════════
# 3. 서울 전역 링크 + 행정동
# ══════════════════════════════════════════════════════════════════════
def load_seoul(files: dict[str, Path | None]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    L = pd.read_parquet(files["seoul_links.parquet"])
    L["lid"] = L.road_lid.astype(str)
    E = pd.read_parquet(files["seoul_edges.parquet"])
    geom = None
    if files.get("seoul_link_geom.parquet"):
        geom = pd.read_parquet(files["seoul_link_geom.parquet"], columns=["road_lid", "xs", "ys"])
        geom["lid"] = geom.road_lid.astype(str)
        geom = geom.set_index("lid")
    return L, E, geom


def load_dong_polygons(data_dir: Path = DATA) -> dict[tuple[str, str], list]:
    """data/*_dong.geojson → {(구, 동): [외곽 링 좌표 배열, ...]}."""
    out: dict[tuple[str, str], list] = {}
    for f in sorted(data_dir.glob("*_dong.geojson")):
        try:
            gj = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for ft in gj.get("features", []):
            pr = ft.get("properties", {})
            name = (pr.get("adm_nm") or "").split()
            if len(name) < 2:
                continue
            gu, dong = (pr.get("sggnm") or name[-2]), name[-1]
            g = ft.get("geometry") or {}
            polys = g.get("coordinates", [])
            if g.get("type") == "Polygon":
                polys = [polys]
            out[(gu, dong)] = [np.asarray(p[0]) for p in polys]
    return out


def assign_dong(L: pd.DataFrame, polys: dict, flow: pd.DataFrame | None) -> pd.Series:
    """링크별 행정동. 유동인구 자료의 동 이름 > 행정동 경계 폴리곤 순."""
    from matplotlib.path import Path as MPath

    dong = pd.Series(index=L.index, dtype=object)
    for (gu, dn), rings in polys.items():
        m = (L.gu == gu).to_numpy()
        if not m.any():
            continue
        pts = L.loc[m, ["lon", "lat"]].to_numpy()
        hit = np.zeros(len(pts), bool)
        for r in rings:
            hit |= MPath(r).contains_points(pts)
        idx = L.index[m][hit]
        dong.loc[idx] = dn
    if flow is not None and len(flow):
        fm = flow.dropna(subset=["dong"]).set_index("link_id").dong
        hit = L.lid.map(fm)
        dong = hit.where(hit.notna(), dong)
    return dong


def region_catalog(L: pd.DataFrame, dong: pd.Series, flow: pd.DataFrame | None) -> pd.DataFrame:
    """자치구 × 행정동 목록과 각 동의 데이터 보유 현황(선택 UI용)."""
    d = pd.DataFrame({"gu": L.gu, "dong": dong, "golmok": L.road_type == "골목길",
                      "has_flow": L.lid.isin(set(flow.link_id)) if flow is not None else False})
    d = d.dropna(subset=["gu"])
    g = (d[d.golmok].groupby(["gu", d.dong.fillna("(동 미상)")])
           .agg(골목길=("golmok", "size"), 유동인구링크=("has_flow", "sum"))
           .reset_index().rename(columns={"dong": "dong"}))
    g.columns = ["gu", "dong", "골목길", "유동인구링크"]
    g["유동인구커버%"] = (100 * g.유동인구링크 / g.골목길).round(1)
    return g


# ══════════════════════════════════════════════════════════════════════
# 4. 지역 그래프
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Region:
    gu: str
    dongs: tuple[str, ...] = ()
    preset: bool = False

    @property
    def label(self) -> str:
        if self.preset:
            return PRESET_LABEL
        return f"{self.gu} " + ("· ".join(self.dongs) if self.dongs else "전체")


def build_region_graph(region: Region, L: pd.DataFrame, E: pd.DataFrame,
                       dong: pd.Series, flow: pd.DataFrame | None,
                       use_precise: bool = True,
                       flow_links_only: bool = False) -> nx.Graph:
    """지역 그래프. 노드 속성은 road_graph.json 과 같은 이름을 쓴다
    (utils.milp 함수를 그대로 쓰기 위해서).

    - 프리셋(4개동): road_graph.json 그대로 — 발표 수치 재현
    - 그 밖: seoul_links/edges 에서 잘라낸다. road_graph.json 에 들어 있는
      링크는 정밀 열선 매칭(heating_installed_precise)으로 덮어쓴다.
    - flow_links_only: 유동인구 자료가 있는 링크만 남긴다. 발표 모형의
      노드 유니버스와 같은 방식(B030 이 잡지 않은 링크는 통행 0으로 두지 않고
      분석에서 뺀다).
    """
    fmap = (dict(zip(flow.link_id, flow.flow_sum)) if flow is not None and len(flow) else {})

    if region.preset and (DATA / "road_graph.json").exists():
        G = M.load_graph()
        for n in G.nodes():
            nd = G.nodes[n]
            lid = str(nd.get("link_id") or n)
            nd["has_flow"] = lid in fmap or (nd.get("flow_sum") or 0) > 0
            if lid in fmap:
                nd["flow_sum"] = fmap[lid]
        _attach_lid_index(G, L)
        return G

    m = (L.gu == region.gu).to_numpy()
    if region.dongs:
        m &= dong.isin(region.dongs).to_numpy()
    if flow_links_only and fmap:
        m &= L.lid.isin(fmap).to_numpy()
    sub = L.loc[m]
    G = nx.Graph()
    for i, r in sub.iterrows():
        lid = r.lid
        G.add_node(lid, link_id=lid, gu=r.gu, dong=dong.get(i),
                   flow_sum=float(fmap.get(lid, 0.0)), has_flow=lid in fmap,
                   slope_pct=None if pd.isna(r.slope_pct) else float(r.slope_pct),
                   z=None if pd.isna(r.z) else float(r.z),
                   road_type=r.road_type, length_m=float(r.dist_m or 0) if not pd.isna(r.dist_m) else 0.0,
                   lon=float(r.lon), lat=float(r.lat),
                   heating_installed_precise=bool(r.heat),
                   heat_year=None if pd.isna(r.heat_year) else int(r.heat_year),
                   _ix=int(i))
    ix2lid = dict(zip(sub.index, sub.lid))
    se = E[E.u.isin(ix2lid) & E.v.isin(ix2lid)]
    G.add_edges_from((ix2lid[u], ix2lid[v]) for u, v in se.itertuples(index=False, name=None))

    if use_precise and (DATA / "road_graph.json").exists():
        P = nx.node_link_graph(json.load(open(DATA / "road_graph.json")), edges="links")
        for n in P.nodes():
            lid = str(P.nodes[n].get("link_id") or n)
            if lid in G:
                G.nodes[lid]["heating_installed_precise"] = bool(
                    P.nodes[n].get("heating_installed_precise"))
        for u, v in P.edges():
            if str(u) in G and str(v) in G:
                G.add_edge(str(u), str(v))
    return G


def _attach_lid_index(G: nx.Graph, L: pd.DataFrame) -> None:
    pos = dict(zip(L.lid, L.index))
    for n in G.nodes():
        G.nodes[n]["_ix"] = pos.get(str(G.nodes[n].get("link_id") or n))


# ══════════════════════════════════════════════════════════════════════
# 5. 점검 · 단계 판정
# ══════════════════════════════════════════════════════════════════════
def downhill_pairs(G: nx.Graph, h_min: float = M.H_MIN) -> list[tuple]:
    """(상류 i, 하류 j, 낙차 m) — 인접하고 표고차가 h_min 이상인 모든 쌍."""
    out = []
    for u, v in G.edges():
        zu, zv = G.nodes[u].get("z"), G.nodes[v].get("z")
        if zu is None or zv is None:
            continue
        d = zu - zv
        if abs(d) < h_min:
            continue
        out.append((u, v, d) if d > 0 else (v, u, -d))
    return out


def check_availability(G: nx.Graph, h_min: float = M.H_MIN) -> dict:
    nodes = list(G.nodes())
    gm = [n for n in nodes if G.nodes[n].get("road_type") == "골목길"]
    base = gm or nodes
    cov = lambda key: (sum(1 for n in base if G.nodes[n].get(key) is not None) / len(base)) if base else 0.0
    heat = [n for n in nodes if G.nodes[n].get("heating_installed_precise")]
    flow_cov = (sum(1 for n in base if G.nodes[n].get("has_flow")) / len(base)) if base else 0.0
    pairs = downhill_pairs(G, h_min)
    hs = set(M.installed_links(G, golmok_only=True))
    bnd = [(i, j) for i, j, _ in pairs if i in hs and j not in hs]

    slope_c, z_c = cov("slope_pct"), cov("z")
    if slope_c < LEVEL_MIN_COVERAGE or z_c < LEVEL_MIN_COVERAGE:
        level = 0
    elif flow_cov < LEVEL_MIN_COVERAGE:
        level = 1
    else:
        level = 2
    return {
        "links": len(nodes), "golmok": len(gm), "edges": G.number_of_edges(),
        "heat_links": len(heat),
        "heat_len_m": sum(G.nodes[n].get("length_m") or 0 for n in heat),
        "slope_cov": slope_c, "z_cov": z_c, "flow_cov": flow_cov,
        "downhill_pairs": len(pairs), "boundary_pairs": len(bnd),
        "level": level,
    }


def risk_table(G: nx.Graph, h_min: float = M.H_MIN) -> pd.DataFrame:
    """링크별 위험 지표 — 1단계(유동인구 없음) 시각화용."""
    pairs = downhill_pairs(G, h_min)
    heat = {n for n in G.nodes() if G.nodes[n].get("heating_installed_precise")}
    down_of_heat = {j for i, j, _ in pairs if i in heat and j not in heat}
    max_drop: dict = {}
    for i, j, d in pairs:
        max_drop[j] = max(max_drop.get(j, 0.0), d)
    rows = []
    for n in G.nodes():
        a = G.nodes[n]
        s = a.get("slope_pct")
        f = a.get("flow_sum") or 0.0
        r = obs_risk(s) if s is not None else None
        rows.append({
            "link_id": str(a.get("link_id") or n), "dong": a.get("dong"),
            "road_type": a.get("road_type"), "length_m": a.get("length_m") or 0.0,
            "slope_pct": s, "z": a.get("z"), "risk": r,
            "steep": (s or 0) >= STEEP_PCT, "heat": n in heat,
            "boundary_down": n in down_of_heat,
            "max_inflow_drop_m": max_drop.get(n, 0.0),
            "flow": f if a.get("has_flow") else None,
            "expected_slips": (r * f) if (r is not None and a.get("has_flow")) else None,
            "lon": a.get("lon"), "lat": a.get("lat"), "_ix": a.get("_ix"),
        })
    return pd.DataFrame(rows)


def boundary_pair_table(G: nx.Graph, h_min: float = M.H_MIN) -> pd.DataFrame:
    """현행 열선 경계쌍 — 상류 골목길 열선 · 하류 미설치 · 낙차 ≥ h_min.
    plan_metrics(현행) 의 위험경계와 같은 정의(발표자료 10쪽 17곳)."""
    heat = set(M.installed_links(G, golmok_only=True))
    rows = []
    for i, j, d in downhill_pairs(G, h_min):
        if i in heat and j not in heat:
            a, b = G.nodes[i], G.nodes[j]
            rows.append({"up": i, "down": j, "drop_m": round(d, 2),
                         "down_slope": b.get("slope_pct"),
                         "down_risk": obs_risk(b.get("slope_pct")) if b.get("slope_pct") is not None else None,
                         "down_flow": b.get("flow_sum") if b.get("has_flow") else None,
                         "lon": b.get("lon"), "lat": b.get("lat"), "dong": b.get("dong")})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 6. 최적화
# ══════════════════════════════════════════════════════════════════════
@dataclass
class OptConfig:
    delta: float = 1.0
    h_min: float = M.H_MIN
    cost_per_m: float = M.COST_PER_M
    weight_mode: str = "current"    # "current"(엣지전류·MSA) | "simple"(하류 risk×flow)
    msa_iters: int = 15   # 4개동 S2(+30%, δ=5)가 10회에 수렴 — 8회는 부족
    time_limit: float | None = None
    n_src: int | None = None


@dataclass
class OptContext:
    G: nx.Graph
    cfg: OptConfig
    cand: list
    risk: dict
    flow: dict
    slope: dict
    D: list
    D_all: list                      # 지역 내 모든 하류쌍 (경계 집계용)
    full_cost: float                 # 후보 전면 시공비(원)
    total_risk: float                # 후보군 기대 미끄러짐 합
    current: dict                    # 현행 열선 지표
    _flow_graph: tuple | None = None
    _w0: dict | None = None
    _ref: float | None = None
    _k0: float | None = None          # 엣지전류 → 전이 가중치 고정 척도 (평가용)
    cache: dict = field(default_factory=dict)


def prepare(G: nx.Graph, cfg: OptConfig) -> OptContext:
    cand = M.candidates(G)
    risk, flow, slope = M.risk_flow(G)
    D = M.direction_set(G, cand, cfg.h_min)
    D_all = [(i, j) for i, j, _ in downhill_pairs(G, cfg.h_min)]
    full = sum(G.nodes[n].get("length_m") or 0 for n in cand) * cfg.cost_per_m
    tot = sum(risk[n] * flow[n] for n in cand)
    cur_links = set(M.installed_links(G, golmok_only=True))
    ctx = OptContext(G, cfg, cand, risk, flow, slope, D, D_all, full, tot, {})
    cur_len = sum(G.nodes[n].get("length_m") or 0 for n in cur_links)
    ctx.current = plan_metrics(ctx, cur_links, cur_len * cfg.cost_per_m)
    if cfg.n_src is None and G.number_of_nodes() > LARGE_REGION_NODES:
        cfg.n_src = LARGE_REGION_N_SRC
    return ctx


def _model_budget(ctx: OptContext, B_won: float) -> float:
    """milp.solve 는 모듈 단가(COST_PER_M)로 예산식을 쓴다. 사용자가 단가를
    바꾸면 예산을 같은 비율로 환산해 넘긴다 (Σℓ·c·x ≤ B ⇔ Σℓ·c₀·x ≤ B·c₀/c)."""
    return B_won * M.COST_PER_M / ctx.cfg.cost_per_m


def _ensure_flow_graph(ctx: OptContext) -> None:
    """current-flow 전파 그래프와 초기 가중치(결빙 없음)를 한 번만 만든다.
    예산과 무관하므로 스윕 전체에서 공유한다 (milp.solve_msa 와 같은 계산)."""
    if ctx._w0 is not None:
        return
    G = ctx.G
    try:
        Gg, nodes, idx, edges, ep = M.build_flow_graph(G, M.FLOW_BUFFER_HOPS)
    except FileNotFoundError:
        Gg, nodes, idx, edges, ep, _ = M._largest_component(G, [])
    Dg = [(i, j) for (i, j) in ctx.D if (i, j) in ep]
    fl = np.array([(Gg.nodes[n].get("flow_sum") or 0.0) for n in nodes])
    ref = float(np.mean([ctx.risk[j] * ctx.flow[j] for (i, j) in Dg])) if Dg else 0.0
    ctx._flow_graph = (Gg, nodes, idx, edges, ep, Dg, fl)
    ctx._ref = ref
    ctx._w0 = _make_w(ctx, frozenset())


def _make_w(ctx: OptContext, ice) -> dict:
    Gg, nodes, idx, edges, ep, Dg, fl = ctx._flow_graph
    if not Dg:
        return {}
    acc = M.edge_currents(Gg, nodes, idx, edges, fl, ice, n_src=ctx.cfg.n_src)
    raw = {k: ctx.risk[k[1]] * acc[ep[k]] for k in Dg}
    s = float(np.mean(list(raw.values())))
    if s <= 0:
        return {k: 0.0 for k in Dg}
    return {k: v * ctx._ref / s for k, v in raw.items()}


def solve_budget(ctx: OptContext, B_won: float, delta: float | None = None) -> dict:
    """예산 B(원)·δ 한 점. δ=0 이면 가중치가 필요 없어 1회 풀이로 끝난다."""
    cfg = ctx.cfg
    delta = cfg.delta if delta is None else delta
    key = (round(B_won), round(delta, 6), cfg.weight_mode, cfg.h_min, cfg.cost_per_m)
    if key in ctx.cache:
        return ctx.cache[key]
    Bm = _model_budget(ctx, B_won)
    G, cand = ctx.G, ctx.cand
    tl = cfg.time_limit

    if delta == 0 or cfg.weight_mode == "simple":
        S, _ = M.solve(G, cand, ctx.D, ctx.risk, ctx.flow, Bm, delta, time_limit=tl)
        iters, conv = 0, True
    else:
        # utils.milp.solve_msa 와 같은 MSA 고정점 (λ = 1/(k+2)).
        _ensure_flow_graph(ctx)
        Gg, nodes, idx, edges, ep, Dg, fl = ctx._flow_graph
        w = dict(ctx._w0)
        prev, conv, iters = None, False, cfg.msa_iters
        S = set()
        for it in range(cfg.msa_iters):
            S, _ = M.solve(G, cand, Dg, ctx.risk, ctx.flow, Bm, delta, pen=w, time_limit=tl)
            if prev is not None and S == prev:
                conv, iters = True, it
                break
            prev = S
            ice = frozenset(n for n in cand if n not in S and n in Gg)
            wn = _make_w(ctx, ice)
            lam = 1.0 / (it + 2)
            w = {k: (1 - lam) * w[k] + lam * wn[k] for k in Dg}

    res = plan_metrics(ctx, S, B_won)
    res.update(delta=delta, msa_iters=iters, converged=conv)
    ctx.cache[key] = res
    return res


def _edge_scale(ctx: OptContext) -> float:
    """평가용 고정 척도 k0. 새 열선이 없는 상태(결빙 없음)의 w0 와 같은 척도:
    w0_ij = k0 · risk_j · I_ij  (I = 엣지전류). 모든 배치를 같은 k0 로 재야
    배치 간 비교가 된다(MSA 안의 w 는 반복마다 평균으로 다시 맞추므로 평가에 못 씀)."""
    _ensure_flow_graph(ctx)
    if ctx._k0 is None:
        Gg, nodes, idx, edges, ep, Dg, fl = ctx._flow_graph
        acc = M.edge_currents(Gg, nodes, idx, edges, fl, frozenset(), n_src=ctx.cfg.n_src)
        raw = [ctx.risk[j] * acc[ep[(i, j)]] for (i, j) in Dg]
        m = float(np.mean(raw)) if raw else 0.0
        ctx._k0 = (ctx._ref / m) if m > 0 else 0.0
    return ctx._k0


def edge_exposure(ctx: OptContext, S: set) -> tuple[float | None, dict]:
    """하류 전이위험 — 발표자료 10쪽 정의: Σ_(잔여 경계) 경계 통행량 × 하류 미끄러짐 확률.

    경계 통행량 = 배치 S 에서 결빙 구간(후보 중 미설치)의 컨덕턴스를 낮춘
    current-flow 엣지전류. 반환: (값, {경계쌍: 기여}) — 지도용.
    """
    key = ("edge", frozenset(S))
    if key in ctx.cache:
        return ctx.cache[key]
    try:
        k0 = _edge_scale(ctx)
    except Exception:
        return None, {}
    Gg, nodes, idx, edges, ep, Dg, fl = ctx._flow_graph
    ice = frozenset(n for n in ctx.cand if n not in S and n in Gg)
    acc = M.edge_currents(Gg, nodes, idx, edges, fl, ice, n_src=ctx.cfg.n_src)
    contrib = {(i, j): k0 * ctx.risk[j] * acc[ep[(i, j)]]
               for (i, j) in ctx.D_all if i in S and j not in S and (i, j) in ep}
    out = (float(sum(contrib.values())), contrib)
    ctx.cache[key] = out
    return out


def boundary_currents(ctx: OptContext) -> dict:
    """모든 하류쌍 D 의 경계 통행량(엣지전류, 결빙 없음 기준) — 네트워크 전이 지도용."""
    _edge_scale(ctx)
    Gg, nodes, idx, edges, ep, Dg, fl = ctx._flow_graph
    key = ("acc0",)
    if key not in ctx.cache:
        ctx.cache[key] = M.edge_currents(Gg, nodes, idx, edges, fl, frozenset(),
                                         n_src=ctx.cfg.n_src)
    acc = ctx.cache[key]
    return {(i, j): float(acc[ep[(i, j)]]) for (i, j) in ctx.D_all if (i, j) in ep}


def plan_metrics(ctx: OptContext, S: set, B_won: float) -> dict:
    """배치 S의 성과지표.

    경사위험 방지          = Σ_{i∈S} risk_i·flow_i (기대 미끄러짐 방지 건수)
    위험경계               = #{(i,j): i∈S 상류, j∉S 하류, 낙차 ≥ h_min}
    하류 전이위험(경계 통행) = Σ_{위험경계} 경계 통행량 × risk_j  ← 발표 10쪽 정의·목적함수와 같은 양
    하류 전이위험(하류 통행) = Σ_{위험경계} risk_j·flow_j         ← 발표 10쪽 표의 수치가 이 값
    """
    G, risk, flow = ctx.G, ctx.risk, ctx.flow
    S = set(S)
    ln = sum(G.nodes[n].get("length_m") or 0 for n in S)
    ben = sum(risk[n] * flow[n] for n in S)
    bnd = [(i, j) for (i, j) in ctx.D_all if i in S and j not in S]
    expo = sum(risk[j] * flow[j] for (i, j) in bnd)
    expo_edge, _ = edge_exposure(ctx, S)
    golmok_len = sum(G.nodes[n].get("length_m") or 0 for n in ctx.cand)
    return {
        "selected": S,
        "boundaries": bnd,
        "budget_won": B_won,
        "n": len(S),
        "length_m": ln,
        "spent_won": ln * ctx.cfg.cost_per_m,
        "benefit": ben,
        "boundary": len(bnd),
        "exposure": expo,
        "exposure_edge": expo_edge,
        "mean_slope": float(np.mean([ctx.slope[n] for n in S])) if S else None,
        "mean_flow": float(np.mean([flow[n] for n in S])) if S else None,
        "coverage_len_pct": 100 * ln / golmok_len if golmok_len else None,
        "coverage_risk_pct": 100 * ben / ctx.total_risk if ctx.total_risk else None,
        "won_per_slip": (ln * ctx.cfg.cost_per_m / ben) if ben else None,
    }


def _r1(v):
    return None if v is None else round(v, 1)


def delta_sweep(ctx: OptContext, B_won: float, deltas: list[float],
                progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
    """예산 고정, δ 격자 스윕 (발표자료 11쪽의 한 예산선)."""
    base = solve_budget(ctx, B_won, 0.0)
    rows = []
    for k, d in enumerate(deltas):
        if progress:
            progress(k / len(deltas), f"δ = {d:g} ({k+1}/{len(deltas)})")
        r = solve_budget(ctx, B_won, d)
        rows.append({
            "δ": d, "선정": r["n"], "연장m": round(r["length_m"]),
            "경사위험방지": round(r["benefit"], 1),
            "편익손실%": round(100 * (1 - r["benefit"] / base["benefit"]), 2) if base["benefit"] else 0.0,
            "위험경계": r["boundary"],
            "전이위험_경계통행": _r1(r["exposure_edge"]),
            "전이위험_하류통행": round(r["exposure"], 1),
            "MSA반복": r["msa_iters"], "수렴": r["converged"],
        })
    if progress:
        progress(1.0, "완료")
    df = pd.DataFrame(rows)
    for col in ("전이위험_경계통행", "전이위험_하류통행"):
        b0 = df.loc[df["δ"] == 0, col]
        b0 = float(b0.iloc[0]) if len(b0) and b0.iloc[0] else None
        df[col.replace("전이위험", "감소%")] = (
            (100 * (1 - df[col] / b0)).round(1) if b0 else 0.0)
    return df


def knee_delta(df: pd.DataFrame, col: str = "감소%_경계통행") -> float | None:
    """편익손실%–전이위험 감소% 곡선의 파레토 전환점 (Kneedle: 양 끝 현에서 가장 먼 점)."""
    d = df.dropna(subset=[col]).sort_values("δ")
    if len(d) < 3:
        return None
    x = d["편익손실%"].to_numpy(float); y = d[col].to_numpy(float)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    nrm = (dx * dx + dy * dy) ** 0.5
    if nrm == 0:
        return None
    dist = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / nrm
    return float(d["δ"].iloc[int(dist.argmax())])


def sweep(ctx: OptContext, budgets_won: list[float],
          progress: Callable[[float, str], None] | None = None) -> pd.DataFrame:
    """예산 격자 × (제안 δ, 기준 δ=0) 스윕. 결과 dict 는 ctx.cache 에 남는다."""
    rows = []
    n = len(budgets_won)
    for k, B in enumerate(budgets_won):
        if progress:
            progress(k / n, f"예산 {B/1e8:,.1f}억 ({k+1}/{n})")
        r = solve_budget(ctx, B)
        b0 = solve_budget(ctx, B, 0.0)
        rows.append({
            "예산억": round(B / 1e8, 2),
            "선정": r["n"], "연장m": round(r["length_m"]),
            "경사위험방지": round(r["benefit"], 1),
            "위험경계": r["boundary"],
            "전이위험_경계통행": _r1(r["exposure_edge"]), "전이위험_하류통행": round(r["exposure"], 1),
            "δ0_경사위험방지": round(b0["benefit"], 1),
            "δ0_위험경계": b0["boundary"],
            "δ0_전이위험_경계통행": _r1(b0["exposure_edge"]), "δ0_전이위험_하류통행": round(b0["exposure"], 1),
            "현행대비배": round(r["benefit"] / ctx.current["benefit"], 1) if ctx.current["benefit"] else None,
            "MSA반복": r["msa_iters"], "수렴": r["converged"],
            "_B": B,
        })
    if progress:
        progress(1.0, "완료")
    return pd.DataFrame(rows)


def default_budgets(ctx: OptContext, n: int = 20) -> list[float]:
    """기본 예산 격자 — 현행 투입액이 있으면 그 0.25~2.5배, 없으면 전면 시공비의 1~20%.
    현행 투입액은 격자에 정확히 넣는다."""
    cur = ctx.current["spent_won"]
    if cur > 0:
        lo, hi = 0.25 * cur, min(2.5 * cur, ctx.full_cost)
    else:
        lo, hi = 0.01 * ctx.full_cost, 0.20 * ctx.full_cost
    g = list(np.linspace(lo, hi, n))
    if cur > 0:
        g = sorted(set([round(x, -6) for x in g] + [cur, 1.3 * cur]))
    return [float(x) for x in g]


def strategy_compare(ctx: OptContext, B_won: float) -> pd.DataFrame:
    """같은 예산에서 선정 기준별 비교 (발표자료 10쪽)."""
    G = ctx.G
    ln = {n: (G.nodes[n].get("length_m") or 0.0) for n in ctx.cand}

    def greedy(key):
        s, left = set(), B_won
        for n in sorted(ctx.cand, key=key, reverse=True):
            c = ln[n] * ctx.cfg.cost_per_m
            if 0 < c <= left:
                s.add(n); left -= c
        return plan_metrics(ctx, s, B_won)

    rows = [("현행 실제", ctx.current),
            ("경사 기준", greedy(lambda n: ctx.slope[n])),
            ("통행량 기준", greedy(lambda n: ctx.flow[n])),
            ("위험×통행량 (δ=0)", solve_budget(ctx, B_won, 0.0)),
            (f"제안 (δ={ctx.cfg.delta:g})", solve_budget(ctx, B_won))]
    out = []
    for name, r in rows:
        out.append({"선정 기준": name, "선정": r["n"],
                    "연장m": round(r["length_m"]),
                    "평균경사%": round(r["mean_slope"], 2) if r["mean_slope"] is not None else None,
                    "평균통행": round(r["mean_flow"], 1) if r["mean_flow"] is not None else None,
                    "경사위험방지": round(r["benefit"], 1),
                    "위험경계": r["boundary"],
                    "전이위험(경계통행)": _r1(r["exposure_edge"]),
                    "전이위험(하류통행)": round(r["exposure"], 1)})
    return pd.DataFrame(out)
