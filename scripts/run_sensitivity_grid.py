"""예산 × δ 이중 탐색 (발표자료 11쪽 민감도 분석) — 앱 밖에서 미리 계산한다.

칸 하나가 MILP(δ>0이면 MSA 반복)라서 격자가 커지면 웹 요청 안에서 돌리기 어렵다.
이 스크립트로 계산해 두면 앱은 결과 파일을 읽어 그리기만 한다.

산출 (기본 data/)
-----------------
    sensitivity_grid.parquet   칸별 지표와 선정 링크 — 앱의 '이중 탐색' 탭에 올린다
    sensitivity_grid.json      계산 조건 (앱이 현재 설정과 같은지 비교)
    sensitivity_grid.html      그림 보고서 (브라우저로 바로 열림)

실행 예
-------
    cd streamlit_app
    python3 scripts/run_sensitivity_grid.py                       # 발표 대상지, 기본 격자
    python3 scripts/run_sensitivity_grid.py --budgets 40 --jobs 4 # 예산 40개, 4개 프로세스
    python3 scripts/run_sensitivity_grid.py --budget-range 5 50 --deltas 0,0.5,1,2,3,5,8,12
    python3 scripts/run_sensitivity_grid.py --gu 강북구 --dongs 수유1동,수유2동

설정(h_min, 설치비, 가중치 방식, MSA 반복)은 앱 기본값과 같다. 앱에서 값을 바꿨다면
같은 값을 옵션으로 넘겨야 앱이 결과를 '같은 조건'으로 인식한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402
from utils import region as R  # noqa: E402

DEFAULT_DELTAS = "0,0.25,0.5,0.75,1,1.5,2,3,4,5,7,10,12"


def build_ctx(a: argparse.Namespace):
    """앱(pages/optimize.py)과 같은 순서로 지역 그래프와 최적화 문맥을 만든다."""
    roots = [Path(a.data_root)]
    files = R.find_seoul_files(roots)
    if not files["seoul_links.parquet"] or not files["seoul_edges.parquet"]:
        sys.exit("seoul_links.parquet / seoul_edges.parquet 를 찾지 못했습니다 (--data-root 확인).")
    L, E, _ = R.load_seoul(files)
    flows = [str(f["path"]) for f in R.scan_flow_files(roots)]
    flow = R.load_flow(flows)[0] if flows else None
    dong = R.assign_dong(L, R.load_dong_polygons(), flow)
    preset = a.gu is None
    gu = R.PRESET_GU if preset else a.gu
    dongs = R.PRESET_DONGS if preset else tuple(x for x in (a.dongs or "").split(",") if x)
    region = R.Region(gu, dongs, preset)
    G = R.build_region_graph(region, L, E, dong, flow, flow_links_only=not preset and not a.all_links)
    cfg = R.OptConfig(h_min=a.h_min, cost_per_m=a.cost, weight_mode=a.weight,
                      msa_iters=a.msa, time_limit=a.time_limit or None, n_src=a.n_src or None)
    return R.prepare(G, cfg), region.label


_CTX = None


def _init(a):
    global _CTX
    _CTX, _ = build_ctx(a)


def _line(B, deltas):
    return R.grid_line(_CTX, B, deltas)


def budget_grid(ctx, a) -> list[float]:
    cur = ctx.current["spent_won"]
    if a.budget_range:
        lo, hi = (v * 1e8 for v in a.budget_range)
        grid = [round(float(x), -6) for x in np.linspace(lo, hi, a.budgets)]
    else:
        base = R.default_budgets(ctx, a.budgets)
        grid = [round(float(x), -6) for x in np.linspace(min(base), max(base), a.budgets)]
    return R.with_anchor_budgets(grid, cur)


def main() -> None:
    p = argparse.ArgumentParser(description="예산 × δ 이중 탐색 사전계산")
    p.add_argument("--gu", help="자치구 (생략하면 발표 대상지 성북구 4개동)")
    p.add_argument("--dongs", help="행정동, 쉼표로 구분 (생략하면 구 전체)")
    p.add_argument("--all-links", action="store_true", help="유동인구 없는 링크도 포함 (기본은 발표 방식대로 제외)")
    p.add_argument("--budgets", type=int, default=20, help="예산 격자 수 (기본 20)")
    p.add_argument("--budget-range", type=float, nargs=2, metavar=("LO", "HI"), help="예산 범위(억)")
    p.add_argument("--deltas", default=DEFAULT_DELTAS, help=f"δ 격자 (기본 {DEFAULT_DELTAS})")
    p.add_argument("--h-min", type=float, default=M.H_MIN)
    p.add_argument("--cost", type=float, default=M.COST_PER_M, help="미터당 설치비(원)")
    p.add_argument("--weight", choices=["current", "simple"], default="current")
    p.add_argument("--msa", type=int, default=R.OptConfig.msa_iters)
    p.add_argument("--time-limit", type=float, default=0)
    p.add_argument("--n-src", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1, help="병렬 프로세스 수")
    p.add_argument("--data-root", default=str(R.PROJECT_ROOT), help="로컬 데이터 폴더 (앱 사이드바와 같은 값)")
    p.add_argument("--out", default=str(R.DATA), help="출력 폴더 (기본 data/)")
    a = p.parse_args()

    t0 = time.time()
    deltas = sorted({0.0, *(float(v) for v in a.deltas.split(",") if v.strip())})
    ctx, label = build_ctx(a)
    budgets = budget_grid(ctx, a)
    n_cell = len(budgets) * len(deltas)
    print(f"{label} · 후보 {len(ctx.cand)} · 예산 {len(budgets)} × δ {len(deltas)} = {n_cell}칸")

    rows, done = [], 0
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs, initializer=_init, initargs=(a,)) as ex:
            futs = {ex.submit(_line, B, deltas): B for B in budgets}
            for f in as_completed(futs):
                rows += f.result(); done += 1
                print(f"  [{done}/{len(budgets)}] {futs[f]/1e8:,.2f}억  {time.time()-t0:,.0f}s", flush=True)
    else:
        for B in budgets:
            rows += R.grid_line(ctx, B, deltas); done += 1
            print(f"  [{done}/{len(budgets)}] {B/1e8:,.2f}억  {time.time()-t0:,.0f}s", flush=True)

    df = R.grid_finalize(pd.DataFrame(rows))
    summ = R.grid_summary(df)
    meta = R.grid_signature(ctx, label) | {
        "budgets": len(budgets), "deltas": deltas, "cells": n_cell,
        "current_budget_eok": round(ctx.current["spent_won"] / 1e8, 2),
        "created": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0),
    }

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pq, js, html = (out / "sensitivity_grid.parquet", out / "sensitivity_grid.json",
                    out / "sensitivity_grid.html")
    df.to_parquet(pq, index=False)
    js.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    from utils import sensitivity_viz as V
    html.write_text(V.report_html(df, summ, meta), encoding="utf-8")

    print(f"\n{n_cell}칸 · {time.time()-t0:,.0f}초")
    print(summ.drop(columns=["_B"]).to_string(index=False))
    print(f"\n저장: {pq}\n      {js}\n      {html}")


if __name__ == "__main__":
    main()
