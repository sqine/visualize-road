"""고해상도 예산 × δ 격자 (100 × 100) — 최적 운영점 탐색.

왜 고해상도인가
---------------
δ=2 는 11점 격자에서 Kneedle 로 뽑은 값일 뿐 원리적 근거가 없다.
예산과 δ를 촘촘히 훑어 **편익-경계노출 평면에서 실제 파레토 전선**을 그리고,
그 위에서 운영점을 고른다.

속도 설계 — 3단 가속
--------------------
한 점마다 MSA(고정점)를 돌리면 ~1.0s → 10,000점 = 3시간.

1. **예산마다 w_ij 를 한 번만 계산**(δ=1 기준 MSA)하고 그 예산의 δ 전체에
   재사용한다. 점당 MILP 1회로 줄어든다.
2. **δ축 파라메트릭 분할** — MILP 해는 δ에 대해 조각별 상수다. 구간 양 끝의
   해가 같으면 사이를 전부 그 해로 채우고, 다르면 반으로 쪼갠다. 실제 풀이
   횟수가 100 → 20~40회로 준다(결과는 100점 전부 기록).
3. **예산 축 4-프로세스 병렬** — 예산끼리 완전 독립이라 그대로 쪼갠다.

  · w 는 "경계를 넘는 사람 수"이며 δ 와 직접 관계가 없다.
  · 선정이 바뀌면 w 도 바뀌지만, 같은 예산 안에서 δ 변화로 인한 선정 변동은
    작다(δ=0↔10 에서 Jaccard 0.65 이상).
  · 최종 채택점은 **전체 MSA 로 재검증**한다(`--verify`).

실행
----
    cd streamlit_app
    python3 scripts/run_grid_hires.py              # 체크포인트 이어서
    python3 scripts/run_grid_hires.py --verify     # 채택점 전체 MSA 재검증

산출
----
    viz/data/p11_grid_hires.csv       격자 전체 (10,000행)
    viz/data/p11_pareto.csv           파레토 전선
    /tmp/grid_hires_state.pkl         체크포인트
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402

VIZ = ROOT / "viz" / "data"
STATE = Path("/tmp/grid_hires_state.pkl")

N_B, N_D = 100, 100
#: 예산 — 전면 시공비 대비 0.5~30%, 저예산 구간 해상도를 위해 기하수열
PCTS = np.geomspace(0.5, 30.0, N_B)
#: δ — 0 과 0.02~12 기하수열
DELTAS = np.concatenate([[0.0], np.geomspace(0.02, 12.0, N_D - 1)])

#: 1회 호출당 처리할 예산 개수 (bash 타임아웃 10분 안에 끝나는 크기)
CHUNK = 12
#: 병렬 워커
NPROC = 4


def setup():
    G = M.load_graph()
    cand = M.candidates(G)
    risk, flow, slope = M.risk_flow(G)
    D = M.direction_set(G, cand)
    ln = {n: (G.nodes[n].get("length_m") or 0.0) for n in G.nodes()}
    full = sum(ln[n] for n in cand) * M.COST_PER_M
    total = sum(risk[n] * flow[n] for n in cand)
    return G, cand, risk, flow, slope, D, ln, full, total


_CTX = None


def _one_budget(idx: int):
    """예산 1개에 대해 δ 격자 전체를 푼다 (δ축 파라메트릭 분할)."""
    G, cand, risk, flow, slope, D, ln, FULL, TOTAL = _CTX
    pct = float(PCTS[idx])
    B = FULL * pct / 100

    # 이 예산의 w 를 δ=1 기준 MSA 로 한 번만 확정
    ref = M.solve_msa(G, cand, D, risk, flow, B, 1.0)
    w, Dg = ref["w"], ref["D_used"]

    cache: dict[int, frozenset] = {}
    nsolve = 0

    def sol(k: int) -> frozenset:
        nonlocal nsolve
        if k not in cache:
            S, _ = M.solve(G, cand, Dg, risk, flow, B, float(DELTAS[k]), pen=w)
            cache[k] = frozenset(S)
            nsolve += 1
        return cache[k]

    def fill(lo: int, hi: int) -> None:
        """[lo, hi] 구간을 채운다. 양 끝 해가 같으면 사이를 그 해로 간주."""
        if hi - lo <= 1:
            sol(lo); sol(hi)
            return
        if sol(lo) == sol(hi):
            for k in range(lo + 1, hi):
                cache[k] = cache[lo]
            return
        mid = (lo + hi) // 2
        fill(lo, mid); fill(mid, hi)

    fill(0, N_D - 1)

    base = sol(0)
    bb = sum(risk[n] * flow[n] for n in base)
    bx = sum(risk[j] * flow[j] for (a, j) in D if a in base and j not in base)

    rows = []
    for k, d in enumerate(DELTAS):
        S = cache[k]
        bd = [(a, b) for (a, b) in D if a in S and b not in S]
        ben = sum(risk[n] * flow[n] for n in S)
        exp = sum(risk[j] * flow[j] for (a, j) in bd)
        L = sum(ln[n] for n in S)
        rows.append(dict(
            예산비율=round(pct, 4), 예산억=round(B / 1e8, 3),
            delta=round(float(d), 4), 선정수=len(S), 선정연장m=round(L),
            기대방지=round(ben, 1), 위험비율=round(100 * ben / TOTAL, 2),
            잔여경계=len(bd), 경계노출=round(exp, 1),
            노출편익비=round(100 * exp / ben, 2) if ben else None,
            편익손실=round(100 * (1 - ben / bb), 3) if bb else None,
            노출감소=round(100 * (1 - exp / bx), 2) if bx else None,
            평균경사=round(float(np.mean([slope[n] for n in S])), 2) if S else None,
            평균통행=round(float(np.mean([flow[n] for n in S])), 1) if S else None))
    return idx, rows, nsolve


def main() -> None:
    verify = "--verify" in sys.argv
    G, cand, risk, flow, slope, D, ln, FULL, TOTAL = setup()

    if verify:
        g = pd.read_csv(VIZ / "p11_grid_hires.csv")
        print(f"격자 {len(g):,}행 로드")
        pick = _pareto_pick(g)
        print("\n=== 채택 후보 전체 MSA 재검증 ===")
        rows = []
        for _, r in pick.iterrows():
            B = FULL * r.예산비율 / 100
            res = M.solve_msa(G, cand, D, risk, flow, B, r.delta)
            S = res["selected"]
            rows.append(dict(
                예산비율=round(r.예산비율, 2), 예산억=round(B / 1e8, 2),
                delta=round(r.delta, 3),
                격자_기대방지=r.기대방지, 검증_기대방지=round(res["benefit"], 1),
                격자_경계노출=r.경계노출, 검증_경계노출=round(res["boundary_exposure"], 1),
                격자_경계수=int(r.잔여경계), 검증_경계수=res["boundary"],
                선정=len(S)))
        v = pd.DataFrame(rows)
        pd.set_option("display.width", 240)
        print(v.to_string(index=False))
        v.to_csv(VIZ / "p11_pareto_verified.csv", index=False, encoding="utf-8-sig")
        return

    st = pickle.load(open(STATE, "rb")) if STATE.exists() else {"rows": [], "done": 0}
    t0 = time.time()
    i = st["done"]
    print(f"예산 {N_B} × δ {N_D} = {N_B*N_D:,}점 | 진행 {i}/{N_B}", flush=True)

    todo = list(range(i, min(i + CHUNK, N_B)))
    if todo:
        ctx = (G, cand, risk, flow, slope, D, ln, FULL, TOTAL)
        global _CTX
        _CTX = ctx
        with mp.Pool(NPROC) as pool:
            for idx, rows, nsolve in pool.imap(_one_budget, todo):
                st["rows"].extend(rows)
                st["done"] = max(st["done"], idx + 1)
                pickle.dump(st, open(STATE, "wb"))
                print(f"  [{idx+1:>3}/{N_B}] {PCTS[idx]:>6.2f}% "
                      f"MILP {nsolve}회  ({time.time()-t0:.0f}s)", flush=True)
        i = st["done"]

    g = pd.DataFrame(st["rows"])
    g.to_csv(VIZ / "p11_grid_hires.csv", index=False, encoding="utf-8-sig")
    if i >= N_B:
        _pareto_pick(g).to_csv(VIZ / "p11_pareto.csv", index=False, encoding="utf-8-sig")
        print(f"\n✅ 완료 — {len(g):,}행 저장. 다음: --verify")
    else:
        print(f"\n⏸ {i}/{N_B} 저장. 같은 명령으로 이어서 실행")


def _pareto_pick(g: pd.DataFrame) -> pd.DataFrame:
    """편익 최대 · 경계노출 최소의 파레토 전선 추출."""
    d = g.dropna(subset=["기대방지", "경계노출"]).copy()
    d = d.sort_values(["기대방지", "경계노출"], ascending=[False, True])
    front, best = [], np.inf
    for _, r in d.iterrows():
        if r.경계노출 < best:
            front.append(r)
            best = r.경계노출
    f = pd.DataFrame(front).reset_index(drop=True)
    return f


if __name__ == "__main__":
    main()
