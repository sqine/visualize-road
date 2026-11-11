"""δ 민감도 실험 — 경계 패널티 가중치를 바꿔가며 트레이드오프를 측정.

질문
----
이미 집행된 예산을 그대로 다시 쓴다면, δ를 올릴수록 재결빙 경계는 얼마나
줄고 편익은 얼마나 손해인가?

설계
----
- 대상: 성북구 4개동 (유동인구 실측을 보유한 유일 구역)
- 예산 2종
    current : 현행 열선 골목길 연장 2,101 m × 882,353원/m = 18.53억 (고정)
    model   : 구 후보군(경사 6%↑) 총연장의 5% = 8.81억 (보고서 기준선)
- 변수: **δ 하나.** h_min은 2.0 m 고정
  (h_min 민감도는 서울 전역 경계 분석에서 2→10 m 검증 완료)
- 위험도: Observed MAA EDF
- 가중치: current-flow 엣지전류 + MSA 고정점

실행
----
    cd streamlit_app
    python3 scripts/run_delta_experiment.py

산출
----
    data/exp_delta.csv       δ × 예산모드 요약표
    data/exp_selection.csv   선정 구간 (long format, 지도용)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402

DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
OUT = ROOT / "data"


def run(mode: str, G, cand, D, risk, flow, slope, B):
    rows, sel = [], []
    base = M.solve_msa(G, cand, D, risk, flow, B, 0.0)
    for d in DELTAS:
        t0 = time.time()
        res = base if d == 0.0 else M.solve_msa(G, cand, D, risk, flow, B, d)
        row = M.summarize(G, res, risk, flow, slope, B,
                          base_benefit=base["benefit"],
                          base_boundary=base["boundary"] or None,
                          D_all=D)
        row.update(예산모드=mode, delta=d, 예산억=round(B / 1e8, 2),
                   소요초=round(time.time() - t0, 1))
        rows.append(row)
        for n in res["selected"]:
            sel.append(dict(
                예산모드=mode, delta=d, node=n,
                dong=G.nodes[n].get("dong"),
                lat=G.nodes[n].get("lat"), lon=G.nodes[n].get("lon"),
                slope_pct=slope[n], flow=flow[n], risk=round(risk[n], 4),
                length_m=G.nodes[n].get("length_m"),
                기대미끄러짐=round(risk[n] * flow[n], 2),
            ))
        print(f"  δ={d:<5} 선정 {row['선정수']:>3}  방지 {row['기대방지']:>8.1f}  "
              f"경계 {row['잔여경계']:>3}  노출 {row['경계노출']:>7.1f}  "
              f"({row['소요초']}s)", flush=True)
    return rows, sel


def main() -> None:
    t0 = time.time()
    G = M.load_graph()
    cand = M.candidates(G)
    risk, flow, slope = M.risk_flow(G)
    D = M.direction_set(G, cand)
    budgets = {
        "current": M.current_budget(G)[0],
        "model": M.model_budget(G),
    }
    total = sum(risk[n] * flow[n] for n in cand)
    print(f"후보 {len(cand)} / 방향엣지 {len(D)} / 후보 총위험 {total:,.1f}건")

    rows, sel = [], []
    for mode, B in budgets.items():
        print(f"\n[{mode}] 예산 {B/1e8:.2f}억")
        r, s = run(mode, G, cand, D, risk, flow, slope, B)
        rows += r
        sel += s

    cols = ["예산모드", "delta", "예산억", "선정수", "선정연장m", "집행액억",
            "기대방지", "편익손실%", "잔여경계", "잔여경계_D전체", "경계감소%", "경계노출",
            "평균경사", "평균통행", "건당만원", "MSA수렴반복", "소요초"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT / "exp_delta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sel).to_csv(OUT / "exp_selection.csv", index=False,
                             encoding="utf-8-sig")
    print(f"\n저장: data/exp_delta.csv ({len(df)}행) / "
          f"data/exp_selection.csv ({len(sel)}행)  총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
