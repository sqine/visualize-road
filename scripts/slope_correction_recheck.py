"""경사 보정의 해 민감도 — 현세대(G4) 재산출.

배경
----
경사 = |Δz| / L 이므로 짧은 링크에서 고도 오차가 증폭된다. 등고선 레벨
홀드아웃 CV에서 σ_Δz(L)은 L≥45 m에서 6.0 m로 포화하고 L=5~12 m에서도
4.1 m다. 따라서 경사 표준오차 σ_s = σ_Δz(L)/L 이 짧은 링크에서 크게 커진다.

보정
----
링크를 도로망 위에서 **고도가 단조인 방향으로만** 연장해 기저선을 늘리고
(무제약 연장은 오르내림이 상쇄돼 진짜 경사를 지운다), 원본과 연장값을
오차분산 역가중으로 결합한다(경험베이즈 수축).
원 구현: `02_분석파일/slope_corrected.py` (구세대 `final_model` 의존).
여기서는 알고리즘만 가져와 현세대 `utils.milp` 그래프에 적용한다.

질문
----
보정 경사로 risk를 다시 계산해 MILP(B=18.53억, δ=4)를 풀면
**선정 구간이 얼마나 바뀌고 잔여 경계는 어떻게 되는가.**

실행: cd streamlit_app && python3 scripts/slope_correction_recheck.py
산출: viz/data/p14_slope_correction.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402
from utils.risk import obs_risk  # noqa: E402

#: 등고선 레벨 홀드아웃 CV에서 추정한 σ_Δz(L) — (링크길이 m, 표준편차 m)
SIG_TAB = [(8.5, 4.100), (16.0, 5.182), (25.0, 5.724), (37.5, 5.965),
           (57.5, 5.964), (90.0, 5.935), (140.0, 6.000)]
#: 단조 연장 반경 상한(m)
R_EXT = 60.0
#: 채택 시나리오 S1
B_ADOPT = 1_853_000_000.0
DELTA_ADOPT = 4.0

OUT = ROOT / "viz" / "data" / "p14_slope_correction.csv"


def sigma_dz(L: float) -> float:
    return float(np.interp(L, [x for x, _ in SIG_TAB], [y for _, y in SIG_TAB]))


def monotone_arm(G, i, start, want_up, half, z, R):
    """i→start 방향으로 고도가 단조인 동안만 연장. 반환 (기저선 거리, 끝 고도)."""
    cur, prev, zc = start, i, z[start]
    d = half[i] + half[start]
    best = (d, zc)
    while d < R:
        nxt = bz = None
        for v in G.neighbors(cur):
            if v == prev or z[v] is None:
                continue
            if (z[v] > zc) if want_up else (z[v] < zc):
                if bz is None or ((z[v] > bz) if want_up else (z[v] < bz)):
                    nxt, bz = v, z[v]
        if nxt is None:
            break
        d += half[cur] + half[nxt]
        prev, cur, zc = cur, nxt, bz
        best = (d, zc)
    return best


def extended_slope(G, R: float = R_EXT) -> dict:
    """노드별 (연장 경사 %, 유효 기저선 m)."""
    half = {n: (G.nodes[n].get("length_m") or 0.0) / 2 for n in G.nodes()}
    z = {n: G.nodes[n].get("z") for n in G.nodes()}
    out = {}
    for i in G.nodes():
        nb = [a for a in G.neighbors(i) if z[a] is not None]
        if z[i] is None or not nb:
            out[i] = (None, None)
            continue
        hi = max(nb, key=lambda a: z[a])
        lo = min(nb, key=lambda a: z[a])
        if hi == lo:  # 이웃이 하나 — 한쪽 팔만
            d, ze = monotone_arm(G, i, hi, z[hi] > z[i], half, z, R)
            out[i] = (abs(ze - z[i]) / d * 100, d)
            continue
        du, zu = monotone_arm(G, i, hi, True, half, z, R)
        dd, zl = monotone_arm(G, i, lo, False, half, z, R)
        L = du + dd
        out[i] = (abs(zu - zl) / L * 100, L)
    return out


def shrink(G) -> pd.DataFrame:
    """원본·연장 경사를 오차분산 역가중으로 결합."""
    ext = extended_slope(G)
    ln = {n: (G.nodes[n].get("length_m") or 0.0) for n in G.nodes()}
    rows = []
    for n in G.nodes():
        s0 = G.nodes[n].get("slope_pct")
        if s0 is None:
            continue
        se, Le = ext[n]
        L0 = max(ln[n], 1.0)
        v0 = (sigma_dz(L0) / L0 * 100) ** 2
        if se is None:
            sc, ve, Le = s0, 1e12, None
        else:
            ve = (sigma_dz(Le) / Le * 100) ** 2
            w = ve / (v0 + ve)
            sc = w * s0 + (1 - w) * se
        sd = float(np.sqrt(1 / (1 / v0 + 1 / ve)))
        rows.append(dict(node=n, dong=G.nodes[n].get("dong"),
                         length_m=round(L0, 1), slope_raw=round(s0, 2),
                         slope_ext=None if se is None else round(se, 2),
                         L_eff=None if Le is None else round(Le, 1),
                         slope_shrunk=round(sc, 2), se_pp=round(sd, 2)))
    return pd.DataFrame(rows)


def main() -> None:
    G = M.load_graph()
    cand = M.candidates(G)
    D = M.direction_set(G, cand)
    flow = {n: (G.nodes[n].get("flow_sum") or 0.0) for n in G.nodes()}

    df = shrink(G)
    SH = dict(zip(df.node, df.slope_shrunk))

    se_raw = float(np.median([sigma_dz(max(l, 1)) / max(l, 1) * 100
                              for l in df.length_m]))
    print(f"링크 {len(df)}개 (단조연장 성공 {df.L_eff.notna().sum()})")
    print(f"유효 기저선 중앙 {df.L_eff.median():.0f} m vs 링크길이 중앙 "
          f"{df.length_m.median():.0f} m ({df.L_eff.median()/df.length_m.median():.1f}배)")
    print(f"경사 표준오차 중앙: 원본 {se_raw:.1f}%p → 보정 {df.se_pp.median():.1f}%p "
          f"({se_raw/df.se_pp.median():.1f}배 개선)")

    r_raw = {n: obs_risk(G.nodes[n].get("slope_pct")) for n in G.nodes()}
    r_cor = {n: obs_risk(SH.get(n, G.nodes[n].get("slope_pct"))) for n in G.nodes()}

    res = {}
    for nm, r in (("원본 경사", r_raw), ("보정 경사", r_cor)):
        res[nm] = M.solve_msa(G, cand, D, r, flow, B_ADOPT, DELTA_ADOPT)
        S = res[nm]["selected"]
        # p10·p12 표는 패널티 대상(Dg) 밖 경계까지 포함한 D 전체 기준을 쓴다
        res[nm]["boundary_all"] = sum(1 for (i, j) in D if i in S and j not in S)
        print(f"\n[{nm}] 선정 {len(S)} · 잔여경계 {res[nm]['boundary']}"
              f"(D전체 {res[nm]['boundary_all']}) · 편익 {res[nm]['benefit']:.1f}")

    S0 = res["원본 경사"]["selected"]
    Sc = res["보정 경사"]["selected"]
    common = S0 & Sc
    print(f"\n공통 {len(common)} · 빠진 곳 {len(S0 - Sc)} · 새로 들어온 곳 {len(Sc - S0)}")
    print(f"Jaccard {len(common)/len(S0 | Sc):.3f} · "
          f"원본해 유지율 {100*len(common)/len(S0):.1f}%")

    # 교차평가 — 어느 경사 추정을 믿든 각 해의 편익
    print("\n=== 교차평가 (행: 해, 열: 평가 기준) ===")
    cross = []
    for nm, S in (("원본해", S0), ("보정해", Sc)):
        cross.append(dict(해=f"{nm} ({len(S)}곳)",
                          원본경사기준=round(sum(r_raw[n] * flow[n] for n in S), 1),
                          보정경사기준=round(sum(r_cor[n] * flow[n] for n in S), 1)))
    print(pd.DataFrame(cross).to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        dict(항목="예산(원)", 값=B_ADOPT),
        dict(항목="delta", 값=DELTA_ADOPT),
        dict(항목="원본해 선정", 값=len(S0)),
        dict(항목="보정해 선정", 값=len(Sc)),
        dict(항목="공통", 값=len(common)),
        dict(항목="교체(빠짐)", 값=len(S0 - Sc)),
        dict(항목="교체(신규)", 값=len(Sc - S0)),
        dict(항목="원본해 유지율(%)", 값=round(100 * len(common) / len(S0), 1)),
        dict(항목="Jaccard", 값=round(len(common) / len(S0 | Sc), 3)),
        dict(항목="잔여경계 원본해", 값=res["원본 경사"]["boundary"]),
        dict(항목="잔여경계 보정해", 값=res["보정 경사"]["boundary"]),
        dict(항목="잔여경계 원본해 D전체", 값=res["원본 경사"]["boundary_all"]),
        dict(항목="잔여경계 보정해 D전체", 값=res["보정 경사"]["boundary_all"]),
        dict(항목="교차평가 원본해·보정기준",
             값=round(sum(r_cor[n] * flow[n] for n in S0), 1)),
        dict(항목="교차평가 보정해·보정기준",
             값=round(sum(r_cor[n] * flow[n] for n in Sc), 1)),
        dict(항목="경사 표준오차 원본(%p)", 값=round(se_raw, 2)),
        dict(항목="경사 표준오차 보정(%p)", 값=round(float(df.se_pp.median()), 2)),
    ]).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n→ {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
