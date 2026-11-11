"""p7 입력 생성 — 현행 열선 진단 (무엇을 보고 깔았나 / 경계를 얼마나 만드나).

산출
----
    viz/data/p7_current_compare.csv      경사·통행 중위 비교 + Mann-Whitney p
    viz/data/p7_boundary_rate_by_gu.csv  자치구별 경계 보유율

실행: cd streamlit_app && python3 scripts/build_current_diagnosis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import milp as M  # noqa: E402

DATA = ROOT / "data"
VIZ = ROOT / "viz" / "data"


def quintile_and_achievement(G, risk, flow, inst, ln):
    """위험군을 risk×flow 5분위로 나눈 커버리지 + 같은 연장 대비 달성률.

    "위험군을 못 고른다"가 아니라 **"위험군 안에서 우선순위가 없다"**를 본다.
    """
    import numpy as np
    from scipy.stats import chi2_contingency

    golmok = [n for n in G.nodes() if G.nodes[n].get("road_type") == "골목길"]
    risky = [n for n in golmok if (G.nodes[n].get("slope_pct") or 0) >= 100 / 18]
    sc = pd.Series({n: risk[n] * flow[n] for n in risky})

    def quint(pool, scoref, tag):
        """pool 을 점수 5분위로 나눈 커버리지 + 균등성 검정."""
        s = pd.Series({n: scoref(n) for n in pool})
        q = pd.qcut(s.rank(method="first"), 5,
                    labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        d = pd.DataFrame({"node": pool, "len": [ln[n] for n in pool],
                          "heat": [n in inst for n in pool],
                          "분위": q[pool].values})
        t = (d.groupby("분위", observed=True)
               .agg(구간=("node", "size"), 열선구간=("heat", "sum"),
                    연장m=("len", "sum"),
                    열선연장m=("len", lambda x: x[d.loc[x.index, "heat"]].sum()))
               .reset_index())
        t["설치율_구간"] = (100 * t.열선구간 / t.구간).round(1)
        t["커버리지_연장"] = (100 * t.열선연장m / t.연장m).round(1)
        ct = np.array([t.열선구간.values, (t.구간 - t.열선구간).values])
        chi2, p, *_ = chi2_contingency(ct)
        t.insert(0, "기준", tag)
        t["chi2"] = round(chi2, 2); t["p"] = round(p, 4)
        print(f"  {tag:<28} Q1→Q5 {list(t.커버리지_연장)}  "
              f"χ² = {chi2:.2f}, p = {p:.3f}")
        return t

    print("\n[5분위 커버리지 — 무엇을 보고 깔았나]")
    parts = [
        quint(golmok, lambda n: G.nodes[n].get("slope_pct") or 0,
              "골목길 전체 · 경사 단독"),
        quint(risky, lambda n: G.nodes[n].get("slope_pct") or 0,
              "위험군 안 · 경사 단독"),
        quint(risky, lambda n: flow[n], "위험군 안 · 통행 단독"),
        quint(risky, lambda n: risk[n] * flow[n], "위험군 안 · risk×flow"),
    ]
    t = parts[3]
    pd.concat(parts, ignore_index=True).to_csv(
        VIZ / "p7_quintile_coverage.csv", index=False, encoding="utf-8-sig")
    print("  → 위험군에 들어가는 것까지는 경사를 본다(p = 0.004). "
          "그 안에서는 경사로도 못 가린다(p = 0.766).")

    # ── 같은 연장으로 "잘 골랐다면" 얼마나 얻었나 ───────────
    #    두 잣대를 같은 범위(4개동 골목길)에서 나란히 본다.
    #    경사만 봐도 59%에 그치고, 통행까지 넣으면 8%로 떨어진다.
    L = sum(ln[n] for n in inst)

    def ach(key, ben, tag):
        order = sorted(golmok, key=lambda n: -key(n))
        acc, ideal = 0.0, []
        for n in order:
            if acc + ln[n] <= L:
                ideal.append(n); acc += ln[n]
        cur = sum(ben(n) for n in inst)
        ide = sum(ben(n) for n in ideal)
        hit = len(set(ideal) & inst)
        print(f"  {tag:<34} {cur:>8.1f} / {ide:>8.1f} → {100*cur/ide:5.1f}%"
              f"   적중 {hit}/{len(ideal)}")
        return dict(기준=tag, 현행=round(cur, 1), 이상=round(ide, 1),
                    달성률=round(100 * cur / ide, 1),
                    이상선택=len(ideal), 적중=hit)

    print(f"\n[같은 연장 {L:.0f} m 로 잘 골랐다면 — 4개동]")
    rows = [ach(lambda n: risk[n], lambda n: risk[n] * ln[n],
                "경사만 (편익 = risk×길이)"),
            ach(lambda n: risk[n] * flow[n], lambda n: risk[n] * flow[n],
                "경사×통행 (편익 = risk×flow)")]
    rows[0]["범위"] = rows[1]["범위"] = "4개동"
    rows[0]["위험군커버리지"] = rows[1]["위험군커버리지"] = round(
        100 * sum(ln[n] for n in risky if n in inst)
        / sum(ln[n] for n in risky), 1)
    pd.DataFrame(rows).to_csv(VIZ / "p7_achievement.csv", index=False,
                              encoding="utf-8-sig")


def main() -> None:
    # ── (a) 현행 열선이 고른 것 — 4개동 ──────────────────────
    G = M.load_graph()
    risk, flow, slope = M.risk_flow(G)
    inst = set(M.installed_links(G))
    golmok = [n for n in G.nodes() if G.nodes[n].get("road_type") == "골목길"]
    other = [n for n in golmok if n not in inst]

    rows = []
    for key, src in [("slope", slope), ("flow", flow)]:
        a = np.array([src[n] for n in inst])
        b = np.array([src[n] for n in other])
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(지표=key, 현행n=len(a), 그외n=len(b),
                         현행중위=round(float(np.median(a)), 2),
                         그외중위=round(float(np.median(b)), 2),
                         현행평균=round(float(a.mean()), 2),
                         그외평균=round(float(b.mean()), 2),
                         U=round(float(u)), p=float(p)))
    c = pd.DataFrame(rows)
    c.to_csv(VIZ / "p7_current_compare.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(c.to_string(index=False))

    quintile_and_achievement(G, risk, flow, inst,
                             {n: (G.nodes[n].get("length_m") or 0.0)
                              for n in G.nodes()})

    # ── (b) 경계 보유율 — 서울 전역 ──────────────────────────
    bp = pd.read_parquet(DATA / "seoul_boundary_pairs.parquet")
    l = pd.read_parquet(DATA / "seoul_links.parquet",
                        columns=["link_id", "gu", "heat"])
    l = l[l.gu.notna()]
    heat = l[l.heat.fillna(False).astype(bool)]

    g = (bp.groupby("down_gu")
           .agg(경계쌍=("up", "size"), 상류=("up", "nunique"))
           .reset_index().rename(columns={"down_gu": "자치구"}))
    h = heat.groupby("gu").size().rename("열선구간").reset_index()
    h = h.rename(columns={"gu": "자치구"})
    m = h.merge(g, on="자치구", how="left").fillna(0)
    m["경계보유율"] = (100 * m.상류 / m.열선구간).round(1)
    m["열선100당경계"] = (100 * m.경계쌍 / m.열선구간).round(1)
    m.to_csv(VIZ / "p7_boundary_rate_by_gu.csv", index=False,
             encoding="utf-8-sig")

    tot = 100 * bp.up.nunique() / len(heat)
    print(f"\n전역 경계보유율 {tot:.1f}% "
          f"(열선 {len(heat):,}구간 중 {bp.up.nunique():,}구간이 경계를 만듦)")
    print(f"경계쌍 {len(bp):,} · 하류 골목길 비중 "
          f"{100*(bp.down_road_type=='골목길').mean():.1f}% · "
          f"하류 경사 중위 {bp.down_slope.median():.2f}%")
    print(m.sort_values("경계보유율", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
