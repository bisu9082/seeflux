# Step 8 round 6, M7b: paired comparisons on the 802 independent cruise points (resampling points, B = 2000) and
# binomial intervals for the surrogate-multiplier coverage (Wilson score, 95%).
#  - ratio of spreads (P95/P5) IEC / AN4589 ABB, IEC / JESD89, ABB / JESD89 on the same resampled points
#  - CARI-7 vs PARMA as reference (totals, monthly-mean conditions; cari_crosscheck_points.csv): spread ratios for IEC and JESD89
# Output: data/step4/paired_bootstrap.json
import json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
SEED, B = 20261018, 2000
rng = np.random.default_rng(SEED)
V = pd.read_csv(OUT / "validation_direct.csv")
C = V[V.alt_km.between(9.0, 12.5)].reset_index(drop=True)
X = pd.read_csv(OUT / "cari_crosscheck_points.csv")
assert len(X) == len(C)
sp = lambda r: np.percentile(r, 95) / np.percentile(r, 5)
R = {m: (C[m] / C.PARMA).values for m in ("S1_IEC", "S2_JESD_mid", "S4_AN_ABB")}
RX = {("IEC", "PARMA"): (X.S1_IEC / X.P_tot).values, ("IEC", "CARI"): (X.S1_IEC / X.CARI_tot).values,
      ("JESD", "PARMA"): (X.S2_JESD_mid / X.P_tot).values, ("JESD", "CARI"): (X.S2_JESD_mid / X.CARI_tot).values}
pairs = {"IEC/ABB": ("S1_IEC", "S4_AN_ABB"), "IEC/JESD89": ("S1_IEC", "S2_JESD_mid"), "ABB/JESD89": ("S4_AN_ABB", "S2_JESD_mid")}
draws = {k: [] for k in pairs}; dx = {"IEC: CARI/PARMA": [], "JESD89: CARI/PARMA": []}
n = len(C)
for _ in range(B):
    i = rng.integers(0, n, n)
    for k, (a, b) in pairs.items():
        draws[k].append(sp(R[a][i]) / sp(R[b][i]))
    dx["IEC: CARI/PARMA"].append(sp(RX[("IEC", "CARI")][i]) / sp(RX[("IEC", "PARMA")][i]))
    dx["JESD89: CARI/PARMA"].append(sp(RX[("JESD", "CARI")][i]) / sp(RX[("JESD", "PARMA")][i]))
ci = lambda v: dict(estimate=None, lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), p_le_1=float(np.mean(np.array(v) <= 1)))
out = {}
for k, (a, b) in pairs.items():
    out[k] = ci(draws[k]); out[k]["estimate"] = float(sp(R[a]) / sp(R[b]))
out["IEC: CARI/PARMA"] = ci(dx["IEC: CARI/PARMA"]); out["IEC: CARI/PARMA"]["estimate"] = float(sp(RX[("IEC", "CARI")]) / sp(RX[("IEC", "PARMA")]))
out["JESD89: CARI/PARMA"] = ci(dx["JESD89: CARI/PARMA"]); out["JESD89: CARI/PARMA"]["estimate"] = float(sp(RX[("JESD", "CARI")]) / sp(RX[("JESD", "PARMA")]))


def wilson(k, m, z=1.959964):
    p = k / m; d = 1 + z * z / m; c = p + z * z / (2 * m); h = z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m))
    return dict(k=int(k), n=int(m), share=p, lo=float((c - h) / d), hi=float((c + h) / d))


A = V[V.in_domain]; corner = A[(A.alt_km > 12.5) & (A.Rc < 2)]
cov = lambda c, s: int(np.sum(c * s.FORMULA >= s.PARMA))
rule = np.where(A.Rc < 2, 1.13, 1.10)
out["coverage_1.10_all"] = wilson(cov(1.10, A), len(A))
out["coverage_1.10_corner"] = wilson(cov(1.10, corner), len(corner))
out["coverage_rule_all"] = wilson(int(np.sum(rule * A.FORMULA >= A.PARMA)), len(A))
out["coverage_1.13_corner"] = wilson(cov(1.13, corner), len(corner))
json.dump(dict(B=B, seed=SEED, results=out), open(OUT / "paired_bootstrap.json", "w"), indent=1)
print(json.dumps(out, indent=1))
