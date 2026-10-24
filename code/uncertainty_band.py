# Step 4 (decision A, 2026-10-15): separate scale-invariant (structural) error from reference-dependent bias.
# Reference normalization factor k = PARMA / measurement spans 0.88 (NYC ground >10 MeV vs the JESD89B reference 12.946 n/cm2/h)
# to 1.37 (ER-2 total fluence, Goldhagen 2004); true model/measurement ratio = (model/PARMA) * k.
# Computed for cruise (9-12.5 km) on the independent direct-PARMA set (primary) and on the geographic grid audit.
import numpy as np, pandas as pd, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
b = pd.read_csv(OUT / "bench_goldhagen2004.csv")
nyc = json.load(open(OUT / "bench_nyc.json"))
K_LO = nyc["PARMA_NYC_Nov2002_gt10_ncm2h"] / nyc["JESD89_NYC_ref_ncm2h"]  # PARMA NYC Nov 2002 / JESD89 reference
K_HI = float(b[b.alt_km > 0].PARMA_over_meas.max())   # worst in-flight overestimate
V = pd.read_csv(OUT / "validation_direct.csv"); V = V[V.alt_km.between(9.0, 12.5)]
A = pd.read_csv(OUT / "audit_nodes.csv"); A = A[A.alt_km.between(9.0, 12.5)]
rows = []
for dname, D, models in [("independent", V, ["FORMULA", "S1_IEC", "S2_JESD_mid", "S4_AN_ABB", "S4b_AN_CoolCAD"]),
                         ("grid", A, ["FORMULA", "S1_IEC", "S1b_IEC_eq3", "S2_JESD_mid", "S2_JESD_min", "S2_JESD_max",
                                      "S4_AN_ABB", "S4b_AN_CoolCAD"])]:
    for m in models:
        s = D[D.in_domain] if (m == "FORMULA" and "in_domain" in D) else D
        r = s[m] / s.PARMA; med = r.median(); rn = r / med
        rows.append(dict(dataset=dname, model=m, n=len(s), median_ratio=med, bias_band_lo=med * K_LO, bias_band_hi=med * K_HI,
                         spread_p95_over_p05=r.quantile(.95) / r.quantile(.05),
                         frac_outside_30pct_after_bias_removal=float(((rn > 1.3) | (rn < 1 / 1.3)).mean()),
                         frac_outside_factor2_after_bias_removal=float(((rn > 2.0) | (rn < 0.5)).mean())))
U = pd.DataFrame(rows); U.to_csv(OUT / "uncertainty_band.csv", index=False, lineterminator="\n", float_format="%.6f")
json.dump(dict(K_LO=K_LO, K_HI=K_HI), open(OUT / "reference_band.json", "w"), indent=1)
print("reference band k = [%.3f, %.3f]" % (K_LO, K_HI)); print(U.round(3).to_string(index=False))
