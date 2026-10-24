# Step 8 round 6, M2: break-even reference scale k* for each scale-dependent finding.
# k = PARMA / true flux, so a model-to-PARMA ratio r corresponds to k r against the true flux, and a PARMA flux F to F / k.
# A finding that holds at k = 1 reverses at k*. Compared with the envelope 0.88-1.37 and with the in-flight anchors
# (ER-2 total fluence ratios and the cruise anchor above 10 MeV).
# Inputs: routes_levels.csv, routes_levels_profiles_FL370.csv, routes_monthly_south.csv, revision_k_flat_jesd.csv,
#         revision2_bounding_by_level.csv, bench_goldhagen2004.csv, er2_gt10_anchor.json, reference_band.json
# Output: data/step4/k_breakeven.csv, k_breakeven.json
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
L = pd.read_csv(OUT / "routes_levels.csv")
PF = pd.read_csv(OUT / "routes_levels_profiles_FL370.csv")
MS = pd.read_csv(OUT / "routes_monthly_south.csv")
T1 = pd.read_csv(OUT / "revision_k_flat_jesd.csv").set_index("model")
BL = pd.read_csv(OUT / "revision2_bounding_by_level.csv"); BL = BL[BL.Rc == 0].set_index("FL")
B = pd.read_csv(OUT / "bench_goldhagen2004.csv")
A = json.load(open(OUT / "er2_gt10_anchor.json"))
band = json.load(open(OUT / "reference_band.json"))
rows = []
add = lambda group, finding, kstar, holds_below: rows.append(dict(group=group, finding=finding, k_star=float(kstar),
                                                                   holds_for=("k < k*" if holds_below else "k > k*")))
# (1) flat 6000 against PARMA n + p, solar-minimum years (worst of 2009 and 2019), per level: peak, cruise mean, half of cruise
for lev in ("FL370", "FL390", "FL410"):
    s = L[(L.level == lev) & L.cond.isin(["y2009", "y2019"])]
    g = s.groupby("route").max_np_cruise.max()
    add("flat 6000", f"peak n+p above 6000 somewhere on the route, {lev} (max over routes: {g.idxmax()})", g.max() / 6000, True)
cr = PF[(PF.cond == "y2009")]
for rt in ("ICN-JFK", "ICN-FRA", "SYD-SCL"):
    p = cr[(cr.route == rt) & (cr.alt_km >= cr[cr.route == rt].alt_km.max() - 1e-9)]
    np_ = p.PARMA_n + p.PARMA_p
    add("flat 6000", f"cruise-mean n+p above 6000, {rt} FL370 2009", np_.mean() / 6000, True)
    add("flat 6000", f"n+p above 6000 for at least half of the cruise, {rt} FL370 2009", np_.median() / 6000, True)
# (2) scaled IEC curves below the flux on GRU-JNB (monthly route fluence, FL370, 2009-2026): k r < 1 <=> k < 1/r
r = MS[MS.route == "GRU-JNB"].IEC_over_PARMA_n
add("scaled IEC, GRU-JNB", "below the flux in at least half of the months", 1 / r.median(), True)
add("scaled IEC, GRU-JNB", "below the flux in at least one month", 1 / r.min(), True)
# (3) median ratio of each model below one on the independent cruise set (Table I): k r_med < 1 <=> k < 1/r_med
for m, lab in (("S1_IEC", "IEC scaled"), ("S2_JESD_mid", "JESD89"), ("S4_AN_ABB", "AN4589 ABB"), ("S4b_AN_CoolCAD", "AN4589 CoolCAD")):
    add("median ratio", f"{lab} median below the flux", 1 / T1.loc[m, "median"], True)
# (4) bounding environment (PARMA n+p, Rc = 0, W = 0) at or above the 6000 minimum
for fl in (370, 410, 510):
    add("bounding environment", f"FL{fl} value ({BL.loc[fl, 'np']:.0f}) at or above 6000", BL.loc[fl, "np"] / 6000, True)
res = pd.DataFrame(rows)
K_LO, K_HI = band["K_LO"], 1.37
inflight_tot = B[B.alt_km > 0].PARMA_over_meas
k_gm = float(np.exp(np.log(inflight_tot).mean()))
anch = dict(envelope=[K_LO, K_HI], inflight_total_ratios=inflight_tot.round(4).tolist(), inflight_total_geomean=k_gm,
            cruise_gt10=A["k_gt10_PARMA_over_measured"], ground_total_goldhagen=float(B[B.alt_km == 0].PARMA_over_meas.iloc[0]),
            ground_gt10_jesd=K_LO)
res["vs_envelope"] = [("holds across the envelope" if (h == "k < k*" and k >= K_HI) else
                       "fails across the envelope" if (h == "k < k*" and k <= K_LO) else "decided inside the envelope")
                      for k, h in zip(res.k_star, res.holds_for)]
res["holds_at_inflight_geomean"] = res.k_star > k_gm
res["holds_at_cruise_anchor"] = res.k_star > anch["cruise_gt10"]
res.to_csv(OUT / "k_breakeven.csv", index=False, lineterminator="\n", float_format="%.4f")
json.dump(dict(anchors=anch, rows=res.to_dict(orient="records")), open(OUT / "k_breakeven.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(json.dumps(anch, indent=1)); print(res.round(3).to_string(index=False))
