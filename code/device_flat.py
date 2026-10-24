# Step 8 round 10 (M2, R1/R5): the device counts of device_r7.py with the flat design value 6000 and with the floor
# (larger of 6000 and the scaled curves) in place of the scaled curves. No PARMA run: the reference counts and the
# IEC-curve ratios come from device_r7.csv, and the flux integrals of the three uses are recomputed on the same route
# profiles, time step and climb/descent model, so ratio_use = ratio_IEC * F_use / F_IEC.
# Output: data/step4/device_flat.csv, device_flat.json
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
R = pd.read_csv(OUT / "device_r7.csv")
PF = pd.read_csv(OUT / "routes_levels_profiles_FL370.csv")
H = {"FL370": 37000 * 0.3048 / 1000, "FL410": 41000 * 0.3048 / 1000}
fac = {}
for rt, g0 in PF[PF.cond == "y2019"].groupby("route", sort=False):
    t = g0.t_min.values; T = t.max(); lat = g0.lat.values
    for lev, hcr in H.items():
        alt = np.minimum(hcr, np.minimum(t, T - t) / 25.0 * hcr); hi = alt >= 6.0
        iec = S.s1_iec_boeing(alt, lat) * 6000 / 5600
        integ = lambda y: float(np.trapezoid(np.where(hi, y, 0), t * 60))
        F_iec = integ(iec / 3600); F_flat = integ(np.full(len(t), 6000.0) / 3600); F_floor = integ(np.maximum(6000.0, iec) / 3600)
        fac[(rt, lev)] = (F_flat / F_iec, F_floor / F_iec)
rows = []
for r in R.itertuples():
    ff, fl = fac[(r.route, r.level)]
    d = dict(route=r.route, level=r.level, cond=r.cond)
    for col in ("IEC_sigM2_s200_over_M2np", "IEC_sigM2_s200_over_M3np", "IEC_sigM3_s200_over_M3np", "IEC_sigM3_s230_over_M3np",
                "IEC_sigM2_sp800_over_M2np", "IEC_sigM3_sp800_over_M3np"):
        v = getattr(r, col)
        d[col.replace("IEC_", "FLAT_")] = v * ff
        d[col.replace("IEC_", "FLOOR_")] = v * fl
    rows.append(d)
D = pd.DataFrame(rows)
D.to_csv(OUT / "device_flat.csv", index=False, lineterminator="\n", float_format="%.6g")
hl = D.route.isin(["ICN-JFK", "ICN-FRA", "SYD-SCL", "ICN-LAX"])
summ = {}
for sel, lab in ((D.level == "FL370", "FL370_all"), ((D.level == "FL370") & hl, "FL370_high_latitude"),
                 ((D.level == "FL370") & ~hl, "FL370_other")):
    s = D[sel]
    summ[lab] = {c: [float(s[c].min()), float(s[c].max())] for c in D.columns if c.startswith(("FLAT_", "FLOOR_"))}
    summ[lab]["routes"] = sorted(s.route.unique().tolist())
json.dump(summ, open(OUT / "device_flat.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
