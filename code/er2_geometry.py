# Step 8 round 8 (M2): ER-2 reference scale under the PARMA developer's own setting.
# EXPACS ships parma/build/Condition/ERII-exp.inp with the four Goldhagen et al. (2004) ER-2 conditions and the ground
# site: date 1997-07-15, PARMA's own cutoff map at the stated lat/lon, the stated depth, and g = -10.5 (cabin geometry,
# airframe scaled to 0.5 of the reference aircraft) in flight, g = 0.15 on the ground. Our benchmark (bench_goldhagen.py)
# used 1997-06-15, the measured cutoff and the free atmosphere (g = 10). Here the three differences are separated.
# Output: data/step4/er2_geometry.csv, er2_geometry.json
import json, pathlib, sys, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
B = pd.read_csv(OUT / "bench_goldhagen2004.csv")
A = json.load(open(OUT / "er2_gt10_anchor.json"))
W_jun = float(run_driver(["w D 1997 6 15 37.4 -122 0 10"])[0][1])
W_jul = float(run_driver(["w D 1997 7 15 37.4 -122 0 10"])[0][1])
# PARMA map cutoff at each site
rc_map = [float(run_driver([f"r D 1997 7 15 {la} {lo} 0 10"])[0][2]) for la, lo in zip(B.lat, B.lon)]
B["Rc_map"] = rc_map
rows = []
SET = {"ours (Jun, measured Rc, free air)": ("jun", "meas", "free"),
       "geometry only (Jun, measured Rc, cabin g=-10.5)": ("jun", "meas", "cabin"),
       "date+map only (Jul, map Rc, free air)": ("jul", "map", "free"),
       "developer (Jul, map Rc, cabin g=-10.5)": ("jul", "map", "cabin")}
for name, (dt, rcs, geo) in SET.items():
    w = W_jun if dt == "jun" else W_jul
    lines = []
    for r in B.itertuples():
        g = 0.15 if r.alt_km == 0 else (10 if geo == "free" else -10.5)
        rc = r.Rc if rcs == "meas" else r.Rc_map
        lines.append(f"{r.site} S {w} {rc} {r.depth} {g}")
    o = run_driver(lines)
    for r, x in zip(B.itertuples(), o):
        rows.append(dict(setting=name, site=r.site, alt_km=r.alt_km, W=w, Rc=float(x[2]), g=float(x[4]),
                         PARMA_total=float(x[8]), PARMA_gt10=float(x[5]), measured=r.measured,
                         ratio_total=float(x[8]) / r.measured))
R = pd.DataFrame(rows)
R.to_csv(OUT / "er2_geometry.csv", index=False, lineterminator="\n", float_format="%.6g")
meas_gt10 = A["measured_gt10_cm2s"]
summ = {"W_1997_06_15": W_jun, "W_1997_07_15": W_jul, "Rc_map": dict(zip(B.site, rc_map))}
for name in SET:
    s = R[(R.setting == name) & (R.alt_km > 0)]
    a = R[(R.setting == name) & (R.site == "ER2_38N_11.9km")].iloc[0]
    gnd = R[(R.setting == name) & (R.alt_km == 0)].iloc[0]
    summ[name] = dict(inflight_total_min=float(s.ratio_total.min()), inflight_total_max=float(s.ratio_total.max()),
                      inflight_total_geomean=float(np.exp(np.log(s.ratio_total).mean())),
                      cruise_gt10_k=float(a.PARMA_gt10 / meas_gt10), cruise_total_k=float(a.ratio_total),
                      ground_total=float(gnd.ratio_total))
json.dump(summ, open(OUT / "er2_geometry.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(R[["setting", "site", "W", "Rc", "g", "ratio_total", "PARMA_gt10"]].round(3).to_string(index=False))
print(json.dumps(summ, indent=1))
