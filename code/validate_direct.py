# Step 4 independent validation (cross-review fix): 2,000 random (date, lat, lon, altitude) points evaluated with
# direct PARMA calls, never through the fitting grid. Locations are area-uniform (uniform in sin(lat)) between
# 80 S and 80 N, altitude uniform 6-15 km, dates uniform from 2009-01-01 to 2026-03-26 (last date in PARMA's
# solar table). W and Rc come from PARMA for each date and place. Formula metrics use points inside its domain.
import sys, pathlib, datetime as dt, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver
from seeflux_formula import flux_gt10, load_params, DOMAIN

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
rng = np.random.default_rng(20261015)
N = 2000
lat = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-80)), np.sin(np.radians(80)), N)))
lon = rng.uniform(-180, 180, N); alt = rng.uniform(6.0, 15.0, N)
d0, d1 = dt.date(2009, 1, 1), dt.date(2026, 3, 26)
days = [d0 + dt.timedelta(days=int(x)) for x in rng.integers(0, (d1 - d0).days + 1, N)]
o = run_driver([f"v{i} D {d.year} {d.month} {d.day} {la:.4f} {lo:.4f} {h:.4f} 10"
                for i, (d, la, lo, h) in enumerate(zip(days, lat, lon, alt))])
V = pd.DataFrame(dict(date=[d.isoformat() for d in days], lat=lat, lon=lon, alt_km=alt,
                      W=[float(x[1]) for x in o], Rc=[float(x[2]) for x in o], depth=[float(x[3]) for x in o],
                      PARMA=[float(x[5]) * 3600 for x in o]))
P = load_params()
V["in_domain"] = V.W.between(*DOMAIN["W"]) & V.Rc.between(*DOMAIN["Rc_GV"])
V["FORMULA"] = flux_gt10(V.alt_km.clip(6, 15), V.Rc.clip(0, 18), V.W.clip(0, 150), P)
V["S1_IEC"] = S.s1_iec_boeing(V.alt_km, V.lat)
V["S1_IEC6000"] = S.s1_iec_boeing_6000(V.alt_km, V.lat)
V["S2_JESD_mid"] = S.s2_jesd89(V.depth, V.Rc, "mid")
V["S4_AN_ABB"] = S.s4_an4589_abb(V.alt_km, V.lat)
V["S4b_AN_CoolCAD"] = S.s4b_an4589_coolcad(V.alt_km, V.lat)
V["frac_below1_flag_IEC"] = V.S1_IEC < V.PARMA  # non-conservative IEC points (ratio < 1)
V.to_csv(OUT / "validation_direct.csv", index=False, lineterminator="\n", float_format="%.5g")

rows = []
for label, sub in [("all points", V), ("cruise 9-12.5 km", V[V.alt_km.between(9, 12.5)])]:
    for m in ["FORMULA", "S1_IEC", "S1_IEC6000", "S2_JESD_mid", "S4_AN_ABB", "S4b_AN_CoolCAD"]:
        s = sub[sub.in_domain] if m == "FORMULA" else sub
        r = s[m] / s.PARMA; a = np.abs(r - 1)
        rows.append(dict(subset=label, model=m, n=len(s), median_ratio=r.median(), p05=r.quantile(.05), p95=r.quantile(.95),
                         median_abs_pct=np.median(a) * 100, p95_abs_pct=np.percentile(a, 95) * 100,
                         frac_outside_30pct=float(((r > 1.3) | (r < 1 / 1.3)).mean()),
                         frac_outside_factor2=float(((r > 2.0) | (r < 0.5)).mean()),
                         frac_nonconservative_below1=float((r < 1.0).mean())))
R = pd.DataFrame(rows); R.to_csv(OUT / "validation_direct_summary.csv", index=False, lineterminator="\n", float_format="%.6f")
print("W range %.1f-%.1f, Rc range %.2f-%.2f, out of formula domain: %d" % (V.W.min(), V.W.max(), V.Rc.min(), V.Rc.max(), (~V.in_domain).sum()))
print(R.round(3).to_string(index=False))
