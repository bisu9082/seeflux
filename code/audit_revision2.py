# Second revision (Step 8 round 2, 2026-10-16), independent cruise set:
#  (1) attribution of ln(model/PARMA) to inputs on the date-sampled points (realistic solar weighting): 5-fold CV R^2 of
#      OLS and gradient-boosted trees for nested input sets (the grid analysis in ml_analysis.py weights all W equally)
#  (2) flat 6000 against PARMA neutrons + protons above 10 MeV (flat 5600 against neutrons is in audit_revision.py)
#  (3) a bounding environment: surrogate at the lowest annual-mean W and Rc = 0 at 40 kft, times the 99% multiplier 1.10
#      and divided by the lower reference-scale bound k = 0.88 (k = PARMA / true flux, so the true flux is at most
#      PARMA / 0.88 within the envelope), with the PARMA proton/neutron ratio at the same condition
import json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver
from seeflux_formula import flux_gt10
from solar_w import daily_w, annual_mean

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/step4"
SEED = 20261016
V = pd.read_csv(OUT / "validation_direct.csv", parse_dates=["date"])
C = V[V.alt_km.between(9.0, 12.5)].copy()
C["abslat"] = C.lat.abs()

# (1) attribution on the date-sampled cruise points
kf = KFold(5, shuffle=True, random_state=SEED)
sets = [("alt,|lat|", ["alt_km", "abslat"]), ("alt,|lat|,W", ["alt_km", "abslat", "W"]),
        ("alt,|lat|,Rc", ["alt_km", "abslat", "Rc"]), ("alt,|lat|,W,Rc", ["alt_km", "abslat", "W", "Rc"])]
rows = []
for m, lab in [("S1_IEC", "IEC 62396-1"), ("S2_JESD_mid", "JESD89"), ("S4_AN_ABB", "AN4589 ABB"), ("S4b_AN_CoolCAD", "AN4589 CoolCAD")]:
    y = np.log(C[m] / C.PARMA).values
    for est_name, est in [("OLS", LinearRegression()),
                          ("GBT", xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED, verbosity=0))]:
        r = dict(model=lab, estimator=est_name, n=len(C))
        for sname, f in sets:
            sc = cross_val_score(est, C[f], y, cv=kf, scoring="r2")
            r["R2 " + sname] = float(sc.mean()); r["R2min " + sname] = float(sc.min()); r["R2max " + sname] = float(sc.max())
        rows.append(r)
A = pd.DataFrame(rows)
A.to_csv(OUT / "revision2_attribution_sampled.csv", index=False, lineterminator="\n", float_format="%.6f")

# (2) protons at the independent points (same dates, places, altitudes as validate_direct.py)
o = run_driver([f"v{i} D {d.year} {d.month} {d.day} {la:.4f} {lo:.4f} {h:.4f} 10 PR"
                for i, (d, la, lo, h) in enumerate(zip(C.date, C.lat, C.lon, C.alt_km))])
C["PARMA_n_check"] = [float(x[5]) * 3600 for x in o]
C["PARMA_p"] = [float(x[9]) * 3600 for x in o]
# validation_direct.csv stores lat/lon/alt with 5 significant digits, so the rerun differs slightly; n and p are
# therefore taken from the same run here
rerun_dev = float(np.max(np.abs(C.PARMA_n_check / C.PARMA - 1)))
assert rerun_dev < 0.01, rerun_dev
npf = C.PARMA_n_check + C.PARMA_p
K = (0.88, 1.0, 1.37)
flat = dict(n=len(C), proton_over_neutron_median=float((C.PARMA_p / C.PARMA_n_check).median()),
            proton_over_neutron_range=[float((C.PARMA_p / C.PARMA_n_check).min()), float((C.PARMA_p / C.PARMA_n_check).max())],
            flat6000_over_np=dict(median=float((6000 / npf).median()), p05=float((6000 / npf).quantile(.05)), p95=float((6000 / npf).quantile(.95)),
                                  **{f"below1_k{k}": float(np.mean(k * 6000 / npf < 1)) for k in K}),
            flat5600_over_n=dict(**{f"below1_k{k}": float(np.mean(k * 5600 / C.PARMA_n_check < 1)) for k in K}), rerun_max_rel_dev=rerun_dev,
            exceed_np6000=dict(Rc_max=float(C[npf > 6000].Rc.max()), alt_min=float(C[npf > 6000].alt_km.min()),
                               W_max=float(C[npf > 6000].W.max()), n=int((npf > 6000).sum())),
            max_np=float(npf.max()), max_n=float(C.PARMA_n_check.max()))

# (3) bounding environment at 40 kft (12.19 km), Rc = 0, lowest annual-mean W of 2009-2025
D = daily_w()
w_lo = min(annual_mean(y, D) for y in range(2009, 2026))
h40 = 40000 * 0.3048 / 1000
s0 = float(flux_gt10(h40, 0.0, w_lo))
g = run_driver([f"b A {w_lo:.3f} 0.0 {h40:.5f} 10 PR"])[0]
pn, pp = float(g[5]) * 3600, float(g[9]) * 3600
flat["bounding_40kft_Rc0_Wmin"] = dict(W=w_lo, surrogate=s0, surrogate_x1p10=1.10 * s0, surrogate_x1p10_over_k088=1.10 * s0 / 0.88,
                                       PARMA_n=pn, PARMA_p=pp, PARMA_np=pn + pp, np_over_n=(pn + pp) / pn,
                                       bound_with_protons=1.10 * s0 / 0.88 * (pn + pp) / pn,
                                       PARMA_np_over_k088=(pn + pp) / 0.88)
# (4) altitude-dependent bounding environment (Step 8 round 3: a single 40 kft value is exceeded at higher levels).
# PARMA at W = 0 (below every annual and monthly mean of 2009-2026) and Rc = 0, 0.5, 1 GV (plateau check), flight levels
# FL350 to FL600 (pressure altitude = US Standard Atmosphere height); neutrons and neutrons + protons above 10 MeV.
FLS = [350, 370, 390, 410, 450, 510, 600]
keys = [(fl, rc) for fl in FLS for rc in (0.0, 0.5, 1.0)]
o = run_driver([f"f{fl}_{rc} A 0 {rc} {fl * 100 * 0.3048 / 1000:.5f} 10 PR" for fl, rc in keys])
rows_b = []
for (fl, rc), x in zip(keys, o):
    n_, p_ = float(x[5]) * 3600, float(x[9]) * 3600
    rows_b.append(dict(FL=fl, alt_km=fl * 100 * 0.3048 / 1000, Rc=rc, n=n_, np=n_ + p_, np_over_k088=(n_ + p_) / 0.88))
Bd = pd.DataFrame(rows_b)
Bd.to_csv(OUT / "revision2_bounding_by_level.csv", index=False, lineterminator="\n", float_format="%.6g")
flat["bounding_by_level_Rc0_W0"] = Bd[Bd.Rc == 0].drop(columns="Rc").to_dict(orient="records")
flat["plateau_max_rel_diff_Rc0_to_1"] = float((Bd.groupby("FL")["np"].max() / Bd.groupby("FL")["np"].min() - 1).max())
json.dump(flat, open(OUT / "revision2_flat_np.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(A.round(3).to_string(index=False))
print(json.dumps(flat, indent=1))
