# Step 4 audit: standard models vs PARMA (>10 MeV neutron flux) on a geographic grid at cruise altitudes.
# PARMA depends only on (W-index, Rc, depth), so geographic nodes are evaluated by interpolating the
# reference grid (data/step4/parma_grid.csv) in Rc (log-linear); interpolation error is checked against
# direct PARMA runs. Rc at each lat/lon is PARMA's own getrcpp (same cutoff model as the reference).
import subprocess, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/step4"
from wslpath_util import run_driver
rng = np.random.default_rng(20261014)


def parma_run(lines):
    return run_driver(lines)


# 1. cutoff rigidity map (PARMA getrcpp), 5 x 10 deg
lats = np.arange(-85, 85.1, 5.0); lons = np.arange(-180, 179.9, 10.0)
geo = pd.DataFrame([(la, lo) for la in lats for lo in lons], columns=["lat", "lon"])
o = parma_run([f"g{i} D 2025 11 15 {la} {lo} 0 10" for i, (la, lo) in enumerate(zip(geo.lat, geo.lon))])
geo["Rc"] = [float(x[2]) for x in o]
geo.to_csv(OUT / "rc_map_parma.csv", index=False, lineterminator="\n")

# 2. reference grid and interpolator
grid = pd.read_csv(ROOT / "data/step4/parma_grid.csv")
cruise = grid[grid.alt_km >= 6.0]


def parma_interp(W, alt, rc):
    g = cruise[(cruise.W == W) & (np.isclose(cruise.alt_km, alt))].sort_values("Rc")
    return np.exp(np.interp(rc, g.Rc, np.log(g.F_gt10))), float(g.depth.iloc[0])


rows = []
for W in sorted(cruise.W.unique()):
    for alt in sorted(cruise.alt_km.unique()):
        f, d = parma_interp(W, alt, geo.Rc.values)
        rows.append(pd.DataFrame({"W": W, "alt_km": alt, "depth": d, "lat": geo.lat, "lon": geo.lon,
                                  "Rc": geo.Rc, "PARMA": f}))
A = pd.concat(rows, ignore_index=True)

# 3. interpolation check against direct PARMA runs (60 random nodes)
chk = A.sample(60, random_state=1)
o = parma_run([f"c{i} S {w} {rc} {d} 10" for i, (w, rc, d) in enumerate(zip(chk.W, chk.Rc, chk.depth))])
direct = np.array([float(x[5]) * 3600 for x in o])
ierr = chk.PARMA.values / direct - 1
print("Rc interpolation vs direct PARMA: median |err| %.4f, max |err| %.4f" % (np.median(abs(ierr)), abs(ierr).max()))

# 4. standard models
A["S1_IEC"] = S.s1_iec_boeing(A.alt_km, A.lat)
A["S1_IEC6000"] = S.s1_iec_boeing_6000(A.alt_km, A.lat)
A["S1b_IEC_eq3"] = S.s1b_iec_boeing_eq3(A.alt_km, A.lat)
for s in ("mid", "min", "max"):
    A[f"S2_JESD_{s}"] = S.s2_jesd89(A.depth, A.Rc, s)
A["S4_AN_ABB"] = S.s4_an4589_abb(A.alt_km, A.lat)
A["S4b_AN_CoolCAD"] = S.s4b_an4589_coolcad(A.alt_km, A.lat)
A.to_csv(OUT / "audit_nodes.csv", index=False, lineterminator="\n")

# 5. summary
models = ["S1_IEC", "S1_IEC6000", "S1b_IEC_eq3", "S2_JESD_mid", "S2_JESD_min", "S2_JESD_max", "S4_AN_ABB", "S4b_AN_CoolCAD"]


def summ(df, label):
    out = []
    for m in models:
        r = df[m] / df.PARMA
        out.append(dict(subset=label, model=m, median_ratio=r.median(), p05=r.quantile(.05), p95=r.quantile(.95),
                        frac_outside_30pct=((r > 1.3) | (r < 1 / 1.3)).mean(),
                        frac_outside_factor2=((r > 2.0) | (r < 0.5)).mean(),
                        median_abs_log10=np.median(abs(np.log10(r)))))
    return out


res = []
res += summ(A, "all (6-15 km, W 0-150)")
res += summ(A[A.alt_km.between(9.0, 12.5)], "cruise 9-12.5 km, all W")
for W in (0, 50, 100, 150):
    res += summ(A[(A.alt_km.between(9.0, 12.5)) & (A.W == W)], f"cruise 9-12.5 km, W={W}")
R = pd.DataFrame(res)
R.to_csv(OUT / "audit_summary.csv", index=False, lineterminator="\n", float_format="%.6f")
pd.set_option("display.width", 200)
print(R[R.subset.isin(["all (6-15 km, W 0-150)", "cruise 9-12.5 km, W=0", "cruise 9-12.5 km, W=150"])]
      .to_string(index=False, float_format=lambda v: "%.3f" % v))

# 6. geographic-latitude ambiguity: PARMA spread across longitude at fixed |lat| (12 km, W=100)
sub = A[(np.isclose(A.alt_km, 12.0)) & (A.W == 100)]
sp = sub.groupby(sub.lat.abs()).agg(PARMA_min=("PARMA", "min"), PARMA_max=("PARMA", "max"),
                                     Rc_min=("Rc", "min"), Rc_max=("Rc", "max"), S1=("S1_IEC", "first"))
sp["max_over_min"] = sp.PARMA_max / sp.PARMA_min
sp.to_csv(OUT / "latitude_ambiguity_12km_W100.csv", lineterminator="\n", float_format="%.3f")
print(sp.loc[[0, 20, 30, 40, 45, 50, 60, 70]].round(2).to_string())
