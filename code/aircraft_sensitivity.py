# Step 4: sensitivity of the >10 MeV flux to the aircraft-structure option of PARMA (g < 0: |g| = aircraft
# mass in units of 100 t, Sato 2016 / PARMA readme) relative to the free atmosphere (g = 10) at cruise.
import subprocess, itertools, pandas as pd, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
from wslpath_util import run_driver
nodes = list(itertools.product([0, 75, 150], [0.0, 5.0, 12.0], [9.0, 11.0, 13.0]))
G = [10, -0.5, -2.0, -4.0]
lines = []
for w, rc, h in nodes:
    for g in G:
        lines.append(f"{w}_{rc}_{h}_{g} A {w} {rc} {h} {g}")
rec = []
for x in run_driver(lines):
    w, rc, h, g = x[0].split("_")
    rec.append(dict(W=float(w), Rc=float(rc), alt_km=float(h), g=float(g), F_gt10=float(x[5]), F_total=float(x[8])))
df = pd.DataFrame(rec)
base = df[df.g == 10].set_index(["W", "Rc", "alt_km"])
df = df[df.g != 10].join(base[["F_gt10", "F_total"]], on=["W", "Rc", "alt_km"], rsuffix="_free")
df["ratio_gt10"] = df.F_gt10 / df.F_gt10_free; df["ratio_total"] = df.F_total / df.F_total_free
df.to_csv(ROOT / "data/step4/aircraft_sensitivity.csv", index=False, lineterminator="\n", float_format="%.4g")
print(df.groupby("g")[["ratio_gt10", "ratio_total"]].agg(["min", "median", "max"]).round(3).to_string())
