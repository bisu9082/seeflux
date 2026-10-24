# Step 4: PARMA 4.10 reference grid (free atmosphere, g = 10)
# altitude 0 and 6.0-15.0 km (0.5 km), Rc 0-18 GV (0.5 GV; PARMA's own cutoff map reaches about 17.6 GV),
# W-index 0-150 (25). Output: data/step4/parma_grid.csv, fluxes in n/cm2/h.
import numpy as np, pandas as pd, pathlib, time
from wslpath_util import run_driver

ROOT = pathlib.Path(__file__).resolve().parents[1]
alts = [0.0] + list(np.round(np.arange(6.0, 15.0001, 0.5), 2))
rcs = list(np.round(np.arange(0.0, 18.0001, 0.5), 2))
ws = [0, 25, 50, 75, 100, 125, 150]
G = 10.0

nodes = [(w, rc, h) for w in ws for rc in rcs for h in alts]
lines = [f"n{i} A {w} {rc} {h} {G}" for i, (w, rc, h) in enumerate(nodes)]
t0 = time.time()
res = run_driver(lines)
rows = [dict(W=float(w), Rc=float(rc), alt_km=float(h), depth=float(o[3]),
             F_gt10=float(o[5]) * 3600, F_gt1=float(o[6]) * 3600,
             F_1to10=float(o[7]) * 3600, F_total=float(o[8]) * 3600) for (w, rc, h), o in zip(nodes, res)]
df = pd.DataFrame(rows)
out = ROOT / "data/step4/parma_grid.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, lineterminator="\n")
print(len(df), "nodes,", round(time.time() - t0, 1), "s ->", out)
