# ML4 follow-up: apply the ML1 decision rule (better on the independent set AND not worse in region hold-outs)
# to the stepwise-selected linear forms. Term sets are frozen from ml4_symbolic_path.csv; coefficients refitted
# on the same reduced training nodes as Eq. (1) in each region hold-out.
import pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ml_analysis import tr, grid, ind, library, errstats, Eq1, X3, OUT

path = pd.read_csv(OUT / "ml4_symbolic_path.csv")
forms = {f"linear k={int(r.k)}": r.terms.split(" + ") for r in path.itertuples() if r.terms != "Eq. (1)" and r.k in (7, 8, 10)}
regions = [("in-domain: independent (1995)", np.ones(len(tr), bool), None),
           ("W>=100 (train W<=75)", (tr.W < 150).values, grid.W >= 100),
           ("Rc>12 (train Rc<=12)", (tr.Rc <= 12).values, grid.Rc > 12),
           ("alt>12.5 (train alt<=12)", (tr.alt_km <= 12).values, grid.alt_km > 12.5),
           ("alt<9 (train alt>=9)", (tr.alt_km >= 9).values, grid.alt_km < 9)]
rows = []
for rname, keep, test in regions:
    T = tr[keep]
    G, F = (ind, ind.PARMA.values) if test is None else (grid[test], grid[test].F_gt10.values)
    m = Eq1().fit(T[X3].values, np.log(T.F_gt10.values))
    rows.append(dict(test=rname, model="Eq. (1)", **errstats(m.predict(G[X3].values), F)))
    for fname, terms in forms.items():
        b, *_ = np.linalg.lstsq(library(T)[terms].values, np.log(T.F_gt10.values), rcond=None)
        rows.append(dict(test=rname, model=fname, **errstats(library(G)[terms].values @ b, F)))
df = pd.DataFrame(rows)
df.to_csv(OUT / "ml4_extrapolation.csv", index=False)
print(df.pivot(index="test", columns="model", values="p95_pct").to_string())
print(df.pivot(index="test", columns="model", values="max_pct").to_string())
