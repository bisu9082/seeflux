# ML4 strict check: term selection repeated inside each region hold-out (no information from the withheld region).
# k fixed at 8 (same parameter count as Eq. (1)) and 10; also the BIC-chosen k within each reduced training set.
import pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ml_analysis import tr, grid, ind, library, errstats, Eq1, X3, OUT


def forward(T, kmax=12):
    L, y = library(T), np.log(T.F_gt10.values)
    chosen, out = ["1"], []
    while len(chosen) < kmax:
        best = None
        for c in L.columns.difference(chosen):
            b, *_ = np.linalg.lstsq(L[chosen + [c]].values, y, rcond=None)
            rss = float(np.sum((L[chosen + [c]].values @ b - y) ** 2))
            if best is None or rss < best[0]:
                best = (rss, c, b)
        chosen = chosen + [best[1]]
        n = len(y)
        out.append((len(chosen), list(chosen), best[2], n * np.log(best[0] / n) + len(chosen) * np.log(n)))
    return out


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
    rows.append(dict(test=rname, model="Eq. (1)", k=8, **errstats(m.predict(G[X3].values), F), terms="Eq. (1)"))
    path = forward(T)
    kbic = min(path, key=lambda p: p[3])[0]
    for k, terms, b, bic in path:
        if k in (8, 10) or k == kbic:
            lab = f"nested k={k}" + (" (BIC)" if k == kbic else "")
            rows.append(dict(test=rname, model=lab, k=k, **errstats(library(G)[terms].values @ b, F), terms=" + ".join(terms)))
df = pd.DataFrame(rows)
df.to_csv(OUT / "ml4_nested.csv", index=False)
pd.set_option("display.width", 250)
print(df.drop(columns="terms").to_string(index=False))
