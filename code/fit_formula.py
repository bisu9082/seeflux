# Step 4 RQ3: minimum-data screening formula for the >10 MeV neutron flux, fitted to a sparse subset of the
# PARMA grid and evaluated on all held-out grid nodes and on the geographic audit nodes.
# Three candidate forms fixed in advance (analysis_design.md section 5); selection by AIC/BIC on training nodes.
#   M1  ln F = b0 + b1 h + b2 h^2 - ln(1 + a Rc^n) + c W                         (separable)
#   M2  ln F = b0 + b1 h + b2 h^2 - (1 + d h) ln(1 + a Rc^n) + c W               (altitude-dependent cutoff effect)
#   M3  ln F = b0 + b1 h + b2 h^2 - (1 + d h) ln(1 + a Rc^n) + c W / (1 + e Rc)  (+ weaker solar effect at high Rc)
# h in km, Rc in GV, W = PARMA W-index. Uncertainty: nonparametric bootstrap over training nodes, B = 10,000.
import numpy as np, pandas as pd, pathlib, json, time
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/step4"
grid = pd.read_csv(OUT / "parma_grid.csv")
grid = grid[grid.alt_km >= 6.0].reset_index(drop=True)

TR_ALT = [6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0]
TR_RC = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
TR_W = [0, 75, 150]
is_tr = grid.alt_km.isin(TR_ALT) & grid.Rc.isin(TR_RC) & grid.W.isin(TR_W)
tr, te = grid[is_tr], grid[~is_tr]


def model(p, h, rc, w, form):
    b0, b1, b2, a, n = p[:5]
    sat = np.log1p(a * np.power(rc, n))
    if form == "M1":
        return b0 + b1 * h + b2 * h * h - sat + p[5] * w
    if form == "M2":
        return b0 + b1 * h + b2 * h * h - (1 + p[6] * h) * sat + p[5] * w
    return b0 + b1 * h + b2 * h * h - (1 + p[6] * h) * sat + p[5] * w / (1 + p[7] * rc)


P0 = {"M1": [3, 1, -0.03, 0.05, 1.5, -0.002], "M2": [3, 1, -0.03, 0.05, 1.5, -0.002, 0.0],
      "M3": [3, 1, -0.03, 0.05, 1.5, -0.002, 0.0, 0.1]}
LB = {"M1": [-50, -10, -1, 1e-6, 0.1, -1], "M2": [-50, -10, -1, 1e-6, 0.1, -1, -0.5],
      "M3": [-50, -10, -1, 1e-6, 0.1, -1, -0.5, 0.0]}
UB = {"M1": [50, 10, 1, 10, 5, 1], "M2": [50, 10, 1, 10, 5, 1, 0.5], "M3": [50, 10, 1, 10, 5, 1, 0.5, 5.0]}


def fit(df, form, p0=None):
    y = np.log(df.F_gt10.values)
    f = lambda p: model(p, df.alt_km.values, df.Rc.values, df.W.values, form) - y
    return least_squares(f, p0 or P0[form], bounds=(LB[form], UB[form]), x_scale="jac", max_nfev=20000)


def errs(p, df, form):
    e = model(p, df.alt_km.values, df.Rc.values, df.W.values, form) - np.log(df.F_gt10.values)
    r = np.exp(e)
    return dict(median_abs_pct=float(np.median(abs(r - 1)) * 100), p95_abs_pct=float(np.percentile(abs(r - 1), 95) * 100),
                max_abs_pct=float(abs(r - 1).max() * 100), frac_outside_30pct=float(((r > 1.3) | (r < 1 / 1.3)).mean()))


sel = {}
for form in ("M1", "M2", "M3"):
    s = fit(tr, form); k = len(s.x); n = len(tr); rss = float(np.sum(s.fun ** 2))
    sel[form] = dict(params=s.x.tolist(), k=k, n=n, rss=rss, AIC=n * np.log(rss / n) + 2 * k,
                     BIC=n * np.log(rss / n) + k * np.log(n), train=errs(s.x, tr, form), heldout=errs(s.x, te, form))
    print(form, "AIC %.1f BIC %.1f | held-out median %.1f%%, p95 %.1f%%, max %.1f%%, outside30 %.3f" %
          (sel[form]["AIC"], sel[form]["BIC"], *[sel[form]["heldout"][x] for x in
           ("median_abs_pct", "p95_abs_pct", "max_abs_pct", "frac_outside_30pct")]))
best = min(sel, key=lambda f: sel[f]["BIC"])
print("selected by BIC:", best)

# bootstrap
rng = np.random.default_rng(20261014)
B = 10000; pb = []; t0 = time.time()
x0 = sel[best]["params"]
for b in range(B):
    idx = rng.integers(0, len(tr), len(tr))
    pb.append(fit(tr.iloc[idx], best, x0).x)
pb = np.array(pb)
names = ["b0", "b1", "b2", "a", "n", "c", "d", "e"][:pb.shape[1]]
ci = {nm: [float(np.percentile(pb[:, i], 2.5)), float(np.percentile(pb[:, i], 97.5))] for i, nm in enumerate(names)}
# held-out error distribution across all bootstrap refits (p95 of |error|). Note: the bootstrap resamples
# training nodes of a deterministic surface, so these intervals describe fit stability, not measurement noise.
p95s = [errs(p, te, best)["p95_abs_pct"] for p in pb]
print("bootstrap %d refits in %.0f s; held-out p95 |err| 95%% CI %.1f-%.1f%%" %
      (B, time.time() - t0, np.percentile(p95s, 2.5), np.percentile(p95s, 97.5)))

# per-group held-out errors (A5: no pooled-only reporting)
pg = []
groups = [("W", te.groupby("W")), ("alt_km", te.groupby(pd.cut(te.alt_km, [5.9, 9, 12.5, 15.1]), observed=True)),
          ("Rc_GV", te.groupby(pd.cut(te.Rc, [-0.1, 3, 8, 18.1]), observed=True))]
for gname, gb in groups:
    for key, grp in gb:
        pg.append(dict(group=f"{gname}={key}", n=len(grp), **errs(np.array(x0), grp, best)))
pd.DataFrame(pg).to_csv(OUT / "formula_heldout_by_group.csv", index=False, lineterminator="\n", float_format="%.3f")
print(pd.DataFrame(pg).round(2).to_string(index=False))

# region hold-out (extrapolation) tests (A5): refit without a whole region of the domain, test on that region
ext = {}
for name, keep, test in [("W: train W<=75, test W>=100", tr.W < 150, grid.W >= 100),
                         ("Rc: train Rc<=12, test Rc>12", tr.Rc <= 12, grid.Rc > 12),
                         ("alt: train alt<=12, test alt>12.5", tr.alt_km <= 12, grid.alt_km > 12.5),
                         ("alt: train alt>=9, test alt<9", tr.alt_km >= 9, grid.alt_km < 9)]:
    sx = fit(tr[keep], best, x0)
    ext[name] = dict(n_train=int(keep.sum()), n_test=int(test.sum()), **errs(sx.x, grid[test], best))
    print("extrapolation %-34s median %.1f%%, p95 %.1f%%, max %.1f%%" %
          (name, ext[name]["median_abs_pct"], ext[name]["p95_abs_pct"], ext[name]["max_abs_pct"]))

json.dump(dict(extrapolation=ext, training_nodes=dict(alt=TR_ALT, Rc=TR_RC, W=TR_W, n=int(is_tr.sum())), heldout_n=int((~is_tr).sum()),
               candidates=sel, selected=best, param_names=names, bootstrap_B=B, param_CI95=ci,
               heldout_p95_CI95=[float(np.percentile(p95s, 2.5)), float(np.percentile(p95s, 97.5))]),
          open(OUT / "formula_fit.json", "w"), indent=1)
np.save(OUT / "formula_bootstrap_params.npy", pb)

# head-to-head on geographic audit nodes
A = pd.read_csv(OUT / "audit_nodes.csv")
A["FORMULA"] = np.exp(model(np.array(x0), A.alt_km.values, A.Rc.values, A.W.values, best))
A.to_csv(OUT / "audit_nodes.csv", index=False, lineterminator="\n")
cr = A[A.alt_km.between(9.0, 12.5)]
for m in ["FORMULA", "S1_IEC", "S2_JESD_mid", "S4_AN_ABB"]:
    r = cr[m] / cr.PARMA
    print("%-12s cruise 9-12.5 km: median ratio %.3f, p05 %.3f, p95 %.3f, outside +/-30%% %.3f" %
          (m, r.median(), r.quantile(.05), r.quantile(.95), ((r > 1.3) | (r < 1 / 1.3)).mean()))
