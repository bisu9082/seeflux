# Step 4 extension after Step 8 review (2026-10-16): analyses on the independent set requested by the reviewers.
#  (1) bias-dependent shares at the reference-scale bounds k = 0.88, 1, 1.37 (true ratio = k * r)
#  (2) flat design values 5600 and 6000 n/cm2/h applied without altitude or latitude scaling
#  (3) breakdown by hemisphere and latitude band; bootstrap 95% intervals for Table I quantities (B = 2000)
#  (4) sensitivity to GLE-flagged days and W outside 0-150
#  (5) JESD89 with a date-matched solar branch (linear and two-branch mappings of W; illustrative)
#  (6) conservative multiplier for the surrogate
import json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from solar_w import daily_w

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/step4"
SEED, B = 20261016, 2000
K = (0.88, 1.0, 1.37)
V = pd.read_csv(OUT / "validation_direct.csv", parse_dates=["date"])
C = V[V.alt_km.between(9.0, 12.5)].copy()
C["FLAT5600"], C["FLAT6000"] = 5600.0, 6000.0
C["S1_IEC_alt45"] = S.s1_iec_boeing(C.alt_km.values, np.full(len(C), 45.0))

# (5) JESD89 date-matched solar branch. Mapping of W to the BSYD branches is not defined by the standard;
# two illustrative choices: linear between the lowest and highest annual-mean W of 2009-2025, and a two-branch switch.
Dw = daily_w()
ann = Dw[~Dw.gle & (Dw.date.dt.year <= 2025)].groupby(Dw.date.dt.year).W.mean()
# JESD89B A.3: interpolate by neutron-monitor level between typical active-sun and quiet-sun values, excluding the
# atypical quiet sun of late 2008 to early 2010 -> quiet bound = lowest annual mean W outside 2008-2010
W_LO, W_HI = float(ann.drop([y for y in (2008, 2009, 2010) if y in ann.index]).min()), float(ann.max())
f = np.clip((C.W - W_LO) / (W_HI - W_LO), 0, 1)
jmin, jmax = S.s2_jesd89(C.depth, C.Rc, "min"), S.s2_jesd89(C.depth, C.Rc, "max")
C["S2_JESD_lin"] = (1 - f) * jmin + f * jmax
C["S2_JESD_2br"] = np.where(C.W < 0.5 * (W_LO + W_HI), jmin, jmax)

MODELS = ["S1_IEC", "S1_IEC6000", "S1_IEC_alt45", "FLAT5600", "FLAT6000", "S2_JESD_mid", "S2_JESD_lin", "S2_JESD_2br", "S4_AN_ABB",
          "S4b_AN_CoolCAD", "FORMULA"]


def stats(r):
    return dict(median=np.median(r), p05=np.percentile(r, 5), p95=np.percentile(r, 95),
                **{f"below1_k{k}": np.mean(k * r < 1) for k in K},
                **{f"out30_k{k}": np.mean((k * r > 1.3) | (k * r < 1 / 1.3)) for k in K},
                out2=np.mean((r > 2) | (r < 0.5)), spread=np.percentile(r, 95) / np.percentile(r, 5))


rows = []
for m in MODELS:
    s = C[C.in_domain] if m == "FORMULA" else C
    rows.append(dict(model=m, n=len(s), **stats((s[m] / s.PARMA).values)))
T1 = pd.DataFrame(rows)
T1.to_csv(OUT / "revision_k_flat_jesd.csv", index=False, lineterminator="\n", float_format="%.6f")

# (3) bootstrap intervals and breakdown
rng = np.random.default_rng(SEED)
bt = []
for m in ["S1_IEC", "S1_IEC6000", "FLAT6000", "S2_JESD_mid", "S4_AN_ABB", "S4b_AN_CoolCAD", "FORMULA"]:
    s = C[C.in_domain] if m == "FORMULA" else C
    r = (s[m] / s.PARMA).values
    draws = np.array([[np.median(x), np.percentile(x, 5), np.percentile(x, 95), np.mean((x > 1.3) | (x < 1 / 1.3)), np.mean(x < 1),
                       np.percentile(x, 95) / np.percentile(x, 5)]
                      for x in (r[rng.integers(0, len(r), len(r))] for _ in range(B))])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    qn = ["median", "p05", "p95", "out30", "below1", "spread"]
    bt.append(dict(model=m, **{f"{q}_lo": a for q, a in zip(qn, lo)}, **{f"{q}_hi": b for q, b in zip(qn, hi)}))
pd.DataFrame(bt).to_csv(OUT / "revision_bootstrap_ci.csv", index=False, lineterminator="\n", float_format="%.6f")

C["hemi"] = np.where(C.lat >= 0, "N", "S")
C["band"] = pd.cut(C.lat.abs(), [0, 30, 60, 90], labels=["0-30", "30-60", "60-90"], include_lowest=True)
br = []
for key in ("hemi", "band"):
    for g, s in C.groupby(key, observed=True):
        r = s.S1_IEC / s.PARMA
        br.append(dict(by=key, group=str(g), n=len(s), share_of_points=len(s) / len(C), IEC_median=r.median(),
                       IEC_below1=float((r < 1).mean()), share_of_all_below1=float((r < 1).sum() / (C.S1_IEC / C.PARMA < 1).sum())))
pd.DataFrame(br).to_csv(OUT / "revision_breakdown.csv", index=False, lineterminator="\n", float_format="%.6f")

# (4) GLE-flagged days and W outside 0-150
gle_days = set(Dw[Dw.gle].date)
flag = C.date.isin(gle_days) | (C.W > 150) | (C.W < 0)
r_all, r_ex = C.S1_IEC / C.PARMA, C[~flag].S1_IEC / C[~flag].PARMA
sens = dict(n_flagged=int(flag.sum()), flagged_dates=sorted(str(d.date()) for d in C[flag].date),
            IEC_all=dict(median=r_all.median(), p05=r_all.quantile(.05), p95=r_all.quantile(.95), below1=float((r_all < 1).mean())),
            IEC_excl=dict(median=r_ex.median(), p05=r_ex.quantile(.05), p95=r_ex.quantile(.95), below1=float((r_ex < 1).mean())),
            all_points_gle_dates=sorted(str(d.date()) for d in V[V.date.isin(gle_days)].date),
            JESD_mapping=dict(W_lo=W_LO, W_hi=W_HI, year_lo=int(ann.drop([y for y in (2008, 2009, 2010) if y in ann.index]).idxmin()), year_hi=int(ann.idxmax())))

# (6) conservative multiplier for the surrogate: PARMA / surrogate quantiles, all in-domain points and Rc < 2 GV
A = V[V.in_domain]
q = lambda x: dict(q95=float(np.percentile(x, 95)), q99=float(np.percentile(x, 99)), max=float(np.max(x)))
sens["surrogate_multiplier"] = dict(all=q(A.PARMA / A.FORMULA), Rc_lt2=q(A[A.Rc < 2].PARMA / A[A.Rc < 2].FORMULA),
                                    n_all=int(len(A)), n_Rc_lt2=int((A.Rc < 2).sum()),
                                    coverage={f"{c:.2f}": dict(all=float(np.mean(c * A.FORMULA >= A.PARMA)),
                                                               Rc_lt2=float(np.mean(c * A[A.Rc < 2].FORMULA >= A[A.Rc < 2].PARMA)))
                                              for c in (1.09, 1.10, 1.11, 1.12)})
# multiplier calibrated on the grid (all nodes 6-15 km, W 0-150), validated on the independent set (Step 8 round 3:
# calibration and validation on separate sets); corner = above 12.5 km with Rc < 2 GV
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from seeflux_formula import flux_gt10
Gd = pd.read_csv(OUT / "parma_grid.csv"); Gd = Gd[Gd.alt_km >= 6.0]
Gd = Gd[~(Gd.alt_km.isin([6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0]) & Gd.Rc.isin([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
         & Gd.W.isin([0, 75, 150]))]  # held-out nodes only (fit_formula.py training design)
rg = Gd.F_gt10 / flux_gt10(Gd.alt_km.values, Gd.Rc.values, Gd.W.values)
cal = {"all": float(np.ceil(100 * np.percentile(rg, 99)) / 100), "Rc_lt2": float(np.ceil(100 * np.percentile(rg[Gd.Rc < 2], 99)) / 100)}
corner = A[(A.alt_km > 12.5) & (A.Rc < 2)]
sens["surrogate_multiplier_grid"] = dict(
    grid_q99=float(np.percentile(rg, 99)), grid_q99_Rc_lt2=float(np.percentile(rg[Gd.Rc < 2], 99)), grid_max=float(rg.max()),
    chosen=cal, n_grid=int(len(Gd)), n_corner=int(len(corner)),
    independent_coverage={k: dict(all=float(np.mean(c * A.FORMULA >= A.PARMA)),
                                  Rc_lt2=float(np.mean(c * A[A.Rc < 2].FORMULA >= A[A.Rc < 2].PARMA)),
                                  corner=float(np.mean(c * corner.FORMULA >= corner.PARMA))) for k, c in cal.items()})
# flat value: where PARMA exceeds the flat value
sens["PARMA_exceeds_flat"] = {str(v): float((C.PARMA > v).mean()) for v in (5600, 6000)}
# IEC curves outside the cruise band (independent set): regional 6-9 km and business-jet 12.5-15 km
bands = {}
for name, lo, hi, incl in (("6-9 km", 6.0, 9.0, "left"), ("12.5-15 km", 12.5, 15.0, "right")):
    s = V[V.alt_km.between(lo, hi, inclusive=incl)]   # the cruise band 9-12.5 km keeps both ends; bands do not overlap it
    r = s.S1_IEC / s.PARMA
    bands[name] = dict(n=int(len(s)), median=float(r.median()), p05=float(r.quantile(.05)), p95=float(r.quantile(.95)),
                       spread=float(r.quantile(.95) / r.quantile(.05)), below1=float((r < 1).mean()))
sens["IEC_bands"] = bands
json.dump(sens, open(OUT / "revision_sensitivity.json", "w"), indent=1, default=float)

pd.set_option("display.width", 250)
print(T1.round(3).to_string(index=False))
print(pd.DataFrame(bt).round(3).to_string(index=False))
print(pd.DataFrame(br).round(3).to_string(index=False))
print(json.dumps(sens, indent=1, default=float))
