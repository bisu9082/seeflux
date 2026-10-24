# Step 8 round 7 (R3): counterfactual substitution instead of R^2 attribution, on the 802 date-sampled cruise points.
#  IEC as used: Boeing altitude x latitude curve at geographic latitude, no solar term.
#  (a) latitude replaced by the cutoff-equivalent latitude: the geographic latitude whose longitude-averaged vertical
#      cutoff (PARMA map, parma/build/input/CORdata.inp) equals the local Rc; this is how Normand built the curve
#      (Shea-Smart cutoffs averaged over longitude), so (a) keeps the curve and corrects only the coordinate
#  (b) (a) plus a solar term: the W dependence of the surrogate, FORMULA(h, Rc, W) / FORMULA(h, Rc, W_ref), with W_ref the
#      median W of the set, so the design value keeps its level
#  (c) solar term alone on the geographic-latitude curve
# P95/P5 of model / PARMA for each, against JESD89 (1.42). Output: data/step4/counterfactual.json
# Step 8 round 9 (F5): order-free (Shapley) split of the reduction in ln(P95/P5) between coordinate and solar term, the
# same on the pooled first and fresh samples, and the zonal-mean cutoff curve written to data/step4/zonal_cutoff.csv so
# that the step reruns without the PARMA input file. Both substitutions take their dependence from the reference (the
# PARMA cutoff map and the surrogate fitted to PARMA), so the reductions are best cases for a user-side correction.
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from seeflux_formula import flux_gt10

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
lat_grid = np.arange(0, 91)
COR = ROOT / "parma/build/input/CORdata.inp"
if COR.exists():
    d = np.loadtxt(COR, skiprows=2)
    zonal = np.array([d[np.abs(np.round(d[:, 1])) == la, 2].mean() for la in lat_grid])   # both hemispheres pooled
    zmono = np.minimum.accumulate(zonal)                                                    # monotone decreasing with |lat|
    pd.DataFrame(dict(abs_lat_deg=lat_grid, zonal_mean_Rc_GV=zonal, monotone_Rc_GV=zmono)).to_csv(
        OUT / "zonal_cutoff.csv", index=False, lineterminator="\n", float_format="%.17g")
else:
    zmono = pd.read_csv(OUT / "zonal_cutoff.csv", float_precision="round_trip").monotone_Rc_GV.to_numpy()
sp = lambda r: float(np.percentile(r, 95) / np.percentile(r, 5))


def cases(C, w_ref):
    # IEC/PARMA ratios as used, (a), (b), (c)
    lat_eq = np.interp(-C.Rc.values, -zmono, lat_grid.astype(float))   # |lat_eq| with zonal(|lat_eq|) = Rc
    iec_a = S.s1_iec_boeing(C.alt_km.values, lat_eq)
    sol = flux_gt10(C.alt_km.values, C.Rc.values, C.W.values) / flux_gt10(C.alt_km.values, C.Rc.values, np.full(len(C), w_ref))
    return C.S1_IEC / C.PARMA, iec_a / C.PARMA, iec_a * sol / C.PARMA, C.S1_IEC * sol / C.PARMA


def shapley(r0, ra, rb, rc):
    l0, la, lb, lc = (np.log(sp(x)) for x in (r0, ra, rb, rc))
    coord = 0.5 * ((l0 - la) + (lc - lb)); solar = 0.5 * ((l0 - lc) + (la - lb))
    return dict(ln_spread_as_used=l0, ln_spread_both=lb, coordinate=float(coord), solar=float(solar),
                coordinate_share_of_reduction=float(coord / (l0 - lb)))


V = pd.read_csv(OUT / "validation_direct.csv")
C = V[V.alt_km.between(9.0, 12.5)].copy()
w_ref = float(C.W.median())
r0, ra, rb, rc = cases(C, w_ref)
res = dict(n=len(C), W_ref=w_ref,
           IEC_as_used=sp(r0),
           a_cutoff_equivalent_latitude=sp(ra),
           b_cutoff_latitude_plus_solar=sp(rb),
           c_solar_only=sp(rc),
           JESD89=sp(C.S2_JESD_mid / C.PARMA),
           median_a=float(np.median(ra)), median_b=float(np.median(rb)),
           zonal_cutoff_GV_at_lat={str(l): float(zmono[l]) for l in (0, 20, 40, 45, 50, 60, 70)})
res["shapley_first"] = shapley(r0, ra, rb, rc)
F = pd.read_csv(OUT / "fresh_sample.csv"); F = F[F.alt_km.between(9.0, 12.5)]
P = pd.concat([C[F.columns.intersection(C.columns)], F[F.columns.intersection(C.columns)]], ignore_index=True)
q = cases(P, w_ref)
res["pooled"] = dict(n=len(P), IEC_as_used=sp(q[0]), a=sp(q[1]), b=sp(q[2]), c=sp(q[3]),
                     JESD89=sp(P.S2_JESD_mid / P.PARMA), shapley=shapley(*q))
json.dump(res, open(OUT / "counterfactual.json", "w"), indent=1)
print(json.dumps(res, indent=1))
