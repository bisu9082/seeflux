# Step 8 round 7 (R3): counterfactual substitution instead of R^2 attribution, on the 802 date-sampled cruise points.
#  IEC as used: Boeing altitude x latitude curve at geographic latitude, no solar term.
#  (a) latitude replaced by the cutoff-equivalent latitude: the geographic latitude whose longitude-averaged vertical
#      cutoff (PARMA map, parma/build/input/CORdata.inp) equals the local Rc; this is how Normand built the curve
#      (Shea-Smart cutoffs averaged over longitude), so (a) keeps the curve and corrects only the coordinate
#  (b) (a) plus a solar term: the W dependence of the surrogate, FORMULA(h, Rc, W) / FORMULA(h, Rc, W_ref), with W_ref the
#      median W of the set, so the design value keeps its level
#  (c) solar term alone on the geographic-latitude curve
# P95/P5 of model / PARMA for each, against JESD89 (1.42). Output: data/step4/counterfactual.json
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from seeflux_formula import flux_gt10

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
d = np.loadtxt(ROOT / "parma/build/input/CORdata.inp", skiprows=2)
lat_grid = np.arange(0, 91)
zonal = np.array([d[np.abs(np.round(d[:, 1])) == la, 2].mean() for la in lat_grid])   # both hemispheres pooled
zmono = np.minimum.accumulate(zonal)                                                    # monotone decreasing with |lat|
V = pd.read_csv(OUT / "validation_direct.csv")
C = V[V.alt_km.between(9.0, 12.5)].copy()
# invert: |lat_eq| such that zonal(|lat_eq|) = Rc (interpolate on the decreasing curve)
lat_eq = np.interp(-C.Rc.values, -zmono, lat_grid.astype(float))
iec_a = S.s1_iec_boeing(C.alt_km.values, lat_eq)
w_ref = float(C.W.median())
sol = flux_gt10(C.alt_km.values, C.Rc.values, C.W.values) / flux_gt10(C.alt_km.values, C.Rc.values, np.full(len(C), w_ref))
sp = lambda r: float(np.percentile(r, 95) / np.percentile(r, 5))
res = dict(n=len(C), W_ref=w_ref,
           IEC_as_used=sp(C.S1_IEC / C.PARMA),
           a_cutoff_equivalent_latitude=sp(iec_a / C.PARMA),
           b_cutoff_latitude_plus_solar=sp(iec_a * sol / C.PARMA),
           c_solar_only=sp(C.S1_IEC * sol / C.PARMA),
           JESD89=sp(C.S2_JESD_mid / C.PARMA),
           median_a=float(np.median(iec_a / C.PARMA)), median_b=float(np.median(iec_a * sol / C.PARMA)),
           zonal_cutoff_GV_at_lat={str(l): float(zmono[l]) for l in (0, 20, 40, 45, 50, 60, 70)})
json.dump(res, open(OUT / "counterfactual.json", "w"), indent=1)
print(json.dumps(res, indent=1))
