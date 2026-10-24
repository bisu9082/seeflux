# Step 8 round 9 (F2, R1/R4): the 6000 design value on the neutron basis that CM-AS-004 states ("a mean neutron flux"),
# next to the neutron-and-proton basis used so far. For the first independent set (validation_direct.csv, all altitudes;
# r7_points_np.csv for protons) and for the bounding environment and routes.
#  (1) the larger of 6000 and the scaled curves (6000 anchor) against PARMA n and n+p: share below one at several k,
#      by altitude band, and where the exceptions lie
#  (2) break-even k* for flat 6000 against the PARMA neutron peak on the routes, solar minimum years, by level
#  (3) bounding environment (Rc = 0, W = 0) and the standard's polar value against PARMA n and n+p
# Output: data/step4/floor_neutron.json
import json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
V = pd.read_csv(OUT / "validation_direct.csv"); Pp = pd.read_csv(OUT / "r7_points_np.csv")
V["n"] = Pp.PARMA_n2; V["np"] = Pp.PARMA_n2 + Pp.PARMA_p
V["floor"] = np.maximum(6000.0, V.S1_IEC * 6000 / 5600)
KS = (1.0, 1.07, 1.23, 1.28, 1.36)
bands = {"6-9 km": (6.0, 9.0), "cruise 9-12.5 km": (9.0, 12.5), "12.5-15 km": (12.5, 15.0)}
res = dict(floor={})
for basis in ("n", "np"):
    d = {}
    for lab, (a, b) in bands.items():
        S = V[V.alt_km.between(a, b, inclusive="right" if a == 12.5 else ("left" if b == 9.0 else "both"))]
        r = S.floor / S[basis]
        ex = S[r < 1]
        d[lab] = dict(n=len(S), median=float(r.median()), **{f"below1_k{k}": float(np.mean(k * r < 1)) for k in KS},
                      exceptions_Rc_max=float(ex.Rc.max()) if len(ex) else None,
                      exceptions_alt_min=float(ex.alt_km.min()) if len(ex) else None)
    res["floor"][basis] = d
# (2) routes, solar minimum years
L = pd.read_csv(OUT / "routes_levels.csv")
res["route_peak_kstar"] = {}
for lev in ("FL370", "FL390", "FL410"):
    s = L[(L.level == lev) & L.cond.isin(["y2009", "y2019"])]
    gn, gnp = s.groupby("route").max_n_cruise.max(), s.groupby("route").max_np_cruise.max()
    res["route_peak_kstar"][lev] = dict(n=float(gn.max() / 6000), n_route=gn.idxmax(), np=float(gnp.max() / 6000), np_route=gnp.idxmax())
# (3) bounding environment and polar value
BL = pd.read_csv(OUT / "revision2_bounding_by_level.csv"); BL = BL[BL.Rc == 0].set_index("FL")
import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent)); import standards as S
pol = {fl: float(S.s1_iec_boeing(np.array([BL.loc[fl, "alt_km"]]), np.array([89.0]))[0] * 6000 / 5600) for fl in BL.index}
res["bounding"] = {f"FL{fl}": dict(n=float(BL.loc[fl, "n"]), np=float(BL.loc[fl, "np"]), kstar_n=float(BL.loc[fl, "n"] / 6000),
                                    kstar_np=float(BL.loc[fl, "np"] / 6000), polar_IEC=pol.get(fl),
                                    polar_over_n=pol[fl] / BL.loc[fl, "n"] if fl in pol else None,
                                    polar_over_np=pol[fl] / BL.loc[fl, "np"] if fl in pol else None)
                   for fl in BL.index}
json.dump(res, open(OUT / "floor_neutron.json", "w"), indent=1)
print(json.dumps(res, indent=1))
