# Step 8 round 7 checks (2026-10-19):
#  (1) certification-compliant use: CM-AS-004 makes 6000 a minimum, so the value in use is max(6000, scaled curves on the
#      6000 basis); compared with PARMA neutrons + protons at the 2000 independent points (cruise 9-12.5 km, 6-9 km,
#      12.5-15 km), shares below one at k = 0.88, 1, 1.28, 1.37
#  (2) offset that varies with Rc or W: true = PARMA / k(Rc) with k(Rc) from a linear fit of the ER-2 total ratios
#      against Rc, normalized to a common scale of one; and k(W) = 1 +/- 0.1 across W = 0-150; spread of IEC and JESD89
#  (3) geographic grid weighted by area (cos latitude) instead of by node: medians at W = 0 and 150, shares outside 30%
#  (4) IEC latitude curve evaluated at geomagnetic (dipole) latitude instead of geographic latitude
# Output: data/step4/r7_checks.json, r7_points_np.csv
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver as _rd

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"


def run_driver(lines, chunk=300):
    out = []
    for i in range(0, len(lines), chunk):
        out += _rd(lines[i:i + chunk])
    return out


V = pd.read_csv(OUT / "validation_direct.csv", parse_dates=["date"])
PTS = OUT / "r7_points_np.csv"
if PTS.exists() and len(pd.read_csv(PTS)) == len(V):
    V["PARMA_n2"], V["PARMA_p"] = pd.read_csv(PTS).PARMA_n2.values, pd.read_csv(PTS).PARMA_p.values
else:
    o = run_driver([f"v{i} D {d.year} {d.month} {d.day} {la:.4f} {lo:.4f} {h:.4f} 10 PR"
                    for i, (d, la, lo, h) in enumerate(zip(V.date, V.lat, V.lon, V.alt_km))])
    V["PARMA_n2"] = [float(x[5]) * 3600 for x in o]; V["PARMA_p"] = [float(x[9]) * 3600 for x in o]
    V[["PARMA_n2", "PARMA_p"]].to_csv(PTS, index=False, lineterminator="\n", float_format="%.6g")
assert float(np.max(np.abs(V.PARMA_n2 / V.PARMA - 1))) < 0.01
V["np"] = V.PARMA_n2 + V.PARMA_p
V["IEC6000"] = V.S1_IEC * 6000 / 5600
V["CM_use"] = np.maximum(6000.0, V.IEC6000)
K = (0.88, 1.0, 1.28, 1.37)
res = {"cm_compliant": {}}
for band, lo, hi, inc in (("cruise 9-12.5 km", 9.0, 12.5, "both"), ("6-9 km", 6.0, 9.0, "left"), ("12.5-15 km", 12.5, 15.0, "right")):
    s = V[V.alt_km.between(lo, hi, inclusive=inc)]
    r_cm = (s.CM_use / s.np).values; r_sc = (s.IEC6000 / s.np).values
    below = {f"k{k}": float(np.mean(k * r_cm < 1)) for k in K}
    sel = s[r_cm < 1]
    res["cm_compliant"][band] = dict(n=len(s), CM_median=float(np.median(r_cm)), CM_below1=below,
                                     scaled_alone_below1={f"k{k}": float(np.mean(k * r_sc < 1)) for k in K},
                                     CM_below1_Rc_max=float(sel.Rc.max()) if len(sel) else None,
                                     CM_below1_alt_min=float(sel.alt_km.min()) if len(sel) else None,
                                     CM_min_ratio=float(r_cm.min()))
# (2) offset varying with Rc or W (cruise points), relative to a common scale
C = V[V.alt_km.between(9.0, 12.5)].copy()
B = pd.read_csv(OUT / "bench_goldhagen2004.csv"); B = B[B.alt_km > 0]
a, b = np.polyfit(B.Rc, B.PARMA_over_meas, 1)
kRc = (a * C.Rc + b); kRc = kRc / kRc.mean()
sp = lambda r: float(np.percentile(r, 95) / np.percentile(r, 5))
off = dict(kRc_fit=dict(slope_per_GV=float(a), intercept=float(b)), base=dict(IEC=sp(C.S1_IEC / C.PARMA), JESD=sp(C.S2_JESD_mid / C.PARMA)))
off["Rc_offset"] = dict(IEC=sp(C.S1_IEC / (C.PARMA / kRc)), JESD=sp(C.S2_JESD_mid / (C.PARMA / kRc)))
for sgn in (+1, -1):
    kW = 1 + sgn * 0.1 * (C.W.clip(0, 150) / 150 - 0.5) * 2
    off[f"W_offset_{'+' if sgn > 0 else '-'}10pct"] = dict(IEC=sp(C.S1_IEC / (C.PARMA / kW)), JESD=sp(C.S2_JESD_mid / (C.PARMA / kW)))
    both = kRc * kW
    off[f"Rc_and_W_{'+' if sgn > 0 else '-'}"] = dict(IEC=sp(C.S1_IEC / (C.PARMA / both)), JESD=sp(C.S2_JESD_mid / (C.PARMA / both)))
off["IEC_over_JESD_spread_range"] = [min(v["IEC"] / v["JESD"] for k_, v in off.items() if isinstance(v, dict) and "IEC" in v),
                                     max(v["IEC"] / v["JESD"] for k_, v in off.items() if isinstance(v, dict) and "IEC" in v)]
res["offset_stress"] = off
# (3) geographic grid, area weighting
G = pd.read_csv(OUT / "audit_nodes.csv"); G = G[G.alt_km.between(9.0, 12.5)]
w = np.cos(np.radians(G.lat))


def wmed(x, wt):
    o = np.argsort(x.values); c = np.cumsum(wt.values[o]); return float(x.values[o][np.searchsorted(c, c[-1] / 2)])


gr = {"polar_node_share_|lat|>=60": float(np.mean(np.abs(G.lat) >= 60)), "polar_area_share_|lat|>=60": float(w[np.abs(G.lat) >= 60].sum() / w.sum())}
for m in ("S1_IEC", "S2_JESD_mid", "S4_AN_ABB"):
    r = G[m] / G.PARMA
    for W in (0.0, 150.0):
        sel = G.W == W
        gr[f"{m}_median_W{int(W)}"] = dict(node=float(np.median(r[sel])), area=wmed(r[sel], w[sel]))
    out = ((r > 1.3) | (r < 1 / 1.3)).astype(float)
    gr[f"{m}_out30"] = dict(node=float(out.mean()), area=float(np.average(out, weights=w)))
res["grid_area_weighting"] = gr
# (4) geomagnetic (centred dipole) latitude, IGRF-13 epoch 2020 north geomagnetic pole 80.65 N, 72.68 W
PL, PO = np.radians(80.65), np.radians(-72.68)
la, lo = np.radians(C.lat), np.radians(C.lon)
mlat = np.degrees(np.arcsin(np.sin(la) * np.sin(PL) + np.cos(la) * np.cos(PL) * np.cos(lo - PO)))
iec_gm = S.s1_iec_boeing(C.alt_km.values, mlat)
r_gm = iec_gm / C.PARMA
res["geomagnetic_latitude_variant"] = dict(spread=sp(r_gm), median=float(np.median(r_gm)), below1=float(np.mean(r_gm < 1)),
                                           spread_geographic=sp(C.S1_IEC / C.PARMA))
CI = pd.read_csv(OUT / "revision_bootstrap_ci.csv").set_index("model")
res["spread_CI"] = {m: [float(CI.loc[m, "spread_lo"]), float(CI.loc[m, "spread_hi"])] for m in ("S1_IEC", "S2_JESD_mid")}
json.dump(res, open(OUT / "r7_checks.json", "w"), indent=1)
print(json.dumps(res, indent=1))
