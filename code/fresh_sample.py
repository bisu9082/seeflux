# Step 8 round 8 (M6, R3): a fresh confirmation sample, drawn after every calibration choice was fixed. Same design as
# validate_direct.py (2000 points, area-uniform 80 S-80 N, 6-15 km, dates 2009-01-01 to 2026-03-26) with a new seed.
# Checks quantities that were tuned or chosen with the first independent set in view: the surrogate multipliers
# (1.10 overall, 1.13 below 2 GV), the surrogate error, the date-matched JESD89 variant (end points W_LO, W_HI fixed
# from the first analysis), and the headline IEC and JESD89 statistics at cruise.
# Output: data/step4/fresh_sample.csv, fresh_sample.json
import sys, json, pathlib, datetime as dt, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver as _rd
from seeflux_formula import flux_gt10, load_params, DOMAIN

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"


def run_driver(lines, chunk=400):
    out = []
    for i in range(0, len(lines), chunk):
        out += _rd(lines[i:i + chunk])
    return out


rng = np.random.default_rng(20261023)
N = 2000
lat = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(-80)), np.sin(np.radians(80)), N)))
lon = rng.uniform(-180, 180, N); alt = rng.uniform(6.0, 15.0, N)
d0, d1 = dt.date(2009, 1, 1), dt.date(2026, 3, 26)
days = [d0 + dt.timedelta(days=int(x)) for x in rng.integers(0, (d1 - d0).days + 1, N)]
o = run_driver([f"f{i} D {d.year} {d.month} {d.day} {la:.4f} {lo:.4f} {h:.4f} 10 PR"
                for i, (d, la, lo, h) in enumerate(zip(days, lat, lon, alt))])
V = pd.DataFrame(dict(date=[d.isoformat() for d in days], lat=lat, lon=lon, alt_km=alt, W=[float(x[1]) for x in o],
                      Rc=[float(x[2]) for x in o], depth=[float(x[3]) for x in o], PARMA=[float(x[5]) * 3600 for x in o],
                      PARMA_p=[float(x[9]) * 3600 for x in o]))
P = load_params()
V["in_domain"] = V.W.between(*DOMAIN["W"]) & V.Rc.between(*DOMAIN["Rc_GV"])
V["FORMULA"] = flux_gt10(V.alt_km.clip(6, 15), V.Rc.clip(0, 18), V.W.clip(0, 150), P)
V["S1_IEC"] = S.s1_iec_boeing(V.alt_km, V.lat)
V["S2_JESD_mid"] = S.s2_jesd89(V.depth, V.Rc, "mid")
m = json.load(open(OUT / "revision_sensitivity.json"))["JESD_mapping"]
f = np.clip((V.W - m["W_lo"]) / (m["W_hi"] - m["W_lo"]), 0, 1)
V["S2_JESD_lin"] = (1 - f) * S.s2_jesd89(V.depth, V.Rc, "min") + f * S.s2_jesd89(V.depth, V.Rc, "max")
V.to_csv(OUT / "fresh_sample.csv", index=False, lineterminator="\n", float_format="%.6g")


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float(p), float((c - h) / d), float((c + h) / d)]


sp = lambda r: float(np.percentile(r, 95) / np.percentile(r, 5))
C = V[V.alt_km.between(9.0, 12.5)]
A = V[V.in_domain]
err = np.abs(A.FORMULA / A.PARMA - 1)
corner = A[(A.alt_km > 12.5) & (A.Rc < 2)]
rule = np.where(A.Rc < 2, 1.13, 1.10)
r_iec = C.S1_IEC / C.PARMA; r_j = C.S2_JESD_mid / C.PARMA; r_jl = C.S2_JESD_lin / C.PARMA
res = dict(seed=20261023, n=N, n_cruise=len(C), n_in_domain=len(A),
           IEC=dict(median=float(r_iec.median()), below1=float((r_iec < 1).mean()), out30=float(((r_iec > 1.3) | (r_iec < 1 / 1.3)).mean()), spread=sp(r_iec)),
           JESD_mid=dict(median=float(r_j.median()), out30=float(((r_j > 1.3) | (r_j < 1 / 1.3)).mean()), spread=sp(r_j)),
           JESD_date_matched=dict(within30=float(((r_jl <= 1.3) & (r_jl >= 1 / 1.3)).mean())),
           surrogate=dict(median_abs_pct=float(np.median(err) * 100), p95_abs_pct=float(np.percentile(err, 95) * 100), max_abs_pct=float(err.max() * 100)),
           coverage_1p10_all=wilson(int((1.10 * A.FORMULA >= A.PARMA).sum()), len(A)),
           coverage_rule_all=wilson(int((rule * A.FORMULA >= A.PARMA).sum()), len(A)),
           coverage_1p10_corner=wilson(int((1.10 * corner.FORMULA >= corner.PARMA).sum()), len(corner)) if len(corner) else None,
           coverage_1p13_corner=wilson(int((1.13 * corner.FORMULA >= corner.PARMA).sum()), len(corner)) if len(corner) else None,
           n_corner=int(len(corner)))
json.dump(res, open(OUT / "fresh_sample.json", "w"), indent=1)
print(json.dumps(res, indent=1))
