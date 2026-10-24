# Step 8 round 6, M7a: epoch of the static cutoff map. PARMA's vertical cutoff map (parma/build/input/CORdata.inp) carries
# no epoch. The FAA CARI-7 distribution tabulates vertical cutoff rigidities at 20 km on a 1x1 deg grid for IGRF epochs
# 1965-2010 (cari7/CARI_7_DVD/CUTOFFS). We (1) find the epoch closest to the PARMA map, (2) extrapolate the 2000-2010
# secular change linearly to 2025 at each grid cell, and (3) apply the change since the matched epoch to the PARMA map
# along the seven routes (FL370, 2009/2019/2025 annual means and GRU-JNB monthly) and at the 802 cruise points.
# Linear extrapolation of the secular change is a first-order estimate; storm-time cutoff lowering is not included.
# Output: data/step4/cutoff_epoch.json, cutoff_epoch_points.csv
import json, re, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver as _run_driver


def run_driver(lines, chunk=400):   # short WSL calls (a single long call can be interrupted)
    out = []
    for i in range(0, len(lines), chunk):
        out += _run_driver(lines[i:i + chunk])
    return out
from solar_w import daily_w, monthly_mean, annual_mean

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
CUT = ROOT / "cari7/CARI_7_DVD/CUTOFFS"


def read_cari(fn):
    """1x1 deg table -> array [lat 90..-90 index lat+90][lon 0..359]"""
    G = np.full((181, 360), np.nan); lons = None
    for line in open(fn, encoding="latin-1"):
        tok = line.split()
        if not tok:
            continue
        if all(re.fullmatch(r"\d+", x) for x in tok) and len(tok) > 3:
            lons = [int(x) for x in tok]; continue
        if lons and re.fullmatch(r"-?\d+", tok[0]) and len(tok) == len(lons) + 1:
            la = int(tok[0])
            if -90 <= la <= 90:
                G[la + 90, lons] = [float(x) for x in tok[1:]]
    return G


def read_parma():
    d = np.loadtxt(ROOT / "parma/build/input/CORdata.inp", skiprows=2)
    G = np.full((181, 360), np.nan)
    for lo, la, rc in d:
        l = int(round(lo)) % 360
        G[int(round(la)) + 90, l] = rc
    return G


EP = {1965: "WGRC1965.1X1", 1980: "IGRF1980.1X1", 1990: "DGRF1990.1X1", 1995: "IGRF1995.1X1", 2000: "IGRF2000.1X1", 2010: "IGRF2010.1X1"}
TAB = {y: read_cari(CUT / f) for y, f in EP.items()}
PM = read_parma()
lat_ax = np.arange(-90, 91)
w = np.cos(np.radians(lat_ax))[:, None] * np.ones((1, 360))
ok = lambda G: np.isfinite(G) & np.isfinite(PM)
match = {}
for y, G in TAB.items():
    m = ok(G) & (np.abs(lat_ax)[:, None] <= 80)
    dlt = (G - PM)[m]
    match[y] = dict(n=int(m.sum()), mean=float(np.average(dlt, weights=w[m])), rms=float(np.sqrt(np.average(dlt ** 2, weights=w[m]))),
                    max_abs=float(np.abs(dlt).max()))
best = min(match, key=lambda y: match[y]["rms"])
rate = (TAB[2010] - TAB[2000]) / 10.0                       # GV per year at each cell
def shift_to(year):
    return rate * (year - best) if best >= 2000 else (TAB[2010] + rate * (year - 2010)) - TAB[best]


def dRc(lat, lon, year):
    """bilinear interpolation of the secular change since the matched epoch"""
    Dg = shift_to(year)
    x = np.mod(lon, 360.0); y = np.asarray(lat) + 90.0
    i0 = np.clip(np.floor(y).astype(int), 0, 179); j0 = np.floor(x).astype(int) % 360; j1 = (j0 + 1) % 360
    fy = y - i0; fx = x - np.floor(x)
    v = (Dg[i0, j0] * (1 - fx) * (1 - fy) + Dg[i0, j1] * fx * (1 - fy) + Dg[i0 + 1, j0] * (1 - fx) * fy + Dg[i0 + 1, j1] * fx * fy)
    return np.nan_to_num(v)


# (3a) routes at FL370: route fluence above 6 km with the map shifted to 2025 vs the static map
P = pd.read_csv(OUT / "routes_levels_profiles_FL370.csv")
D = daily_w(); MW = monthly_mean(D)
res_routes = {}
for rt, g in P[P.cond == "y2025"].groupby("route", sort=False):
    t = g.t_min.values; hi = g.alt_km.values >= 6.0
    rc0 = g.Rc.values; rc1 = np.maximum(rc0 + dRc(g.lat.values, g.lon.values, 2025), 0.0)
    iec = g.S1_IEC.values
    out = {}
    for cname, wv in (("y2009", annual_mean(2009, D)), ("y2019", annual_mean(2019, D)), ("y2025", annual_mean(2025, D))):
        o = run_driver([f"a{i} A {wv:.3f} {r:.5f} {h:.5f} 10" for i, (r, h) in enumerate(zip(np.r_[rc0, rc1], np.r_[g.alt_km.values, g.alt_km.values]))])
        f = np.array([float(x[5]) for x in o]) * 3600; f0, f1 = f[:len(t)], f[len(t):]
        flu = lambda ff: float(np.trapezoid(np.where(hi, ff, 0), t))
        out[cname] = dict(IEC_over_PARMA_static=flu(iec) / flu(f0), IEC_over_PARMA_2025map=flu(iec) / flu(f1), PARMA_change=flu(f1) / flu(f0) - 1)
    out["dRc_route_min"] = float((rc1 - rc0)[hi].min()); out["dRc_route_max"] = float((rc1 - rc0)[hi].max())
    res_routes[rt] = out
    print(rt, {k: (round(v["IEC_over_PARMA_static"], 3), round(v["IEC_over_PARMA_2025map"], 3)) for k, v in out.items() if k.startswith("y")}, flush=True)
# GRU-JNB monthly share below one with the 2025-shifted map (W grid as routes_levels.py)
g = P[(P.cond == "y2025") & (P.route == "GRU-JNB")]; t = g.t_min.values; hi = g.alt_km.values >= 6.0
rc1 = np.maximum(g.Rc.values + dRc(g.lat.values, g.lon.values, 2025), 0.0)
WG = np.arange(0.0, 161.0, 10.0)
o = run_driver([f"m{j}_{i} A {wv:.3f} {r:.5f} {h:.5f} 10" for j, wv in enumerate(WG) for i, (r, h) in enumerate(zip(rc1, g.alt_km.values))])
lnG = np.log(np.array([float(x[5]) for x in o]).reshape(len(WG), len(t)) * 3600)
fi = float(np.trapezoid(np.where(hi, g.S1_IEC.values, 0), t))
mr = np.array([fi / np.trapezoid(np.where(hi, np.exp(np.array([np.interp(wv, WG, lnG[:, k]) for k in range(len(t))])), 0), t) for wv in MW.W])
ms = pd.read_csv(OUT / "routes_monthly_south.csv"); m0 = ms[ms.route == "GRU-JNB"].IEC_over_PARMA_n.values
gru = dict(static=dict(median=float(np.median(m0)), below1=float(np.mean(m0 < 1))), map2025=dict(median=float(np.median(mr)), below1=float(np.mean(mr < 1))))
# (3b) 802 cruise points: IEC spread with the map shifted to the year of each point
V = pd.read_csv(OUT / "validation_direct.csv", parse_dates=["date"])
Cp = V[V.alt_km.between(9.0, 12.5)].reset_index(drop=True)
yr = Cp.date.dt.year + (Cp.date.dt.dayofyear - 1) / 365.25
Cp["Rc_shift"] = np.maximum(Cp.Rc + np.array([dRc(np.array([la]), np.array([lo]), y)[0] for la, lo, y in zip(Cp.lat, Cp.lon, yr)]), 0.0)
o = run_driver([f"v{i} A {wv:.4f} {r:.5f} {h:.5f} 10" for i, (wv, r, h) in enumerate(zip(Cp.W, np.r_[Cp.Rc.values], Cp.alt_km))]
               + [f"w{i} A {wv:.4f} {r:.5f} {h:.5f} 10" for i, (wv, r, h) in enumerate(zip(Cp.W, Cp.Rc_shift, Cp.alt_km))])
f = np.array([float(x[5]) for x in o]) * 3600; n = len(Cp)
Cp["PARMA_A_static"], Cp["PARMA_A_shift"] = f[:n], f[n:]
sp = lambda r: float(np.percentile(r, 95) / np.percentile(r, 5))
pts = dict(dRc_P05=float(np.percentile(Cp.Rc_shift - Cp.Rc, 5)), dRc_P95=float(np.percentile(Cp.Rc_shift - Cp.Rc, 95)),
           IEC_spread_static=sp(Cp.S1_IEC / Cp.PARMA_A_static), IEC_spread_shift=sp(Cp.S1_IEC / Cp.PARMA_A_shift),
           JESD_spread_static=sp(Cp.S2_JESD_mid / Cp.PARMA_A_static), JESD_spread_shift=sp(Cp.S2_JESD_mid / Cp.PARMA_A_shift),
           IEC_median_static=float(np.median(Cp.S1_IEC / Cp.PARMA_A_static)), IEC_median_shift=float(np.median(Cp.S1_IEC / Cp.PARMA_A_shift)),
           note="PARMA in A mode (US Standard Atmosphere) for both, so the comparison isolates the cutoff change")
Cp[["date", "lat", "lon", "alt_km", "W", "Rc", "Rc_shift", "PARMA_A_static", "PARMA_A_shift"]].to_csv(
    OUT / "cutoff_epoch_points.csv", index=False, lineterminator="\n", float_format="%.6g")
SA = (lat_ax[:, None] >= -45) & (lat_ax[:, None] <= 0) & ((np.arange(360)[None, :] >= 300) | (np.arange(360)[None, :] <= 30))
d25 = shift_to(2025)
res = dict(match_rms_GV=match, best_epoch=best, secular_rate_2000_2010=dict(global_mean_GV_per_decade=float(np.nanmean(rate) * 10),
           south_atlantic_mean_GV_per_decade=float(np.nanmean(np.where(SA, rate, np.nan)) * 10)),
           dRc_to_2025=dict(min=float(np.nanmin(d25)), max=float(np.nanmax(d25)), south_atlantic_mean=float(np.nanmean(np.where(SA, d25, np.nan)))),
           routes_FL370=res_routes, GRU_JNB_monthly=gru, cruise_points=pts)
json.dump(res, open(OUT / "cutoff_epoch.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "routes_FL370"}, indent=1))
