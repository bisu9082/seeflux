# Step 4 route demonstration (revised after Step 8 review, 2026-10-16): route-integrated >10 MeV neutron fluence
# along great-circle flights from ICN. Fluence is integrated over the part of the flight at or above 6 km.
# Solar conditions (W from the PARMA solar table, GLE-flagged days excluded from means):
#   min_2019  annual mean W of 2019 (solar minimum year between Cycles 24 and 25)
#   max_2025  annual mean W of 2025 (Cycle 25 maximum year)
#   storm_2025-11-15  the post-GLE 77 day used in the first draft (W = 142.7, top 0.3% of days), kept as a sensitivity case
# Monthly series 2009-01 to 2026-03: PARMA evaluated at 17 W levels per route point and interpolated linearly in
# ln(flux) at the monthly mean W; checked against direct runs at the two annual means.
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver
from seeflux_formula import flux_gt10, load_params
from solar_w import daily_w, annual_mean, monthly_mean

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/step4"
APT = {"ICN": (37.46, 126.44), "JFK": (40.64, -73.78), "FRA": (50.03, 8.57), "LAX": (33.94, -118.41),
       "SIN": (1.36, 103.99), "SYD": (-33.95, 151.18)}
CRUISE_KM, CLIMB_MIN, V_KMH = 11.278, 25.0, 850.0  # FL370 (pressure altitude = US Std 1976 geometric height), climb/descent, speed
D = daily_w()
COND = {"min_2019": round(annual_mean(2019, D), 1), "max_2025": round(annual_mean(2025, D), 1), "storm_2025-11-15": None}
W_GRID = np.arange(0.0, 161.0, 10.0)
P = load_params()
MODELS = ("FORMULA", "S1_IEC", "S2_JESD_mid", "S4_AN_ABB", "S4b_AN_CoolCAD")


def great_circle(p1, p2, n):
    la1, lo1, la2, lo2 = map(np.radians, (*p1, *p2))
    v1 = np.array([np.cos(la1) * np.cos(lo1), np.cos(la1) * np.sin(lo1), np.sin(la1)])
    v2 = np.array([np.cos(la2) * np.cos(lo2), np.cos(la2) * np.sin(lo2), np.sin(la2)])
    om = np.arccos(np.clip(v1 @ v2, -1, 1)); t = np.linspace(0, 1, n)
    v = (np.sin((1 - t) * om)[:, None] * v1 + np.sin(t * om)[:, None] * v2) / np.sin(om)
    return np.degrees(np.arcsin(v[:, 2])), np.degrees(np.arctan2(v[:, 1], v[:, 0])), om * 6371.0


def flux_at_w(rc, dep, w):
    o = run_driver([f"p{i} S {w:.3f} {r:.5f} {d:.4f} 10" for i, (r, d) in enumerate(zip(rc, dep))])
    return np.array([float(x[5]) for x in o]) * 3600


def fluence(t_min, f, hi):
    return float(np.trapezoid(np.where(hi, f, 0.0), t_min / 60))


rows, prof, monthly = [], [], []
MW = monthly_mean(D)
for dest in ("JFK", "FRA", "LAX", "SIN", "SYD"):
    _, _, dist = great_circle(APT["ICN"], APT[dest], 2)
    T = dist / V_KMH * 60
    n = int(T) + 1
    lat, lon, _ = great_circle(APT["ICN"], APT[dest], n)
    t = np.linspace(0, T, n)
    alt = np.minimum(CRUISE_KM, np.minimum(t, T - t) / CLIMB_MIN * CRUISE_KM)
    hi = alt >= 6.0
    # static cutoff and depth along the track (date-independent in PARMA); the D-mode run also gives the storm case
    o = run_driver([f"p{i} D 2025 11 15 {la:.4f} {lo:.4f} {h:.4f} 10" for i, (la, lo, h) in enumerate(zip(lat, lon, alt))])
    rc = np.array([float(x[2]) for x in o]); dep = np.array([float(x[3]) for x in o])
    storm = dict(W=np.array([float(x[1]) for x in o]), PARMA=np.array([float(x[5]) for x in o]) * 3600)
    route = f"ICN-{dest}"
    std = dict(S1_IEC=S.s1_iec_boeing(alt, lat), S2_JESD_mid=S.s2_jesd89(dep, rc, "mid"),
               S4_AN_ABB=S.s4_an4589_abb(alt, lat), S4b_AN_CoolCAD=S.s4b_an4589_coolcad(alt, lat))
    rr = {}
    for cname, w in COND.items():
        wv = storm["W"] if w is None else np.full(n, w)
        par = storm["PARMA"] if w is None else flux_at_w(rc, dep, w)
        df = pd.DataFrame(dict(route=route, cond=cname, t_min=t, lat=lat, lon=lon, alt_km=alt, Rc=rc, W=wv, depth=dep,
                               PARMA=par, FORMULA=flux_gt10(np.clip(alt, 6.0, 15.0), rc, wv, P), **std))
        prof.append(df)
        flu = {k: fluence(t, df[k].values, hi) for k in ("PARMA",) + MODELS}
        below = float(np.trapezoid(np.where(~hi, df.PARMA, 0.0), t / 60))
        rr[cname] = dict(route=route, cond=cname, dist_km=round(dist), block_h=round(T / 60, 2), W=round(float(wv.mean()), 1),
                         Rc_min=rc.min(), Rc_max=rc.max(), lat_max=float(np.abs(lat).max()),
                         PARMA_fluence_ge6km=flu["PARMA"], frac_below6km=below / (below + flu["PARMA"]),
                         **{f"{k}_over_PARMA": flu[k] / flu["PARMA"] for k in MODELS})
    # monthly series via ln-flux interpolation in W
    lnG = np.log(np.array([flux_at_w(rc, dep, w) for w in W_GRID]))  # (nW, n)
    interp = lambda w: np.exp(np.array([np.interp(w, W_GRID, lnG[:, j]) for j in range(n)]))
    for cname in ("min_2019", "max_2025"):  # interpolation check against the direct runs
        rr[cname]["interp_check_rel"] = fluence(t, interp(COND[cname]), hi) / rr[cname]["PARMA_fluence_ge6km"] - 1
    rows += list(rr.values())
    f_std = {k: fluence(t, v, hi) for k, v in std.items()}
    for m, w in zip(MW.month.astype(str), MW.W):
        fp = fluence(t, interp(w), hi)
        monthly.append(dict(route=route, month=m, W=round(w, 2), PARMA_fluence_ge6km=fp,
                            FORMULA_over_PARMA=fluence(t, flux_gt10(np.clip(alt, 6.0, 15.0), rc, np.full(n, w), P), hi) / fp,
                            **{f"{k}_over_PARMA": f_std[k] / fp for k in std}))
    print(route, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(OUT / "routes_summary.csv", index=False, lineterminator="\n", float_format="%.6g")
pd.concat(prof).to_csv(OUT / "routes_profiles.csv", index=False, lineterminator="\n", float_format="%.6g")
M = pd.DataFrame(monthly)
M.to_csv(OUT / "routes_monthly.csv", index=False, lineterminator="\n", float_format="%.6g")
pd.set_option("display.width", 220)
print("conditions:", COND)
print(R.drop(columns=["Rc_min", "Rc_max"]).to_string(index=False, float_format=lambda v: "%.4g" % v))
g = M.groupby("route").S1_IEC_over_PARMA
print("monthly IEC/PARMA by route: min, P5, P95, max\n", pd.DataFrame(dict(min=g.min(), p05=g.quantile(.05), p95=g.quantile(.95), max=g.max())).round(3))
