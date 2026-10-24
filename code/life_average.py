# Step 8 round 6, M4: service-life comparison with the flat 6000. CM-AS-004 calls 6000 a mean flux, to be used as a
# minimum, so the relevant quantity is the flux averaged over flights and over the solar cycle, not the share of cruise
# time above it. Same route geometry as routes_levels.py (great circles, 850 km/h, 25 min climb and descent).
# PARMA neutrons + protons above 10 MeV on a 17-level W grid (0-160), log-interpolated to each monthly-mean W of
# 2009-01 to 2026-03 (GLE days excluded, solar_w.monthly_mean). Points every 3 min along the flight.
# Per route and level, per month: cruise mean and flight mean (all altitudes) of n+p, and the ratio of the scaled IEC curves
# (6000 basis) to PARMA n+p over the flight above 6 km. Averages over Solar Cycle 24 (2009-01 to 2019-12, minimum to
# minimum) and over 2009-01 to 2026-03. Fleet means weight each route by its flight time (one flight per route; not
# traffic-weighted).
# Output: data/step4/life_average.csv (route x level), life_average_monthly.csv, life_average.json
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver
from solar_w import daily_w, monthly_mean

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
APT = {"ICN": (37.46, 126.44), "JFK": (40.64, -73.78), "FRA": (50.03, 8.57), "LAX": (33.94, -118.41), "SIN": (1.36, 103.99),
       "SYD": (-33.95, 151.18), "GRU": (-23.43, -46.47), "JNB": (-26.14, 28.24), "SCL": (-33.39, -70.79)}
ROUTES = [("ICN", "JFK"), ("ICN", "FRA"), ("ICN", "LAX"), ("ICN", "SIN"), ("ICN", "SYD"), ("GRU", "JNB"), ("SYD", "SCL")]
FL = {"FL370": 37000 * 0.3048 / 1000, "FL390": 39000 * 0.3048 / 1000, "FL410": 41000 * 0.3048 / 1000}
CLIMB_MIN, V_KMH, DT = 25.0, 850.0, 3.0
W_GRID = np.arange(0.0, 161.0, 10.0)
MW = monthly_mean(daily_w()); MW["ym"] = MW.month.astype(str)
assert MW.W.max() <= W_GRID.max() and MW.W.min() >= W_GRID.min()


def great_circle(p1, p2, n):   # as in routes_levels.py
    la1, lo1, la2, lo2 = map(np.radians, (*p1, *p2))
    v1 = np.array([np.cos(la1) * np.cos(lo1), np.cos(la1) * np.sin(lo1), np.sin(la1)])
    v2 = np.array([np.cos(la2) * np.cos(lo2), np.cos(la2) * np.sin(lo2), np.sin(la2)])
    om = np.arccos(np.clip(v1 @ v2, -1, 1)); t = np.linspace(0, 1, n)
    v = (np.sin((1 - t) * om)[:, None] * v1 + np.sin(t * om)[:, None] * v2) / np.sin(om)
    return np.degrees(np.arcsin(v[:, 2])), np.degrees(np.arctan2(v[:, 1], v[:, 0])), om * 6371.0


mon, rows = [], []
for a, b in ROUTES:
    route = f"{a}-{b}"
    _, _, dist = great_circle(APT[a], APT[b], 2)
    T = dist / V_KMH * 60; n = int(T / DT) + 1
    lat, lon, _ = great_circle(APT[a], APT[b], n); t = np.linspace(0, T, n)
    rc = np.array([float(x[2]) for x in run_driver([f"p{i} D 2025 1 1 {la:.4f} {lo:.4f} 11 10" for i, (la, lo) in enumerate(zip(lat, lon))])])
    for fl, hcr in FL.items():
        alt = np.minimum(hcr, np.minimum(t, T - t) / CLIMB_MIN * hcr); cr = alt >= hcr - 1e-9; hi = alt >= 6.0
        iec6000 = S.s1_iec_boeing(alt, lat) * 6000.0 / 5600.0
        o = run_driver([f"g{j}_{i} A {w:.3f} {r:.5f} {h:.5f} 10 PR" for j, w in enumerate(W_GRID) for i, (r, h) in enumerate(zip(rc, alt))])
        G = np.array([(float(x[5]) + float(x[9])) * 3600 for x in o]).reshape(len(W_GRID), n)
        lnG = np.log(G)
        for ym, w in zip(MW.ym, MW.W):
            j = np.searchsorted(W_GRID, w, side="right") - 1; j = min(j, len(W_GRID) - 2); f = (w - W_GRID[j]) / (W_GRID[j + 1] - W_GRID[j])
            npf = np.exp((1 - f) * lnG[j] + f * lnG[j + 1])
            mon.append(dict(route=route, level=fl, month=ym, W=w, cruise_mean_np=float(npf[cr].mean()),
                            flight_mean_np=float(np.trapezoid(npf, t) / T), cruise_peak_np=float(npf[cr].max()),
                            IEC6000_over_np=float(np.trapezoid(np.where(hi, iec6000, 0), t) / np.trapezoid(np.where(hi, npf, 0), t)),
                            block_h=T / 60, cruise_h=float(cr.sum() * DT / 60)))
    print(route, "done", flush=True)
M = pd.DataFrame(mon)
M.to_csv(OUT / "life_average_monthly.csv", index=False, lineterminator="\n", float_format="%.6g")
M["ym"] = pd.PeriodIndex(M.month, freq="M")
per = {"cycle24": (pd.Period("2009-01", "M"), pd.Period("2019-12", "M")), "2009_2026": (pd.Period("2009-01", "M"), pd.Period("2026-03", "M"))}
for (route, level), g in M.groupby(["route", "level"], sort=False):
    r = dict(route=route, level=level, block_h=float(g.block_h.iloc[0]))
    for pn, (p0, p1) in per.items():
        s = g[(g.ym >= p0) & (g.ym <= p1)]
        r.update({f"cruise_mean_np_{pn}": s.cruise_mean_np.mean(), f"flight_mean_np_{pn}": s.flight_mean_np.mean(),
                  f"IEC6000_over_np_{pn}": s.IEC6000_over_np.mean(),
                  f"share_months_cruise_mean_gt6000_{pn}": float((s.cruise_mean_np > 6000).mean()),
                  f"k_star_cruise_life_{pn}": s.cruise_mean_np.mean() / 6000})
    r.update(cruise_mean_np_min_month=g.cruise_mean_np.min(), cruise_mean_np_max_month=g.cruise_mean_np.max())
    rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(OUT / "life_average.csv", index=False, lineterminator="\n", float_format="%.6g")
fleet = {}
for level, g in R.groupby("level"):
    for name, sel in (("seven_routes", g.route.notna()), ("northern_long_haul", g.route.isin(["ICN-JFK", "ICN-FRA", "ICN-LAX"]))):
        s = g[sel]; wgt = s.block_h / s.block_h.sum()
        fleet[f"{level}_{name}"] = {c: float((s[c] * wgt).sum()) for c in s.columns if c.startswith(("flight_mean_np", "cruise_mean_np_c", "cruise_mean_np_2"))}
json.dump(dict(periods={k: [str(v[0]), str(v[1])] for k, v in per.items()}, W_range=[float(MW.W.min()), float(MW.W.max())],
               fleet_hours_weighted=fleet), open(OUT / "life_average.json", "w"), indent=1)
pd.set_option("display.width", 260)
print(R.round(3).to_string(index=False)); print(json.dumps(fleet, indent=1))
