# Second revision (Step 8 round 2, 2026-10-16): flat design values and scaled curves on routes, by flight level.
#  routes: the five ICN routes plus two southern-hemisphere routes (GRU-JNB, SYD-SCL)
#  flight levels FL370, FL390, FL410 (pressure altitudes = US Standard Atmosphere geometric heights)
#  solar conditions: annual-mean W of 2009 (lowest annual mean of 2009-2025), 2019, 2025
#  per route, level and condition: route fluence (>= 6 km) of PARMA neutrons and protons above 10 MeV; IEC curves / PARMA;
#  largest PARMA neutron flux and neutron+proton flux at cruise against the flat 5600 and 6000 values, and the share of
#  cruise time above them. Monthly IEC/PARMA series at FL370 for the two southern routes (17-level W interpolation).
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver
from solar_w import daily_w, annual_mean, monthly_mean

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/step4"
APT = {"ICN": (37.46, 126.44), "JFK": (40.64, -73.78), "FRA": (50.03, 8.57), "LAX": (33.94, -118.41), "SIN": (1.36, 103.99),
       "SYD": (-33.95, 151.18), "GRU": (-23.43, -46.47), "JNB": (-26.14, 28.24), "SCL": (-33.39, -70.79)}
ROUTES = [("ICN", "JFK"), ("ICN", "FRA"), ("ICN", "LAX"), ("ICN", "SIN"), ("ICN", "SYD"), ("GRU", "JNB"), ("SYD", "SCL")]
FL = {"FL370": 37000 * 0.3048 / 1000, "FL390": 39000 * 0.3048 / 1000, "FL410": 41000 * 0.3048 / 1000}
CLIMB_MIN, V_KMH = 25.0, 850.0
D = daily_w()
COND = {"y2009": round(annual_mean(2009, D), 1), "y2019": round(annual_mean(2019, D), 1), "y2025": round(annual_mean(2025, D), 1)}
W_GRID = np.arange(0.0, 161.0, 10.0)


def great_circle(p1, p2, n):
    la1, lo1, la2, lo2 = map(np.radians, (*p1, *p2))
    v1 = np.array([np.cos(la1) * np.cos(lo1), np.cos(la1) * np.sin(lo1), np.sin(la1)])
    v2 = np.array([np.cos(la2) * np.cos(lo2), np.cos(la2) * np.sin(lo2), np.sin(la2)])
    om = np.arccos(np.clip(v1 @ v2, -1, 1)); t = np.linspace(0, 1, n)
    v = (np.sin((1 - t) * om)[:, None] * v1 + np.sin(t * om)[:, None] * v2) / np.sin(om)
    return np.degrees(np.arcsin(v[:, 2])), np.degrees(np.arctan2(v[:, 1], v[:, 0])), om * 6371.0


def parma(w, rc, alt, protons=True):
    o = run_driver([f"p{i} A {w:.3f} {r:.5f} {h:.5f} 10" + (" PR" if protons else "") for i, (r, h) in enumerate(zip(rc, alt))])
    n = np.array([float(x[5]) for x in o]) * 3600
    return (n, np.array([float(x[9]) for x in o]) * 3600) if protons else n


flu = lambda t, f, hi: float(np.trapezoid(np.where(hi, f, 0.0), t / 60))
rows, monthly, prof = [], [], []
MW = monthly_mean(D)
for a, b in ROUTES:
    _, _, dist = great_circle(APT[a], APT[b], 2)
    T = dist / V_KMH * 60; n = int(T) + 1
    lat, lon, _ = great_circle(APT[a], APT[b], n); t = np.linspace(0, T, n)
    rc = np.array([float(x[2]) for x in run_driver([f"p{i} D 2025 1 1 {la:.4f} {lo:.4f} 11 10" for i, (la, lo) in enumerate(zip(lat, lon))])])
    route = f"{a}-{b}"
    for fl, hcr in FL.items():
        alt = np.minimum(hcr, np.minimum(t, T - t) / CLIMB_MIN * hcr); hi = alt >= 6.0; cr = alt >= hcr - 1e-9
        iec = S.s1_iec_boeing(alt, lat)
        for cname, w in COND.items():
            pn, pp = parma(w, rc, alt)
            np_ = pn + pp
            rows.append(dict(route=route, level=fl, cond=cname, W=w, block_h=round(T / 60, 2), lat_absmax=float(np.abs(lat).max()),
                             Rc_min=float(rc.min()), PARMA_n_fluence=flu(t, pn, hi), PARMA_p_fluence=flu(t, pp, hi),
                             IEC_over_PARMA_n=flu(t, iec, hi) / flu(t, pn, hi),
                             max_n_cruise=float(pn[cr].max()), max_np_cruise=float(np_[cr].max()),
                             flat5600_over_max_n=5600 / float(pn[cr].max()), flat6000_over_max_np=6000 / float(np_[cr].max()),
                             frac_cruise_n_gt5600=float(np.mean(pn[cr] > 5600)), frac_cruise_np_gt6000=float(np.mean(np_[cr] > 6000)),
                             frac_cruise_IEC_lt_n=float(np.mean(iec[cr] < pn[cr]))))
            if fl == "FL370":
                prof.append(pd.DataFrame(dict(route=route, cond=cname, t_min=t, lat=lat, lon=lon, alt_km=alt, Rc=rc,
                                              PARMA_n=pn, PARMA_p=pp, S1_IEC=iec)))
        if fl == "FL370" and a != "ICN":
            lnG = np.log(np.array([parma(w, rc, alt, protons=False) for w in W_GRID]))
            fi = flu(t, iec, hi)
            for m, w in zip(MW.month.astype(str), MW.W):
                pw = np.exp(np.array([np.interp(w, W_GRID, lnG[:, j]) for j in range(n)]))
                monthly.append(dict(route=route, month=m, W=round(w, 2), IEC_over_PARMA_n=fi / flu(t, pw, hi)))
    print(route, "done", flush=True)

R = pd.DataFrame(rows); R.to_csv(OUT / "routes_levels.csv", index=False, lineterminator="\n", float_format="%.6g")
pd.concat(prof).to_csv(OUT / "routes_levels_profiles_FL370.csv", index=False, lineterminator="\n", float_format="%.6g")
M = pd.DataFrame(monthly); M.to_csv(OUT / "routes_monthly_south.csv", index=False, lineterminator="\n", float_format="%.6g")
pd.set_option("display.width", 260)
print("conditions:", COND)
print(R.drop(columns=["block_h", "lat_absmax"]).to_string(index=False, float_format=lambda v: "%.4g" % v))
print(M.groupby("route").IEC_over_PARMA_n.agg(["min", "median", "max"]).round(3))
