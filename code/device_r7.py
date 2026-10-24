# Step 8 round 7, device level (reviewers R1, R4, R5):
#  (1) test method: the IEC count with a single cross section taken as sigma(200 MeV), sigma at the highest M2 test
#      energy (230 MeV), or a spectrum-averaged sigma over the sea-level (NYC) spectrum from 10 MeV truncated at 800 MeV
#      (the upper end of spallation beams) and untruncated (1e5 MeV); all against the neutron + proton reference
#  (2) energy bands of the M3 reference count: E_th-230, 230-1000, 1000-3000, 3000-1e5 MeV (neutrons + protons)
#  (3) a family between the saturating and the GeV-rising fit, sigma_f = (1-f) M2 + f M3 (the fold is linear in sigma),
#      and the break-even f at which the IEC count equals the reference
#  (4) seven routes at FL370 and FL410, 2019 and 2025 annual means (same geometry as routes_levels.py)
# Device: Renesas R1LV1616R 16-Mbit SRAM (Cecchetto et al. 2019). Output: data/step4/device_r7.csv, device_r7.json
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import standards as S
from wslpath_util import run_driver as _rd
from solar_w import daily_w, annual_mean

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"


def run_driver(lines, chunk=150):
    out = []
    for i in range(0, len(lines), chunk):
        out += _rd(lines[i:i + chunk])
    return out


dev = json.load(open(ROOT / "data/refs/cecchetto2019_tables.json"))["table2_weibull"]["Renesas_R1LV1616R_150nm_16Mbit_SEU"]
P = {m: (dev[m]["sigma_sat_cm2_per_dev"], dev[m]["E_th_MeV"], dev[m]["W_MeV"], dev[m]["s"]) for m in ("M2", "M3")}
sig = lambda m, e: P[m][0] * (1 - np.exp(-((e - P[m][1]) / P[m][2]) ** P[m][3])) if e > P[m][1] else 0.0
BANDS = [(0.0, 230.0), (230.0, 1000.0), (1000.0, 3000.0), (3000.0, 1.0e5)]
# (1) spectrum-averaged cross sections from the NYC sea-level spectrum, 10-800 MeV and 10-1e5 MeV
kw_sp = []
for m in ("M2", "M3"):
    for e2 in (800.0, 1.0e5):
        kw_sp.append(f"WR {P[m][0]:g} {P[m][1]:g} {P[m][2]:g} {P[m][3]:g} 10 {e2:g} 0")
for e2 in (800.0, 1.0e5):
    kw_sp.append(f"WR 0 0 1 1 10 {e2:g} 0")
o = run_driver([f"nyc D 2002 11 15 40.71 -74.01 0 0.15 " + " ".join(kw_sp)])[0]
v = [float(x) for x in o[9:9 + len(kw_sp)]]
flux800, flux1e5 = v[4], v[5]
SIG = {"M2": {"s200": sig("M2", 200.0), "s230": sig("M2", 230.0), "sp800": v[0] / flux800, "sp1e5": v[1] / flux1e5},
       "M3": {"s200": sig("M3", 200.0), "s230": sig("M3", 230.0), "sp800": v[2] / flux800, "sp1e5": v[3] / flux1e5}}
# (2)+(4) routes
D = daily_w(); COND = {"y2019": annual_mean(2019, D), "y2025": annual_mean(2025, D)}
PF = pd.read_csv(OUT / "routes_levels_profiles_FL370.csv")
H = {"FL370": 37000 * 0.3048 / 1000, "FL410": 41000 * 0.3048 / 1000}
kw = []
for m in ("M2", "M3"):
    for ip in (0, 1):
        for e1, e2 in BANDS:
            kw.append(f"WR {P[m][0]:g} {P[m][1]:g} {P[m][2]:g} {P[m][3]:g} {e1:g} {e2:g} {ip}")
NK = len(kw)
rows = []
for rt, g0 in PF[PF.cond == "y2019"].groupby("route", sort=False):
    t = g0.t_min.values; T = t.max(); lat = g0.lat.values; rc = g0.Rc.values
    for lev, hcr in H.items():
        alt = np.minimum(hcr, np.minimum(t, T - t) / 25.0 * hcr); hi = alt >= 6.0
        iec = S.s1_iec_boeing(alt, lat) * 6000 / 5600
        for cond, w in COND.items():
            o = run_driver([f"r{i} A {w:.3f} {r:.5f} {h:.5f} 10 PR " + " ".join(kw) for i, (r, h) in enumerate(zip(rc, alt))])
            A = np.array([[float(x) for x in z[9:10 + NK]] for z in o])   # PR then the band folds
            integ = lambda y: float(np.trapezoid(np.where(hi, y, 0), t * 60))
            F_iec = integ(iec / 3600)
            band = {}
            j = 1
            for m in ("M2", "M3"):
                for ip in ("n", "p"):
                    for (e1, e2) in BANDS:
                        band[(m, ip, e1)] = integ(A[:, j]); j += 1
            ref = {m: sum(band[(m, ip, e1)] for ip in ("n", "p") for e1, _ in BANDS) for m in ("M2", "M3")}
            r = dict(route=rt, level=lev, cond=cond, ref_M2_np=ref["M2"], ref_M3_np=ref["M3"])
            for m in ("M2", "M3"):
                for e1, e2 in BANDS:
                    r[f"{m}_band_{int(e1)}_{int(e2)}"] = (band[(m, "n", e1)] + band[(m, "p", e1)]) / ref[m]
            for sm in ("M2", "M3"):
                for k_, s_ in SIG[sm].items():
                    r[f"IEC_sig{sm}_{k_}_over_M2np"] = F_iec * s_ / ref["M2"]
                    r[f"IEC_sig{sm}_{k_}_over_M3np"] = F_iec * s_ / ref["M3"]
            # (3) mixture: IEC count with sigma_f(200 MeV) (or the spectrum average) against the mixed reference
            for k_ in ("s200", "sp800"):
                fs = np.linspace(0, 1, 1001)
                num = F_iec * ((1 - fs) * SIG["M2"][k_] + fs * SIG["M3"][k_]); den = (1 - fs) * ref["M2"] + fs * ref["M3"]
                rr = num / den
                r[f"mix_breakeven_f_{k_}"] = float(fs[np.argmax(rr < 1)]) if (rr < 1).any() else None
            rows.append(r)
    print(rt, "done", flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "device_r7.csv", index=False, lineterminator="\n", float_format="%.6g")
summ = {"sigma_cm2": SIG}
for c in R.columns:
    if c.startswith(("IEC_sig", "M2_band", "M3_band", "mix_")):
        x = R[c].dropna()
        summ[c] = [float(x.min()), float(x.max())] if len(x) else None
json.dump(summ, open(OUT / "device_r7.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
