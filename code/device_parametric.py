# Step 8 round 6, M6: does the flux-level audit carry to devices? Generic and published Weibull responses folded with
# PARMA neutrons and protons at the 802 independent cruise points (dates, places and altitudes of validation_direct.csv)
# and along ICN-JFK at FL370 for the 2019 and 2025 annual-mean conditions (routes_profiles.csv).
#  generic saturating responses: E_th = 1, 5, 10, 20, 35 MeV x W = 15, 50, 150 MeV, s = 1.5, sigma_sat = 1 (ratios only)
#  published responses (Cecchetto et al. 2019, Table 2): M2 (to 230 MeV) and M3 (to GeV) fits of three parts
# IEC simplified count: flux above 10 MeV x sigma(200 MeV); the scaled IEC value on its 6000 basis against the
# neutron + proton reference (and on its 5600 basis against neutrons), JESD89 (mid level) against neutrons + protons.
# Proton cross section = neutron cross section (as in ser_routes.py).
# Output: data/step4/device_parametric.csv, device_parametric.json
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
C = json.load(open(ROOT / "data/refs/cecchetto2019_tables.json"))["table2_weibull"]
RESP = []
for eth in (1, 5, 10, 20, 35):
    for w in (15, 50, 150):
        RESP.append(dict(name=f"gen_Eth{eth}_W{w}", kind="generic, saturating", sig=1.0, eth=float(eth), w=float(w), s=1.5))
SHORT = {"Renesas_R1LV1616R_150nm_16Mbit_SEU": "Renesas SEU", "Alliance_AS7C34098A_200nm_4Mbit_SEL": "Alliance SEL",
         "Samsung_K6R4016V1D_180nm_4Mbit_SEL": "Samsung SEL"}
for part, fits in C.items():
    for m in ("M2", "M3"):
        f = fits[m]
        RESP.append(dict(name=f"{SHORT[part]} {m}", kind="published, " + ("to 230 MeV" if m == "M2" else "GeV rise"),
                         sig=f["sigma_sat_cm2_per_dev"], eth=float(f["E_th_MeV"]), w=float(f["W_MeV"]), s=f["s"]))
sig_at = lambda r, e: r["sig"] * (1 - np.exp(-((e - r["eth"]) / r["w"]) ** r["s"])) if e > r["eth"] else 0.0
kw = " ".join(f"WB {r['sig']:g} {r['eth']:g} {r['w']:g} {r['s']:g} WBP {r['sig']:g} {r['eth']:g} {r['w']:g} {r['s']:g}" for r in RESP)
NR = len(RESP)


def fold(lines):
    o = []   # chunks keep each WSL call short (a single long call was interrupted)
    for i in range(0, len(lines), 100):
        o += run_driver([f"{l} {kw}" for l in lines[i:i + 100]])
        print("fold", min(i + 100, len(lines)), "/", len(lines), flush=True)
    F10 = np.array([float(x[5]) for x in o])
    Rn = np.array([[float(x[9 + 2 * j]) for j in range(NR)] for x in o])
    Rp = np.array([[float(x[10 + 2 * j]) for j in range(NR)] for x in o])
    return F10, Rn, Rp


# (1) independent cruise points
V = pd.read_csv(OUT / "validation_direct.csv", parse_dates=["date"])
Cp = V[V.alt_km.between(9.0, 12.5)].reset_index(drop=True)
F10, Rn, Rp = fold([f"c{i} D {d.year} {d.month} {d.day} {la:.4f} {lo:.4f} {h:.4f} 10" for i, (d, la, lo, h) in
                    enumerate(zip(Cp.date, Cp.lat, Cp.lon, Cp.alt_km))])
dev = float(np.max(np.abs(F10 * 3600 / Cp.PARMA - 1)))
assert dev < 0.01, dev   # same PARMA flux as validation_direct.csv (stored with 5 significant digits)
# (2) ICN-JFK at FL370 (route profile above 6 km), 2019 and 2025 annual means
P = pd.read_csv(OUT / "routes_profiles.csv"); P = P[(P.route == "ICN-JFK") & (P.alt_km >= 6.0)]
sw = {}
for cond in ("min_2019", "max_2025"):
    g = P[P.cond == cond]
    _, rn, rp = fold([f"r{i} S {w:.4f} {r:.5f} {d:.4f} 10" for i, (w, r, d) in enumerate(zip(g.W, g.Rc, g.depth))])
    t = g.t_min.values * 60
    sw[cond] = np.array([np.trapezoid(rn[:, j] + rp[:, j], t) for j in range(NR)])
    sw[cond + "_flux"] = float(np.trapezoid(g.PARMA.values, t))
pct = lambda x, q: float(np.percentile(x, q))
rows = []
iec6000 = Cp.S1_IEC.values * 6000 / 5600
for j, r in enumerate(RESP):
    s200 = sig_at(r, 200.0) / 3600   # per (n/cm2/h)
    ref_np = (Rn[:, j] + Rp[:, j]); ref_n = Rn[:, j]
    ri = iec6000 * s200 / ref_np; rj = Cp.S2_JESD_mid.values * s200 / ref_np; rf = Cp.PARMA.values * s200 / ref_n
    rows.append(dict(response=r["name"], kind=r["kind"], E_th=r["eth"], W=r["w"], s=r["s"],
                     method_factor_median=float(np.median(rf)), method_factor_p05=pct(rf, 5), method_factor_p95=pct(rf, 95),
                     p_over_n_median=float(np.median(Rp[:, j] / Rn[:, j])),
                     IEC6000_over_np_median=float(np.median(ri)), IEC6000_over_np_p05=pct(ri, 5), IEC6000_over_np_p95=pct(ri, 95),
                     IEC6000_spread=pct(ri, 95) / pct(ri, 5), IEC6000_below1=float(np.mean(ri < 1)),
                     JESD_over_np_median=float(np.median(rj)), JESD_spread=pct(rj, 95) / pct(rj, 5),
                     swing_JFK_FL370_2019_over_2025=float(sw["min_2019"][j] / sw["max_2025"][j])))
D = pd.DataFrame(rows)
D.to_csv(OUT / "device_parametric.csv", index=False, lineterminator="\n", float_format="%.6g")
json.dump(dict(n_points=len(Cp), flux_rerun_max_rel_dev=dev, flux_swing_JFK_FL370_2019_over_2025=sw["min_2019_flux"] / sw["max_2025_flux"],
               flux_level_IEC_spread_neutrons=pct(Cp.S1_IEC / Cp.PARMA, 95) / pct(Cp.S1_IEC / Cp.PARMA, 5)),
          open(OUT / "device_parametric.json", "w"), indent=1)
pd.set_option("display.width", 260)
print(D.round(3).to_string(index=False))
print(json.load(open(OUT / "device_parametric.json")))
