# Step 4 extension (author decision 2026-10-15; revised after Step 8 review 2026-10-16): device-level SEU demonstration.
# Device: Renesas R1LV1616R 16-Mbit SRAM (150 nm, SEU), Weibull cross sections fitted by Cecchetto et al. 2019
# (Aerospace 6:119, Table 2): M2 = standard characterization up to 230 MeV, M3 = response measured to GeV.
# Reference rate: PARMA neutron spectrum folded with sigma(E) along each route point (>= 6 km), per solar condition
# of routes.py (annual-mean W of 2019 and 2025, and the post-GLE storm day).
# Practitioner rate (IEC 62396 simplified method): flux > 10 MeV of each simplified model x one cross section:
#   sigma(200 MeV) of M2 (main case), or a spectrum-averaged cross section sigma_sp = R / F(>10 MeV) for the NYC sea-level
#   spectrum (stand-in for a white spallation beam test), for M2 and M3 (sensitivity).
# Sensitivity of the reference: M3 held constant above 3 GeV (driver keyword WC).
# Validation of the folding: Geneva 12 km, 2016-07-01 vs Cecchetto's MAIRE FIT (Table 3).
# Protons (Step 8 round 6, M1): the same Weibull cross sections folded with the PARMA proton spectrum (driver keywords
# WBP/WCP); the reference becomes neutrons + protons, compared with the IEC value on its 6000 (n + p) basis.
# Pions are not in PARMA and are not included.
import json, sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
C = json.load(open(ROOT / "data/refs/cecchetto2019_tables.json"))
dev = C["table2_weibull"]["Renesas_R1LV1616R_150nm_16Mbit_SEU"]
wb = lambda m: "%g %g %g %g" % (dev[m]["sigma_sat_cm2_per_dev"], dev[m]["E_th_MeV"], dev[m]["W_MeV"], dev[m]["s"])
WB = (f"WB {wb('M2')} WB {wb('M3')} WC {wb('M3')} 3000 "
      f"WBP {wb('M2')} WBP {wb('M3')} WCP {wb('M3')} 3000")
m2 = dev["M2"]
SIG200 = m2["sigma_sat_cm2_per_dev"] * (1 - np.exp(-((200.0 - m2["E_th_MeV"]) / m2["W_MeV"]) ** m2["s"]))
m3 = dev["M3"]  # same 200 MeV point from the GeV fit, for a count consistent with the M3 reference (Step 8 round 2)
SIG200_M3 = m3["sigma_sat_cm2_per_dev"] * (1 - np.exp(-((200.0 - m3["E_th_MeV"]) / m3["W_MeV"]) ** m3["s"]))
H2FIT = 3600 * 1e9  # per second -> per 1e9 device-hours

# 1) validation point
v = run_driver([f"gva D 2016 7 1 46.23 6.10 12.0 10 {WB}"])[0]
ref = C["table3_FIT_12km_Geneva_by_energy"]["Renesas"]
val = dict(PARMA_gt10_ncm2h=float(v[5]) * 3600, FIT_M2_PARMA=float(v[9]) * H2FIT, FIT_M3_PARMA=float(v[10]) * H2FIT,
           FIT_M2_MAIRE=ref["neutrons_below230"]["M2"] + ref["neutrons_above230"]["M2"],
           FIT_M3_MAIRE=ref["neutrons_below230"]["M3"] + ref["neutrons_above230"]["M3"], MAIRE_gt10_ncm2h=2834.0,
           FITp_M2_PARMA=float(v[12]) * H2FIT, FITp_M3_PARMA=float(v[13]) * H2FIT,
           FITp_M2_MAIRE=ref["protons_below230"]["M2"] + ref["protons_above230"]["M2"],
           FITp_M3_MAIRE=ref["protons_below230"]["M3"] + ref["protons_above230"]["M3"])
print("validation:", {k: round(x, 1) for k, x in val.items()})

# 2) spectrum-averaged cross sections from the NYC sea-level spectrum (ground, g = 0.15)
g = run_driver([f"nyc D 2002 11 15 40.71 -74.01 0 0.15 {WB}"])[0]
SIG_SP = dict(M2=float(g[9]) / float(g[5]), M3=float(g[10]) / float(g[5]))
print("sigma(200 MeV, M2) = %.3g; spectrum-averaged sigma M2 = %.3g, M3 = %.3g cm2/dev" % (SIG200, SIG_SP["M2"], SIG_SP["M3"]))

# 3) routes: reuse route profiles (same points and solar conditions as routes.py)
P = pd.read_csv(OUT / "routes_profiles.csv")
P = P[P.alt_km >= 6.0].copy()
lines = [f"r{i} S {w:.4f} {r:.5f} {d:.4f} 10 {WB}" for i, (w, r, d) in enumerate(zip(P.W, P.Rc, P.depth))]
o = run_driver(lines)
P["R_M2"] = [float(x[9]) for x in o]; P["R_M3"] = [float(x[10]) for x in o]; P["R_M3cap"] = [float(x[11]) for x in o]
P["Rp_M2"] = [float(x[12]) for x in o]; P["Rp_M3"] = [float(x[13]) for x in o]; P["Rp_M3cap"] = [float(x[14]) for x in o]
rows = []
for (route, cond), gg in P.groupby(["route", "cond"], sort=False):
    t = gg.t_min.values * 60.0  # s
    up = lambda r: float(np.trapezoid(r, t))
    res = dict(route=route, cond=cond, W=round(float(gg.W.mean()), 1), upsets_M2_PARMA=up(gg.R_M2.values),
               upsets_M3_PARMA=up(gg.R_M3.values), upsets_M3cap_PARMA=up(gg.R_M3cap.values),
               upsets_M2p_PARMA=up(gg.Rp_M2.values), upsets_M3p_PARMA=up(gg.Rp_M3.values), upsets_M3capp_PARMA=up(gg.Rp_M3cap.values))
    for m in ("S1_IEC", "S2_JESD_mid", "S4_AN_ABB", "FORMULA"):
        res[f"upsets_{m}_simplified"] = up(gg[m].values / 3600.0 * SIG200)
    res["upsets_FLAT6000_sigM2"] = up(np.full(len(gg), 6000.0) / 3600.0 * SIG200)       # flat design value, Step 8 round 3
    res["upsets_FLAT6000_sigM3at200"] = up(np.full(len(gg), 6000.0) / 3600.0 * SIG200_M3)
    res["upsets_S1_IEC_sigsp_M2"] = up(gg.S1_IEC.values / 3600.0 * SIG_SP["M2"])
    res["upsets_S1_IEC_sigsp_M3"] = up(gg.S1_IEC.values / 3600.0 * SIG_SP["M3"])
    iec6000 = gg.S1_IEC.values * 6000.0 / 5600.0   # the scaled design value on its neutron + proton basis
    res["upsets_S1_IEC6000_sigM2"] = up(iec6000 / 3600.0 * SIG200)
    res["upsets_S1_IEC6000_sigM3at200"] = up(iec6000 / 3600.0 * SIG200_M3)
    res["upsets_S1_IEC6000_sigsp_M3"] = up(iec6000 / 3600.0 * SIG_SP["M3"])
    rows.append(res)
S = pd.DataFrame(rows)
for m in ("S1_IEC", "S2_JESD_mid", "S4_AN_ABB", "FORMULA"):
    S[f"{m}_over_M2"] = S[f"upsets_{m}_simplified"] / S.upsets_M2_PARMA
    S[f"{m}_over_M3"] = S[f"upsets_{m}_simplified"] / S.upsets_M3_PARMA
S["M3_over_M2"] = S.upsets_M3_PARMA / S.upsets_M2_PARMA
S["S1_IEC_over_M3cap"] = S.upsets_S1_IEC_simplified / S.upsets_M3cap_PARMA
S["S1_IEC_sigspM3_over_M3"] = S.upsets_S1_IEC_sigsp_M3 / S.upsets_M3_PARMA
S["S1_IEC_sigspM2_over_M2"] = S.upsets_S1_IEC_sigsp_M2 / S.upsets_M2_PARMA
S["S1_IEC_sigM3at200_over_M3"] = S.S1_IEC_over_M3 * SIG200_M3 / SIG200
S["FLAT6000_sigM2_over_M3"] = S.upsets_FLAT6000_sigM2 / S.upsets_M3_PARMA
S["FLAT6000_sigM3at200_over_M3"] = S.upsets_FLAT6000_sigM3at200 / S.upsets_M3_PARMA
# neutrons + protons
for m in ("M2", "M3", "M3cap"):
    S[f"upsets_{m}np_PARMA"] = S[f"upsets_{m}_PARMA"] + S[f"upsets_{m}p_PARMA"]
    S[f"p_over_n_{m}"] = S[f"upsets_{m}p_PARMA"] / S[f"upsets_{m}_PARMA"]
S["S1_IEC6000_over_M2np"] = S.upsets_S1_IEC6000_sigM2 / S.upsets_M2np_PARMA
S["S1_IEC6000_over_M3np"] = S.upsets_S1_IEC6000_sigM2 / S.upsets_M3np_PARMA
S["S1_IEC6000_over_M3capnp"] = S.upsets_S1_IEC6000_sigM2 / S.upsets_M3capnp_PARMA
S["S1_IEC6000_sigM3at200_over_M3np"] = S.upsets_S1_IEC6000_sigM3at200 / S.upsets_M3np_PARMA
S["S1_IEC6000_sigspM3_over_M3np"] = S.upsets_S1_IEC6000_sigsp_M3 / S.upsets_M3np_PARMA
S["FLAT6000_sigM2_over_M2np"] = S.upsets_FLAT6000_sigM2 / S.upsets_M2np_PARMA
S["FLAT6000_sigM2_over_M3np"] = S.upsets_FLAT6000_sigM2 / S.upsets_M3np_PARMA
S["FLAT6000_sigM3at200_over_M3np"] = S.upsets_FLAT6000_sigM3at200 / S.upsets_M3np_PARMA
S["FORMULA_over_M3np"] = S.upsets_FORMULA_simplified / S.upsets_M3np_PARMA
S.to_csv(OUT / "ser_routes.csv", index=False, lineterminator="\n", float_format="%.6g")
json.dump(dict(device="Renesas R1LV1616R 16 Mbit SRAM, 150 nm, SEU (Cecchetto et al. 2019, Table 2)",
               sigma_200MeV_M2_cm2_per_dev=SIG200, sigma_200MeV_M3_cm2_per_dev=SIG200_M3,
               sigma_spectrum_avg_NYC_cm2_per_dev=SIG_SP, M3_cap_MeV=3000,
               validation_geneva_12km_2016_07_01=val),
          open(OUT / "ser_meta.json", "w"), indent=1)
pd.set_option("display.width", 250)
cols = ["route", "cond", "W", "upsets_M2_PARMA", "upsets_M3_PARMA", "S1_IEC_over_M2", "S1_IEC_over_M3", "S1_IEC_over_M3cap",
        "S1_IEC_sigspM3_over_M3", "S1_IEC_sigspM2_over_M2", "FORMULA_over_M2", "M3_over_M2"]
print(S[cols].to_string(index=False, float_format=lambda x: "%.3g" % x))
