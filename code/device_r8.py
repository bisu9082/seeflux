# Step 8 round 8 (M5): device-level analysis on band fluxes, so that any cross-section shape can be folded.
# PARMA neutron and proton fluxes in 30 log bands from 1 MeV to 100 GeV (driver keyword WR with sigma_sat = 0) at
#  (a) the 802 independent cruise points, (b) the seven routes at FL370 for the 2019 and 2025 annual means (3-min points),
#  (c) New York City sea level, November 2002 (ground option), the terrestrial reference shape of JESD89B.
# A cross section is folded band by band with its mean over ln E inside the band (flux per ln E taken as constant in a
# band); the result is checked against direct Weibull folds. Responses: Renesas M2 and M3 (Cecchetto et al. 2019); M2
# with thresholds of 5, 10 and 20 MeV (other parameters kept); three alternative GeV shapes for M3 that keep M2 up to
# 230 MeV and reach the M3 value at 3 GeV (log-log linear, then constant; M3 capped at 1 GeV; M3 capped at 3 GeV).
# Test cross sections: sigma(200 MeV), sigma(230 MeV), and the cross section averaged over the sea-level shape from
# 10 MeV to an upper end of 400 MeV, 800 MeV or 100 GeV (JESD89B compares spallation beams with that shape).
# Output: data/step4/device_r8_bands.npz, device_r8.csv (routes), device_r8_points.csv (method factor), device_r8.json
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
        print("  driver", min(i + chunk, len(lines)), "/", len(lines), flush=True)
    return out


EDGES = np.logspace(0, 5, 31)            # 1 MeV to 100 GeV, 6 bands per decade
KW = " ".join(f"WR 0 0 1 1 {a:.6g} {b:.6g} {ip}" for ip in (0, 1) for a, b in zip(EDGES[:-1], EDGES[1:]))
NB = len(EDGES) - 1
CACHE = OUT / "device_r8_bands.npz"


def bands(lines):
    o = run_driver([f"{l} {KW}" for l in lines])
    F10 = np.array([float(x[5]) for x in o])
    B = np.array([[float(v) for v in x[9:9 + 2 * NB]] for x in o])
    return F10, B[:, :NB], B[:, NB:]


V = pd.read_csv(OUT / "validation_direct.csv", parse_dates=["date"])
C = V[V.alt_km.between(9.0, 12.5)].reset_index(drop=True)
PF = pd.read_csv(OUT / "routes_levels_profiles_FL370.csv")
D = daily_w(); COND = {"y2019": annual_mean(2019, D), "y2025": annual_mean(2025, D)}
rt_rows = []
for rt, g in PF[PF.cond == "y2019"].groupby("route", sort=False):
    g = g.iloc[::3]
    for cond, w in COND.items():
        for t_, la, lo, h, rc in zip(g.t_min, g.lat, g.lon, g.alt_km, g.Rc):
            rt_rows.append((rt, cond, t_, la, h, rc, w))
RT = pd.DataFrame(rt_rows, columns=["route", "cond", "t_min", "lat", "alt_km", "Rc", "W"])
if CACHE.exists():
    z = np.load(CACHE)
    Fp, Bpn, Bpp, Fr, Brn, Brp, Fg, Bgn, Bgp = (z[k] for k in ("Fp", "Bpn", "Bpp", "Fr", "Brn", "Brp", "Fg", "Bgn", "Bgp"))
else:
    print("cruise points"); Fp, Bpn, Bpp = bands([f"c{i} D {d.year} {d.month} {d.day} {la:.4f} {lo:.4f} {h:.4f} 10"
                                                  for i, (d, la, lo, h) in enumerate(zip(C.date, C.lat, C.lon, C.alt_km))])
    print("routes"); Fr, Brn, Brp = bands([f"r{i} A {w:.3f} {rc:.5f} {h:.5f} 10" for i, (w, rc, h) in enumerate(zip(RT.W, RT.Rc, RT.alt_km))])
    print("ground"); Fg, Bgn, Bgp = bands(["nyc D 2002 11 15 40.71 -74.01 0 0.15"])
    np.savez(CACHE, Fp=Fp, Bpn=Bpn, Bpp=Bpp, Fr=Fr, Brn=Brn, Brp=Brp, Fg=Fg, Bgn=Bgn, Bgp=Bgp)
dev = json.load(open(ROOT / "data/refs/cecchetto2019_tables.json"))["table2_weibull"]["Renesas_R1LV1616R_150nm_16Mbit_SEU"]
P = {m: (dev[m]["sigma_sat_cm2_per_dev"], dev[m]["E_th_MeV"], dev[m]["W_MeV"], dev[m]["s"]) for m in ("M2", "M3")}


def weib(p):
    sat, eth, w, s = p
    return lambda E: np.where(E > eth, sat * (1 - np.exp(-np.clip((E - eth) / w, 0, None) ** s)), 0.0)


m2, m3 = weib(P["M2"]), weib(P["M3"])
s230, s3000, s1000 = float(m3(230.0)), float(m3(3000.0)), float(m3(1000.0))
m2_230 = float(m2(230.0))
RESP = {
    "M2": m2, "M3": m3,
    "M2, E_th 5 MeV": weib((P["M2"][0], 5.0, P["M2"][2], P["M2"][3])),
    "M2, E_th 10 MeV": weib((P["M2"][0], 10.0, P["M2"][2], P["M2"][3])),
    "M2, E_th 20 MeV": weib((P["M2"][0], 20.0, P["M2"][2], P["M2"][3])),
    "M3 log-log 0.23-3 GeV": lambda E: np.where(E <= 230, m2(E), np.where(E >= 3000, s3000,
                                                np.exp(np.interp(np.log(E), [np.log(230), np.log(3000)], [np.log(m2_230), np.log(s3000)])))),
    "M3 capped at 1 GeV": lambda E: m3(np.minimum(E, 1000.0)),
    "M3 capped at 3 GeV": lambda E: m3(np.minimum(E, 3000.0)),
}


def band_sigma(f):
    """mean of sigma over ln E in each band (64 sub-points)"""
    out = np.zeros(NB)
    for i, (a, b) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        e = np.exp(np.linspace(np.log(a), np.log(b), 65))
        out[i] = np.trapezoid(f(e), np.log(e)) / np.log(b / a)
    return out


SB = {k: band_sigma(f) for k, f in RESP.items()}
fold = lambda sb, Bn, Bp: (Bn + Bp) @ sb
# check against direct Weibull folds at 5 cruise points (neutrons + protons)
chk = []
for i in range(0, len(C), 160):
    d_, la, lo, h = C.date[i], C.lat[i], C.lon[i], C.alt_km[i]
    o = _rd([f"k D {d_.year} {d_.month} {d_.day} {la:.4f} {lo:.4f} {h:.4f} 10 WB {P['M2'][0]:g} {P['M2'][1]:g} {P['M2'][2]:g} {P['M2'][3]:g} "
             f"WBP {P['M2'][0]:g} {P['M2'][1]:g} {P['M2'][2]:g} {P['M2'][3]:g} WB {P['M3'][0]:g} {P['M3'][1]:g} {P['M3'][2]:g} {P['M3'][3]:g} "
             f"WBP {P['M3'][0]:g} {P['M3'][1]:g} {P['M3'][2]:g} {P['M3'][3]:g}"])[0]
    d2, d3 = float(o[9]) + float(o[10]), float(o[11]) + float(o[12])
    chk.append(dict(i=i, M2_band_over_direct=float(fold(SB["M2"], Bpn[i], Bpp[i]) / d2), M3_band_over_direct=float(fold(SB["M3"], Bpn[i], Bpp[i]) / d3)))
# test cross sections from the sea-level shape (neutrons), 10 MeV to E_up
lo_b = EDGES[:-1]


def sig_sp(sb, e_up):
    sel = (lo_b >= 10 - 1e-9) & (EDGES[1:] <= e_up * 1.0001)
    return float((Bgn[0][sel] @ sb[sel]) / Bgn[0][sel].sum())


TEST = {}
for k, f in RESP.items():
    TEST[k] = {"s200": float(f(np.array([200.0]))[0]), "s230": float(f(np.array([230.0]))[0]),
               "sp400": sig_sp(SB[k], 398.2), "sp800": sig_sp(SB[k], 794.4), "sp1e5": sig_sp(SB[k], 1e5)}
# (a) method factor at the cruise points: F10 x sigma(200 MeV) / folded n+p count, and its dependence
pts = C[["lat", "lon", "alt_km", "W", "Rc"]].copy()
for k in ("M2", "M3"):
    pts[f"method_{k}"] = Fp * TEST[k]["s200"] / fold(SB[k], Bpn, Bpp)
pts.to_csv(OUT / "device_r8_points.csv", index=False, lineterminator="\n", float_format="%.6g")
mf = {}
for k in ("M2", "M3"):
    x = pts[f"method_{k}"]
    mf[k] = dict(p05=float(x.quantile(.05)), median=float(x.median()), p95=float(x.quantile(.95)), spread=float(x.quantile(.95) / x.quantile(.05)),
                 spearman_Rc=float(x.corr(pts.Rc, method="spearman")), spearman_alt=float(x.corr(pts.alt_km, method="spearman")),
                 spearman_W=float(x.corr(pts.W, method="spearman")))
# (b) routes: IEC count (6000 basis) / reference for every response and test cross section
RT["iec6000"] = S.s1_iec_boeing(RT.alt_km.values, RT.lat.values) * 6000 / 5600 / 3600   # per s
rows = []
for (rt, cond), g in RT.groupby(["route", "cond"], sort=False):
    idx = g.index.values; t = g.t_min.values * 60
    F_iec = float(np.trapezoid(g.iec6000.values, t))
    for k in RESP:
        ref = float(np.trapezoid(fold(SB[k], Brn[idx], Brp[idx]), t))
        r = dict(route=rt, cond=cond, response=k, ref=ref)
        for tk, sv in TEST[k].items():
            r[f"IEC_{tk}_over_ref"] = F_iec * sv / ref
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(OUT / "device_r8.csv", index=False, lineterminator="\n", float_format="%.6g")
summ = dict(band_check=chk, method_factor=mf, test_sigma=TEST,
            routes={k: {c: [float(R[R.response == k][c].min()), float(R[R.response == k][c].max())] for c in R.columns if c.startswith("IEC_")} for k in RESP})
json.dump(summ, open(OUT / "device_r8.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "test_sigma"}, indent=1))
