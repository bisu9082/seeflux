# Second-model check of the scale-independent findings (Step 8 round 5): the spread of the IEC and JESD89 ratios and
# the solar swing on the routes depend on how the reference varies with cutoff rigidity and solar modulation, not only
# on its absolute scale. CARI-7 (FAA release of 15 Nov 2018) is run at the same points. This release reports the total
# neutron fluence rate, not the part above 10 MeV, so the comparison uses totals for both codes and reports PARMA's
# >10 MeV result next to them to show how far the total stands in for the part above 10 MeV.
#  (1) independent cruise set (802 points): P95/P5 of IEC/ref and JESD/ref, ref = CARI-7 total, PARMA total, PARMA >10 MeV;
#      both codes at the monthly mean of the date (CARI-7 monthly option; PARMA at the monthly mean W, Rc of this study)
#  (2) FL370 routes: ratio of the 2019 to the 2025 annual-mean route fluence (CARI-7: mean of the 12 monthly runs)
# Output: data/step4/cari_crosscheck.json, cari_crosscheck_points.csv
import json, pathlib, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver
from solar_w import daily_w, monthly_mean, annual_mean

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data/step4"
CARI = ROOT / "cari7/CARI_7_DVD"
CHUNK = 400


def cari_total(rows):
    """rows: (lat, lon, alt_km, 'yyyy/mm') -> CARI-7 neutron fluence rate (monthly mean), one value per row."""
    assert "MENUS     = NO!" in (CARI / "CARI.INI").read_text()
    orig = (CARI / "DEFAULT.INP").read_text()
    vals = []
    try:
        for c0 in range(0, len(rows), CHUNK):
            loc = ["C seeflux CARI-7 cross-check", "START-------------------------------------------------"]
            for la, lo, h, ym in rows[c0:c0 + CHUNK]:
                s = (f"{'N' if la >= 0 else 'S'}, {abs(la):.3f}, {'E' if lo >= 0 else 'W'}, {abs(lo):.3f}, K, {h:.3f}, "
                     f"{ym}/00, H0, D1, P1, C4, S0")
                assert len(s) <= 66, s
                loc.append(s)
            loc.append("STOP--------------------------------------------------")
            (CARI / "XCHECK.LOC").write_text("\n".join(loc) + "\n")
            inp = orig.split("\n"); inp[4] = "XCHECK.LOC"
            (CARI / "DEFAULT.INP").write_text("\n".join(inp))
            with open(CARI / "xcheck.log", "w") as lg:
                subprocess.run([str(CARI / "cari-7.exe")], cwd=str(CARI), stdin=subprocess.DEVNULL, stdout=lg, check=True, timeout=3600)
            ans = [l for l in (CARI / "XCHECK.ANS").read_text().split("\n") if "NEUTRONS" in l]
            assert len(ans) == len(rows[c0:c0 + CHUNK]), (len(ans), c0)
            vals += [float(l.split(",")[8]) for l in ans]
            print("cari", c0 + len(ans), "/", len(rows), flush=True)
    finally:
        (CARI / "DEFAULT.INP").write_text(orig)
    return np.array(vals)


D = daily_w()
M = monthly_mean(D); M["ym"] = M.month.astype(str)
Wm = dict(zip(M.ym, M.W))

# (1) independent cruise set
V = pd.read_csv(OUT / "validation_direct.csv", parse_dates=["date"])
C = V[V.alt_km.between(9.0, 12.5)].copy().reset_index(drop=True)
C["ym"] = C.date.dt.strftime("%Y-%m")
C["W_month"] = C.ym.map(Wm)
o = run_driver([f"c{i} A {w:.4f} {rc:.4f} {h:.5f} 10" for i, (w, rc, h) in enumerate(zip(C.W_month, C.Rc, C.alt_km))])
C["P_gt10"] = [float(x[5]) for x in o]
C["P_tot"] = [float(x[8]) for x in o]
PTS = OUT / "cari_crosscheck_points.csv"
if PTS.exists() and len(pd.read_csv(PTS)) == len(C):   # CARI-7 runs are slow; reuse the stored point results
    C["CARI_tot"] = pd.read_csv(PTS).CARI_tot.values
else:
    C["CARI_tot"] = cari_total([(la, lo, h, ym.replace("-", "/")) for la, lo, h, ym in zip(C.lat, C.lon, C.alt_km, C.ym)])
C.to_csv(PTS, index=False, lineterminator="\n", float_format="%.6g")
sp = lambda r: float(np.percentile(r, 95) / np.percentile(r, 5))
res = dict(n=len(C), note="CARI-7 reports total neutrons only; spreads use totals for both codes, PARMA >10 MeV for reference")
for m, lab in (("S1_IEC", "IEC"), ("S2_JESD_mid", "JESD89")):
    res[f"spread_{lab}"] = dict(PARMA_gt10_monthW=sp(C[m] / C.P_gt10), PARMA_total_monthW=sp(C[m] / C.P_tot),
                                CARI_total=sp(C[m] / C.CARI_tot))
lr = np.log(C.CARI_tot / C.P_tot)
res["CARI_over_PARMA_total"] = dict(median=float(np.exp(lr.median())), p05=float(np.exp(lr.quantile(.05))), p95=float(np.exp(lr.quantile(.95))),
                                    spearman_Rc=float(pd.Series(lr).corr(C.Rc, method="spearman")),
                                    spearman_W=float(pd.Series(lr).corr(C.W_month, method="spearman")),
                                    spearman_alt=float(pd.Series(lr).corr(C.alt_km, method="spearman")))
res["PARMA_gt10_over_total"] = dict(p05=float((C.P_gt10 / C.P_tot).quantile(.05)), p95=float((C.P_gt10 / C.P_tot).quantile(.95)))

# (2) route solar swing at FL370, 2019 vs 2025 annual means
P = pd.read_csv(OUT / "routes_levels_profiles_FL370.csv")
P = P[(P.cond == "y2019") & (P.alt_km > 11.0)]
rows, keys = [], []
for rt, g in P.groupby("route", sort=False):
    g = g.iloc[::max(1, len(g) // 15)]   # about 15 cruise points per route, 24 monthly CARI-7 runs each
    for la, lo, h, rc in zip(g.lat, g.lon, g.alt_km, g.Rc):
        keys.append((rt, la, lo, h, rc))
for k in keys:
    for y in (2019, 2025):
        for mth in range(1, 13):
            rows.append((k[1], k[2], k[3], f"{y}/{mth:02d}"))
cv = cari_total(rows).reshape(len(keys), 2, 12).mean(axis=2)
w19, w25 = annual_mean(2019, D), annual_mean(2025, D)
o = run_driver([f"r{i}_{y} A {w:.4f} {k[4]:.4f} {k[3]:.5f} 10" for i, k in enumerate(keys) for y, w in ((19, w19), (25, w25))])
pg = np.array([float(x[5]) for x in o]).reshape(len(keys), 2)
pt = np.array([float(x[8]) for x in o]).reshape(len(keys), 2)
K = pd.DataFrame(keys, columns=["route", "lat", "lon", "alt_km", "Rc"])
K["C19"], K["C25"], K["Pt19"], K["Pt25"], K["Pg19"], K["Pg25"] = cv[:, 0], cv[:, 1], pt[:, 0], pt[:, 1], pg[:, 0], pg[:, 1]
sw = K.groupby("route", sort=False)[["C19", "C25", "Pt19", "Pt25", "Pg19", "Pg25"]].sum()
res["route_swing_2019_over_2025_FL370"] = {rt: dict(CARI_total=float(r.C19 / r.C25), PARMA_total=float(r.Pt19 / r.Pt25),
                                                    PARMA_gt10=float(r.Pg19 / r.Pg25), n_points=int((K.route == rt).sum()))
                                           for rt, r in sw.iterrows()}
res["W_annual"] = dict(y2019=w19, y2025=w25)
json.dump(res, open(OUT / "cari_crosscheck.json", "w"), indent=1)
print(json.dumps(res, indent=1))
