# Step 4: validate the reference models (PARMA, CARI-7) against in-flight measured total neutron fluence rates,
# Goldhagen, Clem & Wilson 2004, Radiat. Prot. Dosim. 110:387, Table 1 (NASA ER-2, June 1997, solar minimum).
# Measured Rc and depth are used as given in the table. PARMA W-index is taken from PARMA's own table for 1997-06-15.
# Also computes PARMA >10 MeV at NYC sea level (ground, g = 0.15) for Nov 2002, the Gordon et al. 2004 measurement
# month, for comparison with the JESD89B reference 12.946 n/cm2/h (Annex A.2).
import subprocess, pathlib, shutil, sys, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver
import standards as S
CARI = ROOT / "cari7/CARI_7_DVD"
T1 = pd.DataFrame([  # lat, lon(E+), Rc GV, depth g/cm2, alt km, measured fluence rate cm-2 s-1
    ("ER2_19N", 19, -127, 11.6, 53.5, 20.3, 1.24),
    ("ER2_54N", 54, -117, 0.8, 56, 20.0, 9.7),
    ("ER2_56N", 56, -121, 0.7, 101, 16.2, 9.5),
    ("ER2_38N_11.9km", 38, -122, 4.3, 201, 11.9, 3.4),
    ("ground_37N", 37, -76, 2.7, 1030, 0.0, 0.0126),
], columns=["site", "lat", "lon", "Rc", "depth", "alt_km", "measured"])


def parma(lines):
    return run_driver(lines)


W = float(parma(["w D 1997 6 15 37.4 -122 0 10"])[0][1])
nyc = parma(["nyc D 2002 11 15 40.71 -74.01 0 0.15"])[0]
NYC_NOV2002_GT10 = float(nyc[5]) * 3600

lines = [f"{r.site} S {W} {r.Rc} {r.depth} {0.15 if r.alt_km == 0 else 10}" for r in T1.itertuples()]
out = parma(lines)
T1["PARMA_total"] = [float(l[8]) for l in out]
T1["PARMA_gt10"] = [float(l[5]) for l in out]

# CARI-7 neutron fluence (D1, P1) at the same lat/lon and depth (G units), June 1997 monthly mean
loc = ["C seeflux Goldhagen 2004 benchmark", "START-------------------------------------------------"]
for r in T1.itertuples():
    ns, ew = ("N" if r.lat >= 0 else "S"), ("E" if r.lon >= 0 else "W")
    loc.append(f"{ns}, {abs(r.lat):.3f}, {ew}, {abs(r.lon):.3f}, G, {r.depth:.1f}, 1997/06/00, H0, D1, P1, C4, S0")
loc.append("STOP--------------------------------------------------")
(CARI / "GOLDBM.LOC").write_text("\n".join(loc) + "\n")
# CARI-7 batch mode reads line 5 of DEFAULT.INP; edit it for this run and restore the original afterwards.
# (CARI.INI must contain MENUS = NO! for unattended runs; checked below.)
assert "MENUS     = NO!" in (CARI / "CARI.INI").read_text(), "set MENUS = NO! in CARI.INI for batch runs"
orig = (CARI / "DEFAULT.INP").read_text()
inp = orig.split("\n"); inp[4] = "GOLDBM.LOC"
try:
    (CARI / "DEFAULT.INP").write_text("\n".join(inp))
    with open(CARI / "goldbm.log", "w") as lg:
        subprocess.run([str(CARI / "cari-7.exe")], cwd=str(CARI), stdin=subprocess.DEVNULL, stdout=lg, check=True, timeout=600)
finally:
    (CARI / "DEFAULT.INP").write_text(orig)
ans = [l for l in (CARI / "GOLDBM.ANS").read_text().split("\n") if "NEUTRONS" in l]
T1["CARI7_total"] = [float(l.split(",")[8]) for l in ans]
T1["CARI7_Rc"] = [float(l.split(",")[6]) for l in ans]
T1["PARMA_over_meas"] = T1.PARMA_total / T1.measured
T1["CARI7_over_meas"] = T1.CARI7_total / T1.measured
T1["PARMA_gt10_frac"] = T1.PARMA_gt10 / T1.PARMA_total
import json
json.dump(dict(W_1997_06_15=W, PARMA_NYC_Nov2002_gt10_ncm2h=NYC_NOV2002_GT10, JESD89_NYC_ref_ncm2h=S.JESD_REF),
          open(ROOT / "data/step4/bench_nyc.json", "w"), indent=1)
print("W(1997-06-15) = %.1f; PARMA NYC Nov 2002 >10 MeV = %.2f n/cm2/h" % (W, NYC_NOV2002_GT10))
T1.to_csv(ROOT / "data/step4/bench_goldhagen2004.csv", index=False, lineterminator="\n", float_format="%.4g")
print(T1[["site", "Rc", "CARI7_Rc", "depth", "measured", "PARMA_total", "CARI7_total", "PARMA_over_meas",
          "CARI7_over_meas", "PARMA_gt10_frac"]].to_string(index=False, float_format=lambda v: "%.3f" % v))
