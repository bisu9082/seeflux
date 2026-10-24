# Step 8 round 8 (M4): the cruise statistics by flight-level band and with a traffic proxy (R3, R4).
# Subsets of the 802 independent cruise points: all 9-12.5 km; FL350-FL410 (10.67-12.50 km); 20-60 deg N, where most
# long-haul traffic flies; both together. A second proxy weights each point by the share of flight hours in its latitude
# band, approximated as 0.1 (south of 20 S), 0.1 (20 S-20 N), 0.7 (20-60 N), 0.1 (north of 60 N), relative to the area
# share of the band; this is an illustration, not traffic data.
# For the scaled IEC curves (5600, neutrons), the 6000 curves and the larger of 6000 and the curves (neutrons + protons)
# and JESD89: median, share below one at k = 1, 1.07 and 1.28, P95/P5; bootstrap 95% intervals (B = 2000).
# Output: data/step4/subsets.csv
import pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
V = pd.read_csv(OUT / "validation_direct.csv"); P = pd.read_csv(OUT / "r7_points_np.csv")
V["np"] = P.PARMA_n2 + P.PARMA_p
C = V[V.alt_km.between(9.0, 12.5)].copy()
C["r_iec"] = C.S1_IEC / C.PARMA
C["r_iec6000_np"] = C.S1_IEC * 6000 / 5600 / C.np
C["r_floor_np"] = np.maximum(6000.0, C.S1_IEC * 6000 / 5600) / C.np
C["r_jesd"] = C.S2_JESD_mid / C.PARMA
fl = C.alt_km.between(35000 * 0.3048 / 1000, 41000 * 0.3048 / 1000)
nl = C.lat.between(20, 60)
band = pd.cut(C.lat, [-90, -20, 20, 60, 90], labels=["S", "T", "N", "P"])
area = {"S": (1 - np.sin(np.radians(20))) / 2, "T": np.sin(np.radians(20)), "N": (np.sin(np.radians(60)) - np.sin(np.radians(20))) / 2,
        "P": (1 - np.sin(np.radians(60))) / 2}
traffic = {"S": 0.1, "T": 0.1, "N": 0.7, "P": 0.1}
wt = band.map(lambda b: traffic[b] / area[b]).astype(float)
rng = np.random.default_rng(20261023)


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]); return x[o][np.searchsorted(c, q * c[-1])]


def stats(x, w):
    x = np.asarray(x); w = np.asarray(w, float)
    p5, p50, p95 = wq(x, w, 0.05), wq(x, w, 0.5), wq(x, w, 0.95)
    return dict(median=p50, below1_k1=float(np.average(x < 1, weights=w)), below1_k107=float(np.average(1.07 * x < 1, weights=w)),
                below1_k128=float(np.average(1.28 * x < 1, weights=w)), spread=p95 / p5)


rows = []
for sub, sel, weighted in (("all cruise (9-12.5 km)", np.ones(len(C), bool), False), ("FL350-FL410", fl.values, False),
                           ("20-60 N", nl.values, False), ("20-60 N, FL350-FL410", (fl & nl).values, False),
                           ("traffic-proxy weights", np.ones(len(C), bool), True)):
    s = C[sel]; w = wt[sel].values if weighted else np.ones(len(s))
    for col, lab in (("r_iec", "IEC curves 5600 / n"), ("r_iec6000_np", "IEC curves 6000 / n+p"), ("r_floor_np", "larger of 6000 and curves / n+p"),
                     ("r_jesd", "JESD89 / n")):
        st = stats(s[col].values, w)
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(s), len(s)); bs.append([stats(s[col].values[i], w[i])[k] for k in ("below1_k1", "spread")])
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        rows.append(dict(subset=sub, n=len(s), model=lab, **st, below1_k1_lo=lo[0], below1_k1_hi=hi[0], spread_lo=lo[1], spread_hi=hi[1]))
R = pd.DataFrame(rows)
R.to_csv(OUT / "subsets.csv", index=False, lineterminator="\n", float_format="%.6g")
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
