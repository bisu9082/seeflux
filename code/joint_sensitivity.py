# Step 8 round 8 (M6, R3): one joint propagation of the three uncertainty sources that were reported separately:
#  (1) reference scale k (PARMA / true flux): log-uniform over the envelope 0.88-1.37, and, as a second case, over the
#      in-flight range 1.07-1.28 (PARMA developer's ER-2 setting to free air; er2_geometry.json)
#  (2) digitization of the Boeing curves: smooth multiplicative perturbations, an offset and a slope in altitude
#      (about 12 km) and in |latitude| (about 45 deg), each normal with sigma = 2% (the digitized latitude ratio, 5.62,
#      is 2% from the stated 5.5)
#  (3) sampling: bootstrap over the 802 cruise points
# B = 4000 joint draws. For the scaled IEC curves (5600, neutrons) and JESD89 (mid, neutrons): median of k r, share of
# k r below one, P95/P5 of r; 95% intervals. Output: data/step4/joint_sensitivity.json
import json, pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
V = pd.read_csv(OUT / "validation_direct.csv")
C = V[V.alt_km.between(9.0, 12.5)].reset_index(drop=True)
rng = np.random.default_rng(20261023)
B, SIG = 4000, 0.02
r_iec0 = (C.S1_IEC / C.PARMA).values; r_j = (C.S2_JESD_mid / C.PARMA).values
h = C.alt_km.values; al = np.abs(C.lat.values); n = len(C)
res = {}
for case, (klo, khi) in (("envelope_0.88_1.37", (0.88, 1.37)), ("inflight_1.07_1.28", (1.07, 1.28))):
    out = {"IEC": [], "JESD": []}
    for _ in range(B):
        k = np.exp(rng.uniform(np.log(klo), np.log(khi)))
        a0, a1, b0, b1 = rng.normal(0, SIG, 4)
        dig = (1 + a0 + a1 * (h - 12.0) / 3.0) * (1 + b0 + b1 * (al - 45.0) / 45.0)
        i = rng.integers(0, n, n)
        for lab, r in (("IEC", r_iec0 * dig), ("JESD", r_j)):
            rr = r[i]
            out[lab].append([np.median(k * rr), np.mean(k * rr < 1), np.percentile(rr, 95) / np.percentile(rr, 5)])
    res[case] = {lab: dict(zip(("median_kr", "below1", "spread"),
                               [[float(np.percentile(np.array(v)[:, j], q)) for q in (2.5, 50, 97.5)] for j in range(3)]))
                 for lab, v in out.items()}
# digitization alone (k = 1, no bootstrap)
dd = []
for _ in range(B):
    a0, a1, b0, b1 = rng.normal(0, SIG, 4)
    r = r_iec0 * (1 + a0 + a1 * (h - 12.0) / 3.0) * (1 + b0 + b1 * (al - 45.0) / 45.0)
    dd.append([np.mean(r < 1), np.percentile(r, 95) / np.percentile(r, 5)])
dd = np.array(dd)
res["digitization_only_IEC"] = dict(below1=[float(np.percentile(dd[:, 0], q)) for q in (2.5, 50, 97.5)],
                                    spread=[float(np.percentile(dd[:, 1], q)) for q in (2.5, 50, 97.5)])
res["settings"] = dict(B=B, sigma_digitization=SIG, seed=20261023)
json.dump(res, open(OUT / "joint_sensitivity.json", "w"), indent=1)
print(json.dumps(res, indent=1))
