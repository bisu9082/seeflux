# Step 8 round 9 (F1, R3): headline spread on the pooled cruise points of the first independent set
# (validation_direct.csv, seed of validate_direct.py) and the fresh sample (fresh_sample.csv, seed 20261023), which share
# one design. Reports, per sample and pooled: P95/P5 with percentile and BCa bootstrap intervals, the SD of ln r as a
# less tail-sensitive spread measure, the IEC/JESD89 spread ratio, and the difference between the two samples.
# Output: data/step4/pooled_spread.json
import json, pathlib, numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
SEED, B = 20261024, 4000
cols = ["alt_km", "PARMA", "S1_IEC", "S2_JESD_mid"]
A = pd.read_csv(OUT / "validation_direct.csv")[cols]
F = pd.read_csv(OUT / "fresh_sample.csv")[cols]
A, F = (d[d.alt_km.between(9.0, 12.5)].reset_index(drop=True) for d in (A, F))
P = pd.concat([A, F], ignore_index=True)

sp = lambda r, axis=-1: np.percentile(r, 95, axis=axis) / np.percentile(r, 5, axis=axis)
sdl = lambda r, axis=-1: np.std(np.log(r), axis=axis, ddof=1)
rng = np.random.default_rng(SEED)


def boot(stat, *data):
    r = stats.bootstrap(data, stat, n_resamples=B, method="BCa", random_state=rng, vectorized=True,
                        paired=len(data) > 1)
    p = stats.bootstrap(data, stat, n_resamples=B, method="percentile", random_state=rng, vectorized=True,
                        paired=len(data) > 1)
    return dict(estimate=float(stat(*data)), bca=[float(r.confidence_interval.low), float(r.confidence_interval.high)],
                percentile=[float(p.confidence_interval.low), float(p.confidence_interval.high)])


def summary(D):
    ri, rj = (D.S1_IEC / D.PARMA).to_numpy(), (D.S2_JESD_mid / D.PARMA).to_numpy()
    return dict(n=len(D),
                IEC_spread=boot(sp, ri), JESD_spread=boot(sp, rj),
                IEC_sdlog=boot(sdl, ri), JESD_sdlog=boot(sdl, rj),
                spread_ratio=boot(lambda a, b, axis=-1: sp(a, axis) / sp(b, axis), ri, rj),
                sdlog_ratio=boot(lambda a, b, axis=-1: sdl(a, axis) / sdl(b, axis), ri, rj),
                IEC_below1=float((ri < 1).mean()))


res = dict(seed=SEED, B=B, first=summary(A), fresh=summary(F), pooled=summary(P))
# difference first minus fresh in IEC spread, independent resampling of the two samples
ra, rf = (A.S1_IEC / A.PARMA).to_numpy(), (F.S1_IEC / F.PARMA).to_numpy()
d = np.array([sp(ra[rng.integers(0, len(ra), len(ra))]) - sp(rf[rng.integers(0, len(rf), len(rf))]) for _ in range(B)])
res["IEC_spread_diff_first_minus_fresh"] = dict(estimate=float(sp(ra) - sp(rf)),
                                                 percentile=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
json.dump(res, open(OUT / "pooled_spread.json", "w"), indent=1)
print(json.dumps(res, indent=1))
