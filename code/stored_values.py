# Final audit (A3, 2026-10-24): values quoted in the text that were computed interactively and not stored.
#  (1) PARMA proton / neutron flux above 10 MeV at 12 km over Geneva on 2016-07-01 (Methods)
#  (2) PARMA flux above 10 MeV at Incheon, FL350, 2025-11-15 for the worked surrogate example (Supplementary S9)
# Output: data/step4/stored_values.json
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver
from seeflux_formula import flux_gt10

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/step4"
g = run_driver(["gva D 2016 7 1 46.23 6.10 12.0 10 PR"])[0]
e = run_driver([f"icn D 2025 11 15 37.47 126.45 {35000 * 0.3048 / 1000:.5f} 10"])[0]
res = dict(geneva_12km_2016_07_01=dict(W=float(g[1]), Rc=float(g[2]), n_gt10=float(g[5]) * 3600, p_gt10=float(g[9]) * 3600,
                                       p_over_n=float(g[9]) / float(g[5])),
           incheon_FL350_2025_11_15=dict(W=float(e[1]), Rc=float(e[2]), alt_km=35000 * 0.3048 / 1000, PARMA_gt10=float(e[5]) * 3600,
                                         surrogate=float(flux_gt10(35000 * 0.3048 / 1000, float(e[2]), float(e[1])))))
json.dump(res, open(OUT / "stored_values.json", "w"), indent=1)
print(json.dumps(res, indent=1))
