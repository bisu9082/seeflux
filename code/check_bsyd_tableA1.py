# Check of the printed BSYD equations (Gordon et al. 2004, Appendix (A-1)-(A-6), as implemented in standards.py)
# against Gordon's own Table A-I at sea level (mid-level solar modulation). Output: data/step4/bsyd_tableA1_check.json
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from standards import _bsyd

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = json.load(open(ROOT / "data/refs/gordon2004_tableA1.json"))
rc, tab = np.array(T["Rc_GV"], float), np.array(T["F_BSYD_Smid"])
eq = np.array([float(_bsyd(r, 1033.2, "mid")) for r in rc])
rel = eq / tab - 1
out = dict(Rc_GV=rc.tolist(), table=tab.tolist(), equations=eq.round(4).tolist(), rel_diff=rel.round(4).tolist(),
           min_rel=float(rel.min()), max_rel=float(rel.max()))
json.dump(out, open(ROOT / "data/step4/bsyd_tableA1_check.json", "w"), indent=1)
print("equations / Table A-I - 1: %.1f%% to %.1f%%" % (100 * rel.min(), 100 * rel.max()))
