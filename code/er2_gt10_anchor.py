# In-flight anchor above 10 MeV at cruise altitude (Step 8 round 4): the ER-2 spectrum measured at 11.9 km
# (201 g/cm2, 4.3 GV) by Goldhagen et al. 2004 (Radiat. Prot. Dosim. 110:387), Fig. 5, plotted as E dPhi/dE x 2.9.
# The curve is extracted from the vector paths of the published PDF (grey line, width 1.11), not digitized by eye.
# Axis calibration from the tick labels; the integral over all energies is checked against the stated total 3.4 cm-2 s-1.
# Output: data/step4/er2_gt10_anchor.json
import json, pathlib, sys, re
import numpy as np
import fitz

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wslpath_util import run_driver

p = fitz.open(ROOT / "data/refs/goldhagen2004airplane.pdf")[3]
FIG = fitz.Rect(100, 276, 285, 400)
# tick labels: x "10-8 ... 10 4" below the axis, y "0.0 0.5 1.0" left of the axis
xt, yt = [], []
for b in p.get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        s = "".join(x["text"] for x in l["spans"]).strip(); bb = fitz.Rect(l["bbox"])
        if FIG.y1 - 10 < bb.y0 < FIG.y1 + 15 and FIG.x0 - 10 < bb.x0 < FIG.x1 and s.startswith("10"):
            spans = l["spans"]
            exps = [sp["text"].strip() for sp in spans if sp["size"] < spans[0]["size"]]  # superscript exponent
            if exps:
                xt.append((float(exps[0].replace("−", "-").replace("ÿ", "-")), (bb.x0 + bb.x1) / 2))
        if re.fullmatch(r"[01]\.[05]", s) and bb.x1 < FIG.x0 + 20 and FIG.y0 - 5 < bb.y0 < FIG.y1:
            yt.append((float(s), (bb.y0 + bb.y1) / 2))
xt, yt = sorted(set(xt)), sorted(set(yt))
assert len(xt) >= 4 and len(yt) >= 3, (xt, yt)
ax = np.polyfit([c for _, c in xt], [e for e, _ in xt], 1)      # page x -> log10(E)
ay = np.polyfit([c for _, c in yt], [v for v, _ in yt], 1)      # page y -> E dPhi/dE
# grey curve (12 km x 2.9): stroke width ~1.11, grey colour
pts = []
for d in p.get_drawings():
    if abs((d.get("width") or 0) - 1.113) < 0.01 and FIG.intersects(d["rect"]):
        for it in d["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
pts = np.array(sorted(set(pts)))
logE = np.polyval(ax, pts[:, 0]); y = np.polyval(ay, pts[:, 1]) / 2.9
# step histogram: keep, for each x, the upper envelope traced by the stroke; integrate E dPhi/dE over ln E
order = np.argsort(logE); logE, y = logE[order], y[order]
ux = np.unique(np.round(logE, 4))
yy = np.array([y[np.round(logE, 4) == u].max() for u in ux])
lnE = ux * np.log(10)
total = float(np.trapezoid(yy, lnE))
hi = ux >= 1.0
gt10 = float(np.trapezoid(yy[hi], lnE[hi]))
# PARMA at the same condition (date of the campaign, measured cutoff and depth)
o = run_driver(["er2 S 28.6 4.3 201 10"])[0]   # W of 1997-06-15 from bench_nyc.json; Rc and depth as measured
parma_gt10, parma_tot = float(o[5]), float(o[8])
out = dict(source="Goldhagen et al. 2004, Fig. 5, 200 g/cm2 (11.9 km, 4.3 GV), vector path of the published PDF",
           axis_ticks_x=xt, axis_ticks_y=yt, n_points=int(len(ux)),
           measured_total_from_curve_cm2s=total, measured_total_stated_cm2s=3.4, total_check_ratio=total / 3.4,
           measured_gt10_cm2s=gt10, measured_gt10_fraction=gt10 / total,
           PARMA_gt10_cm2s=parma_gt10, PARMA_total_cm2s=parma_tot,
           k_gt10_PARMA_over_measured=parma_gt10 / gt10, k_total_PARMA_over_measured=parma_tot / 3.4,
           note="Goldhagen et al. state the unfolded spectrum is lower than calculations above 10 MeV; unfolding uncertainty is not given per energy band")
json.dump(out, open(ROOT / "data/step4/er2_gt10_anchor.json", "w"), indent=1)
print(json.dumps(out, indent=1))
