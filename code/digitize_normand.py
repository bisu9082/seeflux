# Digitize Normand 1996 (IEEE TNS 43(2):461, p.463) Fig. 2 (1-10 MeV flux vs altitude, kft) and
# Fig. 3 (vs geographic latitude). Page rendered at 300 dpi (data/refs/img/n96_p3_300.png), then cropped.
# Axis calibration pixels (crop coordinates) were read from the frame lines:
#   Fig.2 crop (400,300)-(2200,1380): x=0 kft at col 168, x=80 kft at col 1691; y=0 at row 940.5, y=1.4 at row 87
#   Fig.3 crop (400,1450)-(2200,2600): x=0 deg at col 263, x=90 deg at col 1640; y=0 at row 966, y=1.6 at row 78.5
# Fig.3 is checked against the paper's Eq.(3) to estimate digitization error.
import numpy as np, pandas as pd, pathlib
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMG = ROOT / "data/refs/img"
PDF = ROOT / "data/refs/normand1996avionics.pdf"
if PDF.exists():  # regenerate the page render and crops from the publisher PDF (not redistributed)
    import fitz
    IMG.mkdir(parents=True, exist_ok=True)
    fitz.open(PDF)[2].get_pixmap(dpi=300).save(IMG / "n96_p3_300.png")
    page = Image.open(IMG / "n96_p3_300.png").convert("L")
    page.crop((400, 300, 2200, 1380)).save(IMG / "fig2_crop.png")
    page.crop((400, 1450, 2200, 2600)).save(IMG / "fig3_crop.png")
elif not (IMG / "fig2_crop.png").exists():
    # neither the PDF nor the renders are available (they are not redistributed): keep the provided CSVs
    if (ROOT / "data/step4/normand1996_fig2_digitized.csv").exists():
        print("source figure not available; using provided data/step4/normand1996_fig*_digitized.csv")
        raise SystemExit(0)
    raise SystemExit("need data/refs/normand1996avionics.pdf or the provided digitized CSVs")


def trace(path, x0, x1, xmax, y0, y1, ymax, pad=8):
    """Follow the thick curve column by column from right to left, keeping the dark run
    (6-30 px tall) nearest to the previous point, which rejects ticks, grid dots and axis lines."""
    a = np.array(Image.open(path).convert("L")) < 110
    xs, ys, prev = [], [], None
    for c in range(int(x1) - pad, int(x0) + pad, -1):
        r = np.where(a[int(y1) + pad:int(y0) - 2, c])[0]
        if len(r) == 0:
            continue
        runs = [q for q in np.split(r, np.where(np.diff(r) > 1)[0] + 1) if 4 <= len(q) <= 30]
        if not runs:
            continue
        cand = [int(y1) + pad + q.mean() for q in runs]
        yc = cand[int(np.argmin([abs(v - prev) for v in cand]))] if prev is not None else cand[0]
        if prev is not None and abs(yc - prev) > 15:
            continue
        prev = yc
        xs.append((c - x0) / (x1 - x0) * xmax)
        ys.append((y0 - yc) / (y0 - y1) * ymax)
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(ys)[o]


ax, ay = trace(IMG / "fig2_crop.png", 168, 1691, 80.0, 940.5, 87, 1.4)
lx, ly = trace(IMG / "fig3_crop.png", 263, 1640, 90.0, 966, 78.5, 1.6)
ag, lg = np.arange(1, 79.01, 1.0), np.arange(1, 89.01, 1.0)
alt = np.interp(ag, ax, ay)
lat = np.interp(lg, lx, ly)
eq3 = 0.6252 * np.exp(-0.461 * np.cos(np.radians(2 * lg)) ** 2 - 0.94 * np.cos(np.radians(2 * lg)) + 0.252)
# Eq.(3) is Normand's own approximate fit to Fig.3 (Normand & Baker 1993 Fig.3 shows residual deviations),
# so it is reported for context only. Digitization check: Normand & Baker 1993 state the 90/0 deg ratio is 5.5.
print("Fig.3 digitized: 1 deg %.3f, 89 deg %.3f, ratio %.2f (paper: 5.5); Eq.(3) ratio %.2f" %
      (lat[0], lat[-1], lat[-1] / lat[0], eq3[-1] / eq3[0]))
for k in (10, 20, 30, 35, 40, 45, 50, 60):
    print("  Fig.2 %2d kft: %.3f n/cm2/s" % (k, np.interp(k, ag, alt)))
d = ROOT / "data/step4"
pd.DataFrame({"alt_kft": ag, "flux_1to10_ncm2s": alt}).to_csv(d / "normand1996_fig2_digitized.csv", index=False, lineterminator="\n")
pd.DataFrame({"lat_deg": lg, "flux_1to10_ncm2s": lat, "eq3": eq3}).to_csv(d / "normand1996_fig3_digitized.csv", index=False, lineterminator="\n")
