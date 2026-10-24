# Minimum-data screening formula for the >10 MeV atmospheric neutron flux (n/cm2/h), free atmosphere.
#   ln F = b0 + b1 h + b2 h^2 - (1 + d h) ln(1 + a Rc^n) + c W / (1 + e Rc)
# h: altitude (km, US Standard Atmosphere 1976), Rc: vertical cutoff rigidity (GV), W: PARMA W-index.
# Coefficients are read from data/step4/formula_fit.json (fit_formula.py). Validity domain = training domain.
import json, pathlib, warnings
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOMAIN = {"alt_km": (6.0, 15.0), "Rc_GV": (0.0, 18.0), "W": (0.0, 150.0)}


def load_params(path=ROOT / "data/step4/formula_fit.json"):
    fit = json.load(open(path))
    return np.array(fit["candidates"][fit["selected"]]["params"])


def flux_gt10(h, rc, w, params=None, strict=False):
    """>10 MeV neutron flux (n/cm2/h). Warns (or raises if strict) outside the fitted domain."""
    p = load_params() if params is None else np.asarray(params)
    h, rc, w = (np.asarray(x, float) for x in (h, rc, w))
    for name, x in (("alt_km", h), ("Rc_GV", rc), ("W", w)):
        lo, hi = DOMAIN[name]
        if np.any((x < lo) | (x > hi)):
            msg = f"{name} outside fitted domain [{lo}, {hi}]"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)
    b0, b1, b2, a, n, c, d, e = p
    return np.exp(b0 + b1 * h + b2 * h * h - (1 + d * h) * np.log1p(a * rc ** n) + c * w / (1 + e * rc))
